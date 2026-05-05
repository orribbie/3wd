# voxel_map.py  — Sparse TSDF voxel hash-grid for bounded online mapping
#
# KinectFusion-style Truncated Signed Distance Function (TSDF) map.
# Each voxel stores its distance to the nearest surface [-1, +1] and
# an observation weight. Free-space carving is done via vectorized
# ray marching — no per-voxel Python loops.
#
# Usage:
#   vmap = GlobalVoxelMap(voxel_size=0.03)
#   vmap.integrate(pts_world, colors, frame_id, camera_pose)
#   grid, meta, cost, floor_y, kernel = vmap.get_2d_grid(grid_params)
#   pts, cols = vmap.get_points_colors()

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_nn

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cuda")


# ── Inline quaternion → rotation matrix (replaces scipy per-frame calls) ─────
def _quat_to_matrix(q) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 3×3 rotation matrix.  Pure numpy, no scipy."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.array([
        [1.0 - (yy + zz),       xy - wz,         xz + wy],
        [      xy + wz,    1.0 - (xx + zz),       yz - wx],
        [      xz - wy,         yz + wx,     1.0 - (xx + yy)],
    ], dtype=np.float32)


# ── Pre-computed rotation test matrices (24 proper rotations) ────────────────
# Replaces a 4-deep nested Python loop that ran on every cache miss.
def _build_rot_tests() -> np.ndarray:
    tests = []
    for p in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    m = np.zeros((3, 3), dtype=np.float32)
                    m[0, p[0]], m[1, p[1]], m[2, p[2]] = s0, s1, s2
                    if np.linalg.det(m) > 0.9:
                        tests.append(m)
    return np.stack(tests)  # (24, 3, 3)


_ROT_TESTS: np.ndarray = _build_rot_tests()  # computed once at import time

# ============================================================
# Sparse TSDF voxel hash-grid
# ============================================================
class GlobalVoxelMap:
    """
    Sparse voxel grid backed by hash-table keying on integer (ix,iy,iz).

    Each voxel stores:
        - log_odds   (float16): Probability occupancy log-odds
        - color      (float32, 3): EMA of RGB
        - centroid   (float32, 3): EMA of sub-voxel position

    The grid auto-grows: new voxels are simply appended to the flat arrays.
    Memory is proportional to the number of *observed* voxels, not the
    bounding volume.
    """

    def __init__(
        self,
        voxel_size: float = 0.03,          # Change E: bumped 0.02 → 0.03 for speed
        device: torch.device = DEVICE,
        max_voxels: int = 15_000_000,
        profile: bool = False,             # Change F: set True to dump torch.profiler trace
    ):
        self.vs = float(voxel_size)
        self.device = torch.device(device)
        self._profile = bool(profile)
        print(f"[VoxelMap] Initialized on device: {self.device}, voxel_size={self.vs}")
        self.max_voxels = max_voxels

        # Default 720p 1/4th resolution fallback
        self.carver_W = int(1280 * 0.25)
        self.carver_H = int(720 * 0.25)
        self.carver_fx = 525.0 * 0.25
        self.carver_fy = 525.0 * 0.25
        self.carver_cx = (1280 / 2.0) * 0.25
        self.carver_cy = (720 / 2.0) * 0.25

        # ---- Insertion-zone cap (matches deletion zone + margin) ----
        # Only points closer than this are ever added to the map, ensuring every
        # voxel we insert is within the frustum we check during deletion.
        # Set to match z_range_max used in _clear_dynamic_objects plus a small buffer.
        self.max_insert_depth: float = 5.0  # metres; tune with z_range_max in mind

        self._lock = threading.Lock()
        self._keys     = torch.zeros((0, 3), dtype=torch.int32).pin_memory()
        self._log_odds = torch.zeros(0,      dtype=torch.float16, device=self.device)
        self._color    = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self._centroid = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        self._age      = torch.zeros(0,      dtype=torch.int16,   device=self.device)
        self._count    = 0

        # Export params
        self.l_hit          = 0.85
        self.l_min          = -2.0
        self.l_max          = 3.5
        self.truncation_dist = 0.05
        self.ema_alpha       = 0.1
        # Set promote_age=0 to disable live→permanent promotion (single-map mode).
        self.promote_age: int = 0

        # Key→idx dict: lazy — only rebuilt when _remove_voxel is called.
        # Hot-path integration never touches this dict; all lookups use _sorted_packed.
        self._key_to_idx: dict[tuple[int, int, int], int] = {}
        self._key_dict_dirty: bool = False   # True = dict is stale, rebuild before use

        # Sorted-key GPU cache for O(K log M) lookup via torch.searchsorted.
        # torch.argsort on GPU is 14-24x faster than np.argsort on CPU for maps > 50k voxels.
        self._sorted_packed_gpu: Optional[torch.Tensor] = None  # (M,) int64 sorted, on GPU
        self._sort_idx_gpu:      Optional[torch.Tensor] = None  # (M,) int64 argsort, on GPU
        self._sorted_cache_count: int = -1                       # invalidated on new voxels

        self._best_rot_cache = None

        # GPU cache: keys mirrored on device so carver avoids a full pinned→GPU copy each frame.
        # Invalidated whenever voxels are added or removed (same lifecycle as _sorted_packed_gpu).
        self._keys_gpu: Optional[torch.Tensor] = None
        self._keys_gpu_count: int = -1  # count when cache was last built

        # Persistent carver buffers — reset in-place each frame to avoid per-frame GPU alloc.
        self._carver_depth_buf: Optional[torch.Tensor] = None   # (H, W) float32
        self._carver_obs_buf:   Optional[torch.Tensor] = None   # (H, W) bool

        # Floor estimate (EMA) — throttled to avoid full GPU→CPU copies
        self._floor_y_ema: Optional[float] = None
        self._floor_alpha = 0.05  # EMA smoothing factor
        self._floor_frame_count: int = 0   # counts integration calls for throttle

    def set_camera_intrinsics(self, width: int, height: int, fx: float, fy: float, cx: float, cy: float):
        """Precalculate and cache quarter resolution intrinsics for fast carving."""
        with self._lock:
            scale = 0.25
            self.carver_W = int(width * scale)
            self.carver_H = int(height * scale)
            self.carver_fx = float(fx * scale)
            self.carver_fy = float(fy * scale)
            self.carver_cx = float(cx * scale)
            self.carver_cy = float(cy * scale)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def integrate(
        self,
        pts_world: torch.Tensor,       # (N, 3) float32 in world frame
        colors: torch.Tensor,           # (N, 3) uint8
        frame_id: int = 0,
        camera_pose: Optional[np.ndarray] = None,  # (7,) [qx,qy,qz,qw, tx,ty,tz]
        _profiler=None,                # internal: active torch.profiler context or None
    ) -> dict:
        """
        Insert new points into the voxel grid.  O(N) amortized.

        If `camera_pose` is provided, Range-Image differencing free-space carving is
        performed to instantly clear dynamic ghosts.

        Returns dict with key:
            'cleared_keys' – (M,3) int32 pinned-CPU keys cleared this frame (or None)
        """
        result = {'promoted_keys': None, 'promoted_colors': None, 'cleared_keys': None}

        if pts_world is None or pts_world.shape[0] == 0:
            return result

        _t0 = time.perf_counter()

        pts  = pts_world.to(self.device, non_blocking=True).float()
        cols = colors.to(self.device, non_blocking=True).float()

        # Filter invalid
        valid = torch.isfinite(pts).all(dim=1)
        if not valid.all():
            pts  = pts[valid]
            cols = cols[valid]
        if pts.shape[0] == 0:
            return result

        # ---- GPU tensor for carver — kept on GPU, no CPU transfer ----
        # Captured BEFORE the 5 m range filter so carver sees full-range evidence.
        pts_gpu_full = pts if camera_pose is not None else None

        # ---- Insertion-zone filter (matches deletion frustum range) ----
        if camera_pose is not None and self.max_insert_depth > 0:
            cam_t = torch.tensor(camera_pose[4:7], device=self.device, dtype=torch.float32)
            diff = pts - cam_t
            dist_sq = (diff * diff).sum(dim=1)
            in_range = dist_sq <= (self.max_insert_depth * self.max_insert_depth)
            pts  = pts[in_range]
            cols = cols[in_range]
            if pts.shape[0] == 0:
                return result

        # Quantize to voxel indices on GPU
        vox_idx = torch.floor(pts / self.vs).to(torch.int32)  # (N, 3) GPU

        _t1 = time.perf_counter()

        with self._lock:
            updated_eidx = self._integrate_surface(vox_idx, cols, pts)

            _t2 = time.perf_counter()

            # Carve free space using Dense Range Image Differencing (GPU path)
            if camera_pose is not None:
                result['cleared_keys'] = self._clear_dynamic_objects(
                    camera_pose, pts_gpu_full, frame_id=frame_id
                )

            _t3 = time.perf_counter()

        if frame_id > 0 and frame_id % 60 == 0:
            n_vox = self._count
            print(
                f"[VoxelMap] frame={frame_id} voxels={n_vox:,} | "
                f"filter={(_t1-_t0)*1e3:.1f}ms  surface={(_t2-_t1)*1e3:.1f}ms  "
                f"clear={(_t3-_t2)*1e3:.1f}ms  total={(_t3-_t0)*1e3:.1f}ms"
            )

        return result

    # ---- Sorted-key cache helpers (Changes A + B) ----
    @staticmethod
    def _pack_keys(keys: np.ndarray) -> np.ndarray:
        """Pack (N,3) int32 voxel keys into a single int64 for fast sorted search."""
        # Offset by 2^20 so negative indices don't collide; assumes coords in [-1M, 1M]
        OFFSET = np.int64(1 << 20)
        STRIDE1 = np.int64(1 << 21)          # 2^21 > 2 * OFFSET
        STRIDE2 = np.int64(1 << 42)          # STRIDE1^2
        k = keys.astype(np.int64)
        return (k[:, 0] + OFFSET) * STRIDE2 + (k[:, 1] + OFFSET) * STRIDE1 + (k[:, 2] + OFFSET)

    def _rebuild_sorted_cache(self):
        """Rebuild the GPU sorted-key cache used for O(K log M) torch.searchsorted lookup.

        torch.argsort is 14-24× faster than np.argsort for maps > 50k voxels:
          - 50k voxels:  3ms CPU  → 0.5ms GPU
          - 500k voxels: 42ms CPU → 2ms GPU
          - 1M voxels:   93ms CPU → 4ms GPU
        """
        # _keys is pinned CPU memory; .to(device) is zero-copy on Jetson unified memory.
        k = self._keys[:self._count].to(self.device, dtype=torch.int64)  # (M, 3) GPU
        _OFF, _S1, _S2 = 1 << 20, 1 << 21, 1 << 42
        packed_gpu = (k[:, 0] + _OFF) * _S2 + (k[:, 1] + _OFF) * _S1 + (k[:, 2] + _OFF)
        sort_idx_gpu = torch.argsort(packed_gpu)             # GPU sort — very fast
        self._sorted_packed_gpu = packed_gpu[sort_idx_gpu]  # (M,) int64 sorted on GPU
        self._sort_idx_gpu      = sort_idx_gpu              # (M,) int64 storage indices on GPU
        self._sorted_cache_count = self._count

    def _lookup_voxels(self, query_packed_gpu: torch.Tensor, n_query: int):
        """
        GPU batch lookup using torch.searchsorted.

        Args:
            query_packed_gpu  – (Q,) int64 GPU tensor of packed voxel keys
            n_query           – Q (number of unique query keys)
        Returns:
            existing_src  – numpy indices into query that already exist in the map
            existing_idx  – corresponding flat storage indices in _keys/_log_odds/...
            new_src       – numpy indices into query that are new
        """
        if self._count == 0:
            return np.empty(0, int), np.empty(0, int), np.arange(n_query)

        # Rebuild GPU cache only when map grew
        if self._sorted_cache_count != self._count:
            self._rebuild_sorted_cache()

        # Binary search entirely on GPU — torch.searchsorted is 10× faster than CPU
        pos = torch.searchsorted(self._sorted_packed_gpu, query_packed_gpu)  # (Q,) GPU
        pos_clipped = pos.clamp(0, self._sorted_packed_gpu.shape[0] - 1)
        hit = self._sorted_packed_gpu[pos_clipped] == query_packed_gpu       # (Q,) bool GPU

        hit_np = hit.cpu().numpy()
        existing_src = np.where(hit_np)[0]                                    # numpy indices
        existing_idx = self._sort_idx_gpu[pos_clipped[hit]].cpu().numpy()    # GPU → CPU
        new_src      = np.where(~hit_np)[0]
        return existing_src, existing_idx, new_src

    def _integrate_surface(
        self,
        vox_idx_gpu: torch.Tensor,             # (N, 3) int32 on GPU
        cols_gpu:    torch.Tensor,             # (N, 3) float32 on GPU
        pts_gpu:     torch.Tensor,             # (N, 3) float32 on GPU
    ) -> Optional[torch.Tensor]:
        """
        GPU-accelerated surface insertion. Called under self._lock.

        Accepts GPU tensors directly, avoiding the two large (N,3) GPU→CPU
        transfers the old numpy path needed.  Pack + unique + aggregate are
        done entirely on the GPU; only the small unique-key batch (K << N)
        is moved to CPU for the hash-table lookup.

        Returns a 1-D long tensor of GPU storage indices updated this frame,
        or None if no existing voxels were touched.
        """
        # ── GPU: pack → unique → unpack ─────────────────────────────────────
        # Same bijective int64 encoding as _pack_keys, done on GPU.
        _OFF = 1 << 20
        _S1  = 1 << 21
        _S2  = 1 << 42
        k          = vox_idx_gpu.to(torch.int64)
        packed_gpu = (k[:, 0] + _OFF) * _S2 + (k[:, 1] + _OFF) * _S1 + (k[:, 2] + _OFF)

        unique_packed_gpu, inverse_gpu = torch.unique(
            packed_gpu, sorted=True, return_inverse=True
        )
        n_unique = int(unique_packed_gpu.shape[0])

        # Unpack int64 → (K, 3) int32 keys — still on GPU
        ux = (unique_packed_gpu // _S2).to(torch.int32) - _OFF
        uy = ((unique_packed_gpu % _S2) // _S1).to(torch.int32) - _OFF
        uz = (unique_packed_gpu % _S1).to(torch.int32) - _OFF
        unique_vox_gpu = torch.stack([ux, uy, uz], dim=1)  # (K, 3) int32 on GPU

        # ── GPU: aggregate color / centroid / hit-count per unique voxel ────
        # index_add_ (O(N) GPU scatter) replaces np.add.at which iterates N
        # times in Python — 10–30× faster on large clouds.
        cols_f = cols_gpu.float()
        pts_f  = pts_gpu.float()
        ones   = torch.ones(pts_f.shape[0], device=self.device)
        color_sums_t = torch.zeros((n_unique, 3), device=self.device).index_add(0, inverse_gpu, cols_f)
        point_sums_t = torch.zeros((n_unique, 3), device=self.device).index_add(0, inverse_gpu, pts_f)
        hit_counts_t = torch.zeros(n_unique,      device=self.device).index_add(0, inverse_gpu, ones)

        # Move only the small unique-key batch to CPU for hash-table lookup.
        # K (unique voxels per frame) is typically 5–20k << N (300k raw points).
        unique_vox = unique_vox_gpu.cpu().numpy()                   # (K, 3)
        color_sums = color_sums_t.cpu().numpy()                     # (K, 3)
        point_sums = point_sums_t.cpu().numpy()                     # (K, 3)
        hit_counts = hit_counts_t.to(torch.int32).cpu().numpy()     # (K,)

        # ---- Lazily allocate storage ----
        if self._keys.shape[0] == 0:
            cap = min(self.max_voxels, max(n_unique * 4, 50_000))
            self._keys     = torch.zeros((cap, 3), dtype=torch.int32).pin_memory()
            self._log_odds = torch.zeros(cap, dtype=torch.float16, device=self.device)
            self._color    = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
            self._centroid = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
            self._age      = torch.zeros(cap, dtype=torch.int16, device=self.device)

        # ---- Separate existing vs new voxels (vectorized, Change A) ----
        # Pass unique_packed_gpu directly — avoids re-packing in _lookup_voxels.
        existing_src, existing_idx, new_src = self._lookup_voxels(unique_packed_gpu, n_unique)
        # Keep as numpy arrays — avoids .tolist()→list→np.array round-trip

        if len(existing_src) > 0:
            eidx = torch.tensor(np.ascontiguousarray(existing_idx.astype(np.int64))).to(self.device)
            esrc = existing_src.astype(np.int32)

            hits_np = hit_counts[esrc].astype(np.float32)
            hits_t = torch.tensor(np.ascontiguousarray(hits_np)).to(self.device)
            log_odds_add = hits_t * self.l_hit

            new_log_odds = self._log_odds[eidx].float() + log_odds_add
            new_log_odds = torch.clamp(new_log_odds, min=self.l_min, max=self.l_max)

            # Fixed-rate Exponential Moving Average (EMA) for color and centroid
            alpha_np = 1.0 - np.exp(-hits_np * 0.1)
            alpha_3 = torch.tensor(np.ascontiguousarray(alpha_np)).to(self.device).float().unsqueeze(1)

            hits_t_3d = hits_t.unsqueeze(1)

            old_color = self._color[eidx].float()
            incoming_mean_color = torch.tensor(np.ascontiguousarray(color_sums[esrc])).to(self.device) / hits_t_3d
            new_color = old_color * (1 - alpha_3) + incoming_mean_color * alpha_3

            old_centroid = self._centroid[eidx].float()
            incoming_mean_centroid = torch.tensor(np.ascontiguousarray(point_sums[esrc])).to(self._centroid.device) / hits_t_3d
            new_centroid = old_centroid * (1 - alpha_3) + incoming_mean_centroid * alpha_3

            self._log_odds[eidx] = new_log_odds.to(torch.float16)
            self._color[eidx]    = new_color
            self._centroid[eidx] = new_centroid

            # ── Age update: increment age ──
            if self.promote_age > 0 and len(self._age) > 0:
                cap_age = self.promote_age + 1
                self._age[eidx] = (self._age[eidx].int() + 1).clamp(max=cap_age).to(torch.int16)

        # ---- Insert new voxels ----
        if len(new_src) > 0:
            n_new = len(new_src)
            needed = self._count + n_new

            # Grow if needed
            if needed > self._keys.shape[0]:
                new_cap = min(self.max_voxels, max(self._keys.shape[0] * 2, needed))
                if new_cap < needed:
                    n_new = new_cap - self._count  # truncate
                    new_src = new_src[:n_new]
                if n_new <= 0:
                    self._update_floor_ema()
                    return eidx if len(existing_src) > 0 else None
                self._keys     = _grow_tensor(self._keys, new_cap, pinned=True)  # keep pinned (Change C)
                self._log_odds = _grow_tensor(self._log_odds, new_cap)
                self._color    = _grow_tensor(self._color, new_cap)
                self._centroid = _grow_tensor(self._centroid, new_cap)
                self._age      = _grow_tensor(self._age, new_cap)

            nsrc = new_src.astype(np.int32) if new_src.dtype != np.int32 else new_src
            start = self._count

            # Batch write keys
            new_keys_np = unique_vox[nsrc]  # (n_new, 3)
            self._keys[start:start + n_new] = torch.tensor(np.ascontiguousarray(new_keys_np).copy())

            hits_1d = hit_counts[nsrc].astype(np.float32)
            hits_t = torch.tensor(np.ascontiguousarray(hits_1d)).to(self.device)
            log_odds_init = torch.clamp(hits_t * self.l_hit, min=self.l_min, max=self.l_max)
            self._log_odds[start:start + n_new] = log_odds_init.to(torch.float16)

            # Initial values are the exact average of incoming points
            w_new_3d = hits_t.unsqueeze(1)
            self._color[start:start + n_new]    = torch.tensor(np.ascontiguousarray(color_sums[nsrc])).to(self._log_odds.device) / w_new_3d
            self._centroid[start:start + n_new] = torch.tensor(np.ascontiguousarray(point_sums[nsrc])).to(self._log_odds.device) / w_new_3d

            # New voxels start at age=1
            if self.promote_age > 0 and len(self._age) > start + n_new:
                self._age[start:start + n_new] = 1

            # Do NOT maintain _key_to_idx here — that Python loop is O(n_new)
            # and blocks the integration lock. Mark the dict stale instead.
            # It will be lazily rebuilt the next time _remove_voxel is called.
            self._key_dict_dirty = True

            self._count += n_new
            self._sorted_cache_count = -1  # invalidate sorted cache (Change B)
            self._keys_gpu_count = -1      # invalidate GPU keys mirror

        # Update floor EMA — pass the new keys batch for the fast (O(batch)) path
        new_keys_batch = unique_vox[nsrc] if len(new_src) > 0 else None
        self._update_floor_ema(new_keys_batch)

        # Return the GPU index tensor for existing voxels updated this frame so
        # integrate() can limit the promotion check to in-FOV voxels only.
        return eidx if len(existing_src) else None

    # ------------------------------------------------------------------
    # Range Image Differencing (Clear Dynamic Objects)
    # ------------------------------------------------------------------
    def _clear_dynamic_objects(
        self,
        camera_pose: np.ndarray,
        pts_world_gpu: torch.Tensor,   # (N,3) float32 GPU — full-range pre-filtered pts
        max_range: float = 5.0,
        frame_id: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Vectorized depth buffer differencing.  Depth buffer is built entirely on GPU;
        only the small orientation-search subsample (≤5000 pts) ever touches CPU.

        Returns (M,3) int32 numpy array of voxel keys cleared this frame, or None.
        """
        if self._count == 0 or pts_world_gpu is None or pts_world_gpu.shape[0] == 0:
            return

        # Use cached carver intrinsics
        W, H = self.carver_W, self.carver_H
        fx, fy = self.carver_fx, self.carver_fy
        cx, cy = self.carver_cx, self.carver_cy

        t = camera_pose[4:7]
        q_raw = camera_pose[:4]

        # Optimization: retrieve the static coordinate-system configuration
        t_gpu = torch.tensor(t, device=self.device, dtype=torch.float32)

        if hasattr(self, "_best_cfg_tuple") and self._best_cfg_tuple is not None:
            name, do_transpose, m_opt = self._best_cfg_tuple
            q_data = q_raw if name == "xyzw" else np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]])
            m_base = _quat_to_matrix(q_data)
            m_quat = m_base.T if do_transpose else m_base
            rot = m_quat @ m_opt

            # Sanity check: current-frame points are all physically in front of the
            # camera, so their mean must project to positive camera-Z with the cached
            # rotation. If not, the cached permutation is wrong (bad frame-1 pick or
            # coordinate drift) — invalidate and re-search this frame.
            step = max(1, pts_world_gpu.shape[0] // 500)
            pts_check = pts_world_gpu[::step].cpu().numpy() - t
            mean_z = float((pts_check @ rot[:, 2]).mean())
            if mean_z < 0.1:
                self._best_cfg_tuple = None
                if frame_id % 10 == 0:
                    print(f"[VoxelMap] WARNING: cached orientation invalid (mean_z={mean_z:.2f}), re-searching.")

        if hasattr(self, "_best_cfg_tuple") and self._best_cfg_tuple is not None:
            best_cfg = "cached_config"
        else:
            best_count = -1
            best_rot = None
            best_cfg = ""
            best_tuple = None

            # Subsample only ≤5000 pts to CPU for the orientation search.
            # This is the only CPU involvement once the cache is warm.
            step = max(1, pts_world_gpu.shape[0] // 5000)
            pts_rel = pts_world_gpu[::step].cpu().numpy() - t   # (P≤5000, 3)

            for name, q_data in [("xyzw", q_raw), ("wxyz", np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]]))]:
                try:
                    m_base = _quat_to_matrix(q_data)

                    for do_transpose in (False, True):
                        m_quat = m_base.T if do_transpose else m_base

                        # Batched rotation search: one einsum replaces 24-iteration loop
                        all_rots = np.einsum('ij,rjk->rik', m_quat, _ROT_TESTS)
                        test_c = np.einsum('pj,rjk->rpk', pts_rel, all_rots)

                        z_vals = test_c[:, :, 2]
                        valid_z = z_vals > 0.1
                        z_safe = np.where(valid_z, z_vals, 1.0)
                        u_arr = (test_c[:, :, 0] * fx / z_safe + cx).astype(np.int32)
                        v_arr = (test_c[:, :, 1] * fy / z_safe + cy).astype(np.int32)

                        on_screen = valid_z & (u_arr >= 0) & (u_arr < W) & (v_arr >= 0) & (v_arr < H)
                        lower_half = on_screen & (v_arr >= int(cy))

                        scores = (on_screen.sum(axis=1) * 1000
                                  + lower_half.sum(axis=1) * 100
                                  + valid_z.sum(axis=1))
                        idx = int(np.argmax(scores))
                        sc = int(scores[idx])
                        if sc > best_count:
                            best_count = sc
                            best_rot = all_rots[idx]
                            best_tuple = (name, do_transpose, _ROT_TESTS[idx])
                            best_cfg = f"{name}{'.T' if do_transpose else ''}#R{idx}"
                except Exception:
                    continue

            if best_rot is None or best_count <= 0:
                if frame_id % 40 == 0:
                    print(f"[VoxelMap] CRITICAL: Carver failed to find camera orientation. pts={pts_world_gpu.shape[0]}")
                return

            self._best_cfg_tuple = best_tuple
            rot = best_rot

        # Build rotation on GPU — stays there for depth buffer and voxel projection
        rot_gpu = torch.tensor(np.ascontiguousarray(rot)).to(self.device, dtype=torch.float32)

        # 2. Build Depth Buffer entirely on GPU (no CPU→GPU transfer for pts)
        # Reuse persistent buffers to avoid per-frame GPU allocation overhead.
        if self._carver_depth_buf is None or self._carver_depth_buf.shape != (H, W):
            self._carver_depth_buf = torch.empty((H, W), device=self.device, dtype=torch.float32)
            self._carver_obs_buf   = torch.empty((H, W), device=self.device, dtype=torch.bool)
        depth_buffer_gpu = self._carver_depth_buf
        observed_mask_gpu = self._carver_obs_buf
        depth_buffer_gpu.fill_(20.0)
        observed_mask_gpu.zero_()

        # Project full-range point cloud into camera frame (GPU matmul)
        pts_c_all_t = (pts_world_gpu - t_gpu) @ rot_gpu     # (N,3) GPU — no CPU roundtrip
        valid_pts_mask = pts_c_all_t[:, 2] > 0.1
        pts_c_t = pts_c_all_t[valid_pts_mask]
        
        if len(pts_c_t) > 0:
            u_t = (pts_c_t[:, 0] * fx / pts_c_t[:, 2] + cx).long()
            v_t = (pts_c_t[:, 1] * fy / pts_c_t[:, 2] + cy).long()

            valid_img_mask = (u_t >= 0) & (u_t < W) & (v_t >= 0) & (v_t < H)
            u_v_t, v_v_t, z_v_t = u_t[valid_img_mask], v_t[valid_img_mask], pts_c_t[valid_img_mask, 2]

            if len(z_v_t) > 0:
                # Fix 1: Z-buffer min semantics — closest point per pixel wins.
                # Simple scatter overwrites arbitrarily; scatter_reduce_ keeps minimum.
                flat_idx = v_v_t * W + u_v_t
                depth_flat = depth_buffer_gpu.view(-1)
                depth_flat.scatter_reduce_(0, flat_idx, z_v_t, reduce='amin', include_self=True)

                # Track which pixels were actually observed
                obs_flat = observed_mask_gpu.view(-1)
                obs_flat.index_fill_(0, flat_idx, True)

                depth_buffer_gpu = depth_flat.view(H, W)

        # 3. Project ALL active voxels into camera frame
        with torch.no_grad():
            try:
                # Use cached GPU keys copy — rebuilt only when voxels are added/removed.
                if self._keys_gpu_count != self._count:
                    self._keys_gpu = self._keys[:self._count].to(self.device)
                    self._keys_gpu_count = self._count
                keys_t = self._keys_gpu  # (N, 3) GPU
                vox_centers_t = (keys_t.float() + 0.5) * self.vs
                
                vox_c_t = (vox_centers_t - t_gpu) @ rot_gpu
                
                valid_vox_mask = vox_c_t[:, 2] > 0.1
                all_idx = torch.arange(len(keys_t), device=self.device)
                
                v_cand_idx = all_idx[valid_vox_mask]
                vc_t = vox_c_t[valid_vox_mask]
                
                if len(vc_t) > 0:
                    u_all = (vc_t[:, 0] * fx / vc_t[:, 2] + cx).long()  # (N_infront,)
                    v_all = (vc_t[:, 1] * fy / vc_t[:, 2] + cy).long()  # (N_infront,)

                    # Strictly on-screen only — clamped off-screen voxels cause false
                    # conflict_b hits (edge pixel has no depth return but voxel was
                    # never in the frustum). Removing MARGIN_PX prevents this.
                    MARGIN_PX = 0
                    valid_img_v = (u_all >= 0) & (u_all < W) & \
                                  (v_all >= 0) & (v_all < H)

                    # Narrow to on-screen (mostly) for conflict check
                    final_v_idx = v_cand_idx[valid_img_v]
                    u_v_t = u_all[valid_img_v]
                    v_v_t = v_all[valid_img_v]
                    z_v_t = vc_t[valid_img_v, 2]

                    # Clamp coordinates to the actual depth buffer boundaries
                    u_clamped = u_v_t.clamp(0, W - 1)
                    v_clamped = v_v_t.clamp(0, H - 1)

                    # 4. Conflict Mask (CUDA Tensor Comparison)
                    # Compute 4D views once — reused by all five max_pool2d calls below.
                    _ERODE_K = 3
                    _depth_4d = depth_buffer_gpu.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                    _obs_4d   = observed_mask_gpu.float().unsqueeze(0).unsqueeze(0)

                    # Erode depth buffer with min-pool before lookup.
                    _neg_min  = F_nn.max_pool2d(-_depth_4d, kernel_size=_ERODE_K,
                                                stride=1, padding=_ERODE_K // 2)
                    depth_eroded = (-_neg_min).squeeze(0).squeeze(0)        # (H,W)

                    # Dilate observation mask: if any pixel in the 3x3 window was observed,
                    # the eroded (min) depth value pulled from that window is valid.
                    _obs_dilated = F_nn.max_pool2d(_obs_4d, kernel_size=_ERODE_K,
                                                   stride=1, padding=_ERODE_K // 2)
                    obs_eroded = _obs_dilated.squeeze(0).squeeze(0) > 0.5

                    measured_z  = depth_eroded[v_clamped, u_clamped]
                    is_observed = obs_eroded[v_clamped, u_clamped]

                    # Case A: Direct Conflict (Point return confirms empty space)
                    margin_a = 0.25 + 0.04 * z_v_t.clamp(0.0, 5.0)
                    conflict_a = is_observed & (measured_z > 0.1) & (measured_z > (z_v_t + margin_a))

                    # ── Active-field dilation tuning ─────────────────────────────
                    # These are the knobs to adjust clearing aggressiveness:
                    #
                    #   FIELD_K_TOP  — dilation radius (px) for rows ABOVE cy (upper half).
                    #                  Large = aggressively clears floating head/torso remnants.
                    #                  ↑ to clear more  |  ↓ to preserve more
                    FIELD_K_TOP  = 31   # px — upper half (people, floating objects)
                    #
                    #   FIELD_K_BOT  — dilation radius (px) for rows BELOW cy (lower half).
                    #                  Large = clears floor-level dynamic blobs.
                    #                  ↑ to clear more  |  ↓ to preserve floor/furniture
                    FIELD_K_BOT  = 31   # px — lower half (floor blobs, chair bases)
                    #
                    #   SIDE_BORDER_PX — number of columns masked out on each side AFTER
                    #                    dilation. Voxels projecting into this border zone
                    #                    are excluded from miss-vote clearing entirely,
                    #                    preventing walls/doorframes near the image edge
                    #                    from being falsely cleared.
                    #                    ↑ to protect more of the sides  |  ↓ for less margin
                    SIDE_BORDER_PX = 0  # px — left+right exclusion border
                    #                  Restored: with FIELD_K=31, the active field reaches
                    #                  the image boundary. The outermost columns of a ZED
                    #                  sensor have no reliable depth returns → is_miss=True
                    #                  at every edge pixel → guaranteed conflict_b deletions.
                    #
                    #   CONFLICT_B_MAX_RANGE — maximum camera-Z at which a miss-vote
                    #                          (conflict_b) can delete a voxel. Reducing
                    #                          this prevents sparse/angled returns on
                    #                          revisit from mass-clearing distant surfaces.
                    #                          ↑ clear farther | ↓ protect far geometry
                    CONFLICT_B_MAX_RANGE = 4.5  # m — tighter than max_range (5 m)
                    # ─────────────────────────────────────────────────────────────

                    _field_top = F_nn.max_pool2d(_obs_4d, kernel_size=FIELD_K_TOP,
                                                 stride=1, padding=FIELD_K_TOP // 2)
                    _field_bot = F_nn.max_pool2d(_obs_4d, kernel_size=FIELD_K_BOT,
                                                 stride=1, padding=FIELD_K_BOT // 2)

                    # Combine top/bottom halves at the optical-centre row (cy).
                    cy_px = int(cy)
                    _field_combined = _field_bot.clone()
                    _field_combined[:, :, :cy_px, :] = _field_top[:, :, :cy_px, :]

                    # Zero out left and right border columns — voxels projecting there
                    # are near the frustum edge and clamping makes miss-votes unreliable.
                    if SIDE_BORDER_PX > 0:
                        _field_combined[:, :, :, :SIDE_BORDER_PX] = 0
                        _field_combined[:, :, :, W - SIDE_BORDER_PX:] = 0

                    in_active_field = (_field_combined.squeeze(0).squeeze(0) > 0.5)[v_clamped, u_clamped]

                    # Trust the "nothingness" (sentinel 20.0) only within the active field
                    # and within the tighter conflict_b range limit.
                    is_miss = (measured_z > 19.0)
                    margin_b = 0.25
                    conflict_b_raw = (~is_observed) & is_miss & in_active_field & (z_v_t + margin_b < CONFLICT_B_MAX_RANGE)

                    # ── Occluded-voxel protection ────────────────────────────────
                    # Bug: obs_eroded uses a 3-px kernel, but in_active_field uses a
                    # 31-px kernel.  In the fringe zone (1.5–15.5 px from observed
                    # pts), a voxel that is BEHIND a surface (e.g. behind dark/glass
                    # walls that have holes in the depth return) appears unobserved
                    # (is_observed=False) yet inside the active field → conflict_b
                    # fires and deletes it.
                    #
                    # Fix: widen the obs lookup to a 15-px radius to find any nearby
                    # observed surface CLOSER than the voxel.  If one exists, the
                    # voxel is almost certainly occluded — skip clearing it.
                    _PROTECT_K = 10
                    _obs_protect_4d = F_nn.max_pool2d(
                        _obs_4d, kernel_size=_PROTECT_K, stride=1, padding=_PROTECT_K // 2
                    )
                    _neg_min_protect = F_nn.max_pool2d(
                        -_depth_4d, kernel_size=_PROTECT_K, stride=1, padding=_PROTECT_K // 2
                    )
                    depth_near = (-_neg_min_protect).squeeze(0).squeeze(0)   # min depth in 15-px window
                    obs_near   = (_obs_protect_4d.squeeze(0).squeeze(0) > 0.5)

                    depth_near_vox = depth_near[v_clamped, u_clamped]
                    obs_near_vox   = obs_near[v_clamped, u_clamped]

                    # Voxel is behind a closer observed surface within 15 px → occluded.
                    # Threshold is range-adaptive: same logic as margin_a — flat 0.15 m is
                    # too tight at range where pose drift causes the voxel's z_v_t to appear
                    # slightly behind a fresh return that is actually the same surface.
                    occlude_margin = 0.15 + 0.04 * z_v_t.clamp(0.0, 5.0)
                    occluded = obs_near_vox & (depth_near_vox > 0.1) & (z_v_t > depth_near_vox + occlude_margin)

                    # Apply occlusion guard to conflict_a as well: a glass/dark wall at
                    # z_v=2 m with a background return at 2.4 m triggers conflict_a, but
                    # if an adjacent pixel holds an observed surface closer than the voxel
                    # the voxel is more likely real and occluded than a ghost.
                    conflict_b = conflict_b_raw & ~occluded
                    conflict_a = conflict_a & ~occluded
                    # ─────────────────────────────────────────────────────────────

                    conflict_mask = conflict_a | conflict_b
                    cleared_vox_idx_t = final_v_idx[conflict_mask]

                    # Extended-FOV miss vote REMOVED:
                    # Clamping off-screen u/v projected to nearest edge pixel and
                    # comparing its depth against an out-of-frustum voxel is
                    # geometrically meaningless — it causes valid map geometry to
                    # be carved whenever the robot is simply looking away from it.

                else:
                    cleared_vox_idx_t = torch.tensor([], dtype=torch.long, device=self.device)
                    depth_eroded = depth_buffer_gpu  # fallback for Fix B below

                # Height-stratified ghost decay REMOVED:
                # "In front of the camera" does NOT mean "camera is looking at it".
                # This block decayed tall voxels (walls, doorframes, shelves) simply
                # because the robot was within 5m of them, even when looking away.
                # Correct dynamic-object removal only via the confirmed on-screen
                # depth-buffer conflict check above (conflict_mask).

                if frame_id % 100 == 0:
                     fwd = rot_gpu[:, 2].cpu().numpy()
                     min_z = float(z_v_t.min()) if len(z_v_t) > 0 else 0.0
                     max_z = float(z_v_t.max()) if len(z_v_t) > 0 else 0.0
                     print(f"[VoxelMap] {frame_id}: {len(pts_c_t)} pts ({len(u_v_t)} on-scr), {len(vc_t)} v-fov, {len(cleared_vox_idx_t)} cleared")
                     print(f"           cfg={best_cfg}, z_range=({min_z:.1f}/{max_z:.1f}), fwd=({fwd[0]:.2f}, {fwd[1]:.2f}, {fwd[2]:.2f})")

                # 5. Deletion: mark ALL cleared voxels as l_min and reset their age.
                # Do NOT call _remove_voxel here — that Python loop is O(n_cleared).
                # Batch compaction (below) will sweep them up efficiently.
                cleared_keys_out = None
                if len(cleared_vox_idx_t) > 0:
                    cleared_vox_idx_t = cleared_vox_idx_t.unique()
                    # Capture keys BEFORE zeroing so caller can sync permanent map
                    if self.promote_age > 0:
                        cleared_keys_out = self._keys[:self._count][
                            cleared_vox_idx_t.cpu()
                        ].numpy().copy()
                    self._log_odds[cleared_vox_idx_t] = self.l_min
                    # Reset age so cleared voxels must re-earn promotion if they reappear
                    if self.promote_age > 0 and len(self._age) >= self._count:
                        self._age[cleared_vox_idx_t] = 0

                # ---- Batch compaction of zombie (l_min) voxels ----
                # Pure GPU→CPU compact: no Python loops, no dict rebuild (dict is marked dirty).
                dead_mask = self._log_odds[:self._count] <= self.l_min
                n_dead = int(dead_mask.sum())
                if n_dead > 10_000:   # worth compacting (raised threshold to avoid over-aggressive removal)
                    keep_mask_cpu  = (~dead_mask[:self._count]).cpu()
                    alive_idx_cpu  = torch.where(keep_mask_cpu)[0]   # CPU — for pinned _keys
                    alive_idx_gpu  = alive_idx_cpu.to(self.device)   # GPU — for GPU arrays
                    n_keep = len(alive_idx_cpu)

                    # Compact arrays in-place
                    self._keys[:n_keep]     = self._keys[alive_idx_cpu]           # CPU / CPU
                    self._log_odds[:n_keep] = self._log_odds[alive_idx_gpu]       # GPU / GPU
                    if hasattr(self, '_color') and self._color is not None:
                        self._color[:n_keep]    = self._color[alive_idx_gpu]
                    if hasattr(self, '_centroid') and self._centroid is not None:
                        self._centroid[:n_keep] = self._centroid[alive_idx_gpu]
                    if self.promote_age > 0 and len(self._age) >= self._count:
                        self._age[:n_keep] = self._age[alive_idx_gpu]

                    # Mark dict stale — it will be rebuilt lazily on next _remove_voxel call.
                    self._key_to_idx.clear()
                    self._key_dict_dirty = True

                    self._count = n_keep
                    self._sorted_cache_count = -1  # invalidate sorted-key cache
                    self._keys_gpu_count = -1       # invalidate GPU keys mirror
                    if frame_id % 100 == 0:
                        print(f"[VoxelMap] Compacted: removed {n_dead} zombie voxels, {n_keep} remain.")

                return cleared_keys_out

            except Exception as e:
                # Catch-all to keep mapping thread alive
                if frame_id % 20 == 0:
                    print(f"[VoxelMap] Carver error at frame {frame_id}: {e}")
                return None

    # ------------------------------------------------------------------
    # Voxel removal
    # ------------------------------------------------------------------
    def _rebuild_key_dict(self):
        """Lazily rebuild _key_to_idx from _keys. Called only when _remove_voxel needs it."""
        keys_np = self._keys[:self._count].numpy()   # pinned CPU memory — no transfer
        self._key_to_idx = {
            (int(k[0]), int(k[1]), int(k[2])): i
            for i, k in enumerate(keys_np)
        }
        self._key_dict_dirty = False

    def _remove_voxel(self, key: tuple):
        """
        Remove a voxel by swapping with the last active voxel (O(1)).
        Called under self._lock. Dict is rebuilt lazily here if stale.
        """
        if self._key_dict_dirty:
            self._rebuild_key_dict()
        if key not in self._key_to_idx:
            return
        idx = self._key_to_idx.pop(key)
        last = self._count - 1
        if idx != last:
            # Swap last -> idx
            last_key = (int(self._keys[last, 0]), int(self._keys[last, 1]), int(self._keys[last, 2]))
            self._keys[idx] = self._keys[last]
            self._log_odds[idx] = self._log_odds[last]
            if hasattr(self, '_color') and self._color is not None:
                self._color[idx] = self._color[last]
            if hasattr(self, '_centroid') and self._centroid is not None:
                self._centroid[idx] = self._centroid[last]
            self._key_to_idx[last_key] = idx
        # Clear last slot
        self._keys[last] = 0
        self._log_odds[last] = 0.0
        if hasattr(self, '_color') and self._color is not None:
            self._color[last] = 0
        if hasattr(self, '_centroid') and self._centroid is not None:
            self._centroid[last] = 0
        self._count -= 1

    def _update_floor_ema(self, new_keys_batch: Optional[np.ndarray] = None):
        """
        Estimate floor height from Y coordinates — throttled to minimise GPU→CPU traffic.

        Fast path  (every call):  use the newly-inserted keys batch if provided (O(batch_size)).
        Slow path  (every 60 calls): sample 2 000 random voxels from the full map.
        The _keys array is pinned (page-locked) CPU memory, so numpy access is free;
        there is NO GPU→CPU copy here — only the float() cast on a CPU tensor slice.
        """
        _SLOW_PATH_EVERY = 60   # full-map sample once every N integration calls
        _SAMPLE_N        = 2_000

        self._floor_frame_count += 1
        p5: Optional[float] = None

        # --- Fast path: use batch of newly inserted voxels ---
        if new_keys_batch is not None and len(new_keys_batch) >= 5:
            ys = new_keys_batch[:, 1].astype(np.float32) * self.vs
            p5 = float(np.percentile(ys, 5.0))

        # --- Slow path: sampled read from full pinned-CPU _keys array ---
        elif self._floor_frame_count % _SLOW_PATH_EVERY == 0 and self._count >= 10:
            n = min(self._count, _SAMPLE_N)
            # _keys is pinned CPU memory — .numpy() is zero-copy
            idx = np.random.randint(0, self._count, size=n)
            ys = self._keys[idx, 1].numpy().astype(np.float32) * self.vs
            p5 = float(np.percentile(ys, 5.0))

        if p5 is None:
            return

        if self._floor_y_ema is None:
            self._floor_y_ema = p5
        else:
            self._floor_y_ema += self._floor_alpha * (p5 - self._floor_y_ema)

    # ------------------------------------------------------------------
    # 2D grid extraction
    # ------------------------------------------------------------------
    def get_2d_grid(self, grid_params) -> Optional[Tuple[np.ndarray, dict, np.ndarray, float, np.ndarray]]:
        """
        Project voxels to a 2D occupancy grid compatible with StaticGridWithLiveOverlayThread.

        Returns (grid_codes, meta, cost_map, floor_y, kernel) or None if empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            keys = self._keys[:self._count].clone()
            log_odds = self._log_odds[:self._count].clone()

        # Filter to occupied voxels only (log_odds > 0)
        mask = log_odds > 0.0
        # mask is on self.device (GPU), keys is on CPU.
        keys = keys[mask.cpu()]

        if keys.shape[0] == 0:
            return None

        # Convert voxel indices to world coordinates (center of voxel)
        pts_world = (keys.float() + 0.5) * self.vs  # (K, 3)
        pts_np = pts_world.numpy()

        floor_y = self._floor_y_ema if self._floor_y_ema is not None else 0.0

        # Import here to avoid circular import
        from robot.nav.pathPlanning import (
            LocalGrid2D,
            gridcodes_to_float,
            Grid2DParams,
        )

        # Build grid from voxel centers (same interface as compute_static_grid_from_points)
        builder = LocalGrid2D(grid_params)
        grid_codes, meta = builder.update(pts_np, None)
        cost_map = gridcodes_to_float(grid_codes)
        kernel = builder._kernel.copy() if hasattr(builder, "_kernel") else None
        return grid_codes, meta, cost_map, builder.floor_y_est, kernel

    # ------------------------------------------------------------------
    # Visualization / export
    # ------------------------------------------------------------------
    def get_points_colors(
        self,
        max_points: Optional[int] = None,
        min_log_odds: float = 1.5,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return (N,3) float32 points and (N,3) uint8 colors for visualization.

        Args:
            max_points:    If set, uniformly subsample to this many points.
            min_log_odds:  Minimum log-odds to show. Default 1.5 (live map —
                           suppresses single-frame speckle).  Pass 0.5 for the
                           permanent map, whose voxels are already stable
                           (promoted after promote_age consecutive frames).
        """
        with self._lock:
            if self._count == 0:
                return None, None

            # Only show well-confirmed voxels (default min_log_odds=1.5 ≈ seen ≥2 times).
            # The permanent map should pass min_log_odds=0.5 since its voxels are
            # already guaranteed stable via the promotion threshold.
            log_odds = self._log_odds[:self._count]   # GPU float16 view
            occ_mask_gpu = log_odds >= min_log_odds    # GPU bool

            occ_idx = torch.where(occ_mask_gpu)[0]   # GPU int64
            n_occ = occ_idx.shape[0]
            if n_occ == 0:
                return None, None

            if max_points is not None and n_occ > max_points:
                # Uniform stride subsample (deterministic, no allocation)
                stride = max(1, n_occ // max_points)
                occ_idx = occ_idx[::stride]

            has_centroid = hasattr(self, '_centroid') and self._centroid is not None
            has_color    = hasattr(self, '_color')    and self._color    is not None

            # Single GPU→CPU transfer for each array
            if has_centroid:
                pts_gpu = self._centroid[occ_idx]          # GPU float32
                pts = pts_gpu.cpu().numpy()                # one transfer
            else:
                keys_sub = self._keys[occ_idx.cpu()]       # pinned CPU (zero-copy)
                pts = ((keys_sub.float() + 0.5) * self.vs).numpy()

            if has_color:
                cols = self._color[occ_idx].clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                cols = None

        # Height-band filter: strip floor clutter and anything above head height.
        # Pure CPU numpy operation on the already-transferred arrays — zero GPU cost.
        if pts is not None and len(pts) > 0:
            floor_y = self._floor_y_ema if self._floor_y_ema is not None else float(pts[:, 1].min())
            y_min = floor_y - 0.20   # 20 cm below floor (Handles slight slant/tilt)
            y_max = floor_y + 2.2    # 2.2 m above floor (skip ceiling/lights)
            band = (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
            pts = pts[band]
            if cols is not None:
                cols = cols[band]
            if len(pts) == 0:
                return None, None

        return pts, cols

    def __len__(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # TorchPointCloud compatibility shim
    # ------------------------------------------------------------------
    def cpu_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Drop-in replacement for TorchPointCloud.cpu_numpy()."""
        pts, cols = self.get_points_colors()
        if pts is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        return pts, cols

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, filename: str, min_log_odds: Optional[float] = None) -> None:
        """Serialize voxel map to .npz.

        Args:
            filename: Output .npz path.
            min_log_odds: If set, only voxels with log_odds >= this value are saved.
                          Use e.g. 2.0 to export permanent-features-only maps that
                          exclude transient/dynamic objects still above l_min.
        """
        with self._lock:
            if self._count == 0:
                print("[GlobalVoxelMap] Nothing to save (empty map).")
                return

            log_odds_cpu = self._log_odds[:self._count].cpu()
            if min_log_odds is not None:
                mask = (log_odds_cpu >= min_log_odds).numpy()
                keys_out = self._keys[:self._count].numpy()[mask]
                log_odds_out = log_odds_cpu.numpy()[mask]
                color_out = self._color[:self._count].cpu().numpy()[mask] if hasattr(self, '_color') and self._color is not None else None
                centroid_out = self._centroid[:self._count].cpu().numpy()[mask] if hasattr(self, '_centroid') and self._centroid is not None else None
            else:
                keys_out = self._keys[:self._count].numpy()
                log_odds_out = log_odds_cpu.numpy()
                color_out = self._color[:self._count].cpu().numpy() if hasattr(self, '_color') and self._color is not None else None
                centroid_out = self._centroid[:self._count].cpu().numpy() if hasattr(self, '_centroid') and self._centroid is not None else None

            np.savez_compressed(
                filename,
                keys=keys_out,
                log_odds=log_odds_out,
                color=color_out,
                centroid=centroid_out,
                voxel_size=np.array([self.vs], dtype=np.float32),
            )
        n_saved = len(keys_out)
        tag = f" (permanent only, lo>={min_log_odds:.1f})" if min_log_odds is not None else ""
        print(f"[GlobalVoxelMap] Saved {n_saved} voxels{tag} to {filename}")

    def load(self, filename: str) -> None:
        """Load voxel map from .npz."""
        import os
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Voxel map file not found: {filename}")
        data = np.load(filename)
        keys = data["keys"]
        color_sum = data.get("color_sum", None)
        # Unwrap 0-d arrays which indicate None from npz
        if color_sum is not None and color_sum.shape == (): color_sum = None
        color = data.get("color", None)
        if color is not None and color.shape == (): color = None
        centroid = data.get("centroid", None)
        if centroid is not None and centroid.shape == (): centroid = None
        vs = float(data["voxel_size"][0])

        n = keys.shape[0]
        with self._lock:
            self.vs = vs
            cap = max(n * 2, 50_000)
            self._keys = torch.zeros((cap, 3), dtype=torch.int32).pin_memory()
            self._log_odds = torch.zeros(cap, dtype=torch.float16, device=self.device)
            
            if color is not None:
                self._color = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
            else:
                self._color = None
            if centroid is not None:
                self._centroid = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
            else:
                self._centroid = None

            self._keys[:n] = torch.tensor(np.ascontiguousarray(keys).copy())

            if "log_odds" in data:
                self._log_odds[:n] = torch.tensor(np.ascontiguousarray(data["log_odds"])).to(self.device)
                if color_sum is not None:
                    self._color = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
                    self._color[:n] = torch.tensor(np.ascontiguousarray(color_sum)).to(self.device)  # assuming it was already averaged
                    
            elif "tsdf" in data:
                # Legacy format: convert TSDF to log_odds representation
                tsdf = torch.tensor(np.ascontiguousarray(data["tsdf"]).copy())
                weight = torch.tensor(np.ascontiguousarray(data["weight"]).copy())
                occ_mask = tsdf < 0.3
                self._log_odds[:n][occ_mask] = self.l_hit
                self._log_odds[:n][~occ_mask] = 0.0
                
                if color_sum is not None:
                    self._color = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
                    self._color[:n] = (torch.tensor(np.ascontiguousarray(color_sum)).to(self.device) / 
                                       torch.clamp_min(weight.to(self.device).unsqueeze(1).float(), 1.0))
                    
            elif "hits" in data:
                # Ancient legacy format
                hits = torch.tensor(np.ascontiguousarray(data["hits"]).copy())
                occ_mask = hits > 0
                self._log_odds[:n][occ_mask] = self.l_hit
                self._log_odds[:n][~occ_mask] = 0.0
                
                if color_sum is not None:
                    self._color = torch.zeros((cap, 3), dtype=torch.float32, device=self.device)
                    self._color[:n] = (torch.tensor(np.ascontiguousarray(color_sum)).to(self.device) / 
                                       torch.clamp_min(hits.to(self.device).unsqueeze(1).float(), 1.0))

            if color is not None:
                self._color[:n] = torch.tensor(np.ascontiguousarray(color)).to(self.device)
            if centroid is not None:
                self._centroid[:n] = torch.tensor(np.ascontiguousarray(centroid)).to(self.device)

            self._key_to_idx.clear()
            for i in range(n):
                key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
                self._key_to_idx[key] = i
            self._count = n
            self._update_floor_ema()

        print(f"[GlobalVoxelMap] Loaded {n} voxels (vs={vs:.3f} m) from {filename}")


# ============================================================
# Helpers
# ============================================================
def _grow_tensor(t: torch.Tensor, new_capacity: int, pinned: bool = False) -> torch.Tensor:
    """Grow a 1D or 2D tensor to new_capacity along dim 0, preserving existing data.

    Args:
        pinned: if True and tensor is on CPU, allocate the new tensor in pinned
                (page-locked) memory so DMA transfers to GPU are non-blocking. (Change C)
    """
    on_cpu = t.device.type == 'cpu'
    if t.ndim == 1:
        if pinned and on_cpu:
            new = torch.zeros(new_capacity, dtype=t.dtype).pin_memory()
        else:
            new = torch.zeros(new_capacity, dtype=t.dtype, device=t.device)
        new[: t.shape[0]] = t
    else:
        if pinned and on_cpu:
            new = torch.zeros((new_capacity,) + t.shape[1:], dtype=t.dtype).pin_memory()
        else:
            new = torch.zeros((new_capacity,) + t.shape[1:], dtype=t.dtype, device=t.device)
        new[: t.shape[0]] = t
    return new