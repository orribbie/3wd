# mapping_torch.py  (Open3D-free, GPU-first point cloud mapping)
# - Uses ZED native point cloud (XYZRGBA) when available via datastream.get_pcd_pose()
# - RGB/Depth fallback path kept intact
# - Point cloud stored as torch tensors: {points: (N,3), colors: (N,3)}
# - Same MapManager structure (save/load/visualize + live mapping thread)
# - Simple voxel downsampling on GPU

from PIL import Image
import numpy as np
import os
import queue                             # Change D: ring-buffer ingest queue
from typing import List, Tuple, Optional

import threading
import time
import torch
import torch.nn.functional as F
from loop_rate_limiters import RateLimiter

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ============================================================
# Point cloud container (torch tensors)
# ============================================================
class TorchPointCloud:
    """
    Stores a point cloud on GPU (if available).

    Attributes:
        points: (N,3) float32
        colors: (N,3) uint8 in [0,255]
    """
    def __init__(self, points: torch.Tensor, colors: torch.Tensor):
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
        assert colors.ndim == 2 and colors.shape[1] == 3, "colors must be (N,3)"
        if points.device != DEVICE:
            points = points.to(DEVICE)
        if colors.device != DEVICE:
            colors = colors.to(DEVICE)
        # Enforce dtypes
        points = points.float()
        if colors.dtype != torch.uint8:
            colors = torch.clamp(colors, 0, 255).to(torch.uint8)

        self.points = points
        self.colors = colors

    def __len__(self):
        return self.points.shape[0]

    def clone(self) -> "TorchPointCloud":
        return TorchPointCloud(self.points.clone(), self.colors.clone())

    def to(self, device: torch.device) -> "TorchPointCloud":
        return TorchPointCloud(self.points.to(device), self.colors.to(device))

    def cpu_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points_np, colors_np) for serialization/viz."""
        return self.points.detach().cpu().numpy(), self.colors.detach().cpu().numpy()

    def append(self, other: "TorchPointCloud"):
        """In-place concatenation."""
        if other is None or len(other) == 0:
            return
        # Move other to this device if needed
        if other.points.device != self.points.device:
            other = other.to(self.points.device)
        self.points = torch.cat([self.points, other.points], dim=0)
        self.colors = torch.cat([self.colors, other.colors], dim=0)

    def transformed(self, T_4x4: torch.Tensor) -> "TorchPointCloud":
        """Return a new cloud with transform applied (4x4, row-major)."""
        return TorchPointCloud(apply_transform(self.points, T_4x4), self.colors.clone())

    def transform_(self, T_4x4: torch.Tensor):
        """In-place transform."""
        self.points = apply_transform(self.points, T_4x4)

# ============================================================
# Math helpers (torch)
# ============================================================
def _quat_to_matrix_np(q) -> np.ndarray:
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


def pose_to_matrix(quat_xyzw: np.ndarray, trans_xyz: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """
    Convert pose into 4x4 torch transform.  Pure numpy + torch, no scipy.
    quat_xyzw: [x,y,z,w]
    trans_xyz: [tx,ty,tz]
    """
    rot = _quat_to_matrix_np(quat_xyzw)
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, :3] = torch.tensor(np.ascontiguousarray(rot)).to(device)
    T[:3, 3] = torch.tensor(np.ascontiguousarray(trans_xyz.astype(np.float32))).to(device)
    return T

def apply_transform(points: torch.Tensor, T_4x4: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transform to Nx3 points (torch) -> Nx3.

    Uses R @ p + t directly — avoids allocating an (N,1) ones column
    and an (N,4) homogeneous expansion (saves ~30 % memory + time on Jetson).
    """
    return points @ T_4x4[:3, :3].T + T_4x4[:3, 3]

def make_flip_transform(device: torch.device = DEVICE) -> torch.Tensor:
    """Open3D-style flip used previously (for some RGB-D sensors)."""
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return T

# ============================================================
# RGB-D -> TorchPointCloud (GPU)  [fallback path]
# ============================================================

@torch.no_grad()
def voxel_downsample_(pc: TorchPointCloud, voxel_size: float) -> TorchPointCloud:
    """
    Open3D-style voxel grid downsampling for TorchPointCloud.
    Filters out NaN/Inf rows first, then averages XYZ and RGB within each voxel.
    Runs on CPU or CUDA depending on pc.points.device.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    device = pc.points.device
    pts = pc.points                         # (N,3) float32
    cols_u8 = pc.colors                     # (N,3) uint8 (per your class)

    # 1) Filter invalid rows: points must be finite; if colors are float, check them too.
    pts_ok = torch.isfinite(pts).all(dim=1)
    if cols_u8.dtype.is_floating_point:
        cols_ok = torch.isfinite(cols_u8).all(dim=1)
    else:
        cols_ok = torch.ones_like(pts_ok, dtype=torch.bool, device=device)

    ok = pts_ok & cols_ok
    if not ok.any():
        # Return an empty cloud on the same device/dtypes
        return TorchPointCloud(pts[:0], cols_u8[:0])

    pts = pts[ok]
    cols_u8 = cols_u8[ok]
    cols = cols_u8.to(torch.float32)        # accumulate in float

    # 2) Open3D-like min bound: min - 0.5 * voxel_size
    min_bound = pts.amin(dim=0) - (voxel_size * 0.5)

    # 3) Integer voxel indices
    vox_idx = torch.floor((pts - min_bound) / voxel_size).to(torch.long)  # (M,3)

    # 4) Unique voxel cells + mapping per point (deterministic order)
    unique_vox, inverse = torch.unique(
        vox_idx, dim=0, return_inverse=True, sorted=True
    )  # unique_vox: (K,3), inverse: (M,)

    K = unique_vox.shape[0]
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=torch.float32)

    # 5) Sum positions/colors per voxel
    pts_sum = torch.zeros((K, 3), device=device, dtype=torch.float32)
    pts_sum.index_add_(0, inverse, pts)

    col_sum = torch.zeros((K, 3), device=device, dtype=torch.float32)
    col_sum.index_add_(0, inverse, cols)

    # 6) Counts per voxel
    cnts = torch.zeros((K, 1), device=device, dtype=torch.float32)
    cnts.index_add_(0, inverse, ones)

    # 7) Means
    pts_mean = (pts_sum / cnts.clamp_min(1.0)).to(torch.float32)
    col_mean = col_sum / cnts.clamp_min(1.0)
    colors_ds = torch.clamp(torch.round(col_mean), 0, 255).to(torch.uint8)

    return TorchPointCloud(pts_mean, colors_ds)

@torch.no_grad()
def clean_outliers_torch(
    cloud: Optional["TorchPointCloud"],
    radius: float = 0.12,
    min_neighbors: int = 3,
    max_points: int = 4000,
) -> Optional["TorchPointCloud"]:
    """
    Remove isolated points using neighbor counting (torch.cdist).
    NOTE: O(N^2) → keep max_points small (few thousand).
    """
    if cloud is None:
        return None
    if len(cloud) == 0:
        return cloud

    points = cloud.points
    colors = cloud.colors

    # Uniform subsample to cap O(N^2) cost
    if max_points is not None and points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points = points.index_select(0, idx)
        colors = colors.index_select(0, idx)

    if points.shape[0] == 0:
        return TorchPointCloud(points, colors)

    dists = torch.cdist(points, points)
    neighbor_counts = (dists < radius).sum(dim=1) - 1
    mask = neighbor_counts >= min_neighbors

    return TorchPointCloud(points[mask], colors[mask])


# ============================================================
# ZED PCD -> TorchPointCloud (GPU)  [preferred path]
# ============================================================
def _rgba_float_to_rgb_u8(rgba_float) -> np.ndarray:
    """
    Vectorized reinterpretation of ZED packed RGBA stored as a float32.
    Returns (H,W,3) uint8 RGB. Supports both NumPy and PyTorch Tensors.
    """
    if torch.is_tensor(rgba_float):
        # GPU path: Use PyTorch bitwise operations
        f = rgba_float.to(torch.float32)
        u32 = f.view(torch.int32)
        r = (u32 & 0x000000FF).to(torch.uint8)
        g = ((u32 >> 8) & 0x000000FF).to(torch.uint8)
        b = ((u32 >> 16) & 0x000000FF).to(torch.uint8)
        return torch.stack([r, g, b], dim=-1)
    
    # CPU path: NumPy
    f = np.ascontiguousarray(rgba_float.astype(np.float32, copy=False))
    u32 = f.view(np.uint32)
    # MEASURE.XYZRGBA packs bytes as 0xAABBGGRR in little-endian float layout.
    r = (u32 & 0x000000FF).astype(np.uint8)
    g = ((u32 >> 8) & 0x000000FF).astype(np.uint8)
    b = ((u32 >> 16) & 0x000000FF).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb

@torch.no_grad()
def zed_pcd_to_pointcloud_torch(
    zed_pcd,                         # sl.Mat or np.ndarray (H,W,4) float32, in CAMERA frame
    pose_qt: np.ndarray,             # [qx,qy,qz,qw, tx,ty,tz], WORLD_T_CAM
    return_extra: bool = False,       # if True, also returns (valid_mask_gpu, arr_t_gpu)
) -> TorchPointCloud | Tuple[TorchPointCloud, torch.Tensor, torch.Tensor]:
    """
    Build a TorchPointCloud directly from ZED native point cloud.
    - Points are taken from XYZ (meters) and transformed to WORLD using pose_qt.
    - Colors are decoded from packed RGBA float (we keep RGB, drop alpha).
    - Invalid points (nan/inf) are removed on GPU.
    """
    try:
        import pyzed.sl as sl
        SL_AVAILABLE = True
    except Exception:
        SL_AVAILABLE = False

    if SL_AVAILABLE and hasattr(zed_pcd, "get_data"):
        arr_np = zed_pcd.get_data(sl.MEM.CPU)
    else:
        arr_np = np.asarray(zed_pcd)

    assert arr_np.ndim == 3 and arr_np.shape[2] >= 3, "Expected (H,W,4) or (H,W,3) array from ZED"

    # Fix: PyTorch requires writable arrays
    if not arr_np.flags.writeable:
        arr_np = np.copy(arr_np)

    # Move to GPU ASAP
    arr_t = torch.tensor(np.ascontiguousarray(arr_np)).to(DEVICE, non_blocking=True)
    H, W = arr_t.shape[:2]

    # 1. Separate XYZ and Colors (GPU)
    xyz_t = arr_t[..., :3].float()
    
    if arr_t.shape[2] >= 4:
        # GPU Unpacking of colors
        colors_t = _rgba_float_to_rgb_u8(arr_t[..., 3])
        colors_t = colors_t.view(-1, 3)
    else:
        colors_t = torch.full((H * W, 3), 255, dtype=torch.uint8, device=DEVICE)

    # 2. Filter invalid points (GPU)
    valid_t = torch.isfinite(xyz_t).all(dim=2)

    # ── Flying-pixel / depth-edge rejection ─────────────────────────────────
    # At depth discontinuities (object edges vs background) the ZED stereo
    # matcher interpolates depth between two surfaces, producing points that
    # float in free space between the foreground and background.  These are
    # called "flying pixels" and they survive the confidence filter because
    # the ZED SDK considers them valid stereo matches.
    #
    # We reject them by computing the local depth range in a 3×3 window.  Any
    # pixel whose neighbourhood spans > EDGE_THR metres (where EDGE_THR is
    # roughly half the smallest inter-object depth gap we care about) is
    # discarded.  The op costs ~0.3 ms on a Jetson for a 720p/4 cloud.
    #
    # EDGE_THR tuning:
    #   ↑ larger  →  fewer rejections, more flying pixels survive (noisier map)
    #   ↓ smaller →  more rejections, some thin objects get clipped at edges
    EDGE_THR = 0.25   # metres — typical foreground/background gap threshold

    depth_t = xyz_t[..., 2]   # ZED camera Z = depth in metres, (H, W)
    # Replace invalid pixels with NaN so they don't pollute the pool
    depth_valid = torch.where(valid_t, depth_t, torch.tensor(float('nan'), device=DEVICE))
    d4 = depth_valid.unsqueeze(0).unsqueeze(0)   # (1,1,H,W) for F ops

    POOL_K = 3
    PAD    = POOL_K // 2

    # nan-safe max/min: treat NaN as the neutral element for each op
    # torch max_pool2d ignores NaN (treats as -inf), so we gate on valid_t separately.
    d_max = F.max_pool2d(
        torch.nan_to_num(d4, nan=-1e6), kernel_size=POOL_K, stride=1, padding=PAD
    ).squeeze()
    d_min = F.max_pool2d(
        -torch.nan_to_num(d4, nan=1e6), kernel_size=POOL_K, stride=1, padding=PAD
    ).squeeze().neg()

    depth_range = d_max - d_min   # (H, W) — local depth spread in 3×3 window
    edge_free   = depth_range < EDGE_THR   # True = pixel is NOT on a depth edge

    valid_t = valid_t & edge_free
    # ─────────────────────────────────────────────────────────────────────────

    valid_flat = valid_t.view(-1)

    if not valid_flat.any():
        pcd = TorchPointCloud(
            points=torch.zeros((0, 3), dtype=torch.float32, device=DEVICE),
            colors=torch.zeros((0, 3), dtype=torch.uint8, device=DEVICE),
        )
        return (pcd, valid_t, arr_t) if return_extra else pcd

    pts_cam = xyz_t.view(-1, 3)[valid_flat]
    cols = colors_t[valid_flat]

    # 3. Transform CAM -> WORLD (GPU)
    pose_qt = np.asarray(pose_qt, dtype=np.float32).reshape(-1)
    quat, trans = pose_qt[:4], pose_qt[4:7]
    T_world_cam = pose_to_matrix(quat, trans, device=DEVICE)

    pts_world = apply_transform(pts_cam, T_world_cam)
    
    pcd = TorchPointCloud(points=pts_world, colors=cols)
    return (pcd, valid_t, arr_t) if return_extra else pcd


@torch.no_grad()
def rgbd_to_pointcloud_torch(
    image: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
    focal,
    resolution,
    device: torch.device = DEVICE,
) -> TorchPointCloud:
    """
    Convert an RGB-D frame + pose into a TorchPointCloud in WORLD frame.

    image:      H x W x 3   (BGR from ZED)
    depth:      H x W       (meters)
    confidence: H x W       (currently unused)
    pose:       [qx, qy, qz, qw, tx, ty, tz]  (WORLD_T_CAM)
    focal:      [fx, fy] or [fx, fy, cx, cy]
    resolution: [W, H]  (optional, we infer from depth anyway)
    """
    # --- Depth & intrinsics ---
    depth_m = np.asarray(depth, dtype=np.float32)
    H, W = depth_m.shape

    if focal is None:
        fx = fy = 720.0
        cx = W / 2.0
        cy = H / 2.0
    else:
        focal_list = list(focal)
        if len(focal_list) >= 4:
            fx, fy, cx, cy = focal_list[:4]
        else:
            fx, fy = focal_list[:2]
            cx = W / 2.0
            cy = H / 2.0

    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)

    # Valid depth mask
    Z = depth_m
    valid = np.isfinite(Z) & (Z > 0.0)
    if not np.any(valid):
        return TorchPointCloud(
            points=torch.zeros((0, 3), dtype=torch.float32, device=device),
            colors=torch.zeros((0, 3), dtype=torch.uint8, device=device),
        )

    # --- Back-project to camera frame ---
    u = np.arange(W, dtype=np.float32)[None, :]   # (1, W)
    v = np.arange(H, dtype=np.float32)[:, None]   # (H, 1)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pts_cam = np.stack([X[valid], Y[valid], Z[valid]], axis=1)  # (N, 3)
    pts_cam_t = torch.tensor(np.ascontiguousarray(pts_cam)).to(device=device, dtype=torch.float32)

    # --- Colors from RGB image ---
    img_np = np.asarray(image)
    if img_np.ndim == 3 and img_np.shape[2] >= 3:
        # ZED gives BGR; convert to RGB so it's consistent with zed_pcd_to_pointcloud_torch
        rgb = img_np[..., ::-1]  # BGR -> RGB
        cols = rgb[valid]
        cols_t = torch.tensor(np.ascontiguousarray(cols)).to(device=device, dtype=torch.uint8)
    else:
        cols_t = torch.zeros((pts_cam_t.shape[0], 3), dtype=torch.uint8, device=device)

    # --- Transform CAM -> WORLD using pose ---
    quat = np.asarray(pose[:4], dtype=np.float32)
    trans = np.asarray(pose[4:7], dtype=np.float32)
    T_world_cam = pose_to_matrix(quat, trans, device=device)

    pts_world = apply_transform(pts_cam_t, T_world_cam)

    return TorchPointCloud(points=pts_world, colors=cols_t)

# ============================================================
# MapManager
# ============================================================
class MapManager:
    """
    Single live voxel-map pipeline.

    One GlobalVoxelMap, one ingest thread (lean ZED poll), one integration
    worker (GPU surface insert + dynamic clearing), one latest-frame buffer.

    External API:
        start_mapping(datastream) / stop_mapping()
        get_map() / get_voxel_map()       → GlobalVoxelMap
        get_latest_frame_pts()            → (pts_world, pose_qt) for grid overlay
        set_camera_info(dict)             → pass ZED intrinsics to carver
        save_map(filename) / load_map(filename)
    """

    def __init__(self, voxel_size: float = 0.03):
        # thread state
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.paused: bool = False
        self.last_error: Optional[str] = None
        self.datastream = None

        # pose history (used by _log_status / state_monitor)
        self._lock = threading.Lock()
        self.all_poses: List[np.ndarray] = []

        # ── Single live voxel map ──────────────────────────────────────────────
        # 1 000 000 voxels @ 3 cm resolution = ~40 MB GPU (color+centroid+log_odds)
        # + ~12 MB pinned CPU (keys).  Covers ~30 m × 30 m at typical indoor density.
        # Dynamic obstacle clearing compacts zombie voxels automatically so the
        # ceiling is rarely approached during normal operation.
        from robot.nav.mapping.voxel_map import GlobalVoxelMap
        self._voxel_map = GlobalVoxelMap(
            voxel_size=voxel_size,
            max_voxels=1_000_000,
        )
        self._voxel_map.max_insert_depth = 5.0   # matches carver z_range_max
        self._voxel_map.promote_age = 0           # single-map mode: no promotion

        self._frame_count = 0

        # 2-slot ring-buffer queue: ZED ingest → integration worker.
        # Capacity=2: worker always gets the freshest frame; stale ones are dropped.
        self._ingest_queue: queue.Queue = queue.Queue(maxsize=2)
        self._integration_thread: Optional[threading.Thread] = None

        # ── Latest-frame shared pts buffer ────────────────────────────────────
        # Integration worker publishes pre-projected world-frame points here so
        # the grid overlay thread never needs to re-poll ZED or re-project.
        self._pts_lock = threading.Lock()
        self._latest_pts_world: Optional[np.ndarray] = None   # (N, 3) float32 world
        self._latest_frame_pose: Optional[np.ndarray] = None  # (7,) [qx,qy,qz,qw,tx,ty,tz]
        self._pts_buffer_max: int = 30_000


    # --- I/O ---
    def save_map(self, _unused=None, filename: str = "", min_log_odds: float | None = None):
        """Save voxel map to .npz."""
        if self._voxel_map is not None and filename:
            self._voxel_map.save(filename, min_log_odds=min_log_odds)

    def load_map(self, filename: str):
        """Load voxel map from .npz."""
        if self._voxel_map is not None:
            self._voxel_map.load(filename)


    # --- Live mapping control ---
    def start_mapping(self, datastream, *, load: bool = False, target_hz: float = 10.0, map_path: str = None):
        """
        Start the background mapping thread.

        datastream: object with .get_pcd_pose() -> (pcd, pose_qt)
        target_hz:  ZED ingest rate cap (0 = as fast as possible).
        """
        if self._running:
            print("[MapManager] Mapping already running.")
            return

        self._running = True
        self.paused = False
        self.last_error = None
        self._frame_count = 0
        self.datastream = datastream

        if load and map_path:
            self.load_map(map_path)
            print(f"[MapManager] Loaded map from {map_path}")

        # Start integration worker first, then the lean ingest loop.
        self._integration_thread = threading.Thread(
            target=self._integration_worker, daemon=True, name="map-integrate"
        )
        self._integration_thread.start()

        self._thread = threading.Thread(
            target=self._mapping_loop, args=(datastream, load, target_hz),
            daemon=True, name="map-ingest"
        )
        self._thread.start()
        print(f"[MapManager] Mapping started (target_hz={target_hz}).")               
        

    def stop_mapping(self, join_timeout: Optional[float] = 2.0):
        """Signal the thread to stop and optionally join."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        # Change D: drain queue and wake integration worker so it exits cleanly
        try:
            self._ingest_queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass
        if self._integration_thread is not None:
            self._integration_thread.join(timeout=join_timeout)
        self._thread = None
        self._integration_thread = None
        print("[MapManager] Mapping thread stopped.")

    def get_state(self) -> Tuple[None, List[np.ndarray]]:
        """Return (None, pose_list). Pose list is a snapshot for logging only."""
        with self._lock:
            poses_copy = list(self.all_poses)
        return None, poses_copy

    def get_latest_frame_pts(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return the most recently integrated frame's world-frame points and
        pose as a (pts_world, pose_qt) tuple, or (None, None) if not yet available.

        pts_world : (N≤3000, 3) float32 — already in world frame (Y-up).
                    Uniform stride-subsampled from the full ZED cloud so the
                    buffer stays small and fast to copy.
        pose_qt   : (7,) float32 [qx, qy, qz, qw, tx, ty, tz].

        IMPORTANT: caller must NOT modify the returned arrays in-place.
        """
        with self._pts_lock:
            return self._latest_pts_world, self._latest_frame_pose
    
    def get_map(self):
        """Return the live GlobalVoxelMap."""
        return self._voxel_map

    def get_voxel_map(self):
        """Return the live GlobalVoxelMap."""
        return self._voxel_map

    def get_live_map(self):
        """Return the live GlobalVoxelMap."""
        return self._voxel_map

    def set_camera_info(self, cam_info: dict):
        if self._voxel_map is not None:
            self._voxel_map.set_camera_intrinsics(
                cam_info["width"], cam_info["height"],
                cam_info["fx"], cam_info["fy"],
                cam_info["cx"], cam_info["cy"]
            )



    def _integration_worker(self):
        """
        Integration worker thread — single voxel-map path, no legacy fallback.

        Drains _ingest_queue at full GPU speed, decoupled from ZED ingest cadence.
        Sentinel: None in the queue → exit cleanly.

        Timing printed every 60 frames (wall-clock sections):
          ingest  = ZED PCD decode + GPU transfer
          surface = _integrate_surface (voxel hash insert)
          clear   = _clear_dynamic_objects (depth-buffer differencing)
          buffer  = latest-frame pts CPU copy
        """
        from collections import deque
        _hz_window: deque = deque(maxlen=60)   # rolling frame timestamps for Hz
        _T_LOG = 60                             # log every N frames

        # Per-frame timing accumulators (reset every _T_LOG frames)
        _t_ingest = _t_surface = _t_clear = _t_buffer = _t_total = 0.0
        _n_logged = 0

        while True:
            item = self._ingest_queue.get()
            if item is None:
                break

            zed_pcd, pose_zed = item
            _frame_t0 = time.perf_counter()

            try:
                # ── 1. Decode ZED PCD (CPU→GPU) ─────────────────────────────
                try:
                    import pyzed.sl as sl
                    _arr_np = (zed_pcd.get_data(sl.MEM.CPU)
                               if hasattr(zed_pcd, 'get_data')
                               else np.asarray(zed_pcd))
                except ImportError:
                    _arr_np = np.asarray(zed_pcd)
                if not _arr_np.flags.writeable:
                    _arr_np = _arr_np.copy()

                pcd, _, _ = zed_pcd_to_pointcloud_torch(_arr_np, pose_zed, return_extra=True)
                _t1 = time.perf_counter()

                if len(pcd) == 0:
                    continue

                pose_arr = np.asarray(pose_zed, dtype=np.float32).reshape(-1)

                # ── 2. Voxel integration (surface insert + dynamic clearing) ─
                result = self._voxel_map.integrate(
                    pcd.points, pcd.colors, self._frame_count,
                    camera_pose=pose_arr,
                )
                _t2 = time.perf_counter()

                # ── 3. Publish latest-frame pts to shared buffer ─────────────
                # One stride-sampled GPU→CPU copy; grid overlay reads this
                # instead of re-polling ZED or re-projecting on CPU.
                pts_gpu = pcd.points
                n = pts_gpu.shape[0]
                if n > self._pts_buffer_max:
                    stride = max(1, n // self._pts_buffer_max)
                    pts_gpu = pts_gpu[::stride]
                pts_cpu = pts_gpu.cpu().numpy()
                with self._pts_lock:
                    self._latest_pts_world = pts_cpu
                    self._latest_frame_pose = pose_arr
                _t3 = time.perf_counter()

                with self._lock:
                    self.all_poses.append(pose_arr)

                # ── Hz + timing ─────────────────────────────────────────────
                _hz_window.append(_t3)
                _t_ingest  += _t1 - _frame_t0
                _t_surface += _t2 - _t1
                _t_buffer  += _t3 - _t2
                _t_total   += _t3 - _frame_t0
                _n_logged  += 1

                if _n_logged >= _T_LOG:
                    span = float(_hz_window[-1] - _hz_window[0]) if len(_hz_window) > 1 else 1.0
                    hz = (len(_hz_window) - 1) / span if span > 0 else 0.0
                    scale = 1e3 / _n_logged
                    print(
                        f"[MapManager] {hz:.1f} Hz | "
                        f"ingest={_t_ingest*scale:.1f}ms  "
                        f"surface={_t_surface*scale:.1f}ms  "
                        f"buffer={_t_buffer*scale:.1f}ms  "
                        f"frame={_t_total*scale:.1f}ms  "
                        f"voxels={self._voxel_map._count:,}"
                    )
                    _t_ingest = _t_surface = _t_clear = _t_buffer = _t_total = 0.0
                    _n_logged = 0

            except Exception as e:
                self.last_error = f"integration_worker failed: {e}"
                import traceback; traceback.print_exc()

    def _mapping_loop(self, datastream, load: bool, target_hz: float):
        """
        Lean ingest loop: polls ZED, deduplicates repeated frames, enqueues for worker.
        """
        rate = RateLimiter(target_hz, name="map_manager") if (target_hz and target_hz > 0) else None
        _last_pose_bytes: bytes = b""   # dedup: skip if same pose as previous call

        # Loop timing instrumentation — prints every 60 iterations
        _loop_t0 = time.perf_counter()
        _loop_n = 0
        _t_fetch = _t_dedup = _t_enqueue = _t_ratelim = 0.0
        _n_dedup = 0
        _LOG_EVERY = 60

        while self._running:
            if self.paused:
                time.sleep(0.05)
                continue
            try:
                _ta = time.perf_counter()
                zed_pkt = datastream.get_pcd_pose()
                _tb = time.perf_counter()
                _t_fetch += _tb - _ta
                if isinstance(zed_pkt, tuple) and len(zed_pkt) >= 2:
                    zed_pcd, pose_zed = zed_pkt[0], zed_pkt[1]

                    # Dedup: ZED subscriber returns the latest cached frame; if the
                    # publisher hasn't pushed a new one yet, skip rather than integrating
                    # the same cloud twice (wastes GPU and inflates log-odds).
                    # NOTE: do NOT call rate.sleep() here — it would cause a double-sleep
                    # (once here + once at the bottom for the next new frame = 2× period).
                    _tc = time.perf_counter()
                    pose_bytes = np.asarray(pose_zed, dtype=np.float32).tobytes()
                    if pose_bytes == _last_pose_bytes:
                        _n_dedup += 1
                        time.sleep(0.003)  # 3ms spin-wait; bottom rate.sleep paces new frames
                        continue
                    _last_pose_bytes = pose_bytes
                    _td = time.perf_counter()
                    _t_dedup += _td - _tc

                    self._frame_count += 1
                    # Non-blocking put: always keep the freshest frame.
                    try:
                        self._ingest_queue.put_nowait((zed_pcd, pose_zed))
                    except queue.Full:
                        try:
                            self._ingest_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self._ingest_queue.put_nowait((zed_pcd, pose_zed))
                        except queue.Full:
                            pass
                    _t_enqueue += time.perf_counter() - _td

            except Exception as e:
                self.last_error = f"ingest failed: {e}"

            _te = time.perf_counter()
            if rate is not None:
                rate.sleep()
            _t_ratelim += time.perf_counter() - _te

            _loop_n += 1
            if _loop_n >= _LOG_EVERY:
                elapsed = time.perf_counter() - _loop_t0
                scale = 1e3  # → ms
                print(
                    f"[MapLoop] {_loop_n/elapsed:.1f} Hz | "
                    f"fetch={_t_fetch/elapsed*scale:.1f}ms  "
                    f"dedup={_t_dedup/elapsed*scale:.1f}ms  "
                    f"enqueue={_t_enqueue/elapsed*scale:.1f}ms  "
                    f"ratelim={_t_ratelim/elapsed*scale:.1f}ms  "
                    f"dedup_hits={_n_dedup}/{_loop_n}",
                    flush=True,
                )
                _loop_t0 = time.perf_counter()
                _loop_n = 0
                _t_fetch = _t_dedup = _t_enqueue = _t_ratelim = 0.0
                _n_dedup = 0

