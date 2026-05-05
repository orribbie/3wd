#!/usr/bin/env python3

"""SLAM node for ZED camera mapping and navigation."""

import argparse
import concurrent.futures
import os
import signal
import sys
import threading
import time
from typing import Optional, Tuple

import numpy as np
from commlink import Subscriber
from robot.nav.mapping.mapping_torch import MapManager
from robot.nav.pathPlanning import (
    Grid2DParams,
    compute_static_grid_from_points,
    StaticGridWithLiveOverlayThread,
    AStarPlannerThread,
)
from robot.nav.viserBridge import start_viser_server, ViserMirrorThread
from robot.nav.semantic_labels import SemanticLabelStore

from robot.nav.odometry.omni3_odom import Omni3Odom
from robot.nav.odometry.robot_ekf import RobotEKF
from robot.sir_bridge import SirBridge
from robot.utils.utils import waitKey


# Pub/sub topics from zed_pub_node.py
POSE_TOPIC = "zed/pose"
IMAGE_TOPIC = "zed/image"
DEPTH_TOPIC = "zed/depth"
PCD_TOPIC = "zed/pcd"
CAMERA_INFO_TOPIC = "zed/camera_info"
ZED_PUB_PORT = 6000

# RPC from SLAM -> Yor (follow_path via RPC)
YOR_RPC_HOST = "194.168.1.10" 
YOR_RPC_PORT = 5557

def _quat_to_matrix(q) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3×3 rotation matrix (inline, no scipy)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)


def _matrix_to_quat(m) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [x, y, z, w] (inline, no scipy).
    Uses Shepperd's method for numerical stability."""
    m = np.asarray(m, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def xyzw_xyz_to_matrix(qt7):
    """
    qt7: [qx, qy, qz, qw, tx, ty, tz]
    """
    qt7 = np.asarray(qt7, dtype=np.float32).reshape(-1)
    if qt7.shape[0] < 7:
        raise ValueError(f"Expected 7 values [qx,qy,qz,qw,tx,ty,tz], got {qt7.shape}")
    q = qt7[:4]
    t = qt7[4:7]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_to_matrix(q)
    T[:3, 3] = t
    return T


class ZedSub:
    """Wrapper around Subscriber to get the latest RGB image, depth map, pose, and point cloud from the ZED camera, with optional Z-up to Y-up conversion."""
    def __init__(self, host: str = "194.168.1.11", port: int = ZED_PUB_PORT, up_axis: str = "y",
                 topics: list = None):
        self._up_axis = str(up_axis).lower()
        if self._up_axis not in ("y", "z"):
            self._up_axis = "y"
        self._zup_to_yup = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        if topics is None:
            topics = [IMAGE_TOPIC, DEPTH_TOPIC, POSE_TOPIC, PCD_TOPIC, CAMERA_INFO_TOPIC]
        self._sub = Subscriber(host=host, port=port, topics=topics)
        self._sub_lock = threading.Lock()


    def _zup_to_yup_transform(self, T: np.ndarray) -> np.ndarray:
        """Convert a 4x4 transformation matrix from Z-up to Y-up by applying the appropriate rotation."""
        return self._zup_to_yup @ T @ self._zup_to_yup.T

    def _zup_to_yup_pose(self, pose_qt: np.ndarray) -> np.ndarray:
        """Convert pose from Z-up to Y-up by applying the appropriate transformation. Expects pose_qt as [qx, qy, qz, qw, tx, ty, tz]."""
        pose_qt = np.asarray(pose_qt, dtype=np.float32).reshape(-1)
        if pose_qt.size < 7:
            return pose_qt
        quat = pose_qt[:4]
        trans = pose_qt[4:7]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = _quat_to_matrix(quat)
        T[:3, 3] = trans
        T = self._zup_to_yup_transform(T)
        quat_y = _matrix_to_quat(T[:3, :3])
        trans_y = T[:3, 3].astype(np.float32)
        return np.concatenate([quat_y, trans_y])
    
    def _sub_get(self, topic):
        """Thread-safe get for subscriber topics."""
        with self._sub_lock:
            return self._sub[topic]


    def _zup_to_yup_points(self, pcd):
        """Convert point cloud from Z-up to Y-up by swapping axes. Expects shape (..., 3) or (..., N) with XYZ in the first 3 channels."""
        arr = np.asarray(pcd)
        if arr.ndim < 2 or arr.shape[-1] < 3:
            return arr
        xyz = arr[..., :3].astype(np.float32, copy=False)
        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]
        xyz_yup = np.stack([x, z, -y], axis=-1)
        if arr.shape[-1] > 3:
            out = arr.copy()
            out[..., :3] = xyz_yup
            return out
        return xyz_yup

    def stop(self):
        self._sub.stop()
    
    def ready(self) -> bool:
        """Return True once at least the pose topic has been received."""
        ready_attr = getattr(self._sub, "ready", None)
        if callable(ready_attr):
            return bool(ready_attr())
        if ready_attr is not None:
            return bool(ready_attr)

        try:
            pose_msg = self._sub_get(POSE_TOPIC)
        except Exception:
            return False
        return pose_msg is not None

    def get_rgb_depth_pose(self):
        """Get the latest RGB image, depth map, and pose. Pose is returned as [qx, qy, qz, qw, tx, ty, tz]."""
        img_msg = self._sub_get(IMAGE_TOPIC)
        depth_msg = self._sub_get(DEPTH_TOPIC)
        pose_msg = self._sub_get(POSE_TOPIC)

        if img_msg is None or depth_msg is None or pose_msg is None:
            raise RuntimeError("ZedSub not ready yet")
        
        image_rgb = img_msg["image"]
        depth_m = depth_msg["depth"]
        pose_qt = pose_msg[7:14]
        if self._up_axis == "z":
            pose_qt = self._zup_to_yup_pose(pose_qt)

        return image_rgb, depth_m, pose_qt
    
    def get_pcd_pose(self):
        """Get the latest point cloud and pose.

        Returns
        -------
        pcd      : point cloud array
        pose_qt  : [qx, qy, qz, qw, tx, ty, tz]
        confidence: ZED pose_confidence (0–100); extracted from the same POSE
                    message so no extra subscriber call is needed.
        """
        pcd_msg = self._sub_get(PCD_TOPIC)
        pose_msg = self._sub_get(POSE_TOPIC)

        if pcd_msg is None or pose_msg is None:
            raise RuntimeError("ZedSub not ready yet or Points not being streamed")

        pcd = pcd_msg["points"]
        pose_qt = pose_msg[7:14]  # [qx, qy, qz, qw, tx, ty, tz]
        # Confidence is at index 19 in the 20-float pose message.
        # Extracting it here avoids a second _sub_get(POSE_TOPIC) call in EKFSlamSource.
        confidence = float(pose_msg[19]) if len(pose_msg) > 19 else 100.0
        if self._up_axis == "z":
            pose_qt = self._zup_to_yup_pose(pose_qt)
            pcd = self._zup_to_yup_points(pcd)
        return pcd, pose_qt, confidence
    
    def get_pose(self):
        """Get the latest pose as (translation, yaw, full_transform). Translation is (x, z) in world coordinates, yaw is in radians."""
        pose_msg = self._sub_get(POSE_TOPIC)
        base_quat = pose_msg[0:7]
        base_transform = xyzw_xyz_to_matrix(base_quat)
        if self._up_axis == "z":
            base_transform = self._zup_to_yup_transform(base_transform)
        translation = base_transform[:3, 3].astype(np.float32)
        yaw = float(np.arctan2(-base_transform[2, 0], base_transform[0, 0]))
        return translation, yaw, base_transform

    def get_tracking_confidence(self) -> float:
        """Return ZED pose_confidence (0–100) from the latest pose message.

        Published as element [19] of the 20-float pose message by zed_pub_node.py.
        Returns 100.0 (full trust) if the field is not yet available (older nodes).
        """
        try:
            pose_msg = self._sub_get(POSE_TOPIC)
            if pose_msg is not None and len(pose_msg) > 19:
                return float(pose_msg[19])
        except Exception:
            pass
        return 100.0

    def get_camera_info(self):
        """Return the latest camera intrinsics dictionary if available."""
        return self._sub_get(CAMERA_INFO_TOPIC)

class EKFSlamSource:
    """
    Drop-in replacement for ZedSub as the pose_source for ViserMirrorThread and
    the camera_pose feed into the voxel map.

    Architecture
    ------------
    - A background predict-thread polls get_base_encoders() over the YOR RPC at
      ~20 Hz and calls ekf.predict(u, dt).
    - get_pcd_pose() / get_pose() intercept the ZED pose and call ekf.update(z)
      before returning the fused state, so every ZED frame triggers a correction.
    - All other methods (get_rgb_depth_pose, ready, stop, …) delegate directly
      to the wrapped ZedSub so the rest of the SLAM node is unaffected.

    Coordinate conventions
    ----------------------
    ZedSub.get_pose() returns (translation[3], yaw, T_wr[4,4]) in Y-up world
    frame where translation = [x, y, z] and the robot moves in the X-Z plane.
    The EKF state is [px, pz, yaw] to match this (px=x, pz=z).
    """

    def __init__(
        self,
        zed_sub: ZedSub,
        yor_client,
        predict_hz: float = 20.0,
    ):
        self._zed = zed_sub
        self._yor  = yor_client
        self._odom = Omni3Odom()
        self._ekf  = RobotEKF()       # uses calibrated defaults from robot_ekf.py
        self._lock = threading.Lock()

        self._predict_hz = max(1.0, float(predict_hz))
        self._predict_dt = 1.0 / self._predict_hz
        self._stop_evt  = threading.Event()
        self._predict_thr = threading.Thread(
            target=self._predict_loop, name="EKF-predict", daemon=True
        )
        self._last_enc_t: Optional[float] = None
        self._initialized = False   # True once first ZED update is applied
        self._consecutive_rejections: int = 0   # Mahalanobis rejection counter
        # Debounce: track the last ZED message timestamp so the same frame
        # cannot update the EKF twice (get_pcd_pose + get_pose can both fire
        # in the same mapping iteration, reading the same stale subscriber slot).
        self._last_zed_msg_ts: float = -1.0
        self._predict_thr.start()
        print("[EKFSlamSource] Started EKF predict thread at",
              self._predict_hz, "Hz")

    # ------------------------------------------------------------------ #
    # Background predict loop                                             #
    # ------------------------------------------------------------------ #
    def _predict_loop(self):
        zupt_count = 0            # frames robot appears stationary
        ZUPT_FRAMES = 4           # require 4× quiet ticks (~200 ms at 20 Hz)
        ZUPT_VEL_THR  = 0.015    # m/s — below this wheel speed → ZUPT eligible
        ZUPT_OMEG_THR = 0.025    # rad/s — below this turn rate → ZUPT eligible

        # Physical velocity limits — clamp encoder output to sane values
        MAX_VEL   = 2.0    # m/s   — robot cannot physically exceed this
        MAX_OMEGA = 3.0    # rad/s — max turn rate

        # RPC timeout: the ZMQ REQ socket has no built-in timeout — if yor.py is
        # offline it blocks forever, stalling this entire thread. We wrap every
        # call in a thread-pool future with a hard deadline instead.
        RPC_TIMEOUT_S = 0.5
        _rpc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix="ekf-rpc")

        _predict_count  = 0
        _none_count     = 0   # consecutive None returns (yor.py offline)
        _timeout_count  = 0   # consecutive timeouts

        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                # ---- non-blocking RPC call with timeout ----
                future = _rpc_executor.submit(self._yor.get_base_encoders)
                try:
                    enc = future.result(timeout=RPC_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    _timeout_count += 1
                    if _timeout_count % 20 == 1:   # every ~10 s at 20 Hz
                        print(f"[EKF-predict] WARN: get_base_encoders() timed out "
                              f"{_timeout_count}× — is yor.py running at "
                              f"{self._yor._host if hasattr(self._yor, '_host') else '?'}?")
                    time.sleep(self._predict_dt)
                    continue

                _timeout_count = 0  # reset on success

                if enc is None:
                    _none_count += 1
                    if _none_count % 20 == 1:
                        print(f"[EKF-predict] WARN: get_base_encoders() returned None "
                              f"{_none_count}× — yor.py may not be initialized yet.")
                    time.sleep(self._predict_dt)
                    continue

                _none_count = 0  # reset on success

                steer  = np.asarray(enc["steer_rad"],    dtype=float)
                counts = np.asarray(enc["drive_counts"], dtype=float)

                if not (np.all(np.isfinite(steer)) and np.all(np.isfinite(counts))):
                    print(f"[EKF-predict] WARN: NaN in encoders — steer={steer}  counts={counts}")
                    time.sleep(self._predict_dt)
                    continue

                now = float(enc.get("timestamp", time.time()))
                dt  = (now - self._last_enc_t) if self._last_enc_t is not None else self._predict_dt
                self._last_enc_t = now
                dt  = float(np.clip(dt, 1e-4, 0.5))

                u = self._odom.update(steer, counts, dt)   # [vx, vy, omega]

                # Clamp to physical limits — protects EKF from encoder spikes
                u_raw = u.copy()
                u[0]  = float(np.clip(u[0], -MAX_VEL,   MAX_VEL))
                u[1]  = float(np.clip(u[1], -MAX_VEL,   MAX_VEL))
                u[2]  = float(np.clip(u[2], -MAX_OMEGA,  MAX_OMEGA))

                _predict_count += 1
                if _predict_count % 100 == 0:
                    speed = float(np.hypot(u[0], u[1]))
                    print(f"[EKF-predict] step={_predict_count}  dt={dt:.4f}  "
                          f"u=[{u[0]:.3f}, {u[1]:.3f}, {u[2]:.3f}]  "
                          f"speed={speed:.3f} m/s")
                if np.any(np.abs(u_raw) > np.array([MAX_VEL, MAX_VEL, MAX_OMEGA])):
                    print(f"[EKF-predict] CLAMPED u_raw="
                          f"[{u_raw[0]:.3f}, {u_raw[1]:.3f}, {u_raw[2]:.3f}]")

                if self._initialized:
                    with self._lock:
                        self._ekf.predict(u, dt)

                    # ---- ZUPT: zero-velocity update during stops ----
                    speed = float(np.hypot(u[0], u[1]))
                    if speed < ZUPT_VEL_THR and abs(u[2]) < ZUPT_OMEG_THR:
                        zupt_count += 1
                        if zupt_count >= ZUPT_FRAMES:
                            with self._lock:
                                self._ekf.zupt()
                    else:
                        zupt_count = 0

            except Exception as e:
                print(f"[EKF-predict] Exception: {e}")
                zupt_count = 0   # reset on RPC error

            elapsed = time.time() - t0
            sleep_s = self._predict_dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ------------------------------------------------------------------ #
    # EKF update helper (call when a fresh ZED pose arrives)              #
    # ------------------------------------------------------------------ #
    def _adaptive_R(self, confidence: float) -> np.ndarray:
        """Scale measurement noise R inversely with ZED tracking confidence.

        confidence=100  →  R × 1   (fully trust ZED)
        confidence=50   →  R × 10  (moderate doubt)
        confidence=0    →  R × 100 (nearly reject ZED, ride on encoders)

        Uses exponential scaling so the transition is smooth rather than
        a binary accept/reject, which would cause the filter to jerk.
        """
        trust = float(np.clip(confidence / 100.0, 0.0, 1.0))
        scale = np.exp((1.0 - trust) * np.log(100.0))   # 1× → 100×
        return self._ekf.R * scale

    def _apply_zed_update(
        self, translation: np.ndarray, yaw: float, confidence: float = 100.0,
        zed_ts: float = -1.0,
    ) -> None:
        """
        Feed a ZED measurement into the EKF with adaptive noise.

        translation: Y-up world-frame [x, y, z] — robot moves in XZ plane.
        yaw:         heading around Y axis [rad].
        confidence:  ZED pose_confidence (0–100). Low values inflate R.
        zed_ts:      ZED message timestamp (ns or s). When provided, the same
                     frame is silently skipped if it was already applied, so
                     two call-sites (get_pcd_pose + get_pose) can't double-update.
        """
        # Debounce: skip if this exact ZED frame was already applied.
        if zed_ts > 0 and zed_ts == self._last_zed_msg_ts:
            return
        if zed_ts > 0:
            self._last_zed_msg_ts = zed_ts
        px = float(translation[0])
        pz = float(translation[2])
        
        if not (np.isfinite(px) and np.isfinite(pz) and np.isfinite(yaw)):
            return
            
        z  = np.array([px, pz, yaw], dtype=float)


        with self._lock:
            if not self._initialized:
                # Seed the filter at first ZED fix
                self._ekf.reset(x=px, y=pz, theta=yaw)
                self._odom.reset(x=px, y=pz, theta=yaw)
                self._initialized = True
                self._consecutive_rejections = 0
                print(f"[EKFSlamSource] Initialized at ({px:.2f}, {pz:.2f}), "
                      f"yaw={np.degrees(yaw):.1f}°")
                return

            # Adaptive R: inflate when ZED is struggling
            R_adapt = self._adaptive_R(confidence)

            # Log innovation (predict drift since last ZED frame) for diagnostics
            innov = z - self._ekf.get_state()
            innov[2] = ((innov[2] + np.pi) % (2 * np.pi)) - np.pi  # wrap yaw
            innov_norm = float(np.hypot(innov[0], innov[1]))
            innov_yaw_deg = float(np.degrees(abs(innov[2])))
            if innov_norm > 0.05 or innov_yaw_deg > 5.0:
                print(f"[EKF] LARGE innovation: pos={innov_norm:.3f}m  "
                      f"yaw={innov_yaw_deg:.1f}°  conf={confidence:.0f}  "
                      f"ekf=[{self._ekf.x[0]:.2f},{self._ekf.x[1]:.2f},{np.degrees(self._ekf.x[2]):.1f}°]  "
                      f"zed=[{px:.2f},{pz:.2f},{np.degrees(yaw):.1f}°]")

            # No Mahalanobis gating — the adaptive R (confidence-based) already
            # handles bad ZED frames smoothly. The gate was causing rejection
            # cascades because ZUPT collapsed P so tightly that even normal
            # ZED sensor noise (~0.3°, ~4mm) exceeded the chi-squared threshold.
            accepted = self._ekf.update(z, R_override=R_adapt, gate_chi2=None)

            if accepted:
                self._consecutive_rejections = 0
            else:
                self._consecutive_rejections += 1
                print(f"[EKF] REJECTED update (Mahalanobis gate): "
                      f"innov_pos={innov_norm:.3f}m  innov_yaw={innov_yaw_deg:.1f}°  "
                      f"conf={confidence:.0f}")
                # Safety net: if ZED is consistently rejected (e.g. robot was
                # physically carried) re-seed the filter to avoid permanent lock-out.
                if self._consecutive_rejections >= 15:
                    self._ekf.reset(x=px, y=pz, theta=yaw)
                    self._odom.reset(x=px, y=pz, theta=yaw)
                    self._consecutive_rejections = 0
                    print("[EKFSlamSource] 15 consecutive ZED rejections — "
                          "re-seeded filter at ZED position.")

    # ------------------------------------------------------------------ #
    # Fused pose extraction                                               #
    # ------------------------------------------------------------------ #
    def _get_fused_transform(self) -> np.ndarray:
        """Build a 4x4 Y-up transform from the EKF state [px, pz, yaw]."""
        with self._lock:
            state = self._ekf.get_state()   # [px, pz, yaw]
        px, pz, yaw = float(state[0]), float(state[1]), float(state[2])
        cy, sy = np.cos(yaw), np.sin(yaw)
        T = np.array([
            [ cy,  0.0, sy, px],
            [ 0.0, 1.0, 0.0, 0.0],
            [-sy,  0.0, cy, pz],
            [ 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return T

    # ------------------------------------------------------------------ #
    # ZedSub interface — delegated or fused                               #
    # ------------------------------------------------------------------ #
    def ready(self) -> bool:
        return self._zed.ready()

    def stop(self):
        self._stop_evt.set()
        self._zed.stop()

    def get_rgb_depth_pose(self):
        """Returns (image_rgb, depth_m, pose_qt) — pose_qt is raw ZED (not fused)."""
        return self._zed.get_rgb_depth_pose()

    def get_pcd_pose(self):
        """
        Returns (pcd, pose_qt, confidence) where pose_qt is the raw 7-float ZED
        quaternion+translation.  Also triggers an EKF update so the fused pose
        stays current.

        Confidence is forwarded from ZedSub.get_pcd_pose() — no extra subscriber
        call is made, which previously caused a ~100 ms stall (second blocking recv
        on POSE_TOPIC) and halved the effective mapping rate.
        """
        pcd, pose_qt, confidence = self._zed.get_pcd_pose()
        # EKF update uses BASE pose (indices 0:7), not camera pose (7:14).
        # Pass the ZED message timestamp for debounce — same frame won't update twice.
        try:
            pose_msg = self._zed._sub_get("zed/pose")
            zed_ts = float(pose_msg[18]) if (pose_msg is not None and len(pose_msg) > 18) else -1.0
            translation, yaw, _ = self._zed.get_pose()
            self._apply_zed_update(translation, yaw, confidence=confidence, zed_ts=zed_ts)
        except Exception:
            pass
        # Return camera pose_qt — map integration needs camera-frame coords.
        return pcd, pose_qt, confidence

    def get_pose(self):
        """
        Returns (translation, yaw, T_wr) using the EKF-fused estimate.
        Falls back to raw ZED if EKF is not yet initialized.
        """
        raw_trans, raw_yaw, raw_T = self._zed.get_pose()
        confidence = self._zed.get_tracking_confidence()
        # Update EKF with ZED measurement, with timestamp debounce to avoid
        # double-updating if get_pcd_pose() already applied this same frame.
        try:
            pose_msg = self._zed._sub_get("zed/pose")
            zed_ts = float(pose_msg[18]) if (pose_msg is not None and len(pose_msg) > 18) else -1.0
            self._apply_zed_update(raw_trans, raw_yaw, confidence=confidence, zed_ts=zed_ts)
        except Exception:
            pass

        if not self._initialized:
            return raw_trans, raw_yaw, raw_T   # pre-initialization fallback

        T_fused    = self._get_fused_transform()
        trans_fused = T_fused[:3, 3].astype(np.float32)
        yaw_fused   = float(np.arctan2(-T_fused[2, 0], T_fused[0, 0]))
        return trans_fused, yaw_fused, T_fused

    def get_ekf_uncertainty(self) -> np.ndarray:
        """Return 1-sigma [sigma_px, sigma_pz, sigma_yaw] from the EKF covariance."""
        with self._lock:
            return self._ekf.get_uncertainty()

    def get_camera_info(self):
        return self._zed.get_camera_info()

class Slam:
    """
    Shared SLAM container:
      - Holds latest ZED connection, map, grid, goal, and path.
      - Spawns threads for mapping (MapManager), visualization (Viser), and planning (A*).
      - Keeps Yor RPC streaming updated paths.
    """

    def __init__(
        self,
        *,
        target_hz: float,
        duration_s: float,
        load_map: bool,
        save_map: bool,
        map_path: Optional[str],
        yor_host: str = YOR_RPC_HOST,
        yor_port: int = YOR_RPC_PORT,
        zed_host: str = "127.0.0.1",
        zed_port: int = ZED_PUB_PORT,
        zed_up_axis: str = "y",
        path_step_m: Optional[float] = None,
        use_ekf: bool = False,
        traction_model: Optional[str] = None,
        traction_scaler: Optional[str] = None,
    ):
        self.target_hz = target_hz
        self.duration_s = duration_s
        self.load_map = load_map
        self.save_map = save_map
        self.map_path = map_path
        self._traction_model_path  = traction_model
        self._traction_scaler_path = traction_scaler
        self._traction_thread: Optional[threading.Thread] = None

        # Full subscriber — used by everything except the mapping loop:
        # camera_info, rgb/depth for Viser, pose display, _wait_for_datastream.
        _zed_raw = ZedSub(host=zed_host, port=zed_port, up_axis=zed_up_axis)

        # Mapping-only subscriber: PCD + POSE only — no IMAGE or DEPTH traffic.
        # At 30 Hz, IMAGE (~900 KB) + DEPTH (~1.2 MB) = ~63 MB/s that the full
        # subscriber must deserialize even though mapping never uses those topics.
        # A dedicated slim subscriber cuts fetch time from ~140 ms → ~33 ms.
        _zed_mapping = ZedSub(
            host=zed_host, port=zed_port, up_axis=zed_up_axis,
            topics=[PCD_TOPIC, POSE_TOPIC],
        )

        # Single SirBridge instance shared across EKF sources and path sender
        self._sir_bridge = SirBridge()
        self._sir_bridge.connect()

        if use_ekf:
            print("[Slam] EKF fusion enabled — wrapping ZedSub with EKFSlamSource")
            self.datastream = EKFSlamSource(
                zed_sub=_zed_raw,
                yor_client=self._sir_bridge,
                predict_hz=20.0,
            )
            # Slim EKF source for mapping: same EKF predict thread is not shared,
            # but mapping only needs get_pcd_pose() which is PCD+POSE.
            self.mapping_datastream = EKFSlamSource(
                zed_sub=_zed_mapping,
                yor_client=self._sir_bridge,
                predict_hz=20.0,
            )
        else:
            self.datastream = _zed_raw
            self.mapping_datastream = _zed_mapping
        self.map_manager = MapManager()
        self.label_store = SemanticLabelStore()
        self._cone_rpc_lock = threading.Lock()
        self.yor_client = self._sir_bridge

        # Give the bridge access to the EKF/ZED fused pose for slip correction
        self._sir_bridge.set_pose_provider(self.datastream.get_pose)
        self.server = start_viser_server(host="0.0.0.0", port=8099)
        self.path_step_m = None if path_step_m is None else max(0.0, float(path_step_m))


        self.static_map_pts = None
        self.static_map_cols = None

        # Planner/grid configuration
        self.grid_params = Grid2DParams(
            res_m=0.05,
            x_half_m=4.0,
            z_front_m=6.0,
            z_back_m=2.0,
            floor_band_m=0.25,
            min_obst_h_m=0.25,
            max_obst_h_m=1.80,
            robot_radius_m=0.15,       # reduced: 0.3 inflated everything to red
            ego_centric=False,
            auto_size_from_map=True,
            auto_size_margin_m=0.5,
            min_pts_per_obst_cell=5,   # increased: filter noise/robot-body artifacts
            ttl=40,
            min_world_width_m=4.0,
            min_world_height_m=4.0,
        )

        # Mutable latest-state fields
        self.latest_map = None
        self.latest_grid: Optional[Tuple[np.ndarray, dict, np.ndarray]] = None
        self.latest_goal: Optional[Tuple[float, float]] = None
        self.latest_path = None

        # Threads / runtime flags
        self.grid_thread: Optional[StaticGridWithLiveOverlayThread] = None
        self.planner: Optional[AStarPlannerThread] = None

        self.viser_mirror: Optional[ViserMirrorThread] = None
        self.path_thread: Optional[threading.Thread] = None
        self.state_thread: Optional[threading.Thread] = None

        self.running = False
        self.nav_initialized = False
        self.map_loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def run(self):
        self.running = True

        if not self._wait_for_datastream():
            self.stop()
            return

        self._start_mapping()

        # Start traction inference thread if model files are provided
        if self._traction_model_path and self._traction_scaler_path:
            self._traction_thread = threading.Thread(
                target=self._traction_loop, name="traction", daemon=True
            )
            self._traction_thread.start()

        # Start monitoring latest state in the background
        self.state_thread = threading.Thread(target=self._state_monitor_loop, daemon=True)
        self.state_thread.start()

        # If a map was preloaded, bring up navigation immediately
        if self.map_loaded:
            self._start_planning_stack()
        else:
            # Stage 1: auto-start planning stack once enough map points exist
            self._planning_autostart_thread = threading.Thread(
                target=self._autostart_planning_loop, daemon=True
            )
            self._planning_autostart_thread.start()

        start = time.time()
        try:
            while self.running:
                if self.duration_s and (time.time() - start) >= self.duration_s:
                    break

                self._log_status()

                key = waitKey(1) & 0xFF
                if key == ord("q"):
                    self._on_freeze_and_nav()
                elif key == ord("w"):
                    break

                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\n[slam_node_new] Ctrl+C received; shutting down.")
        finally:
            # ZMQ context.term() can block if the publisher is still running.
            # Schedule a hard exit after 4 s so the terminal never freezes.
            def _force_exit():
                time.sleep(4.0)
                print("[slam_node_new] Forced exit (ZMQ did not drain in time).")
                os._exit(0)
            threading.Thread(target=_force_exit, daemon=True).start()
            self.stop()
 
    def set_goal(self, x: float, z: float):
        """Public entry point to set a navigation goal in world coordinates."""
        self.latest_goal = (float(x), float(z))
        if self.planner is not None:
            self.planner.set_goal_world(x, z)
            print(f"[slam_node_new] Goal set to ({x:.2f}, {z:.2f})")
        else:
            print("[slam_node_new] Cannot set goal: Planner not initialized.")
 
    def stop(self):
        self.running = False


        self.map_manager.stop_mapping()
        self.datastream.stop()
        if self.grid_thread is not None:
            self.grid_thread.stop()
        if self.planner is not None:
            self.planner.stop()
        if self.viser_mirror is not None:
            self.viser_mirror.stop()


        # Optional save
        if self.save_map and self.map_path:
            try:
                map_cloud = self.map_manager.get_map()
            except Exception:
                map_cloud = None

            if map_cloud is not None and len(map_cloud) > 0:
                try:
                    self.map_manager.save_map(map_cloud, self.map_path, min_log_odds=None)
                    print(f"[slam_node_new] Saved unified map to: {self.map_path}")
                except Exception as e:
                    print(f"[slam_node_new] Failed to save map '{self.map_path}': {e}", file=sys.stderr)

            # Save semantic labels sidecar (even if map was empty — labels are independent)
            if len(self.label_store) > 0:
                try:
                    labels_path = SemanticLabelStore.sidecar_path(self.map_path)
                    self.label_store.save(labels_path)
                except Exception as e:
                    print(f"[slam_node_new] Failed to save labels: {e}", file=sys.stderr)

        print("[slam_node_new] Stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_floating_points(self, pts: np.ndarray, cols: np.ndarray | None,
                            voxel_m: float | None = None, min_pts: int = 3,
                            floor_y: float | None = None, floor_band_m: float = 0.25):
        """Filter out floating points from the static map by voxelizing and keeping only voxels with enough points, while always keeping points close to the floor. Optimized with PyTorch/GPU."""
        if pts is None or len(pts) == 0:
            return pts, cols
        
        # Use GPU-accelerated filtering for speed
        import torch
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        if voxel_m is None:
            voxel_m = float(getattr(self.grid_params, "res_m", 0.05))

        # Robust floor estimate from lowest 1% of points (y-up).
        if floor_y is None:
            floor_y = float(np.percentile(pts[:, 1], 1.0))

        # 1. Point Cloud to GPU
        pts_t = torch.tensor(np.ascontiguousarray(np.ascontiguousarray(pts))).to(device).float()
        
        # 2. Voxelize and Unique on GPU (Fast CUDA kernel)
        vox_t = torch.floor(pts_t / voxel_m).to(torch.int32)
        _, inverse, counts = torch.unique(vox_t, dim=0, return_inverse=True, return_counts=True)
        keep_t = counts[inverse] >= int(min_pts)

        # 3. Always keep points close to floor
        keep_floor_t = torch.abs(pts_t[:, 1] - floor_y) <= float(floor_band_m)
        keep_t = keep_t | keep_floor_t

        # 4. Result back to CPU
        pts_f = pts_t[keep_t].cpu().numpy()
        if cols is None:
            return pts_f, None
        
        cols_t = torch.tensor(np.ascontiguousarray(np.ascontiguousarray(cols))).to(device)
        return pts_f, cols_t[keep_t].cpu().numpy()
        
    def _wait_for_datastream(self) -> bool:
        """Wait for the ZED datastream to be ready, with an optional timeout."""
        t0 = time.time()
        while not self.datastream.ready():
            if self.duration_s and (time.time() - t0) >= self.duration_s:
                print("[slam_node_new] Timed out waiting for first ZED frames.")
                return False
            if (time.time() - t0) > 5.0:
                print("[slam_node_new] No frames in 5s; continuing anyway.")
                break
            time.sleep(0.01)

        while self.datastream.get_camera_info() is None:
            if self.duration_s and (time.time() - t0) >= self.duration_s:
                print("[slam_node_new] Timed out waiting for camera info.")
                return False
            if (time.time() - t0) > 7.0:
                print("[slam_node_new] No camera info in 7s; continuing anyway.")
                break
            time.sleep(0.01)

        cam_info = self.datastream.get_camera_info()
        if cam_info is not None:
            self.map_manager.set_camera_info(cam_info)

        return True

    def _start_mapping(self):
        """Start the mapping thread, optionally loading from a previous map."""
        ds = self.mapping_datastream  # slim PCD+POSE subscriber — avoids IMAGE/DEPTH overhead
        if self.load_map and self.map_path:
            try:
                self.map_manager.start_mapping(ds, load=True, map_path=self.map_path)
                self.map_loaded = True
                print(f"[slam_node_new] Loaded map: {self.map_path}")
                # Load semantic labels from sidecar file if it exists
                labels_path = SemanticLabelStore.sidecar_path(self.map_path)
                try:
                    self.label_store.load(labels_path)
                except Exception as e:
                    print(f"[slam_node_new] Could not load labels: {e}", file=sys.stderr)
                return
            except Exception as e:
                print(f"[slam_node_new] Failed to load map '{self.map_path}': {e}", file=sys.stderr)

        self.map_manager.start_mapping(ds, target_hz=self.target_hz)
        print("[slam_node_new] Mapping started. Press 'q' to stop mapping and freeze the static map.")

    def _on_freeze_and_nav(self):
        """Handle the 'q' key: stop mapping, freeze the static map, and start the planning stack."""
        if not self.map_loaded:
            print("\n[slam_node_new] 'q' pressed: stopping mapping and freezing static map.")
            self.map_manager.stop_mapping()
            self.map_loaded = True

        if not self._start_planning_stack():
            return

    def _autostart_planning_loop(self):
        """Poll the map every 0.5 s; start the planning stack once ≥500 voxels exist."""
        MIN_VOXELS = 500
        while self.running and (self.grid_thread is None):
            try:
                vmap = self.map_manager.get_voxel_map()
                if vmap is not None and len(vmap) >= MIN_VOXELS:
                    print(f"\n[slam_node_new] Map has {len(vmap):,} voxels — auto-starting planning stack.")
                    self._start_planning_stack()
                    return
            except Exception:
                pass
            time.sleep(0.5)

    def _start_planning_stack(self) -> bool:
        """Start the grid and planner threads if they aren't already running. Returns True if the planner is ready to accept goals."""
        if self.grid_thread is not None and self.planner is not None:
            return True

        voxel_map_init = self.map_manager.get_voxel_map()
        if voxel_map_init is None or len(voxel_map_init) == 0:
            print("[slam_node_new] Map not ready yet.")
            return False

        # Use voxel map's native 2D extraction for the initial grid.
        # get_points_colors() already applies the height-band filter internally.
        init_result = voxel_map_init.get_2d_grid(self.grid_params)
        if init_result is None:
            print("[slam_node_new] get_2d_grid returned None — map may be too sparse.")
            return False
        base_grid, base_meta, base_cost, floor_y, kernel = init_result

        pts_np, cols_np = voxel_map_init.get_points_colors()
        if pts_np is not None:
            print(f"[slam_node_new] Initial map: {len(pts_np):,} occupied voxels")
        self.static_map_pts = pts_np
        self.static_map_cols = cols_np

        # Provider: refresh the 2D grid from the live voxel map (called at base_refresh_hz)
        voxel_map = self.map_manager.get_voxel_map()

        def _base_grid_provider():
            """Return a fresh (grid, meta, cost, floor_y, kernel) from the live voxel map."""
            try:
                if voxel_map is None or len(voxel_map) == 0:
                    return None
                result = voxel_map.get_2d_grid(self.grid_params)
                if result is not None:
                    pts, cols = voxel_map.get_points_colors()
                    if pts is not None:
                        self.static_map_pts = pts
                        self.static_map_cols = cols
                return result
            except Exception as e:
                print(f"[slam_node_new] base_grid_provider error: {e}")
                return None

        # Only supply the live provider when mapping is still running
        provider = _base_grid_provider if not self.map_loaded else None

        self.grid_thread = StaticGridWithLiveOverlayThread(
            datastream=self.datastream,
            base_grid=base_grid,
            base_meta=base_meta,
            base_cost_map=base_cost,
            floor_y=floor_y,
            kernel=kernel,
            grid_params=self.grid_params,
            hz=10.0,
            base_grid_provider=provider,
            base_refresh_hz=1.0,
            # Use pre-projected world pts from the integration worker buffer —
            # avoids ZED subscriber race and redundant CPU point-cloud re-projection.
            pts_provider=self.map_manager.get_latest_frame_pts,
        )
        self.grid_thread.start()

        self.planner = AStarPlannerThread(
            self.grid_thread,
            treat_unknown_as_obstacle=False,
            near_obstacle_radius_cells=4,
            near_obstacle_penalty=0.5,
            log_entity_path_3d="world/path",
            log_entity_grid_overlay="world/local_grid_with_path",
            hold_last_good=False,
        )

        def ensure_nav_started():
            """Start the planner thread if it hasn't been started yet. 
                This is called lazily on the first set_goal_world call, 
                which allows us to defer starting the planner until we have a goal and the latest grid available."""
            if self.nav_initialized:
                return

            print("\n[slam_node_new] Starting planner (auto).")


            try:
                self.planner.start()
                self.nav_initialized = True
            except Exception as e:
                print(f"[slam_node_new] Failed to start planner: {e}")

        _orig_set_goal_world = self.planner.set_goal_world

        def _set_goal_world_autostart(xw: float, zw: float):
            """Wrapper around planner.set_goal_world that also ensures the planner thread is started and the latest goal is stored."""
            self.latest_goal = (float(xw), float(zw))
            ensure_nav_started()
            _orig_set_goal_world(xw, zw)

        self.planner.set_goal_world = _set_goal_world_autostart

        if self.server is not None:
            def map_provider():
                # If frozen/static map loaded, just return the static points
                if self.map_loaded:
                    return self.static_map_pts, self.static_map_cols

                # Single unified map provider (1M points max) with density filtering for "pretty" visuals
                _MAX_PTS = 1_000_000
                try:
                    vmap = self.map_manager.get_voxel_map()
                    if vmap is not None and len(vmap) > 0:
                        # Goldilocks Threshold: Balanced between density and noise (seen ~3 times)
                        pts, cols = vmap.get_points_colors(max_points=_MAX_PTS, min_log_odds=1.8)
                        if pts is not None and len(pts) > 0:
                            fy = getattr(self.grid_thread, "floor_y", None)
                            # Identify voxels in the floor zone (30cm band) for protection
                            is_floor = np.abs(pts[:, 1] - fy) <= 0.30 if fy is not None else np.zeros(len(pts), dtype=bool)

                            # 1. Luminance Filter: remove dark speckles (R+G+B < 20)
                            # But ALWAYS keep floor points regardless of color
                            lum = cols.sum(axis=1)
                            keep_lum = (lum >= 20) | is_floor
                            pts = pts[keep_lum]
                            cols = cols[keep_lum]

                            if len(pts) > 0:
                                # 2. Balanced Density filter: 4 voxels in 8cm radius
                                # Re-calculate is_floor for the potentially filtered points
                                fy = getattr(self.grid_thread, "floor_y", None)
                                pts, cols = self._filter_floating_points(
                                    pts, cols, 
                                    voxel_m=0.08, 
                                    min_pts=4, 
                                    floor_y=fy,
                                    floor_band_m=0.30
                                )
                        return pts, cols

                    live = self.map_manager.get_map()
                    if live is not None and len(live) > 0:
                        return live.cpu_numpy()
                except Exception as e:
                    print(f"[slam_node_] Viser map_provider error: {e}")

                return self.static_map_pts, self.static_map_cols

            origin_xy = (0.0, 0.0)
            grid_res_viser = self.grid_params.res_m
            floor_y_viser = 0.0

            for _ in range(50):  # ~2.5 s of retries
                grid_codes, meta, T_wr = self.grid_thread.get_grid()
                if grid_codes is not None and meta is not None:
                    grid_res_viser = float(meta.get("cell_size_m", self.grid_params.res_m))
                    if not meta.get("ego_centric", True) and "x0" in meta and "z_top" in meta:
                        H, W = grid_codes.shape[:2]
                        x0 = float(meta["x0"])
                        z_top = float(meta["z_top"])
                        z_min = z_top - H * grid_res_viser
                        origin_xy = (x0, z_min)
                    floor_y_viser = float(meta.get("floor_y_est", 0.0))
                    break
                time.sleep(0.05)

            # If meta didn't provide floor_y, estimate it from the live map's
            # minimum Y so that the Viser robot marker and click-ray land correctly.
            if floor_y_viser == 0.0:
                try:
                    vmap = self.map_manager.get_voxel_map()
                    if vmap is not None:
                        pts_check, _ = vmap.get_points_colors(max_points=5000)
                        if pts_check is not None and len(pts_check) > 0:
                            floor_y_viser = float(np.percentile(pts_check[:, 1], 2.0))
                except Exception:
                    pass

            self.viser_mirror = ViserMirrorThread(
                self.server,
                grid_thread=self.grid_thread,
                planner_thread=self.planner,
                pose_source=self.datastream,
                origin_xy=origin_xy,
                grid_res_m=grid_res_viser,
                floor_y=floor_y_viser,
                hz=20.0,
                grid_update_hz=10.0,
                map_update_hz=5.0,
                map_provider=map_provider,
                static_map_once=False,
                robot_radius_m=self.grid_params.robot_radius_m,
                label_store=self.label_store,
                traction_source=self._sir_bridge.get_traction_info,
            )
            # Propagate voxel size so Viser uses overlapping point splats
            vmap = self.map_manager.get_voxel_map()
            if vmap is not None:
                self.viser_mirror.voxel_size = float(vmap.vs)
            self.viser_mirror.start()

        if self.path_thread is None:
            self.path_thread = threading.Thread(target=self._path_sender_loop, daemon=True)
            self.path_thread.start()

        return True


    def _auto_path_step_m(self) -> float:
        """Choose a reasonable densify step from the current grid resolution.

        - Uses the live grid meta if available (cell_size_m).
        - Falls back to Grid2DParams.res_m.
        """
        cell = float(getattr(self.grid_params, "res_m", 0.05))
        try:
            if self.grid_thread is not None:
                _, meta, _ = self.grid_thread.get_grid()
                if isinstance(meta, dict):
                    cell = float(meta.get("cell_size_m", cell))
        except Exception as e:
            print(f"[slam_node_new] Failed to get cell_size_m: {e}")

        # Heuristic: ~2 cells per waypoint, clamped.
        return float(np.clip(2.0 * cell, 0.05, 0.15))


    def _densify_path(self, path_world):
        """Densify the path by adding intermediate waypoints so that consecutive points are at most self.path_step_m apart."""
        if not path_world or len(path_world) < 2:
            return path_world
        step = self.path_step_m
        if step is None:
            step = self._auto_path_step_m()
        if step <= 0.0:
            return path_world
        out = [path_world[0]]
        for (x0, z0), (x1, z1) in zip(path_world, path_world[1:]):
            dx = float(x1) - float(x0)
            dz = float(z1) - float(z0)
            dist = float(np.hypot(dx, dz))
            if dist <= step:
                out.append((float(x1), float(z1)))
                continue
            n = max(1, int(dist / step))
            for i in range(1, n):
                t = i / float(n)
                out.append((float(x0 + t * dx), float(z0 + t * dz)))
            out.append((float(x1), float(z1)))
        return out

    def _state_monitor_loop(self):
        """Background loop to keep latest map, grid, and path updated for the main thread and RPC calls."""
        while self.running:
            try:
                self.latest_map = self.map_manager.get_map()
            except Exception:
                self.latest_map = None

            if self.grid_thread is not None:
                try:
                    self.latest_grid = self.grid_thread.get_grid()
                except Exception:
                    self.latest_grid = None

            if self.planner is not None:
                try:
                    self.latest_path = self.planner.get_latest_path_world()
                except Exception:
                    self.latest_path = None

            time.sleep(0.2)

    def _reset_yor_client(self):
        """No-op: SirBridge manages its own serial reconnection."""
        pass


    def _path_sender_loop(self):
        """Background loop to send the latest path to Yor via RPC whenever it changes."""
        last_sent = None
        last_fail_t = 0.0

        while self.running:
            time.sleep(0.1)
            if self.planner is None or not self.nav_initialized:
                continue

            try:
                path_world = self.planner.get_latest_path_world()
            except Exception as e:
                path_world = None
                now = time.time()
                if now - last_fail_t > 1.0:
                    print(f"[slam_node_new] planner get_latest_path_world failed: {e}")
                    last_fail_t = now

            if not path_world:
                continue

            path_dense = self._densify_path(path_world)

            # Normalize to plain Python floats (helps serialization + consistent equality)
            path_dense = [(float(x), float(z)) for (x, z) in path_dense]

            if path_dense == last_sent:
                continue

            # ---- RPC call (REQ sockets must be serialized) ----
            try:
                with self._cone_rpc_lock:
                    self.yor_client.follow_path(path_dense)
                last_sent = list(path_dense)
                self.latest_path = path_dense

            except Exception as e:
                msg = str(e)
                now = time.time()

                # Don't spam console
                if now - last_fail_t > 1.0:
                    print(f"[slam_node_new] follow_path RPC failed: {e}")
                    last_fail_t = now

                # EFSM / stuck REQ socket -> recreate client
                if "Operation cannot be accomplished in current state" in msg:
                    print("[slam_node_new] RPC socket stuck (EFSM). Resetting RPCClient...")
                    with self._cone_rpc_lock:
                        self._reset_yor_client()

                # allow retry later
                last_sent = None
                time.sleep(0.25)


    def _traction_loop(self):
        """Background thread: run traction inference and push value to SirBridge."""
        import cv2
        import onnxruntime as ort
        import joblib

        IMG_SIZE = 224
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        try:
            session = ort.InferenceSession(
                self._traction_model_path,
                providers=["CPUExecutionProvider"],
            )
            scaler = joblib.load(self._traction_scaler_path)
        except Exception as e:
            print(f"[Traction] Failed to load model/scaler: {e}")
            return

        input_name  = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"[Traction] Running inference from {self._traction_model_path}")

        while self.running:
            try:
                img_msg, _, _ = self.datastream.get_rgb_depth_pose()
            except Exception:
                time.sleep(0.05)
                continue

            try:
                bgr = img_msg
                h = bgr.shape[0]
                floor = bgr[h // 2:, :]
                rgb   = cv2.cvtColor(floor, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE),
                                     interpolation=cv2.INTER_LINEAR)
                img = resized.astype(np.float32) / 255.0
                img = (img - MEAN) / STD
                tensor = img.transpose(2, 0, 1)[np.newaxis]

                pred_norm = session.run([output_name], {input_name: tensor})[0]
                traction  = float(
                    scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()[0]
                )
                self._sir_bridge.set_traction(traction)
            except Exception as e:
                print(f"[Traction] inference error: {e}")

            time.sleep(0.033)   # ~30 Hz

    def _log_status(self):
        vmap = self.map_manager.get_voxel_map()
        n_vox = len(vmap) if vmap is not None else 0
        _, poses = self.map_manager.get_state()
        print(f"\r[slam_node_new] voxels={n_vox:,}  poses={len(poses)}", end="")

        if self.map_manager.last_error:
            print("\n[slam_node_new] MapManager error:", self.map_manager.last_error)
            self.map_manager.last_error = None


def main():
    parser = argparse.ArgumentParser("SLAM node (pubsub ZED + MapManager + grid + A* + Viser)")
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="target mapping rate (Hz); 0 = as fast as possible",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after N seconds (0 = run until Ctrl+C)",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="whether to load previous map instead of starting new mapping",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="whether to save the map on exit",
    )
    parser.add_argument(
        "--map-path",
        type=str,
        default=None,
        help="optional .npz path to save/load map",
    )
    parser.add_argument(
        "--yor-host",
        type=str,
        default=YOR_RPC_HOST,
        help="Yor RPC host (follow_path via RPC)",
    )
    parser.add_argument(
        "--yor-port",
        type=int,
        default=YOR_RPC_PORT,
        help="Yor RPC port (follow_path via RPC)",
    )
    parser.add_argument(
        "--path-step-m",
        type=float,
        default=None,
        help="dense waypoint spacing for follow_path (meters; 0 disables)",
    )

    parser.add_argument(
        "--zed-up-axis",
        type=str,
        default="y",
        choices=["y", "z"],
        help="up axis for incoming ZED frames (y=default, z=swap Y/Z into Y-up)",
    )
    parser.add_argument(
        "--ekf",
        dest="ekf",
        action="store_true",
        default=False,
        help="enable EKF fusion (off by default): swerve odometry predicts at 20 Hz, "
             "ZED corrects each frame with adaptive noise + Mahalanobis gating",
    )
    parser.add_argument(
        "--traction-model",
        type=str,
        default=None,
        help="path to traction ONNX model (e.g. ~/robot/model.onnx); "
             "if provided, traction-based speed scaling is enabled",
    )
    parser.add_argument(
        "--traction-scaler",
        type=str,
        default=None,
        help="path to traction label scaler (e.g. ~/robot/label_scaler.pkl)",
    )
    # ---- End options ----
    args = parser.parse_args()

    slam = Slam(
        target_hz=args.hz,
        duration_s=args.duration,
        load_map=args.load,
        save_map=args.save,
        map_path=args.map_path,
        yor_host=args.yor_host,
        yor_port=args.yor_port,
        path_step_m=args.path_step_m,
        zed_up_axis=args.zed_up_axis,
        use_ekf=args.ekf,
        traction_model=args.traction_model,
        traction_scaler=args.traction_scaler,
    )

    slam.run()


if __name__ == "__main__":
    main()
    import os
    os._exit(0)