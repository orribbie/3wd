#!/usr/bin/env python3
"""
zed_pose_check.py — Live ZED pose monitor.

Connects to zed_pub_node.py on commlink port 6000 and prints
the current px / pz / yaw at ~5 Hz so you can confirm:

  1. ZED is publishing (not frozen)
  2. Pose updates when you move the robot by hand
  3. Axis signs are correct (forward = px, lateral = pz, CCW = +yaw)

Run from the robot/ directory:
    conda activate slam
    python tools/zed_pose_check.py

Requirements:
  - zed_pub_node.py must be running in another terminal.
"""

import math
import os
import sys
import time
import numpy as np
import zmq

# ── Path setup ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from commlink import Subscriber   # noqa: E402


# ── Pose math (inlined — no torch dependency) ───────────────────────────────

def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x+x, y+y, z+z
    xx, xy, xz = x*x2, x*y2, x*z2
    yy, yz, zz = y*y2, y*z2, z*z2
    wx, wy, wz = w*x2, w*y2, w*z2
    return np.array([
        [1.-(yy+zz), xy-wz,      xz+wy],
        [xy+wz,      1.-(xx+zz), yz-wx],
        [xz-wy,      yz+wx,      1.-(xx+yy)],
    ], dtype=np.float32)


def _pose7_to_pxpzpyyaw(data):
    """Extract (px, pz, py, yaw_rad) from [qx,qy,qz,qw,tx,ty,tz]."""
    qt7 = np.asarray(data[:7], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_to_matrix(qt7[:4])
    T[:3, 3]  = qt7[4:7]
    px  = float(T[0, 3])
    pz  = -float(T[2, 3])   # negated: right = -pz (corrected convention)
    py  = float(T[1, 3])
    yaw = float(math.atan2(-T[2, 0], T[0, 0]))
    return px, pz, py, yaw


# ── Config ───────────────────────────────────────────────────────────────────

ZED_HOST   = "127.0.0.1"
ZED_PORT   = 6000
POSE_TOPIC = "zed/pose"
PRINT_HZ   = 5.0
TIMEOUT_MS = 500


# ── Socket helper ────────────────────────────────────────────────────────────

def get_latest_pose(socket, serializer):
    """Drain ZMQ socket and return the most recent pose, or None."""
    last_frames = None
    while True:
        try:
            frames = socket.recv_multipart(flags=zmq.NOBLOCK)
            last_frames = frames
        except zmq.Again:
            break
    if last_frames is None:
        return None
    msg = serializer.deserialize(last_frames)
    _, data = msg
    return _pose7_to_pxpzpyyaw(data)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ZED Live Pose Monitor")
    print(f"  Connecting to zed_pub_node on {ZED_HOST}:{ZED_PORT} ...")
    print("  Press Ctrl-C to quit.")
    print("=" * 64)

    sub        = Subscriber(host=ZED_HOST, port=ZED_PORT, topics=[POSE_TOPIC])
    socket     = sub._topic_sockets[POSE_TOPIC]
    serializer = sub._serializer

    print("\n  Waiting for first pose...")
    if not socket.poll(timeout=5000):
        print("\n  ERROR: No pose received in 5 s.")
        print("  -> Is zed_pub_node.py running?")
        sub.stop()
        return

    pose0 = get_latest_pose(socket, serializer)
    if pose0 is None:
        print("\n  ERROR: Could not deserialize first pose.")
        sub.stop()
        return

    px0, pz0, py0, yaw0 = pose0
    print(f"\n  Origin  px={px0:+.3f}  pz={pz0:+.3f}  yaw={math.degrees(yaw0):+.1f} deg")
    print("-" * 72)
    print(f"  {'px':>7}  {'pz':>7}  {'yaw':>8}    {'dpx':>7}  {'dpz':>7}  {'dyaw':>8}    rate")
    print("-" * 72)

    interval   = 1.0 / PRINT_HZ
    msg_count  = 0
    rate_t0    = time.time()
    last_print = 0.0
    pose       = pose0

    try:
        while True:
            if socket.poll(timeout=TIMEOUT_MS):
                p = get_latest_pose(socket, serializer)
                if p is not None:
                    pose = p
                    msg_count += 1

            now = time.time()
            if now - last_print >= interval:
                px, pz, py, yaw = pose
                dx_world = px - px0
                dz_world = -(pz - pz0)
                cy0, sy0 = math.cos(yaw0), math.sin(yaw0)
                dpx = dx_world * cy0 - dz_world * sy0
                dpz = dx_world * sy0 + dz_world * cy0
                dyaw = math.degrees(yaw - yaw0)
                dyaw = ((dyaw + 180) % 360) - 180
                elapsed = now - rate_t0
                hz = msg_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"  {px:+7.3f}  {pz:+7.3f}  {math.degrees(yaw):+8.1f}  "
                    f"  {dpx:+7.3f}  {dpz:+7.3f}  {dyaw:+8.1f}  "
                    f"  {hz:4.1f} Hz"
                )
                last_print = now

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
    finally:
        sub.stop()


if __name__ == "__main__":
    main()
