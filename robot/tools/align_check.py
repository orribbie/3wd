#!/usr/bin/env python3
"""
align_check.py — Axis alignment verification for SIR 3WD + SLAM.

Uses the same command pattern as the working teleop (g command only):
  - Forward: g STEP 0 0
  - CCW Turn: g 0 0 +90

Run with zed_pub_node.py already running:
    conda activate slam && python tools/align_check.py
"""

import math, os, sys, time
import numpy as np
import zmq

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JETSON = os.path.join(_ROOT, "..", "sir", "jetson")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _JETSON)

from commlink import Subscriber
from serial_link import SerialLink

# ── Config ────────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 115200
ZED_HOST, ZED_PORT, POSE_TOPIC = "127.0.0.1", 6000, "zed/pose"
MOVE_STEP    = 5.0    # grid units = 0.5 m
SETTLE_S     = 1.5
DONE_TIMEOUT = 45.0
MIN_MOTION_M = 0.20   # minimum displacement to count as PASS

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# ── Pose helpers ──────────────────────────────────────────────────────────────

def _quat_to_R(q):
    x,y,z,w = float(q[0]),float(q[1]),float(q[2]),float(q[3])
    x2,y2,z2=x+x,y+y,z+z
    xx,xy,xz=x*x2,x*y2,x*z2
    yy,yz,zz=y*y2,y*z2,z*z2
    wx,wy,wz=w*x2,w*y2,w*z2
    return np.array([[1-(yy+zz),xy-wz,xz+wy],
                     [xy+wz,1-(xx+zz),yz-wx],
                     [xz-wy,yz+wx,1-(xx+yy)]], dtype=np.float32)

def pose_from_data(data):
    """Return (px, pz, yaw_rad) from [qx,qy,qz,qw,tx,ty,tz]."""
    q = data[:4]; t = data[4:7]
    R = _quat_to_R(q)
    px, pz = float(t[0]), float(t[2])
    yaw = float(math.atan2(-R[2,0], R[0,0]))
    return px, pz, yaw

def get_pose(sub, timeout_s=5.0):
    sock = sub._topic_sockets[POSE_TOPIC]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not sock.poll(timeout=500): continue
        last = None
        while True:
            try: last = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again: break
        if last is None: continue
        _, data = sub._serializer.deserialize(last)
        return pose_from_data(data[0:7])
    raise RuntimeError(f"No ZED pose in {timeout_s}s")

def wait_for_motion(sub, px0, pz0, yaw0, min_d=0.06, timeout=8.0):
    """Poll until ZED shows meaningful motion, then return final pose."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        px1, pz1, yaw1 = get_pose(sub)
        dyaw = ((yaw1-yaw0+math.pi)%(2*math.pi))-math.pi
        if abs(px1-px0)>min_d or abs(pz1-pz0)>min_d or abs(dyaw)>min_d:
            return px1, pz1, yaw1
        time.sleep(0.2)
    return get_pose(sub)

# ── Serial helpers ────────────────────────────────────────────────────────────

_SKIP = ("[log auto-stopped]", "[log off]", "[speed]", "[pos set]")

def wait_done(link, timeout=DONE_TIMEOUT):
    """Drain serial until [done] or timeout. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = link.get_response(timeout=0.3)
        if r is None: continue
        if any(s in r for s in _SKIP): continue
        if "[done]" in r: return True
        if "[stopped]" in r: return False
    print("  ⚠  wait_done timed out")
    return False

def send_cmd(link, x, y, th):
    """Send 'g x y th' and consume ack lines. Does NOT wait for [done]."""
    cmd = f"g {x:.2f} {y:.2f} {th:.1f}"
    link.send(cmd)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ack = link.get_response(timeout=0.5)
        if ack is None: break
        if any(s in ack for s in _SKIP): continue
        if ack.strip() == cmd.strip(): continue      # echo
        if "[goto accepted]" in ack: break           # success
        if "[done]" in ack: break                    # immediate finish (0-dist move)
        print(f"  ⚠  Unexpected ack: {ack!r}")
        break
    return cmd

def send_zero(link):
    """Reset robot internal pose to (0,0,0)."""
    link.send("z")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        r = link.get_response(timeout=0.5)
        if r is None or "[reset]" in r: break
    time.sleep(0.3)

# ── Tests ─────────────────────────────────────────────────────────────────────

def run_test(label, link, sub, cmd_x, cmd_y, cmd_th, is_rotation=False):
    print(f"\n{'─'*60}")
    print(f"  TEST: {label}")
    print(f"  Command: g {cmd_x:.1f} {cmd_y:.1f} {cmd_th:.1f}")

    send_zero(link)
    time.sleep(0.2)
    px0, pz0, yaw0 = get_pose(sub)
    print(f"  Before: px={px0:.3f}  pz={pz0:.3f}  yaw={math.degrees(yaw0):.1f}°")

    send_cmd(link, cmd_x, cmd_y, cmd_th)
    done = wait_done(link)
    if not done:
        print(f"  ⚠  [done] not received — continuing anyway")
    time.sleep(SETTLE_S)

    px1, pz1, yaw1 = wait_for_motion(sub, px0, pz0, yaw0,
                                      min_d=0.04 if is_rotation else 0.06)
    dpx  = px1 - px0
    dpz  = pz1 - pz0
    dyaw = ((yaw1 - yaw0 + math.pi) % (2*math.pi)) - math.pi

    print(f"  After:  px={px1:.3f}  pz={pz1:.3f}  yaw={math.degrees(yaw1):.1f}°")
    print(f"  Δpx={dpx:+.3f}m   Δpz={dpz:+.3f}m   Δyaw={math.degrees(dyaw):+.1f}°")

    if is_rotation:
        ok = abs(dyaw) > 0.7
        sign = "CCW ✓" if dyaw > 0 else "CW ✗ (negate heading in bridge)"
        print(f"  {'PASS' if ok else FAIL}  Δyaw={math.degrees(dyaw):+.1f}°  ({sign})")
    else:
        total = math.hypot(dpx, dpz)
        ok = total >= MIN_MOTION_M
        dom = "pz" if abs(dpz) >= abs(dpx) else "px"
        dom_sign = "−" if (dpz < 0 if dom=="pz" else dpx < 0) else "+"
        print(f"  {PASS if ok else FAIL}  Total={total:.3f}m  dominant={dom_sign}{dom}")

    return ok, dpx, dpz, dyaw

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SIR 3WD Axis Alignment Check  (Forward + CCW Turn)")
    print("  Ensure zed_pub_node.py is running.")
    print("  Clear ~0.6m in front of and around the robot.")
    print("=" * 60)
    input("\n  Press Enter to start…")

    link = SerialLink(port=SERIAL_PORT, baud=BAUD_RATE)
    print(f"\n  Opening {SERIAL_PORT} — waiting 4s for Arduino boot…")
    link.connect()
    link.drain_responses()
    print("  Serial connected.")

    sub = Subscriber(host=ZED_HOST, port=ZED_PORT, topics=[POSE_TOPIC])
    time.sleep(1.0)
    print("  ZED subscriber connected.")

    results = []
    try:
        # Test 1: Forward (+X Arduino)
        ok, dpx, dpz, _ = run_test(
            "Forward  (g 5 0 0 — same as teleop UP×5)",
            link, sub,
            cmd_x=MOVE_STEP, cmd_y=0, cmd_th=0,
        )
        results.append(("Forward", ok))
        fwd_dpx, fwd_dpz = dpx, dpz

        # Test 2: CCW rotation (+theta Arduino)
        ok, _, _, dyaw = run_test(
            "CCW Rotation  (g 0 0 90 — same as teleop LEFT×6)",
            link, sub,
            cmd_x=0, cmd_y=0, cmd_th=90,
            is_rotation=True,
        )
        results.append(("CCW Rotation", ok))

    finally:
        link.send("stop")
        link.disconnect()
        sub.stop()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results:
        print(f"  {PASS if ok else FAIL}  {name}")

    print(f"""
  AXIS MAP (copy these into sir_bridge._slam_to_arduino):
  ──────────────────────────────────────────────────────
  Arduino +X (Forward) → SLAM: Δpx={fwd_dpx:+.3f}  Δpz={fwd_dpz:+.3f}
  Arduino +θ (CCW)     → SLAM: Δyaw={math.degrees(dyaw):+.1f}°

  Expected:
    Arduino +X  →  dominant in SLAM −pz  (pz should decrease)
    Arduino +θ  →  SLAM yaw increases    (dyaw > 0)

  If pz increases instead of decreasing: negate x_grid in bridge.
  If dyaw is negative: negate heading_deg in bridge.
{'='*60}""")


if __name__ == "__main__":
    main()
