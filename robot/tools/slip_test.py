#!/home/slam/miniconda3/envs/slam/bin/python3
"""
slip_test.py — Iterative slip compensation test for the SIR 3WD base.

Sends a single forward command (default: g 10 0 0 = 1.0 m), then uses the
ZED pose to measure actual displacement and issues corrective g commands until
the robot is within TOLERANCE_M of the target, or max iterations is reached.

Speed is derived from surface traction via the same formula as sir_bridge.py.
If the Arduino telemetry traction value is not available, a fixed speed is used.

Usage:
    conda activate slam && python tools/slip_test.py
    conda activate slam && python tools/slip_test.py --dist 0.5    # 0.5 m
    conda activate slam && python tools/slip_test.py --dist 1.0 --spd 80
    conda activate slam && python tools/slip_test.py --max-iter 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import zmq

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JETSON = os.path.join(_ROOT, "..", "sir", "jetson")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _JETSON)

from commlink import Subscriber
from serial_link import SerialLink

# ── Config ────────────────────────────────────────────────────────────────────
SERIAL_PORT   = "/dev/ttyACM0"
BAUD_RATE     = 115200
ZED_HOST      = "127.0.0.1"
ZED_PORT      = 6000
POSE_TOPIC    = "zed/pose"

GRID_UNIT_M   = 0.10          # 1 grid unit = 100 mm
TOLERANCE_M   = 0.05          # 5 cm — stop correcting within this
MAX_ITER      = 6             # safety cap on correction attempts
DONE_TIMEOUT  = 30.0          # s per goto command
SETTLE_S      = 1.0           # s to wait after [done] before reading pose

# Traction → speed mapping (same constants as sir_bridge.py)
TRACTION_LOW  = 900.0
TRACTION_HIGH = 1300.0
SPEED_MIN     = 60
SPEED_MAX     = 100

# Lines from Arduino that are safe to discard
_SKIP = ("[log auto-stopped]", "[log off]", "[speed]", "[pos set]")

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

# ── Pose helpers ──────────────────────────────────────────────────────────────

def _quat_to_R(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    x2, y2, z2 = x+x, y+y, z+z
    xx, xy, xz = x*x2, x*y2, x*z2
    yy, yz, zz = y*y2, y*z2, z*z2
    wx, wy, wz = w*x2, w*y2, w*z2
    return np.array([
        [1-(yy+zz),  xy-wz,     xz+wy],
        [xy+wz,      1-(xx+zz), yz-wx],
        [xz-wy,      yz+wx,     1-(xx+yy)],
    ], dtype=np.float32)


def pose_from_data(data):
    """Return (px, pz, yaw_rad) from 7-float [qx,qy,qz,qw,tx,ty,tz].

    Convention (verified by align_check.py):
      px  = T[0,3]   (decreases when moving right)
      pz  = -T[2,3]  (increases when moving forward)
      yaw = atan2(-R[2,0], R[0,0])
    """
    q = data[:4]
    t = data[4:7]
    R = _quat_to_R(q)
    px  = float(t[0])
    pz  = -float(t[2])    # negated: forward = +pz
    yaw = float(math.atan2(-R[2, 0], R[0, 0]))
    return px, pz, yaw


def get_pose(sub, timeout_s=5.0):
    """Return (px, pz, yaw_rad) from the freshest ZED frame (drains queue)."""
    sock = sub._topic_sockets[POSE_TOPIC]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not sock.poll(timeout=500):
            continue
        last = None
        while True:
            try:
                last = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        if last is None:
            continue
        _, data = sub._serializer.deserialize(last)
        return pose_from_data(data[0:7])
    raise RuntimeError(f"No ZED pose received within {timeout_s} s — is zed_pub_node running?")


# ── Serial helpers ─────────────────────────────────────────────────────────────

def wait_done(link, timeout=DONE_TIMEOUT):
    """Drain serial responses until [done] or timeout. Returns True on done."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = link.get_response(timeout=0.3)
        if r is None:
            continue
        if any(s in r for s in _SKIP):
            continue
        if "[done]" in r:
            return True
        if "[stopped]" in r or "[reset]" in r:
            print(f"  {YELLOW}⚠  Arduino sent: {r!r}{RESET}")
            return False
    print(f"  {YELLOW}⚠  wait_done timed out after {timeout:.0f}s{RESET}")
    return False


def send_spd(link, pct):
    """Send speed command (0–100 %)."""
    pct = max(0, min(100, int(pct)))
    link.send(f"spd {pct}")
    time.sleep(0.05)


def send_goto(link, x_grid, y_grid, heading_deg):
    """Send g command and consume the ack. Does NOT wait for [done]."""
    cmd = f"g {x_grid:.4f} {y_grid:.4f} {heading_deg:.1f}"
    link.send(cmd)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ack = link.get_response(timeout=0.5)
        if ack is None:
            break
        if any(s in ack for s in _SKIP):
            continue
        if ack.strip() == cmd.strip():   # Arduino echo
            continue
        if "[goto accepted]" in ack:
            break
        if "[done]" in ack:              # immediate (zero-distance move)
            return True
        print(f"  {YELLOW}⚠  Unexpected ack: {ack!r}{RESET}")
        break
    return False


def send_zero(link):
    """Reset Arduino pose to (0, 0, 0°)."""
    link.send("z")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        r = link.get_response(timeout=0.5)
        if r is None or "[reset]" in r:
            break
    time.sleep(0.3)


# ── Traction → speed ──────────────────────────────────────────────────────────

def traction_to_speed(traction_value):
    """Map traction sensor value → speed percentage (same as sir_bridge.py)."""
    t = max(TRACTION_LOW, min(TRACTION_HIGH, traction_value))
    frac = (t - TRACTION_LOW) / (TRACTION_HIGH - TRACTION_LOW)
    return int(round(SPEED_MIN + frac * (SPEED_MAX - SPEED_MIN)))


def read_traction(link, timeout=1.5):
    """Try to read a traction value from Arduino telemetry; returns None if unavailable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pkt = link.get_telemetry(block=False)
        if pkt and "traction" in pkt:
            try:
                return float(pkt["traction"])
            except (ValueError, TypeError):
                pass
        time.sleep(0.05)
    return None


# ── Main slip-compensation loop ────────────────────────────────────────────────

def run_slip_test(target_m: float, fixed_spd: int | None, max_iter: int):
    target_grid = target_m / GRID_UNIT_M     # e.g. 1.0 m → 10.0 grid units

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  Slip Compensation Test{RESET}")
    print(f"  Target : {target_m:.2f} m  ({target_grid:.1f} grid units)")
    print(f"  Tolerance : {TOLERANCE_M*100:.0f} cm")
    print(f"  Max iterations : {max_iter}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    # ── Connect ZED subscriber ────────────────────────────────────────────────
    sub = Subscriber(host=ZED_HOST, port=ZED_PORT, topics=[POSE_TOPIC])
    sub.start()
    print("Waiting for ZED pose…", end="", flush=True)
    try:
        px0, pz0, yaw0 = get_pose(sub, timeout_s=10.0)
    except RuntimeError as e:
        print(f"\n{RED}FATAL: {e}{RESET}")
        sub.stop()
        return
    print(f" OK  (px={px0:.3f}  pz={pz0:.3f}  yaw={math.degrees(yaw0):.1f}°)")

    # ── Connect Arduino ───────────────────────────────────────────────────────
    link = SerialLink(port=SERIAL_PORT, baud=BAUD_RATE)
    try:
        link.connect()
    except Exception as e:
        print(f"{RED}FATAL: cannot open {SERIAL_PORT}: {e}{RESET}")
        sub.stop()
        return

    time.sleep(0.5)
    link.drain_responses()

    # Start telemetry so we can read traction
    link.send("log 1 ekf")
    link.get_response(timeout=1.0)

    # Zero Arduino pose — physical robot position is now the reference origin
    print("Zeroing Arduino pose…")
    send_zero(link)

    # Re-read ZED after zero (brief settle)
    time.sleep(0.5)
    px0, pz0, yaw0 = get_pose(sub)
    pz_origin = pz0    # pz at Arduino zero = forward reference
    print(f"ZED origin: px={px0:.3f}  pz={pz0:.3f}  yaw={math.degrees(yaw0):.1f}°\n")

    # ── Iterative goto + correction ───────────────────────────────────────────
    cumulative_x_grid = 0.0   # total Arduino x_grid commanded so far (absolute)
    success = False

    for iteration in range(1, max_iter + 1):
        # Determine speed
        if fixed_spd is not None:
            spd = fixed_spd
        else:
            tval = read_traction(link)
            spd  = traction_to_speed(tval) if tval is not None else SPEED_MAX
        send_spd(link, spd)

        # Compute what we still need to drive
        px_now, pz_now, yaw_now = get_pose(sub)
        actual_forward_m  = pz_now - pz_origin          # pz increases forward
        remaining_m       = target_m - actual_forward_m  # how far still to go
        remaining_grid    = remaining_m / GRID_UNIT_M

        # New absolute Arduino target = current cumulative + remaining
        cumulative_x_grid += remaining_grid

        print(f"{CYAN}── Iteration {iteration}/{max_iter} ──{RESET}")
        print(f"  ZED now    : pz={pz_now:.3f}  (forward so far: {actual_forward_m:+.3f} m)")
        print(f"  Remaining  : {remaining_m:+.3f} m  →  correction: {remaining_grid:+.2f} grid")
        print(f"  Sending    : g {cumulative_x_grid:.4f} 0.0000 0.0  @ spd={spd}%")

        if abs(remaining_m) <= TOLERANCE_M:
            print(f"\n  {GREEN}✓ Already within tolerance ({remaining_m*100:.1f} cm) — no move needed{RESET}")
            success = True
            break

        send_goto(link, cumulative_x_grid, 0.0, 0.0)
        done = wait_done(link, timeout=DONE_TIMEOUT)

        if not done:
            print(f"  {YELLOW}⚠  [done] not received; checking pose anyway{RESET}")

        time.sleep(SETTLE_S)

        # Read pose after move
        px1, pz1, yaw1 = get_pose(sub)
        actual_forward_m = pz1 - pz_origin
        error_m          = target_m - actual_forward_m
        slip_pct         = (1.0 - actual_forward_m / (target_m - (target_m - actual_forward_m - remaining_m))) * 100 if iteration == 1 else float("nan")

        print(f"  ZED after  : pz={pz1:.3f}  (forward so far: {actual_forward_m:+.3f} m)")
        print(f"  Error      : {error_m:+.3f} m  ({error_m*100:+.1f} cm)")
        if iteration == 1:
            commanded_m = remaining_m   # first iter, remaining = full target
            if abs(commanded_m) > 1e-3:
                slip = (1.0 - actual_forward_m / commanded_m) * 100
                print(f"  Slip       : {slip:.1f}%  (moved {actual_forward_m/commanded_m*100:.1f}% of commanded)")

        if abs(error_m) <= TOLERANCE_M:
            print(f"\n  {GREEN}✓ Within tolerance! Error = {error_m*100:.1f} cm{RESET}")
            success = True
            break
        else:
            print(f"  → Will correct in next iteration")

    # ── Summary ───────────────────────────────────────────────────────────────
    px_f, pz_f, yaw_f = get_pose(sub)
    total_forward = pz_f - pz_origin
    final_error   = target_m - total_forward

    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"  Final position : {total_forward:+.3f} m forward  (target {target_m:.2f} m)")
    print(f"  Final error    : {final_error*100:+.1f} cm")
    if success:
        print(f"  Result : {GREEN}PASS — within {TOLERANCE_M*100:.0f} cm tolerance{RESET}")
    else:
        print(f"  Result : {RED}FAIL — did not converge in {max_iter} iterations{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    # Stop and clean up
    link.send("stop")
    time.sleep(0.1)
    link.disconnect()
    sub.stop()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR slip compensation test")
    parser.add_argument(
        "--dist", type=float, default=1.0,
        help="Target forward distance in metres (default: 1.0)",
    )
    parser.add_argument(
        "--spd", type=int, default=None,
        help="Fixed speed %% (0-100). Omit to use traction-based speed.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=MAX_ITER,
        help=f"Max correction iterations (default: {MAX_ITER})",
    )
    args = parser.parse_args()

    run_slip_test(
        target_m  = args.dist,
        fixed_spd = args.spd,
        max_iter  = args.max_iter,
    )
