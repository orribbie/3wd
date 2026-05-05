"""
sir_bridge.py — Bridge between slam_node_.py and the SIR 3WD Arduino base.

Replaces the RPCClient("Yor") used in the original SLAM stack with a direct
serial connection to /dev/ttyACM0.

Provides two interfaces consumed by slam_node_.py:

  1. get_base_encoders()
        Called by EKFSlamSource predict thread at ~20 Hz.
        Returns latest encoder counts from Arduino telemetry for odometry.

  2. follow_path(path)
        Called by Slam._path_sender_loop when a new A* path is computed.
        Executes the path by issuing sequential  g x y theta  commands.
        Speed is scaled by the latest traction value (0.0–1.0 normalised).

Coordinate mapping  (verified by align_check.py, session 4+5)
------------------
SLAM world frame: Y-up, robot moves in XZ plane.
    path = [(x_m, z_m), ...]   — world metres (absolute ZED world frame)
    Forward motion  → T[2,3] (z_m) DECREASES
    Rightward motion → T[0,3] (x_m) DECREASES

Arduino frame: robot moves in XY plane (Z-up), positions relative to z-reset origin.
    g  x_grid  y_grid  theta_deg
        x_grid = (z_origin - z_m) / GRID_UNIT   (SLAM -Δz → Arduino +X, forward)
        y_grid = (x_origin - x_m) / GRID_UNIT   (SLAM -Δx → Arduino +Y, rightward)
        theta_deg = atan2(dx·sin(yaw0)+dz·cos(yaw0), dx·cos(yaw0)-dz·sin(yaw0))
            — heading toward next waypoint in Arduino frame, CCW positive
            — yaw0 = robot SLAM yaw at the time z (zero) was sent

Yaw convention
--------------
BNO055 heading is CW-positive (compass), but the Arduino zeros it at startup and
planGoto uses atan2 (CCW) to compute the path heading, then passes it directly to
beginRotateTo().  We replicate that same atan2 calculation here so the headings
are consistent.
"""

from __future__ import annotations

import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

# ── Import SerialLink from sir/jetson ────────────────────────────────────────
_SIR_JETSON = os.path.join(os.path.dirname(__file__), "..", "sir", "jetson")
if _SIR_JETSON not in sys.path:
    sys.path.insert(0, _SIR_JETSON)

from serial_link import SerialLink   # noqa: E402  (path set above)

# ── Constants ────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 115200
GRID_UNIT_M  = 0.10          # 1 grid unit = 100 mm = 0.1 m

# ── Arduino flashing ──────────────────────────────────────────────────────────
ARDUINO_IDE   = "/home/slam/arduino-1.8.15/arduino"
SKETCH_PATH   = os.path.join(os.path.dirname(__file__), "..", "sir", "arduino",
                              "sir_3wd_base", "sir_3wd_base.ino")
BOARD_FQBN    = "arduino:avr:mega"

# Waypoint-arrival tolerance: consider a waypoint reached if the Arduino
# signals [done].  Timeout per waypoint (generous for slow/slippy surfaces).
WAYPOINT_TIMEOUT_S = 5.0

# ZED-based slip correction
CORRECTION_THRESHOLD_M = 0.10   # residual error after [done] that triggers a correction goto
MAX_CORRECTIONS        = 3      # max correction attempts per waypoint before giving up

# Traction speed scaling
TRACTION_LOW    = 900.0      # observed minimum (slippery) → minimum speed
TRACTION_HIGH   = 1300.0     # observed maximum (grippy)   → maximum speed
SPEED_PCT_MIN   = 60         # minimum PWM percentage sent to Arduino (0-100)
SPEED_PCT_MAX   = 100        # maximum PWM percentage

# Log renewal: re-send  log 1 ekf  every N seconds so telemetry keeps flowing
# between goto commands (the Arduino auto-stops log on any non-log command, but
# restarts it automatically on  g  with label "goto"; we renew during idle).
LOG_RENEW_S = 50.0


class SirBridge:
    """Drop-in replacement for RPCClient that drives the SIR Arduino base.

    Usage (matches original RPCClient interface used in slam_node_.py):
        bridge = SirBridge()
        bridge.connect()

        # EKF predict thread
        enc = bridge.get_base_encoders()   # → dict

        # Path sender loop
        bridge.follow_path([(x1, z1), (x2, z2), ...])

        bridge.disconnect()
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = BAUD_RATE,
    ):
        self._link = SerialLink(port=port, baud=baud)

        # Latest encoder counts — updated by background telemetry consumer
        self._enc_lock = threading.Lock()
        self._latest_enc: Optional[dict] = None   # dict with enc1/enc2/enc3 + ts

        # Traction value — updated by TractionReader (set externally or internally)
        self._traction_lock = threading.Lock()
        self._traction_value: float = TRACTION_HIGH   # default to max until we know

        # ZED pose provider — set after EKFSlamSource is initialised
        # Callable: () -> (translation[3], yaw, T_wr[4,4])  or raises
        self._pose_fn = None

        # SLAM origin snapshot — captured on first follow_path() call.
        # The Arduino is zeroed (z command) at connect() time, so all Arduino
        # absolute positions are relative to the robot's SLAM pose at that moment.
        # We capture that SLAM (x_origin, z_origin, yaw0) here so that SLAM world
        # waypoints can be converted to Arduino-frame positions correctly.
        self._slam_origin: Optional[Tuple[float, float]] = None  # (x_m, z_m)
        self._slam_yaw0: Optional[float] = None                  # robot yaw at zero

        # Path execution state
        self._path_lock    = threading.Lock()
        self._stop_evt     = threading.Event()
        self._current_path: List[Tuple[float, float]] = []
        self._last_heading_deg = 0.0
        self._path_updated = threading.Event()   # pulsed when _current_path changes
        self._follow_thread: Optional[threading.Thread] = None

        # Telemetry consumer thread
        self._tel_thread: Optional[threading.Thread] = None

        # Log renewal thread
        self._log_thread: Optional[threading.Thread] = None

        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _flash_arduino(self) -> bool:
        """Compile and upload the firmware to the Arduino.

        Returns True on success, False on failure (non-fatal — connect() will
        still attempt to open the serial port with whatever firmware is there).
        """
        sketch = os.path.abspath(SKETCH_PATH)
        if not os.path.isfile(sketch):
            print(f"[SirBridge] WARNING: sketch not found at {sketch} — skipping flash")
            return False

        print(f"[SirBridge] Flashing {sketch} → {self._link.port}  (board: {BOARD_FQBN})")
        print("[SirBridge] This takes ~30 s…")

        cmd = [
            ARDUINO_IDE,
            "--upload",
            "--port", self._link.port,
            "--board", BOARD_FQBN,
            "--preserve-temp-files",
            sketch,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            output = result.stdout.decode(errors="replace")
            if result.returncode == 0:
                print("[SirBridge] Flash successful.")
                return True
            else:
                print(f"[SirBridge] Flash FAILED (exit {result.returncode}):")
                for line in output.splitlines()[-10:]:
                    print(f"  {line}")
                return False
        except subprocess.TimeoutExpired:
            print("[SirBridge] Flash timed out after 120 s")
            return False
        except Exception as e:
            print(f"[SirBridge] Flash error: {e}")
            return False

    def connect(self) -> None:
        """Flash the Arduino, open serial port, zero pose, and start background threads.

        If the port is not available (Arduino unplugged / dry-run), prints a
        warning and continues in no-op mode so the SLAM stack still starts.
        """
        self._flash_arduino()
        # Arduino reboots after upload — give it time to come back up
        time.sleep(2.0)

        print(f"[SirBridge] Connecting to {self._link.port} @ {self._link.baud} baud…")
        try:
            self._link.connect()
        except Exception as e:
            print(f"[SirBridge] WARNING: could not open serial port ({e})")
            print("[SirBridge] Running in NO-OP mode — robot motion disabled.")
            return

        time.sleep(0.5)   # let Arduino finish boot banner

        # Drain any stale responses
        self._link.drain_responses()

        # Zero pose and yaw — SLAM and Arduino both start from origin
        self._link.send("z")
        resp = self._link.get_response(timeout=3.0)
        print(f"[SirBridge] Zero response: {resp}")

        # Start telemetry stream so encoder data flows immediately
        self._link.send("log 1 ekf")
        resp = self._link.get_response(timeout=2.0)
        print(f"[SirBridge] Log start: {resp}")

        self._connected = True

        # Background threads
        self._tel_thread = threading.Thread(
            target=self._telemetry_loop, name="sir-telemetry", daemon=True
        )
        self._tel_thread.start()

        self._log_thread = threading.Thread(
            target=self._log_renew_loop, name="sir-log-renew", daemon=True
        )
        self._log_thread.start()

        print("[SirBridge] Ready.")

    def disconnect(self) -> None:
        self._stop_evt.set()
        self._link.send("stop")
        time.sleep(0.1)
        self._link.disconnect()
        self._connected = False

    # ── EKF interface: get_base_encoders() ───────────────────────────────────

    def get_base_encoders(self) -> Optional[dict]:
        """Return latest encoder counts for the EKF predict step.

        Returns dict matching the interface expected by EKFSlamSource:
            {
                "steer_rad":    [0.0, 0.0, 0.0],   # unused for omni
                "drive_counts": [enc1, enc2, enc3], # cumulative counts
                "timestamp":    float,              # time.time()
            }
        Returns None if no telemetry has arrived yet.
        """
        with self._enc_lock:
            if self._latest_enc is None:
                return None
            return dict(self._latest_enc)   # shallow copy

    # ── Path execution interface: follow_path() ───────────────────────────────

    def follow_path(self, path: List[Tuple[float, float]]) -> None:
        """Update the target path without interrupting in-flight motion.

        Called by slam_node_._path_sender_loop at ~10 Hz.  Rather than
        cancel/restart on every call, we just update the shared _current_path
        and let the persistent execution loop pick it up after the current
        waypoint finishes.  This means:
          • Obstacle-avoidance re-plans take effect within one waypoint (~1-2 s).
          • The Arduino is never interrupted mid-motion — no unwanted stops.
        """
        if not self._connected:
            return
        if not path:
            return

        with self._path_lock:
            self._current_path = list(path)
            self._path_updated.set()
            # Start the persistent execution loop if it is not already running.
            if self._follow_thread is None or not self._follow_thread.is_alive():
                self._follow_thread = threading.Thread(
                    target=self._execute_path_loop,
                    name="sir-follow",
                    daemon=True,
                )
                self._follow_thread.start()

    def set_traction(self, value: float) -> None:
        """Update the traction value used for speed scaling.

        value: raw traction output from traction_node.py
               (< 1001 = LOW, 1001-1121 = MEDIUM, > 1121 = HIGH)
        """
        with self._traction_lock:
            self._traction_value = float(value)

    def set_pose_provider(self, fn) -> None:
        """Register a callable that returns the current ZED/EKF pose.

        fn: () -> (translation, yaw, T_wr)
            translation: np.ndarray [px, py, pz]  (Y-up world frame)
            Used after each [done] to detect slip and issue correction gotos.
        """
        self._pose_fn = fn

    # ── Internal: path execution ──────────────────────────────────────────────

    def _traction_speed_pct(self) -> int:
        """Map current traction value → Arduino speed percentage (SPEED_PCT_MIN–MAX)."""
        with self._traction_lock:
            t = self._traction_value
        # Linear interpolation across the traction range
        t_clamped = max(TRACTION_LOW, min(TRACTION_HIGH, t))
        frac = (t_clamped - TRACTION_LOW) / (TRACTION_HIGH - TRACTION_LOW)
        return int(round(SPEED_PCT_MIN + frac * (SPEED_PCT_MAX - SPEED_PCT_MIN)))

    def get_traction_info(self) -> tuple:
        """Return (raw_value, speed_pct, label) for display. Thread-safe."""
        with self._traction_lock:
            t = self._traction_value
        pct = self._traction_speed_pct()
        if t < 1001:
            label = "LOW 🔴"
        elif t < 1121:
            label = "MEDIUM 🟡"
        else:
            label = "HIGH 🟢"
        return t, pct, label

    def _send_speed(self, pct: int) -> None:
        """Send  spd <pct>  command to update Arduino PWM limits."""
        pct = max(0, min(100, pct))
        self._link.send(f"spd {pct}")
        # Don't wait for response — it's a best-effort hint
        time.sleep(0.02)

    def _get_zed_pos(self) -> Optional[Tuple[float, float]]:
        """Return (x_m, z_m) from the EKF/ZED pose, or None if unavailable."""
        if self._pose_fn is None:
            return None
        try:
            translation, _yaw, _T = self._pose_fn()
            return float(translation[0]), float(translation[2])
        except Exception:
            return None

    def _capture_slam_origin(self) -> None:
        """Snapshot the current SLAM pose as the Arduino coordinate origin.

        Called once before the first path execution. The Arduino was zeroed at
        connect() time; this records the corresponding SLAM (x, z, yaw) so that
        absolute SLAM waypoints can be converted to Arduino-relative positions.
        """
        if self._slam_origin is not None or self._pose_fn is None:
            return
        try:
            trans, yaw, _ = self._pose_fn()
            self._slam_origin = (float(trans[0]), float(trans[2]))
            self._slam_yaw0   = float(yaw)
            print(
                f"[SirBridge] SLAM origin captured: "
                f"x={self._slam_origin[0]:.3f}  z={self._slam_origin[1]:.3f}  "
                f"yaw={math.degrees(yaw):.1f}°"
            )
        except Exception as e:
            print(f"[SirBridge] WARN: could not capture SLAM origin ({e}); "
                  f"falling back to (0, 0, π/2)")
            self._slam_origin = (0.0, 0.0)
            self._slam_yaw0   = math.pi / 2

    def _slam_to_arduino(
        self, x_m: float, z_m: float, nx: float, nz: float
    ) -> Tuple[float, float, float]:
        """Convert a SLAM world waypoint to (x_grid, y_grid, heading_deg).

        Verified by align_check.py on physical hardware (yaw0 ≈ 90°):
          - Arduino +X (forward) → SLAM Δpz decreases  (dominant −pz)
          - Arduino +θ (CCW)     → SLAM yaw increases   (CCW positive)

        Robot forward direction in SLAM (px, pz) space at startup yaw0:
          forward = ( cos(yaw0), -sin(yaw0) )
          left    = ( sin(yaw0),  cos(yaw0) )

        Parameters
        ----------
        x_m, z_m : waypoint in SLAM world metres (px, pz)
        nx, nz   : next waypoint (for heading only)
        """
        x_origin, z_origin = self._slam_origin if self._slam_origin else (0.0, 0.0)
        yaw0 = self._slam_yaw0 if self._slam_yaw0 is not None else 0.0

        dx_world = x_m - x_origin   # SLAM Δpx
        dz_world = z_m - z_origin   # SLAM Δpz

        cy0, sy0 = math.cos(yaw0), math.sin(yaw0)

        # Project SLAM displacement onto Arduino axes (verified empirically):
        # Arduino +X = forward = SLAM ( cos y0, -sin y0) in (px, pz)
        # Arduino +Y = left    = SLAM ( sin y0,  cos y0) in (px, pz)
        x_grid = (dx_world *  cy0 + dz_world * -sy0) / GRID_UNIT_M
        y_grid = (dx_world *  sy0 + dz_world *  cy0) / GRID_UNIT_M

        # Heading: angle from startup direction to next-waypoint direction, CCW+.
        v_dx, v_dz = nx - x_m, nz - z_m
        if v_dx*v_dx + v_dz*v_dz < 0.0001:   # ~1 cm — final waypoint
            return x_grid, y_grid, None

        heading_deg = math.degrees(math.atan2(
            v_dx * sy0 + v_dz *  cy0,   # left  (y) component
            v_dx * cy0 + v_dz * -sy0    # fwd   (x) component
        ))
        return x_grid, y_grid, heading_deg

    def _execute_path_loop(self) -> None:
        """Velocity-control loop: drives robot toward each waypoint using ZED feedback.

        Sends 'v fwd turn' at CTRL_HZ using real-time ZED pose error.
        No Arduino dead-reckoning — ZED is the only ground truth.
        """
        # ── Velocity controller constants ─────────────────────────────────────
        ARRIVE_M      = 0.14    # waypoint reached threshold — wider to offset ZED lag
        SLOW_M        = 0.55    # begin decelerating beyond this distance from target
        TURN_THRESH   = 20.0    # degrees — drive forward when heading error is below this
        KP_TURN       = 0.8     # bearing_deg  → turn PWM
        V_MAX_FWD     = 68      # max forward PWM (slightly reduced to ease overshoot)
        V_MIN_FWD     = 20      # min forward PWM in slow zone (just above deadband)
        V_MAX_TURN    = 52      # max turn PWM
        CTRL_HZ       = 20.0

        target_idx  = 0
        last_path   = []   # detect path updates to reset index

        while not self._stop_evt.is_set():
            # ── Fetch latest path ─────────────────────────────────────────────
            with self._path_lock:
                path = list(self._current_path)

            if not path:
                self._link.send("v 0 0")
                self._path_updated.wait(timeout=0.5)
                target_idx = 0
                last_path  = []
                continue

            # Reset index whenever the path is replaced by the planner
            if path != last_path:
                target_idx = 0
                last_path  = path
                print(f"[SirBridge] New path: {len(path)} waypoints")

            # ── Get current pose from ZED ─────────────────────────────────────
            pose = self._get_full_pose()
            if pose is None:
                time.sleep(0.1)
                continue
            px, pz, yaw = pose

            # ── Advance index past waypoints already reached ──────────────────
            # Only moves FORWARD — never goes back to previous waypoints
            while target_idx < len(path):
                x_t, z_t = path[target_idx]
                if math.hypot(x_t - px, z_t - pz) >= ARRIVE_M:
                    break
                target_idx += 1
                print(f"[SirBridge] wp[{target_idx-1}] reached → next wp[{target_idx}]")

            if target_idx >= len(path):
                # All waypoints reached — hard-stop then clear path
                self._link.send("v 0 0")
                time.sleep(0.15)          # let motors spin down before declaring done
                self._link.send("v 0 0")  # second send for reliability
                with self._path_lock:
                    if self._current_path == path:
                        self._current_path = []
                print("[SirBridge] Path complete.")
                self._path_updated.wait(timeout=0.5)
                target_idx = 0
                last_path  = []
                continue

            x_t, z_t = path[target_idx]
            ex = x_t - px
            ez = z_t - pz
            dist = math.hypot(ex, ez)

            # ── Project error into robot frame ────────────────────────────────
            # forward direction in SLAM at current yaw: (cos(yaw), -sin(yaw)) in (px,pz)
            fwd_err =  ex * math.cos(yaw) - ez * math.sin(yaw)  # >0 = target ahead
            lat_err =  ex * math.sin(yaw) + ez * math.cos(yaw)  # >0 = target to left

            bearing_deg = math.degrees(math.atan2(lat_err, fwd_err))

            # ── Compute PWM commands ──────────────────────────────────────────
            spd = self._traction_speed_pct() / 100.0
            max_fwd  = int(V_MAX_FWD  * spd)
            max_turn = int(V_MAX_TURN * spd)

            turn = int(max(-max_turn, min(max_turn, KP_TURN * bearing_deg)))

            if abs(bearing_deg) < TURN_THRESH:
                # Ramp down speed as robot enters SLOW_M zone
                if dist < SLOW_M:
                    # Linear ramp: V_MIN_FWD at ARRIVE_M, max_fwd at SLOW_M
                    ramp = (dist - ARRIVE_M) / (SLOW_M - ARRIVE_M)
                    fwd = int(V_MIN_FWD + ramp * (max_fwd - V_MIN_FWD))
                else:
                    fwd = max_fwd
            else:
                fwd = 0   # turn in place until aligned

            self._link.send(f"v {fwd} {turn}")
            time.sleep(1.0 / CTRL_HZ)

        # Stop on exit
        self._link.send("v 0 0")

    def _get_full_pose(self):
        """Return (px, pz, yaw_rad) from ZED, or None on failure."""
        if self._pose_fn is None:
            return None
        try:
            trans, yaw, _ = self._pose_fn()
            return float(trans[0]), float(trans[2]), float(yaw)
        except Exception:
            return None

    # ── Internal: telemetry consumer ─────────────────────────────────────────

    def _telemetry_loop(self) -> None:
        """Drain telemetry queue and update latest encoder snapshot."""
        while not self._stop_evt.is_set():
            pkt = self._link.get_telemetry(block=True, timeout=0.1)
            if pkt is None:
                continue
            try:
                enc1 = int(pkt["enc1"])
                enc2 = int(pkt["enc2"])
                enc3 = int(pkt["enc3"])
            except (KeyError, TypeError, ValueError):
                continue

            with self._enc_lock:
                self._latest_enc = {
                    "steer_rad":    [0.0, 0.0, 0.0],
                    "drive_counts": [enc1, enc2, enc3],
                    "timestamp":    time.time(),
                }

    # ── Internal: log renewal ─────────────────────────────────────────────────

    def _log_renew_loop(self) -> None:
        """Periodically re-enable telemetry in case it was auto-stopped."""
        while not self._stop_evt.is_set():
            time.sleep(LOG_RENEW_S)
            if self._stop_evt.is_set():
                break
            # Only renew if no follow thread is running (goto restarts log itself)
            with self._path_lock:
                follow_active = (
                    self._follow_thread is not None
                    and self._follow_thread.is_alive()
                )
            if not follow_active:
                self._link.send("log 1 ekf")
                # Don't read response — it may race with other traffic
