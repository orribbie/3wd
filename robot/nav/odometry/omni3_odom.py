"""
omni3_odom.py — 3-wheel omnidirectional drive forward kinematics + dead-reckoning.

Wheel layout (angle = direction of active rolling from robot +X axis):
    Wheel 1:   0°  (front)
    Wheel 2: 120°  (back-right)
    Wheel 3: 240°  (back-left)

Encoders come from Arduino telemetry as cumulative counts (already sign-corrected:
telemetry sends -enc.read() so positive delta = forward wheel motion).

Interface matches SwerveOdom so EKFSlamSource works unchanged:
    u = odom.update(steer_rad, drive_counts, dt)   # [vx, vy, omega]
    odom.reset(x, y, theta)
"""

import numpy as np

# ── Robot geometry (from sir_3wd_base.ino) ───────────────────────────────────
WHEEL_RADIUS_M  = 0.060   # 60 mm
ROBOT_RADIUS_M  = 0.155   # 155 mm — wheel-to-centre (matches sir_3wd_base.ino)

# ── Encoder calibration (from sir_3wd_base.ino) ──────────────────────────────
# COUNTS_PER_REV = ENCODER_PPR_MOTOR * GEAR_RATIO * 16 = 4 * 41.6 * 16 = 2662.4
COUNTS_PER_REV  = 4 * 41.6 * 16          # counts per wheel revolution = 2662.4
COUNTS_PER_MM   = COUNTS_PER_REV / (2.0 * np.pi * (WHEEL_RADIUS_M * 1000.0))
METERS_PER_COUNT = 1.0 / (COUNTS_PER_MM * 1000.0)   # m per encoder count

# ── Physical sanity limits (per update step) ─────────────────────────────────
MAX_STEP_DIST_M  = 0.05    # ~0.5 m/s at 10 ms
MAX_STEP_ANGLE   = 0.10    # ~5.7° per step

_RT3 = float(np.sqrt(3.0))


class Omni3Odom:
    """Dead-reckoning odometry for the SIR 3-wheel omnidirectional base.

    Forward kinematics (wheels at 0°, 120°, 240°):
        vx    =  (√3/3) * (v3 − v2)
        vy    =  (2/3)*v1 − (1/3)*(v2 + v3)
        omega =  (v1 + v2 + v3) / (3 * L)

    where v_i is wheel-i linear speed [m/s] and L = robot radius [m].

    EKF frame convention (matches robot_ekf.py):
        vx    = robot forward speed  (+X robot frame)
        vy    = robot left-lateral   (+Y robot frame)
        omega = CCW yaw rate [rad/s]
    """

    def __init__(self):
        self.prev_counts = None
        # World-frame pose (Y-up: robot moves in XZ plane, stored as x/z)
        self.x = 0.0
        self.z = 0.0
        self.theta = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset pose. y argument maps to world-Z (matches SwerveOdom interface)."""
        self.x = float(x)
        self.z = float(y)
        self.theta = float(theta)
        self.prev_counts = None

    def update(self, steer_rad, drive_counts, dt: float) -> np.ndarray:
        """Process one encoder sample. Returns [vx, vy, omega] in robot frame.

        Args:
            steer_rad:    ignored (no steer angles on omni base); kept for
                          interface compatibility with SwerveOdom.
            drive_counts: array-like, length ≥ 3 — cumulative encoder counts
                          from Arduino telemetry (enc1, enc2, enc3).
            dt:           time step [s]

        Returns:
            np.ndarray([vx, vy, omega]) — robot-frame velocities for EKF predict.
        """
        counts = np.asarray(drive_counts, dtype=float)[:3]

        if not np.all(np.isfinite(counts)) or not np.isfinite(dt):
            return np.zeros(3)

        if self.prev_counts is None:
            self.prev_counts = counts.copy()
            return np.zeros(3)

        dc = counts - self.prev_counts
        self.prev_counts = counts.copy()

        if dt < 1e-6:
            return np.zeros(3)

        # Wheel linear speeds [m/s]
        v = dc * METERS_PER_COUNT / dt   # shape (3,)
        v1, v2, v3 = float(v[0]), float(v[1]), float(v[2])

        # ── Forward kinematics ──
        # Forward (vx) is along Wheel 1 (90° from robot-right)
        # Left (vy) is 180° from robot-right
        vx    =  (2.0 / 3.0) * v1 - (1.0 / 3.0) * (v2 + v3)
        vy    =  (_RT3 / 3.0) * (v2 - v3)
        omega =  (v1 + v2 + v3) / (3.0 * ROBOT_RADIUS_M)

        # ── Clamp to physical limits ──
        speed = float(np.hypot(vx, vy))
        if speed > MAX_STEP_DIST_M / max(dt, 1e-6):
            scale = (MAX_STEP_DIST_M / dt) / speed
            vx *= scale
            vy *= scale
        omega = float(np.clip(omega, -MAX_STEP_ANGLE / dt, MAX_STEP_ANGLE / dt))

        # ── Integrate world pose (matches robot_ekf.py kinematics) ──
        # Standard Y-up convention (robot moves in XZ plane):
        # theta=0 is pointing toward world -Z.
        # vx+ (Forward) -> dz = -vx*cos(theta)
        # vy+ (Left)    -> dx = +vy*cos(theta)  (if theta=0, Left is +X)
        c, s = np.cos(self.theta), np.sin(self.theta)
        self.x     += (vx * s + vy * c) * dt
        self.z     += (-vx * c + vy * s) * dt
        self.theta += omega * dt
        self.theta  = float(((self.theta + np.pi) % (2 * np.pi)) - np.pi)

        if not np.all(np.isfinite([self.x, self.z, self.theta])):
            self.x = self.z = self.theta = 0.0

        return np.array([vx, vy, omega])

    def get_pose(self) -> np.ndarray:
        """Return world-frame pose [x, z, theta] (Y-up convention)."""
        return np.array([self.x, self.z, self.theta])
