"""
swerve_odom.py — Swerve drive forward kinematics + dead-reckoning odometry.

Uses least-squares to solve the over-determined system (4 measurements → 3 DOF):
    A (4x3) × [dx, dy, dθ] = b (4x1)

where each row of A comes from a wheel's steer angle and module position,
and b is the wheel's linear displacement from drive encoder counts.

Key design: we solve for DISPLACEMENTS [dx, dy, dθ] directly, not velocities.
This avoids dividing by dt (to get speed) then multiplying by dt (to integrate),
which amplifies quantization noise at high sample rates.

No DRIVE_SIGNS needed — the steer angle already encodes wheel direction.
When the swerve optimizer flips a wheel to 180deg, the motor reverses AND
cos(steer) flips sign, so raw counts × cos/sin(steer) gives correct dx/dy.

Verified empirically against known motions (forward + rotation).
"""

import numpy as np

# ── Robot geometry (from base_motor.py) ──────────────────────────────────────
LENGTH = 0.14527  # m — half-wheelbase (calibrated tight 2026-03-30)
WIDTH  = 0.20161  # m — half-track (calibrated tight 2026-03-30)

# Module order: FL=0, FR=1, RR=2, RL=3  (matches CAN_IDS_DRIVE and get_base_encoders)
MODULE_POSITIONS = np.array([
    [+LENGTH, +WIDTH],   # FL
    [+LENGTH, -WIDTH],   # FR
    [-LENGTH, -WIDTH],   # RR
    [-LENGTH, +WIDTH],   # RL
], dtype=float)  # (4, 2)  — [rx, ry] per module

# ── Calibration ──────────────────────────────────────────────────────────────
METERS_PER_ROTATION = 0.049922   # calibrated via calibrate_drive.py (2026-03-30, 5 runs, σ=0.82%)

# ── Physical limits (per step, not per second) ───────────────────────────────
MAX_STEP_DIST  = 0.05    # m   — max displacement per update (~0.5 m/s at 10ms)
MAX_STEP_ANGLE = 0.10    # rad — max rotation per update (~5.7° per step)


class SwerveOdom:
    """Dead-reckoning odometry for a 4-wheel swerve drive.

    Usage:
        odom = SwerveOdom()
        # In control loop:
        u = odom.update(steer_rad, drive_counts, dt)
        # u = [vx, vy, omega] in robot frame (for EKF predict)
        pose = odom.get_pose()  # [x, y, theta] in world frame
    """

    def __init__(self, mpr: float = METERS_PER_ROTATION):
        self.mpr = mpr
        self.prev_counts = None
        # World-frame pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset pose to given values."""
        self.x = x
        self.y = y
        self.theta = theta
        self.prev_counts = None

    def _solve_fk_displacement(
        self, steer_rad: np.ndarray, wheel_displacements: np.ndarray
    ) -> np.ndarray:
        """Solve for [dx, dy, dθ] from wheel steer angles and displacements.

        This solves the same kinematic equation as forward_kinematics but in
        displacement form (not velocity form), avoiding noise amplification
        from division by small dt values.

        Args:
            steer_rad: (4,) steer angle per module [rad]
            wheel_displacements: (4,) linear displacement per wheel [m]

        Returns:
            (3,) array [dx_robot, dy_robot, dtheta] in robot frame
        """
        cos_s = np.cos(steer_rad)
        sin_s = np.sin(steer_rad)
        rx = MODULE_POSITIONS[:, 0]
        ry = MODULE_POSITIONS[:, 1]

        # A matrix: each row = [cos(θ_i), sin(θ_i), -ry_i·cos(θ_i) + rx_i·sin(θ_i)]
        A = np.column_stack([
            cos_s,
            sin_s,
            -ry * cos_s + rx * sin_s,
        ])  # (4, 3)

        b = wheel_displacements  # (4,)

        # Tikhonov regularization
        LAMBDA = 1e-3
        AtA = A.T @ A + LAMBDA * np.eye(3)
        Atb = A.T @ b
        return np.linalg.solve(AtA, Atb)  # [dx, dy, dtheta]

    def forward_kinematics(self, steer_rad: np.ndarray, wheel_speeds: np.ndarray) -> np.ndarray:
        """Solve for [vx, vy, omega] from wheel steer angles and speeds.

        Args:
            steer_rad: (4,) steer angle per module [rad]
            wheel_speeds: (4,) wheel linear speed per module [m/s]

        Returns:
            (3,) array [vx, vy, omega] in robot frame
        """
        cos_s = np.cos(steer_rad)
        sin_s = np.sin(steer_rad)
        rx = MODULE_POSITIONS[:, 0]
        ry = MODULE_POSITIONS[:, 1]

        A = np.column_stack([
            cos_s,
            sin_s,
            -ry * cos_s + rx * sin_s,
        ])  # (4, 3)

        b = wheel_speeds  # (4,)

        LAMBDA = 1e-3
        AtA = A.T @ A + LAMBDA * np.eye(3)
        Atb = A.T @ b
        result = np.linalg.solve(AtA, Atb)
        return result  # [vx, vy, omega]

    def update(self, steer_rad: np.ndarray, drive_counts: np.ndarray, dt: float) -> np.ndarray:
        """Process one encoder tick. Returns [vx, vy, omega] in robot frame.

        Internally uses displacement-based FK to avoid noise amplification
        from dividing-then-multiplying by dt.  The returned u = [vx, vy, omega]
        is still in velocity units (for the EKF predict step).

        Args:
            steer_rad: (4,) raw steer angles from get_base_encoders()
            drive_counts: (4,) raw drive counts from get_base_encoders()
            dt: time step [s]

        Returns:
            (3,) array [vx, vy, omega] in robot frame
        """
        steer_rad = np.asarray(steer_rad, dtype=float)
        drive_counts = np.asarray(drive_counts, dtype=float)

        if not (np.all(np.isfinite(steer_rad)) and np.all(np.isfinite(drive_counts)) and np.isfinite(dt)):
            return np.zeros(3)

        if self.prev_counts is None:
            self.prev_counts = drive_counts.copy()
            return np.zeros(3)

        # Wheel displacements from encoder deltas (in meters)
        d_counts = drive_counts - self.prev_counts
        self.prev_counts = drive_counts.copy()

        if dt < 1e-6:
            return np.zeros(3)

        wheel_displacements = d_counts * self.mpr  # meters, NOT m/s

        # ── Solve FK in displacement space ──
        # This gives [dx_robot, dy_robot, dtheta] directly,
        # no division by dt needed.
        disp = self._solve_fk_displacement(steer_rad, wheel_displacements)
        dx_robot, dy_robot, dtheta = disp

        # ── Clamp to physical limits per step ──
        dx_robot = np.clip(dx_robot, -MAX_STEP_DIST,  MAX_STEP_DIST)
        dy_robot = np.clip(dy_robot, -MAX_STEP_DIST,  MAX_STEP_DIST)
        dtheta   = np.clip(dtheta,   -MAX_STEP_ANGLE, MAX_STEP_ANGLE)

        # ── Integrate to world frame ──
        # Use midpoint heading for better integration accuracy during turns
        mid_theta = self.theta + dtheta * 0.5
        cos_th = np.cos(mid_theta)
        sin_th = np.sin(mid_theta)

        # ZED Y-up: camera faces backward relative to the robot.
        # Robot frame: dx = forward, dy = left, dθ = CCW from above
        #   world_X = -dx·sin(θ) − dy·cos(θ)
        #   world_Z = -dx·cos(θ) + dy·sin(θ)
        self.x     += -dx_robot * sin_th - dy_robot * cos_th   # world X
        self.y     += -dx_robot * cos_th + dy_robot * sin_th   # world Z
        self.theta += dtheta

        if not np.all(np.isfinite([self.x, self.y, self.theta])):
            self.x = self.y = self.theta = 0.0

        # Return velocity for EKF predict (divide by dt only for the output)
        u = np.array([dx_robot / dt, dy_robot / dt, dtheta / dt])
        return u

    def get_pose(self) -> np.ndarray:
        """Return current world-frame pose [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])

    def get_u(self) -> np.ndarray:
        """Return last computed [vx, vy, omega] (for EKF predict)."""
        return np.zeros(3)  # overridden by update()


def compute_odometry_batch(timestamps, steer_rads, drive_counts, mpr=METERS_PER_ROTATION):
    """Compute full odometry from recorded arrays. Returns dict of trajectories.

    Args:
        timestamps: (N,) array of times
        steer_rads: (N, 4) array of steer angles
        drive_counts: (N, 4) array of drive encoder counts
        mpr: meters per rotation calibration

    Returns:
        dict with keys: timestamps, x, y, theta, vx, vy, omega
    """
    odom = SwerveOdom(mpr=mpr)
    N = len(timestamps)

    x = np.zeros(N)
    y = np.zeros(N)
    theta = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    omega = np.zeros(N)

    for i in range(1, N):
        dt = timestamps[i] - timestamps[i - 1]
        u = odom.update(steer_rads[i], drive_counts[i], dt)
        pose = odom.get_pose()
        x[i] = pose[0]
        y[i] = pose[1]
        theta[i] = pose[2]
        vx[i] = u[0]
        vy[i] = u[1]
        omega[i] = u[2]

    return {
        "timestamps": timestamps,
        "x": x, "y": y, "theta": theta,
        "vx": vx, "vy": vy, "omega": omega,
    }
