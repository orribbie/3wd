"""
robot_ekf.py — 3-state Extended Kalman Filter for swerve drive.

State:       x = [px, pz, theta]     (world frame, Y-up: robot moves in XZ plane)
Control:     u = [vx, vy, omega]     (robot frame, from swerve FK)
Measurement: z = [px, pz, theta]     (from ZED visual odometry)

Coordinate convention
---------------------
Theta (yaw) follows the ZED measurement convention:
  theta = 0   →  robot faces world +X
  theta = pi/2 →  robot faces world -Z
  Extracted as arctan2(-T[2,0], T[0,0]) from the Y-up world→base transform.

The predict kinematics therefore use the standard non-holonomic model:
  dx = ( vx·cos(θ) − vy·sin(θ)) · dt
  dz = (−vx·sin(θ) − vy·cos(θ)) · dt
where vx = robot forward speed, vy = robot left-lateral speed (from SwerveOdom).

Features
--------
- Motion-proportional process noise Q (scales with |u|⋅dt)
- Adaptive measurement noise via R_override parameter (caller inflates R on bad VIO frames)
- Mahalanobis chi-squared gate: rejects statistical outliers (loop-closure jumps)
- Joseph-form covariance update for numerical stability
- ZUPT: zero-velocity update that tightly anchors state when robot is stationary

Carpet tuning
-------------
Default Q values are increased ~3–4× vs. the original calibration to reflect two
carpet-specific sources of encoder uncertainty:
  1. Wheel compression under load changes effective rolling radius by 2–5 %.
  2. Lateral slip on carpet is significantly higher than on hard floors.
These larger Q values make the filter appropriately skeptical of encoder predictions
on carpet, letting the ZED VIO (when confident) carry more weight.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


class RobotEKF:
    """3-state EKF: predict with swerve odometry, update with ZED pose.

    Coordinate convention: Y-up world frame.  Robot moves in the XZ plane.
    State = [px, pz, yaw_around_y].
    """

    def __init__(
        self,
        # Process noise coefficients (per unit travel) — carpet-tuned
        q_x:  float = 0.030,
        q_y:  float = 0.080,
        q_th: float = 0.020,
        # ZED measurement noise std
        r_xy: float = 0.0024,
        r_th: float = 0.0006,
        # Camera offset relative to center (Forward is -Z in our frame)
        zed_offset_forward: float = 0.08, 
    ):
        # State [px, pz, theta]
        self.x = np.zeros(3)
        # Covariance
        self.P = np.eye(3) * 0.1

        self.q_x  = q_x
        self.q_y  = q_y
        self.q_th = q_th
        self.zed_f = zed_offset_forward

        # Nominal measurement noise matrix
        self.R = np.diag([r_xy ** 2, r_xy ** 2, r_th ** 2])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset filter state and covariance."""
        self.x = np.array([x, y, theta], dtype=float)
        self.P = np.eye(3) * 0.1

    def predict(self, u: np.ndarray, dt: float) -> None:
        """EKF predict step using swerve odometry control input."""
        if dt < 1e-6 or not np.isfinite(dt):
            return

        vx, vy, omega = float(u[0]), float(u[1]), float(u[2])
        theta = self.x[2]
        c = np.cos(theta)
        s = np.sin(theta)

        # Process model — Must match omni3_odom.py integration exactly.
        # Forward (vx) reduces Z, Left (vy) increases X.
        dx  = ( vx * s + vy * c) * dt
        dz  = (-vx * c + vy * s) * dt
        dth =  omega * dt

        self.x[0] += dx
        self.x[1] += dz
        self.x[2]  = wrap_pi(self.x[2] + dth)

        # Jacobian F = df/dx
        F = np.array([
            [1.0, 0.0, ( vx * c - vy * s) * dt],
            [0.0, 1.0, ( vx * s + vy * c) * dt],
            [0.0, 0.0, 1.0],
        ])

        # Process noise Q — scales with motion magnitude
        # The constant floor terms (not 1e-6!) are critical: they represent
        # unconditional uncertainty (motor backlash, carpet deformation, IMU
        # drift) that prevent the covariance from collapsing after a ZED update
        # and locking out the Mahalanobis gate.
        Q_FLOOR_POS = 1e-4    # ~1 cm/s position wander
        Q_FLOOR_YAW = 5e-4   # ~1.3°/s heading wander
        Q = np.diag([
            self.q_x  * abs(vx)    * dt + Q_FLOOR_POS * dt,
            self.q_y  * abs(vy)    * dt + Q_FLOOR_POS * dt,
            self.q_th * abs(omega) * dt + Q_FLOOR_YAW * dt,
        ])

        new_P = F @ self.P @ F.T + Q
        if np.all(np.isfinite(new_P)):
            self.P = new_P
        else:
            # If covariance exploded, reset it to something sane
            self.P = np.eye(3) * 0.1

    def update(
        self,
        z: np.ndarray,
        R_override: Optional[np.ndarray] = None,
        gate_chi2: Optional[float] = 12.0,
    ) -> bool:
        """EKF update step using ZED pose measurement."""
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)):
            return False

        # Measurement model H is now identity because zed_pub_node.py 
        # already transforms the camera pose to the robot's base center.
        H = np.eye(3)
        R_use = R_override if R_override is not None else self.R

        # Innovation
        y    = z - self.x
        y[2] = wrap_pi(y[2])      # angle wrapping

        # Innovation covariance
        S = H @ self.P @ H.T + R_use

        # ---- Mahalanobis gating ----
        if gate_chi2 is not None and gate_chi2 > 0.0:
            try:
                d2 = float(y @ np.linalg.solve(S, y))
            except np.linalg.LinAlgError:
                return False
            if d2 > gate_chi2:
                return False

        # ---- Kalman gain ----
        K = np.linalg.solve(S.T, H @ self.P.T).T

        # ---- State update ----
        self.x    = self.x + K @ y
        self.x[2] = wrap_pi(self.x[2])

        # ---- Covariance update (Joseph form) ----
        I_KH   = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_use @ K.T

        return True

    def zupt(self, R_zupt: Optional[np.ndarray] = None) -> None:
        """Zero-Velocity Update: assert the robot hasn't moved.

        Injects a pseudo-measurement z = current_state with very tight R,
        resetting accumulated drift while the robot is stationary.

        Args:
            R_zupt: custom noise for ZUPT. Default: ~1 mm position, ~0.03 deg heading.
        """
        if R_zupt is None:
            R_zupt = np.diag([0.001 ** 2, 0.001 ** 2, 0.0005 ** 2])
        # Assert "I'm here" — no gating (we trust the stopped state)
        self.update(self.x.copy(), R_override=R_zupt, gate_chi2=None)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return current state estimate [px, pz, theta]."""
        return self.x.copy()

    def get_covariance(self) -> np.ndarray:
        """Return current 3×3 covariance matrix."""
        return self.P.copy()

    def get_uncertainty(self) -> np.ndarray:
        """Return 1-sigma uncertainties [sigma_px, sigma_pz, sigma_theta]."""
        return np.sqrt(np.diag(self.P))