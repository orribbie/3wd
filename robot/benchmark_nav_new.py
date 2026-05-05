#!/usr/bin/env python3
"""
Repeatability benchmark for YOR SLAM navigation.

Measures how repeatable the robot's A↔B navigation is over multiple runs,
with live mapping and dynamic obstacle avoidance.

Usage:
    # With EKF (default):
    python benchmark_nav.py --runs 5 --goal-radius 0.10

    # Without EKF:
    python benchmark_nav.py --runs 5 --goal-radius 0.10 --no-ekf

Assumes zed_pub_node and yor.py are already running.
"""

import argparse
import json
import os
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── YOR imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot.slam_node_ import Slam, EKFSlamSource


# ═════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PoseSample:
    """Single timestamped pose sample."""
    t: float
    x: float
    z: float
    yaw: float
    ekf_sigma_x: Optional[float] = None
    ekf_sigma_z: Optional[float] = None
    ekf_sigma_yaw: Optional[float] = None
    voxel_count: Optional[int] = None
    speed_m_s: Optional[float] = None


@dataclass
class VoxelSample:
    """Timestamped voxel map size sample."""
    t: float          # seconds since benchmark start
    count: int        # number of voxels in the map


class VoxelTracker:
    """Background thread that periodically records voxel map size."""

    def __init__(self, slam: 'Slam', sample_hz: float = 1.0):
        self.slam = slam
        self.dt = 1.0 / max(0.1, sample_hz)
        self.samples: List[VoxelSample] = []
        self._stop = threading.Event()
        self._t0: Optional[float] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                vmap = self.slam.map_manager.get_voxel_map()
                n = len(vmap) if vmap is not None else 0
                self.samples.append(VoxelSample(
                    t=time.time() - self._t0, count=n))
            except Exception:
                pass
            self._stop.wait(self.dt)


@dataclass
class LegData:
    """Data collected for one navigation leg (e.g. A→B)."""
    run_idx: int
    leg_label: str          # "A→B" or "B→A"
    goal_label: str         # "B" or "A"
    start_x: float = 0.0
    start_z: float = 0.0
    start_yaw: float = 0.0
    goal_x: float = 0.0
    goal_z: float = 0.0
    goal_yaw: float = 0.0
    end_x: float = 0.0
    end_z: float = 0.0
    end_yaw: float = 0.0
    distance_m: float = 0.0
    duration_s: float = 0.0
    samples: List[PoseSample] = field(default_factory=list)
    # Path length changes (detour detection)
    path_length_changes: int = 0


@dataclass
class WaypointRef:
    """Reference waypoint as labelled by the user."""
    label: str
    x: float
    z: float
    yaw: float


# ═════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════

def _wrap_pi(a: float) -> float:
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def compute_stats(legs: List[LegData], ref_a: WaypointRef, ref_b: WaypointRef,
                  use_ekf: bool) -> dict:
    """Compute all benchmark statistics from collected leg data."""
    stats: dict = {
        "mode": "EKF" if use_ekf else "ZED-only",
        "num_runs": (len(legs) + 1) // 2,
        "num_legs": len(legs),
        "ref_A": {"x": ref_a.x, "z": ref_a.z, "yaw_deg": np.degrees(ref_a.yaw)},
        "ref_B": {"x": ref_b.x, "z": ref_b.z, "yaw_deg": np.degrees(ref_b.yaw)},
    }

    ab_dist = float(np.hypot(ref_b.x - ref_a.x, ref_b.z - ref_a.z))
    stats["AB_euclidean_m"] = ab_dist

    # Per-leg stats
    per_leg: List[dict] = []
    arrivals_A: List[Tuple[float, float, float]] = []   # (x, z, yaw)
    arrivals_B: List[Tuple[float, float, float]] = []

    total_distance = 0.0
    total_duration = 0.0

    for leg in legs:
        ref = ref_b if leg.goal_label == "B" else ref_a
        err_x = leg.end_x - ref.x
        err_z = leg.end_z - ref.z
        err_pos = float(np.hypot(err_x, err_z))
        err_yaw = float(np.degrees(abs(_wrap_pi(leg.end_yaw - ref.yaw))))

        efficiency = ab_dist / max(leg.distance_m, 1e-6) if ab_dist > 0 else 0.0
        avg_speed = leg.distance_m / max(leg.duration_s, 1e-6)

        ld = {
            "run": leg.run_idx + 1,
            "leg": leg.leg_label,
            "err_x_m": round(err_x, 4),
            "err_z_m": round(err_z, 4),
            "err_pos_m": round(err_pos, 4),
            "err_yaw_deg": round(err_yaw, 2),
            "distance_m": round(leg.distance_m, 3),
            "duration_s": round(leg.duration_s, 1),
            "avg_speed_m_s": round(avg_speed, 3),
            "path_efficiency": round(efficiency, 3),
            "detour_events": leg.path_length_changes,
        }

        # EKF uncertainty (mean over leg)
        if use_ekf and leg.samples:
            sx = [s.ekf_sigma_x for s in leg.samples if s.ekf_sigma_x is not None]
            sz = [s.ekf_sigma_z for s in leg.samples if s.ekf_sigma_z is not None]
            sy = [s.ekf_sigma_yaw for s in leg.samples
                  if s.ekf_sigma_yaw is not None]
            if sx:
                ld["mean_ekf_sigma_x_m"] = round(float(np.mean(sx)), 5)
                ld["mean_ekf_sigma_z_m"] = round(float(np.mean(sz)), 5)
                ld["mean_ekf_sigma_yaw_deg"] = round(
                    float(np.degrees(np.mean(sy))), 3)

        per_leg.append(ld)
        total_distance += leg.distance_m
        total_duration += leg.duration_s

        if leg.goal_label == "A":
            arrivals_A.append((leg.end_x, leg.end_z, leg.end_yaw))
        else:
            arrivals_B.append((leg.end_x, leg.end_z, leg.end_yaw))

    stats["per_leg"] = per_leg
    stats["total_distance_m"] = round(total_distance, 3)
    stats["total_duration_s"] = round(total_duration, 1)

    # RMS position error at A and B
    def _rms_pos(arrivals, ref):
        if not arrivals:
            return None
        errs = [np.hypot(a[0] - ref.x, a[1] - ref.z) for a in arrivals]
        return round(float(np.sqrt(np.mean(np.square(errs)))), 4)

    def _rms_yaw(arrivals, ref):
        if not arrivals:
            return None
        errs = [abs(_wrap_pi(a[2] - ref.yaw)) for a in arrivals]
        return round(float(np.degrees(np.sqrt(np.mean(np.square(errs))))), 3)

    stats["rms_pos_error_A_m"] = _rms_pos(arrivals_A, ref_a)
    stats["rms_pos_error_B_m"] = _rms_pos(arrivals_B, ref_b)
    stats["rms_yaw_error_A_deg"] = _rms_yaw(arrivals_A, ref_a)
    stats["rms_yaw_error_B_deg"] = _rms_yaw(arrivals_B, ref_b)

    # Overall RMS
    all_pos_errs = []
    all_yaw_errs = []
    for leg in legs:
        ref = ref_b if leg.goal_label == "B" else ref_a
        all_pos_errs.append(np.hypot(leg.end_x - ref.x, leg.end_z - ref.z))
        all_yaw_errs.append(abs(_wrap_pi(leg.end_yaw - ref.yaw)))
    stats["rms_pos_error_overall_m"] = round(
        float(np.sqrt(np.mean(np.square(all_pos_errs)))), 4) if all_pos_errs else None
    stats["rms_yaw_error_overall_deg"] = round(
        float(np.degrees(np.sqrt(np.mean(np.square(all_yaw_errs))))), 3) if all_yaw_errs else None

    # Additional aggregate metrics
    if all_pos_errs:
        stats["max_pos_error_m"] = round(float(np.max(all_pos_errs)), 4)
        stats["std_pos_error_m"] = round(float(np.std(all_pos_errs)), 4)
        stats["mean_pos_error_m"] = round(float(np.mean(all_pos_errs)), 4)
    if all_yaw_errs:
        stats["max_yaw_error_deg"] = round(float(np.degrees(np.max(all_yaw_errs))), 2)

    # Per-leg speed stats
    speeds = [l["avg_speed_m_s"] for l in per_leg]
    if speeds:
        stats["mean_speed_m_s"] = round(float(np.mean(speeds)), 3)

    # Path efficiency summary
    effs = [l["path_efficiency"] for l in per_leg]
    if effs:
        stats["mean_path_efficiency"] = round(float(np.mean(effs)), 3)

    return stats


# ═════════════════════════════════════════════════════════════════════════
# Pretty printing
# ═════════════════════════════════════════════════════════════════════════

def print_report(stats: dict):
    """Print a formatted report to the terminal."""
    SEP = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  REPEATABILITY BENCHMARK — {stats['mode']}")
    print(f"{'═' * 72}")
    print(f"  Runs: {stats['num_runs']}   Legs: {stats['num_legs']}")
    print(f"  A→B euclidean: {stats['AB_euclidean_m']:.3f} m")
    print(f"  Total distance traveled: {stats['total_distance_m']:.3f} m")
    print(f"  Total duration: {stats['total_duration_s']:.1f} s")
    print(SEP)

    # Header
    has_ekf = any("mean_ekf_sigma_x_m" in l for l in stats["per_leg"])
    hdr = (f"{'Run':>4} {'Leg':>6} {'ΔPos(m)':>8} {'Δx(m)':>7} {'Δz(m)':>7} "
           f"{'Δyaw°':>6} {'Dist(m)':>8} {'T(s)':>6} {'v(m/s)':>7} "
           f"{'Eff':>5} {'Det':>4}")
    if has_ekf:
        hdr += f" {'σx(mm)':>7} {'σz(mm)':>7} {'σψ(°)':>6}"
    print(hdr)
    print(SEP)

    for l in stats["per_leg"]:
        row = (f"{l['run']:>4} {l['leg']:>6} {l['err_pos_m']:>8.4f} "
               f"{l['err_x_m']:>7.4f} {l['err_z_m']:>7.4f} "
               f"{l['err_yaw_deg']:>6.2f} {l['distance_m']:>8.3f} "
               f"{l['duration_s']:>6.1f} {l['avg_speed_m_s']:>7.3f} "
               f"{l['path_efficiency']:>5.3f} {l['detour_events']:>4}")
        if has_ekf:
            sx = l.get("mean_ekf_sigma_x_m")
            sz = l.get("mean_ekf_sigma_z_m")
            sy = l.get("mean_ekf_sigma_yaw_deg")
            if sx is not None:
                row += (f" {sx*1000:>7.2f} {sz*1000:>7.2f} "
                        f"{sy:>6.3f}")
            else:
                row += f" {'—':>7} {'—':>7} {'—':>6}"
        print(row)

    print(SEP)
    print(f"  RMS position error at A: "
          f"{stats.get('rms_pos_error_A_m', '—')} m")
    print(f"  RMS position error at B: "
          f"{stats.get('rms_pos_error_B_m', '—')} m")
    print(f"  RMS position error overall: "
          f"{stats.get('rms_pos_error_overall_m', '—')} m")
    print(f"  RMS heading error at A: "
          f"{stats.get('rms_yaw_error_A_deg', '—')}°")
    print(f"  RMS heading error at B: "
          f"{stats.get('rms_yaw_error_B_deg', '—')}°")
    print(SEP)
    print(f"  Mean position error: "
          f"{stats.get('mean_pos_error_m', '—')} m")
    print(f"  Max position error:  "
          f"{stats.get('max_pos_error_m', '—')} m")
    print(f"  Std position error:  "
          f"{stats.get('std_pos_error_m', '—')} m")
    print(f"  Max heading error:   "
          f"{stats.get('max_yaw_error_deg', '—')}°")
    print(f"  Mean speed:          "
          f"{stats.get('mean_speed_m_s', '—')} m/s")
    print(f"  Mean path efficiency:"
          f" {stats.get('mean_path_efficiency', '—')}")
    print(f"{'═' * 72}\n")


# ═════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """Orchestrates labelling, navigation runs, and data collection."""

    def __init__(self, slam: Slam, *, goal_radius_m: float = 0.10,
                 yaw_tol_deg: float = 5.0,
                 settle_s: float = 1.0, timeout_s: float = 120.0,
                 sample_hz: float = 5.0, use_ekf: bool = False):
        self.slam = slam
        self.goal_radius = goal_radius_m
        self.yaw_tol_rad = float(np.radians(yaw_tol_deg))
        self.settle_s = settle_s
        self.timeout_s = timeout_s
        self.sample_dt = 1.0 / max(0.5, sample_hz)
        self.use_ekf = use_ekf

    # ── helpers ──────────────────────────────────────────────────────
    def _get_pose(self) -> Tuple[float, float, float]:
        """Return (x, z, yaw) from the SLAM datastream."""
        trans, yaw, _ = self.slam.datastream.get_pose()
        return float(trans[0]), float(trans[2]), float(yaw)

    def _get_ekf_sigma(self) -> Optional[Tuple[float, float, float]]:
        if self.use_ekf and isinstance(self.slam.datastream, EKFSlamSource):
            s = self.slam.datastream.get_ekf_uncertainty()
            return float(s[0]), float(s[1]), float(s[2])
        return None

    def _get_path_length(self) -> Optional[float]:
        """Return the current A* path length in meters, or None."""
        if self.slam.planner is None:
            return None
        pw = self.slam.planner.get_latest_path_world()
        if not pw or len(pw) < 2:
            return None
        d = 0.0
        for (x0, z0), (x1, z1) in zip(pw, pw[1:]):
            d += np.hypot(x1 - x0, z1 - z0)
        return d

    # ── labelling ────────────────────────────────────────────────────
    def label_point(self, name: str) -> WaypointRef:
        """Block until user presses Enter, then record current pose."""
        input(f"\n>>> Drive robot to point {name}, then press ENTER... ")
        x, z, yaw = self._get_pose()
        print(f"    [{name}] recorded at x={x:.4f}, z={z:.4f}, "
              f"yaw={np.degrees(yaw):.1f}°")
        return WaypointRef(label=name, x=x, z=z, yaw=yaw)

    # ── single leg ──────────────────────────────────────────────────
    def run_leg(self, goal: WaypointRef, start_label: str,
                run_idx: int) -> LegData:
        """Navigate to `goal` and collect data until arrival or timeout.

        Two-phase approach:
          Phase 1 — A* path-following delivers the robot near the goal position.
          Phase 2 — Once within goal_radius, issue a move_to RPC with the
                    target yaw so the BaseController rotates in place.
        Arrival requires BOTH position ≤ goal_radius AND yaw ≤ yaw_tol
        held for settle_s seconds.
        """
        leg_label = f"{start_label}→{goal.label}"
        sx, sz, syaw = self._get_pose()
        print(f"\n  ▶ Run {run_idx+1} | {leg_label} | "
              f"start=({sx:.3f},{sz:.3f}) → goal=({goal.x:.3f},{goal.z:.3f}) "
              f"yaw_target={np.degrees(goal.yaw):.1f}°")

        leg = LegData(
            run_idx=run_idx, leg_label=leg_label, goal_label=goal.label,
            start_x=sx, start_z=sz, start_yaw=syaw,
            goal_x=goal.x, goal_z=goal.z, goal_yaw=goal.yaw,
        )

        # Send goal to planner (position only — A* path-following)
        self.slam.set_goal(goal.x, goal.z)

        t_start = time.time()
        t_last_sample = 0.0
        prev_x, prev_z = sx, sz
        dist_accum = 0.0
        converged_since: Optional[float] = None
        prev_path_len: Optional[float] = None
        detour_count = 0
        move_to_sent = False  # True once we've issued the final-yaw move_to

        while True:
            now = time.time()
            elapsed = now - t_start

            # Timeout
            if elapsed > self.timeout_s:
                print(f"    ⚠ TIMEOUT after {elapsed:.1f}s")
                break

            # Sample pose
            cx, cz, cyaw = self._get_pose()

            # Accumulate odometric distance
            step = float(np.hypot(cx - prev_x, cz - prev_z))
            if step < 1.0:  # ignore teleport glitches
                dist_accum += step
            prev_x, prev_z = cx, cz

            # Record sample at sample_hz
            if (now - t_last_sample) >= self.sample_dt:
                t_last_sample = now
                sigma = self._get_ekf_sigma()
                sample = PoseSample(
                    t=elapsed, x=cx, z=cz, yaw=cyaw,
                    ekf_sigma_x=sigma[0] if sigma else None,
                    ekf_sigma_z=sigma[1] if sigma else None,
                    ekf_sigma_yaw=sigma[2] if sigma else None,
                )
                leg.samples.append(sample)

                # Detour detection: path length spike > 20%
                cur_path_len = self._get_path_length()
                if (cur_path_len is not None and prev_path_len is not None
                        and prev_path_len > 0.1):
                    if cur_path_len > prev_path_len * 1.20:
                        detour_count += 1
                prev_path_len = cur_path_len

            # ── Phase 2: once near the goal, command final yaw via move_to ──
            d_to_goal = float(np.hypot(cx - goal.x, cz - goal.z))
            if d_to_goal <= self.goal_radius and not move_to_sent:
                try:
                    self.slam.yor_client.move_to(
                        (goal.x, goal.z, goal.yaw))
                    print(f"    ⟳ Near goal — commanding final yaw "
                          f"{np.degrees(goal.yaw):.1f}°")
                    move_to_sent = True
                except Exception as e:
                    print(f"    ⚠ move_to RPC failed: {e}")

            # ── Goal-reached: position AND yaw within tolerance ──
            yaw_err = abs(_wrap_pi(cyaw - goal.yaw))
            pos_ok = d_to_goal <= self.goal_radius
            yaw_ok = yaw_err <= self.yaw_tol_rad

            if pos_ok and yaw_ok:
                if converged_since is None:
                    converged_since = now
                elif (now - converged_since) >= self.settle_s:
                    print(f"    ✓ Arrived at {goal.label} in {elapsed:.1f}s "
                          f"(d={d_to_goal:.4f}m, "
                          f"Δyaw={np.degrees(yaw_err):.1f}°)")
                    break
            else:
                converged_since = None

            time.sleep(0.05)  # ~20 Hz poll

        # Final pose
        ex, ez, eyaw = self._get_pose()
        leg.end_x = ex
        leg.end_z = ez
        leg.end_yaw = eyaw
        leg.distance_m = dist_accum
        leg.duration_s = time.time() - t_start
        leg.path_length_changes = detour_count
        return leg

    # ── full benchmark ──────────────────────────────────────────────
    def run_all(self, ref_a: WaypointRef, ref_b: WaypointRef,
                n_runs: int) -> List[LegData]:
        """Execute n_runs round-trips A→B→A, returning all leg data."""
        legs: List[LegData] = []
        # First leg is always A→B, then alternates
        targets = []
        for i in range(n_runs):
            targets.append(("A", ref_b))   # A→B
            targets.append(("B", ref_a))   # B→A

        for i, (start_label, goal) in enumerate(targets):
            run_idx = i // 2
            leg = self.run_leg(goal, start_label, run_idx)
            legs.append(leg)
            print(f"    Leg {i+1}/{len(targets)} done | "
                  f"dist={leg.distance_m:.3f}m  dur={leg.duration_s:.1f}s")
        return legs


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════

def plot_results(legs: List[LegData], ref_a: WaypointRef, ref_b: WaypointRef,
                 voxel_samples: List[VoxelSample], stats: dict,
                 save_prefix: str, goal_radius: float = 0.10):
    """Generate and save benchmark plots."""
    mode = stats.get("mode", "")
    has_ekf = any(s.ekf_sigma_x is not None
                  for leg in legs for s in leg.samples)

    n_plots = 4 if has_ekf else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
    fig.suptitle(f"Repeatability Benchmark — {mode}", fontsize=14, fontweight="bold")

    # ── colours per leg ──
    cmap = plt.cm.tab10
    colours = [cmap(i % 10) for i in range(len(legs))]

    # ═══════════════════════════════════════════════════════════════════
    # Plot 1: X-Z Trajectory
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[0]
    for i, leg in enumerate(legs):
        xs = [s.x for s in leg.samples]
        zs = [s.z for s in leg.samples]
        ax.plot(xs, zs, color=colours[i], linewidth=1.0, alpha=0.7,
                label=f"R{leg.run_idx+1} {leg.leg_label}")

    # Waypoints and goal radii
    for ref, marker, color in [(ref_a, 'o', '#2ecc71'), (ref_b, 's', '#e74c3c')]:
        ax.plot(ref.x, ref.z, marker, color=color, markersize=10, zorder=5,
                label=f"{ref.label} ({ref.x:.2f}, {ref.z:.2f})")
        circle = Circle((ref.x, ref.z), goal_radius, fill=False,
                        edgecolor=color, linestyle='--', linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)

    # Arrival scatter
    for leg in legs:
        ref = ref_b if leg.goal_label == "B" else ref_a
        c = '#e74c3c' if leg.goal_label == "B" else '#2ecc71'
        ax.plot(leg.end_x, leg.end_z, 'x', color=c, markersize=7,
                markeredgewidth=2, zorder=6)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Trajectory (X-Z plane)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Plot 2: Voxel Count vs Time
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[1]
    if voxel_samples:
        vt = [s.t for s in voxel_samples]
        vc = [s.count for s in voxel_samples]
        ax.plot(vt, vc, color='#3498db', linewidth=1.5)
        ax.fill_between(vt, 0, vc, alpha=0.15, color='#3498db')

        # Mark leg boundaries
        t_accum = 0.0
        for i, leg in enumerate(legs):
            ax.axvline(t_accum, color='gray', linestyle=':', linewidth=0.7, alpha=0.5)
            t_accum += leg.duration_s

        if len(vt) >= 2 and vt[-1] > vt[0]:
            rate = (vc[-1] - vc[0]) / (vt[-1] - vt[0])
            ax.text(0.02, 0.95, f"Growth: {rate:.0f} vox/s",
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No voxel data", ha='center', va='center',
                transform=ax.transAxes)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voxel Count")
    ax.set_title("Map Growth")
    ax.grid(True, alpha=0.3)

    # ═══════════════════════════════════════════════════════════════════
    # Plot 3: Per-Leg Position Error Bar Chart
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[2]
    labels = [f"R{l['run']}\n{l['leg']}" for l in stats["per_leg"]]
    pos_errs = [l["err_pos_m"] * 100 for l in stats["per_leg"]]  # cm
    yaw_errs = [l["err_yaw_deg"] for l in stats["per_leg"]]

    x_pos = np.arange(len(labels))
    bar_w = 0.35
    bars1 = ax.bar(x_pos - bar_w/2, pos_errs, bar_w, label='Position (cm)',
                   color='#3498db', alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x_pos + bar_w/2, yaw_errs, bar_w, label='Heading (°)',
                    color='#e67e22', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Position Error (cm)", color='#3498db')
    ax2.set_ylabel("Heading Error (°)", color='#e67e22')
    ax.set_title("Arrival Error per Leg")
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # ═══════════════════════════════════════════════════════════════════
    # Plot 4: EKF Uncertainty (only if EKF mode)
    # ═══════════════════════════════════════════════════════════════════
    if has_ekf:
        ax = axes[3]
        t_offset = 0.0
        for i, leg in enumerate(legs):
            ts = [s.t + t_offset for s in leg.samples if s.ekf_sigma_x is not None]
            sx = [s.ekf_sigma_x * 1000 for s in leg.samples if s.ekf_sigma_x is not None]  # mm
            sz = [s.ekf_sigma_z * 1000 for s in leg.samples if s.ekf_sigma_z is not None]
            if ts:
                ax.plot(ts, sx, color=colours[i], linewidth=0.8, alpha=0.8)
                ax.plot(ts, sz, color=colours[i], linewidth=0.8, alpha=0.8, linestyle='--')
            t_offset += leg.duration_s

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EKF 1σ (mm)")
        ax.set_title("EKF Uncertainty")
        ax.legend(["σx (solid)", "σz (dashed)"], fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = f"{save_prefix}_plots.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plots saved to: {fig_path}")
    return fig_path


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="YOR SLAM repeatability benchmark")
    parser.add_argument("--runs", type=int, default=5,
                        help="number of round-trip runs (default: 5)")
    parser.add_argument("--goal-radius", type=float, default=0.10,
                        help="arrival radius in metres (default: 0.10)")
    parser.add_argument("--yaw-tol", type=float, default=5.0,
                        help="heading tolerance in degrees (default: 5.0)")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="seconds within radius to declare arrival")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="max seconds per leg before giving up")
    parser.add_argument("--sample-hz", type=float, default=5.0,
                        help="pose sampling rate during navigation")
    parser.add_argument("--no-ekf", dest="ekf", action="store_false",
                        help="disable EKF fusion")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="SLAM mapping rate")
    parser.add_argument("--yor-host", type=str, default="194.168.1.10")
    parser.add_argument("--yor-port", type=int, default=5557)
    parser.add_argument("--zed-host", type=str, default="127.0.0.1")
    parser.add_argument("--zed-port", type=int, default=6000)
    parser.add_argument("--zed-up-axis", type=str, default="y",
                        choices=["y", "z"])
    parser.add_argument("--save-map", action="store_true",
                        help="save the map on exit")
    parser.add_argument("--map-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (auto-generated if omitted)")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOR SLAM Repeatability Benchmark")
    print(f"  Mode: {'EKF' if args.ekf else 'ZED-only'}")
    print(f"  Runs: {args.runs}  |  Goal radius: {args.goal_radius} m  |  "
          f"Yaw tol: {args.yaw_tol}°")
    print("=" * 60)

    # ── 1. Start SLAM stack ──────────────────────────────────────────
    print("\n[benchmark] Starting SLAM stack...")
    slam = Slam(
        target_hz=args.hz,
        duration_s=0.0,
        load_map=False,
        save_map=args.save_map,
        map_path=args.map_path,
        yor_host=args.yor_host,
        yor_port=args.yor_port,
        zed_host=args.zed_host,
        zed_port=args.zed_port,
        zed_up_axis=args.zed_up_axis,
        use_ekf=args.ekf,
    )

    # Boot mapping + wait for data (replicates Slam.run() init but without
    # the blocking main loop)
    if not slam._wait_for_datastream():
        print("[benchmark] ERROR: ZED datastream not ready. Exiting.")
        return
    slam._start_mapping()
    slam.running = True

    # State monitor (keeps latest_map / latest_grid / latest_path updated)
    import threading
    slam.state_thread = threading.Thread(
        target=slam._state_monitor_loop, daemon=True)
    slam.state_thread.start()

    # Wait for enough map to initialise the planning stack
    print("[benchmark] Waiting for map to accumulate (≥500 voxels)...")
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            vmap = slam.map_manager.get_voxel_map()
            if vmap is not None and len(vmap) >= 500:
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not slam._start_planning_stack():
        print("[benchmark] ERROR: Could not start planning stack. "
              "Map may be too sparse.")
        slam.stop()
        return
    print("[benchmark] Planning stack ready.\n")

    # Start voxel tracker
    voxel_tracker = VoxelTracker(slam, sample_hz=1.0)
    voxel_tracker.start()

    # ── 2. Label waypoints ───────────────────────────────────────────
    runner = BenchmarkRunner(
        slam, goal_radius_m=args.goal_radius,
        yaw_tol_deg=args.yaw_tol,
        settle_s=args.settle, timeout_s=args.timeout,
        sample_hz=args.sample_hz, use_ekf=args.ekf,
    )

    ref_a = runner.label_point("A")
    ref_b = runner.label_point("B")

    ab_dist = float(np.hypot(ref_b.x - ref_a.x, ref_b.z - ref_a.z))
    print(f"\n  A→B euclidean distance: {ab_dist:.3f} m")
    if ab_dist < 0.3:
        print("  ⚠ Points are very close together — results may not be "
              "meaningful.")

    input("\n>>> Press ENTER to start the benchmark (robot will move!)... ")

    # ── 3. Run benchmark ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  BENCHMARK STARTING")
    print("─" * 60)

    legs = runner.run_all(ref_a, ref_b, n_runs=args.runs)
    voxel_tracker.stop()

    # ── 4. Statistics ────────────────────────────────────────────────
    stats = compute_stats(legs, ref_a, ref_b, use_ekf=args.ekf)
    print_report(stats)

    # ── 4b. Plots ────────────────────────────────────────────────────

    # ── 5. Save JSON ─────────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    mode_str = "ekf" if args.ekf else "zed"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_prefix = os.path.join(base_dir, f"benchmark_{mode_str}_{ts}")

    plot_results(legs, ref_a, ref_b, voxel_tracker.samples, stats,
                 save_prefix=save_prefix, goal_radius=args.goal_radius)

    if args.output:
        out_path = args.output
    else:
        out_path = f"{save_prefix}.json"

    # Convert samples to serialisable form
    def _serialise(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    raw_legs = []
    for leg in legs:
        d = asdict(leg)
        d["samples"] = [asdict(s) for s in leg.samples]
        raw_legs.append(d)
    stats["raw_legs"] = raw_legs

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=_serialise)
    print(f"  Results saved to: {out_path}")

    # ── 6. Shutdown ──────────────────────────────────────────────────
    slam.stop()
    print("[benchmark] Done.")


if __name__ == "__main__":
    main()
    os._exit(0)
