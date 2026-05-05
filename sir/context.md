# SIR Project — Work Context

_Last updated: 2026-05-03 (session 3)_

---

## Project overview

Control and navigation codebase for a 3-wheel omnidirectional (3WD) robot.

Architecture:
```
ZED 2i camera  ──────────────────────────────────────────────────┐
                                                                  ▼
Jetson  ──commlink pubsub──▶  slam_node_.py  ──SirBridge──▶  /dev/ttyACM0  ──▶  Arduino Mega
  │                               │                 │
  │                          EKF (ZED+odom)    serial_link.py
  │                          A* path planner   controller.py
  │                          Viser UI (8099)
  │
  └── traction inference (MobileNetV3 ONNX, ~30 Hz in-process)
```

The ESP/XIAO wireless mux layer from the original prototype has been removed.
The Jetson is now the sole command interface, sending text commands directly
over USB serial. The Arduino handles all real-time motor control.

---

## Hardware

| Component | Role |
|---|---|
| NVIDIA Jetson (Orin / Xavier / Nano) | SLAM, path planning, traction inference, Python CLI |
| Arduino Mega | Real-time motor control, sensor polling |
| ZED 2i stereo camera | Visual odometry, depth/point-cloud, traction images |
| BNO055 (I2C 0x28) | Primary IMU on Arduino: yaw + linear accel |
| ADXL345 (I2C 0x53) | Secondary accel on Arduino: vibration / surface data |
| INA260 × 2 (0x40, 0x41) | Current sensors for M2 and M3 |
| Cytron motor drivers × 3 | PWM_DIR mode, 3WD omnidirectional layout |
| Encoders × 3 | Quadrature, wired to interrupt-capable pins |

**Motor / encoder pin map (Arduino Mega)**

| | PWM | DIR | ENC_A | ENC_B |
|---|---|---|---|---|
| M1 | 6 | 7 | 13 | 3 |
| M2 | 4 | 5 | 2 | 9 |
| M3 | 8 | 10 | 18 | 11 |

**Key robot constants**
- `GRID_UNIT_MM` = 100 mm (1 grid unit)
- `WHEEL_RADIUS_MM` = 60 mm
- `GEAR_RATIO` = 41.6×, `ENCODER_PPR_MOTOR` = 4, quadrature ×16 → `COUNTS_PER_REV` = 2662.4
- `ROBOT_RADIUS_MM` = 150 mm (wheel-to-center)
- Wheel layout: 0° (front), 120° (back-right), 240° (back-left)

---

## Repository structure

```
~/
├── sir/
│   ├── arduino/
│   │   └── sir_3wd_base/
│   │       └── sir_3wd_base.ino       ← ACTIVE firmware (auto-flashed by slam_node_.py)
│   ├── jetson/
│   │   ├── main.py                    ← standalone CLI entry point
│   │   ├── serial_link.py             ← background reader thread, two queues
│   │   ├── controller.py              ← goto / stop / zero / status / wait_done
│   │   ├── telemetry.py               ← TEL CSV parser + TelemetryLogger
│   │   ├── yaw_provider.py            ← YawProvider ABC
│   │   └── config.py                  ← port, baud, column names, paths
│   ├── logs/
│   ├── environment.yml                ← conda env "3wd"
│   └── context.md                     ← this file
│
└── robot/
    ├── slam_node_.py                  ← MAIN entry point (SLAM + navigation + traction)
    ├── zed_pub_node.py                ← ZED camera publisher (commlink port 6000)
    ├── traction_node.py               ← standalone traction monitor (optional, not needed)
    ├── model.onnx                     ← MobileNetV3 traction regression model
    ├── label_scaler.pkl               ← inverse scaler for traction output
    ├── sir_bridge.py                  ← serial bridge (replaces Yor RPC)
    ├── nav/
    │   ├── odometry/
    │   │   ├── omni3_odom.py          ← 3WD forward kinematics (replaces swerve_odom.py)
    │   │   ├── robot_ekf.py           ← 3-state EKF [px, pz, yaw]
    │   │   └── swerve_odom.py         ← UNUSED (kept for reference)
    │   ├── mapping/
    │   │   └── mapping_torch.py       ← GPU voxel map accumulation
    │   ├── pathPlanning.py            ← A* on 2D occupancy grid
    │   └── viserBridge.py             ← Viser web UI (port 8099)
    └── tools/
        └── align_check.py             ← axis alignment verification script
```

---

## How to run

### Full stack (SLAM + navigation + traction + robot control)

```bash
# Terminal 1 — ZED publisher
conda activate slam
python ~/robot/zed_pub_node.py

# Terminal 2 — SLAM node (also auto-flashes Arduino on startup)
conda activate slam
cd ~
python ~/robot/slam_node_.py \
  --traction-model ~/robot/model.onnx \
  --traction-scaler ~/robot/label_scaler.pkl \
  --zed-up-axis y
```

### Viser UI over SSH

```bash
# On your laptop:
ssh -L 8099:localhost:8099 user@<jetson-ip>
# Open browser: http://localhost:8099
```

### Dry run (no Arduino, mapping only)

Omit the Arduino — `SirBridge` detects the missing port and runs in no-op mode.
Robot motion is disabled but SLAM mapping and Viser work normally.

### Axis alignment check (run once after hardware changes)

```bash
# zed_pub_node.py must be running
conda activate slam
cd ~/robot
python tools/align_check.py
```

---

## What was built / changed (2026-05-03 — session 3)

### Integration: robot/ SLAM ↔ sir/ hardware

The SLAM stack (`robot/`) was originally written for a 4-wheel swerve drive
robot controlled via a ZMQ RPC server ("Yor") on a remote machine. This session
replaced all of that with a direct serial bridge to the SIR 3WD Arduino.

#### New files

| File | Purpose |
|---|---|
| `robot/nav/odometry/omni3_odom.py` | 3WD forward kinematics replacing `SwerveOdom`. Wheels at 0°/120°/240°, correct pseudo-inverse FK. Matches `SwerveOdom` interface so `EKFSlamSource` works unchanged. |
| `robot/sir_bridge.py` | Drop-in replacement for `RPCClient("Yor")`. Manages serial link, feeds encoder counts to EKF, executes `follow_path` waypoints, traction-based speed scaling, ZED slip correction. Also auto-flashes Arduino on startup. |
| `robot/tools/align_check.py` | Moves robot in each axis and reads ZED pose before/after. Reports PASS/FAIL and exact fix instructions if any axis is swapped or sign-flipped. |

#### Modified files

**`robot/slam_node_.py`**
- Removed `RPCClient` import; added `SirBridge` and `Omni3Odom` imports
- `EKFSlamSource`: swapped `SwerveOdom` → `Omni3Odom`
- `Slam.__init__`: single shared `SirBridge` instance passed to both `EKFSlamSource` and `_path_sender_loop`; `SirBridge.set_pose_provider()` wired to `datastream.get_pose`
- Added `_traction_loop()` thread: runs MobileNetV3 ONNX inference in-process, calls `sir_bridge.set_traction(value)` at ~30 Hz
- Added `--traction-model` / `--traction-scaler` CLI args
- `_reset_yor_client()` is now a no-op

**`sir/arduino/sir_3wd_base/sir_3wd_base.ino`**
- `PWM_POS_MAX` and `PWM_YAW_MAX` changed from `const int` to `int` (mutable)
- Added `const int PWM_MOVE_MIN = 35` — minimum PWM to overcome static friction
- Added `deadbandBoost(int pwm)` helper — boosts any non-zero PWM to at least `PWM_MOVE_MIN`
- Applied `deadbandBoost` in both `updatePositionControl()` and `updateYawControl()`
- Added `spd <0-100>` serial command: scales `PWM_POS_MAX` (0–100) and `PWM_YAW_MAX` (0–50)
- **Disabled RLS slip compensation in `updatePositionControl()`** — `alpha_x_use` is still estimated and reported in telemetry but no longer applied to target encoder counts (ZED feedback loop is now the correction authority)

---

## Architecture details (session 3)

### SirBridge

Replaces the `RPCClient("Yor")` ZMQ interface. Single instance shared across
EKF predict thread and path sender loop.

**Startup sequence (every `slam_node_.py` launch):**
1. Compile + upload `sir_3wd_base.ino` via Arduino IDE (`~30 s`)
2. Wait 2 s for Arduino reboot
3. Open `/dev/ttyACM0` @ 115200 baud
4. Send `z` — zero Arduino pose + yaw
5. Send `log 1 ekf` — start telemetry stream
6. Start telemetry consumer thread + log renewal thread

**`get_base_encoders()`** — called by EKF predict thread at ~20 Hz:
- Returns `{"steer_rad": [0,0,0], "drive_counts": [enc1,enc2,enc3], "timestamp": float}`
- Populated by background telemetry consumer reading `TEL,...` rows
- Returns `None` until first telemetry packet arrives (EKF handles gracefully)

**`follow_path([(x,z), ...])`** — called by `_path_sender_loop`:
- Runs in a background thread; new call cancels previous
- For each waypoint: set traction speed → send `g x y theta` → wait `[done]` → ZED correction loop
- No-op if Arduino not connected

**ZED slip correction (after each `[done]`):**
- Reads EKF-fused pose via `pose_fn()`
- If residual error > `CORRECTION_THRESHOLD_M = 0.10 m` → sends correction goto
- Up to `MAX_CORRECTIONS = 3` attempts per waypoint
- Silently skips if ZED unavailable

### Traction → speed scaling

| Traction value | Speed |
|---|---|
| ≤ 900 (slippiest) | 20% PWM |
| 1100 (medium) | ~60% PWM |
| ≥ 1300 (grippiest) | 100% PWM |

Linear interpolation across 900–1300 range.
Sent as `spd <pct>` command before each waypoint.

### 3WD forward kinematics (`omni3_odom.py`)

Wheels at 0°, 120°, 240° from robot +X axis:

```
vx    =  (√3/3) * (v3 − v2)          # forward speed
vy    =  (2/3)*v1 − (1/3)*(v2 + v3)  # left-lateral speed
omega =  (v1 + v2 + v3) / (3 * L)    # yaw rate [rad/s]
```

Telemetry encoder values (`-enc.read()`) are used directly:
- Positive delta = wheel spinning in active rolling direction

### Coordinate mapping (SLAM ↔ Arduino)

| SLAM world frame | Arduino frame |
|---|---|
| `displacement D = (dx, dz)` | `x_grid = D · f_vec / 0.10` |
| `relative to origin` | `y_grid = D · l_vec / 0.10` |
| `yaw` (rad, CCW+) | `theta = atan2(V_next · l_vec, V_next · f_vec)` |

Both frames zeroed together at startup (`z` command + EKF init from ZED).

---

## Serial command interface

| Command | Arduino response |
|---|---|
| `g x y theta` | `[goto accepted]` … later `[done]` |
| `stop` | `[stopped]` |
| `z` | `[reset]` |
| `s` | `[status] phase=N yaw=N ax=F ay=F I_sum=F I_dlt=F adxl=N,N,N` |
| `y` | `[yaw] N` |
| `spd <0-100>` | `[speed] N` — scales PWM_POS_MAX and PWM_YAW_MAX |
| `log 1 label` | `[log on] label=...` |
| `log 0` | `[log off]` |

---

## Telemetry CSV column order (17 fields after "TEL")

```
t_ms, label, enc1, enc2, enc3,
vx_enc_mps, ax_raw_mps2, ax_f_mps2, vx_imu_mps,
alpha_x_hat, alpha_x_use,
i_sum_mA, i_delta_mA,
adxl_ax, adxl_ay, adxl_az
```

Emitted at 40 ms intervals when `LOG_ON = true`.
Note: `alpha_x_hat` / `alpha_x_use` still reported but no longer applied to
motion control (slip correction is now done via ZED feedback on the Jetson).

---

## Conda environments

| Name | Used for |
|---|---|
| `slam` | `slam_node_.py`, `zed_pub_node.py` — ZED SDK, torch, commlink, onnxruntime |
| `3wd` | `sir/jetson/main.py` — standalone robot CLI (pyserial, numpy, pandas, rich) |

`pyserial` must also be installed in `slam` env (needed by `sir_bridge.py`):
```bash
conda activate slam && pip install pyserial
```

---

## Design decisions (session 3)

| Decision | Rationale |
|---|---|
| Single `SirBridge` shared by EKF + path sender | One serial port, one connection — avoids port contention |
| Traction inference in-process (`_traction_loop`) | Avoids IPC complexity; ZED image already available via `datastream` |
| ZED as slip correction authority | Arduino's RLS estimator is encoder+IMU only, inherits wheel slip. ZED is independent ground truth. |
| Arduino RLS compensation disabled | Encoder-based pre-compensation and ZED post-correction fight each other. Disabling one gives clean separation of concerns. |
| `deadbandBoost` in firmware | P-controller outputs small PWMs near target. At low traction (20% max) the robot would stall without a minimum PWM floor. |
| Auto-flash on startup | Ensures firmware is always in sync with repo. ~30 s overhead at start is acceptable. |
| No-op mode when Arduino absent | Allows SLAM mapping dry-runs and SSH-only Viser sessions without hardware. |

---

## Debug log

### 2026-05-02 — BNO08x I2C hang on startup

**Symptom:** Serial monitor printed "Starting BNO08x..." then hung with no further output.

**Root cause:** `bno08x.begin_I2C()` blocks indefinitely when the I2C bus is stuck
(SDA held low from a previous session) or when the sensor is not responding.

**Fix applied (`sir_3wd_base.ino`):**
1. Added `i2c_recover()` — bit-bangs up to 9 SCL pulses to free a stuck SDA line
2. Added `i2c_ping(addr)` — fast check before calling `begin_I2C`
3. Auto-detects BNO08x at 0x4B (ADR=HIGH) or 0x4A (ADR=LOW)

**I2C wiring (Arduino Mega):** SDA → pin 20, SCL → pin 21, 4.7 kΩ pull-ups

---

### 2026-05-02 (session 2) — Yaw control continuous spinning

**Symptom:** Robot spun continuously and never stopped.

**Root cause:** `applyRotationPWM` negated the P-controller output — positive feedback.

**Fix:** Removed negation in `applyRotationPWM`:
```cpp
motor1.setSpeed(pwm);  // was -pwm
motor2.setSpeed(pwm);
motor3.setSpeed(pwm);
```

---

### 2026-05-03 (session 3) — `RPCClient` NameError at startup

**Symptom:** `NameError: name 'RPCClient' is not defined` at class definition time.

**Root cause:** Type annotation `yor_client: RPCClient` remained in `EKFSlamSource.__init__`
after `RPCClient` was removed from imports.

**Fix:** Changed annotation to untyped `yor_client`.

---

### 2026-05-03 (session 3) — `ModuleNotFoundError: No module named 'serial'`

**Symptom:** `sir_bridge.py` import failed in the `slam` conda env.

**Fix:** `pip install pyserial` in the `slam` env (zero dependencies, safe).

---

## Known limitations / future work

- **Axis alignment not yet empirically verified** — run `python tools/align_check.py` after first hardware test to confirm SLAM X/Z → Arduino X/Y mapping and yaw sign are correct. Fix instructions are printed by the script.
- **`PWM_MOVE_MIN = 35` needs tuning** — raise if robot stalls at low traction; lower if it lurches at start of each move.
- **Arduino IDE flash path hardcoded** — `ARDUINO_IDE` in `sir_bridge.py` points to `/home/slam/arduino-1.8.15/arduino`. Update if IDE moves.
- **Board FQBN hardcoded** — `arduino:avr:mega` in `sir_bridge.py`. Verify with `arduino --board-details` if flash fails.
- **No ROS 2 integration** — `controller.py` is designed so a ROS 2 node can wrap it without touching serial logic.
- **Traction model output range assumed 900–1300** — if model is retrained, update `TRACTION_LOW` / `TRACTION_HIGH` in `sir_bridge.py`.
