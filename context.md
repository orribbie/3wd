# SIR Project — Work Context

_Last updated: 2026-05-02 (session 2)_

---

## Project overview

Control codebase for a 3-wheel omnidirectional (3WD) robot.
Architecture: **Jetson ↔ USB Serial (PySerial) ↔ Arduino Mega**

The ESP/XIAO wireless mux layer from the original prototype has been removed.
The Jetson is now the sole command interface, sending text commands directly
over USB serial. The Arduino handles all real-time motor control.

---

## Hardware

| Component | Role |
|---|---|
| NVIDIA Jetson (Orin / Xavier / Nano) | High-level command interface, Python CLI |
| Arduino Mega | Real-time motor control, sensor polling |
| ZED 2i depth camera | Optional: IMU yaw source from Jetson side |
| BNO08x (I2C 0x4B) | Primary IMU on Arduino: yaw + linear accel |
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
- `GEAR_RATIO` = 41.6×, `ENCODER_PPR_MOTOR` = 4, quadrature ×4 → `COUNTS_PER_REV` = 2662.4
- `ROBOT_RADIUS_MM` = 150 mm (wheel-to-center)

---

## Repository structure

```
sir/
├── arduino.ino                        ← original reference sketch (keep for reference)
├── arduino/
│   └── sir_3wd_base/
│       └── sir_3wd_base.ino           ← ACTIVE firmware (upload this)
├── jetson/
│   ├── main.py                        ← CLI entry point
│   ├── serial_link.py                 ← background reader thread, two queues
│   ├── controller.py                  ← goto / stop / zero / status / wait_done
│   ├── telemetry.py                   ← TEL CSV parser + TelemetryLogger
│   ├── yaw_provider.py                ← YawProvider ABC, Arduino + Dummy impls
│   ├── zed_yaw_provider.py            ← optional ZED 2i IMU (isolated import)
│   └── config.py                      ← port, baud, column names, paths
├── logs/                              ← auto-generated telemetry CSVs
├── environment.yml                    ← conda env "3wd"
├── README.md
└── context.md                         ← this file
```

---

## What was built (2026-05-02)

### Arduino firmware (`sir_3wd_base.ino`)

Refactored from `arduino.ino` (Rev 2: BNO08x + ADXL345).

**Removed:**
- `Serial3` / `ESP_SERIAL` / `ESP_BAUD`
- UART MUX framing (`muxPoll`, `muxSendFrame`, `RxState` state machine)
- `CmdStream` class, `WIFI_PORT`, `BT_SERIAL` alias
- Binary `TelemetryPacket` struct and `#pragma pack` block
- `logHeaderPrinted` (was never used)

**Kept (unchanged logic):**
- 3WD motor control: `stopAll`, `applyRotationPWM`, `beginRotateTo`, `updateYawControl`
- Phase state machine: `IDLE → ROTATE_TO_PATH → DRIVE_STRAIGHT → ROTATE_TO_FINAL`
- Encoder-based position control: `startMoveAlongWheel1`, `updatePositionControl`
- Goto planner: `planGoto`
- BNO08x: game rotation vector → yaw, linear accel → IMU velocity proxy
- ADXL345: raw Wire reads, 100 Hz, ±2 g
- INA260 × 2: `readCurrents`, i_sum, i_delta with LPF baseline
- RLS slip estimator + IMU velocity proxy + forward kinematics
- Auto-goto logging: starts TEL on `g` command, stops after 60 s

**Changed:**
- Commands arrive only on `Serial` (USB)
- Telemetry changed from binary packed struct → ASCII CSV prefixed `TEL,`
- `s` status response now includes `yaw=<deg>`
- New `y` command for quick yaw query: `[yaw] <int>`
- `readLineFrom` simplified to single-port, single-buffer

**Telemetry CSV column order (17 fields after "TEL"):**
```
t_ms, label, enc1, enc2, enc3,
vx_enc_mps, ax_raw_mps2, ax_f_mps2, vx_imu_mps,
alpha_x_hat, alpha_x_use,
i_sum_mA, i_delta_mA,
adxl_ax, adxl_ay, adxl_az
```
Emitted at 40 ms intervals when `LOG_ON = true`.

### Python package (`jetson/`)

| File | Key class / function |
|---|---|
| `serial_link.py` | `SerialLink` — background thread, `_response_q`, `_telemetry_q` |
| `controller.py` | `Controller` — `goto`, `stop`, `zero`, `status`, `yaw`, `log_start/stop`, `wait_done` |
| `telemetry.py` | `TelemetryRecord` dataclass, `TelemetryLogger` (context manager, CSV) |
| `yaw_provider.py` | `YawProvider` ABC, `DummyYawProvider`, `ArduinoYawProvider`, `build_yaw_provider()` |
| `zed_yaw_provider.py` | `ZedYawProvider` — opens ZED in `DEPTH_MODE.NONE`, polls IMU at 200 Hz |
| `config.py` | `SERIAL_PORT`, `BAUD_RATE`, `TEL_COLUMNS`, `TEL_NUMERIC`, `LOG_DIR` |
| `main.py` | Interactive CLI, `rich` + readline, `--port`, `--log-tel`, `--yaw-src` flags |

**SerialLink stream parsing:**
- Lines beginning with `TEL,` → `_telemetry_q` (parsed to dict)
- All other lines → `_response_q` (raw strings)
- No binary framing needed (Arduino now sends ASCII only)

**YawProvider hierarchy:**
```
YawProvider (ABC)
├── DummyYawProvider       — constant / test
├── ArduinoYawProvider     — polls 'y' command, caches with interval
└── ZedYawProvider         — ZED SDK, 200 Hz IMU thread (isolated import)
```

---

## Conda environment

Name: `3wd`
Python: 3.10
Dependencies: `pyserial`, `numpy`, `pandas`, `rich`

ZED SDK: manual install only (`bash ZED_SDK_*.run`, then `python /usr/local/zed/get_python_api.py`).

---

## Serial command interface

| Command | Arduino response |
|---|---|
| `g x y theta` | `[goto accepted]` … later `[done]` |
| `stop` | `[stopped]` |
| `z` | `[reset]` |
| `s` | `[status] phase=N yaw=N ax=F ay=F I_sum=F I_dlt=F adxl=N,N,N` |
| `y` | `[yaw] N` |
| `log 1 label` | `[log on] label=...` |
| `log 0` | `[log off]` |

---

## Design decisions

| Decision | Rationale |
|---|---|
| ASCII CSV telemetry instead of binary | Easier to debug, bandwidth is not a bottleneck on USB serial at 115200 |
| Isolated `zed_yaw_provider.py` | ZED SDK import error must not break the rest of the package |
| `YawProvider` abstraction | Future: fuse BNO08x + ZED, or swap to external SLAM pose |
| `wait_done()` polls response queue | Avoids blocking the Arduino; intermediate TEL rows still flow |
| `autoGotoLog` (Arduino-side) | Auto-logs 60 s of telemetry on every goto — captures full motion profile |

---

## Debug log

### 2026-05-02 — BNO08x I2C hang on startup

**Symptom:** Serial monitor printed "Starting BNO08x..." then hung with no further output.

**Root cause:** `bno08x.begin_I2C()` blocks indefinitely when the I2C bus is stuck
(SDA held low from a previous session) or when the sensor is not responding. The
original code had no pre-flight ping, so a stuck bus caused a silent hang.

**Fix applied (`sir_3wd_base.ino`):**
1. Added `i2c_recover()` — bit-bangs up to 9 SCL pulses to free a stuck SDA line,
   then issues a STOP condition and re-inits `Wire`. Runs before any sensor init.
2. Added `i2c_ping(addr)` — a fast `beginTransmission/endTransmission` check that
   returns immediately, so the I2C scan is done BEFORE calling `begin_I2C`.
3. Setup now prints the full I2C scan result (0x4B, 0x4A, 0x40, 0x41, 0x53) so you
   can see at a glance what's alive on the bus.
4. Auto-detects BNO08x at 0x4B (ADR=HIGH) OR 0x4A (ADR=LOW) — picks whichever
   responds. This handles boards where ADR is wired to GND by default.
5. If no devices are found at all, prints wiring hints and halts with a clear message.

**I2C wiring reference for Arduino Mega:**
- SDA → pin 20
- SCL → pin 21
- Both lines need 4.7 kΩ pull-ups to 3.3 V (if BNO08x is 3.3 V) or 5 V

**BNO08x I2C address:**
- ADR pin → GND : 0x4A (factory default)
- ADR pin → VCC : 0x4B

**Diagnostic sketch added:** `/home/slam/sketchbook/i2c_scanner/i2c_scanner.ino`
Upload this to see every I2C device address responding on the bus.

### 2026-05-02 (session 2) — Yaw control continuous spinning

**Symptom:** Robot spun continuously and never stopped when given a yaw or goto
command. Error never converged to zero.

**Root cause:** `applyRotationPWM` negated the P-controller output:
```cpp
motor1.setSpeed(-pwm);  // was wrong
```
A positive `err` (need to turn CCW) produced a negative PWM, which spun the
motors CW — moving yaw *away* from the target and increasing the error every
cycle (positive feedback).

**Fix applied (`sir_3wd_base.ino`, line ~262):**
```cpp
void applyRotationPWM(int pwm) {
  motor1.setSpeed(pwm);   // sign flipped: was -pwm
  motor2.setSpeed(pwm);
  motor3.setSpeed(pwm);
}
```

**Fallback if still spinning (opposite direction):** the BNO08x yaw convention
on this hardware is opposite to the motor wiring convention. In that case,
negate the error instead of the PWM:
```cpp
// in updateYawControl:
float err = (float)yaw_deg - target_yaw_deg;  // was target - current
```

**Yaw control tuning constants (currently):**
- `YAW_KP = 3.0`, `YAW_DEADBAND = 3.0°`, `PWM_YAW_MAX = 50`

---

## Known limitations / future work

- **No closed-loop yaw from Jetson yet** — the ZED yaw provider exists but is
  not wired into the goto command path. Future: intercept goto, compute
  yaw-corrected heading from ZED, send corrected `g` command.
- **No odometry feedback to Jetson** — encoder counts arrive in telemetry CSV
  but are not used to update a Jetson-side pose estimate.
- **Single-port assumption** — `config.py` hardcodes `/dev/ttyACM0`; on Jetson
  with multiple USB devices, check `ls /dev/ttyACM*` after each plug.
- **No ROS 2 integration** — `controller.py` is designed so a ROS 2 node can
  wrap it without touching serial logic.
- **Surface/traction classification** — ADXL345 + i_delta_mA data is logged;
  classification model not yet built.
