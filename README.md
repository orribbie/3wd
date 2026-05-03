# SIR 3WD Base — Jetson ↔ Arduino Control

Direct USB-serial control of a 3-wheel omnidirectional robot from a Jetson.

```
Jetson ──────────── USB / PySerial ──────────── Arduino Mega
  main.py                                         sir_3wd_base.ino
  controller.py                                   BNO08x  (yaw, accel)
  serial_link.py                                  ADXL345 (vibration)
  telemetry.py                                    INA260 × 2 (current)
  yaw_provider.py                                 3× Cytron motors
  zed_yaw_provider.py  ← optional ZED 2i IMU      3× encoders
```

---

## Project structure

```
sir/
├── arduino/
│   └── sir_3wd_base/
│       └── sir_3wd_base.ino   ← upload this to the Arduino
├── jetson/
│   ├── main.py                ← entry point (CLI)
│   ├── serial_link.py         ← low-level serial I/O + reader thread
│   ├── controller.py          ← high-level command interface
│   ├── telemetry.py           ← TEL packet parser + CSV logger
│   ├── yaw_provider.py        ← abstract YawProvider + Arduino + Dummy
│   ├── zed_yaw_provider.py    ← optional ZED 2i IMU yaw
│   └── config.py              ← port, baud, column names, paths
├── logs/                      ← auto-created CSV telemetry files
├── environment.yml
└── README.md
```

---

## Setup

### 1 — Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate 3wd
```

### 2 — Install Python dependencies (if updating later)

```bash
conda activate 3wd
pip install pyserial rich
```

### 3 — Upload Arduino firmware

Open `arduino/sir_3wd_base/sir_3wd_base.ino` in the Arduino IDE and upload
to your Arduino Mega (or compatible board).

Required Arduino libraries (install via the Library Manager):
- `Adafruit BNO08x`
- `Adafruit INA260`
- `CytronMotorDriver`
- `Encoder` (Paul Stoffregen)

### 4 — Find the Arduino serial port on Jetson / Linux

```bash
ls /dev/ttyACM* /dev/ttyUSB*
# typically /dev/ttyACM0 for Arduino Mega over USB
```

If you get a permission error when running the CLI, add yourself to the
`dialout` group and re-login:

```bash
sudo usermod -aG dialout $USER
# log out and back in, then verify:
groups | grep dialout
```

Alternatively, set the permissions directly for the current session:

```bash
sudo chmod a+rw /dev/ttyACM0
```

---

## Running the CLI

```bash
conda activate 3wd
python jetson/main.py
# or specify the port:
python jetson/main.py --port /dev/ttyACM1
# with automatic CSV telemetry logging:
python jetson/main.py --log-tel
```

---

## Command reference

| Command | Description |
|---|---|
| `g x y theta` | Move to position (x, y in grid units) and face `theta` degrees |
| `stop` | Halt all motors immediately |
| `z` | Zero the pose and IMU yaw reference |
| `s` | Print status: phase, yaw, slip estimates, current, ADXL |
| `y` | Print current BNO08x yaw only |
| `log 1 label` | Start emitting TEL CSV rows with the given label |
| `log 0` | Stop telemetry stream |
| `help` | Show command list |
| `quit` / `exit` | Disconnect and exit |

### Example session

```
sir> z
  << [reset]

sir> g 1 0 0
  << [goto accepted]
  Wait for motion to complete? [Y/n]: y
  Waiting for [done]...
  << [done] received.

sir> g 0.5 0.5 90
  << [goto accepted]
  Wait for motion to complete? [Y/n]: y

sir> s
  << [status] phase=0 yaw=88 ax=0.997 ay=0.998 I_sum=312.5 I_dlt=4.1 adxl=0,1,-255

sir> stop
  << [stopped]

sir> quit
Disconnected.
```

Grid unit is 100 mm by default (`GRID_UNIT_MM` in the Arduino sketch).
So `g 1 0 0` moves 100 mm along the X axis and faces 0°.

---

## Telemetry CSV

When `log 1 <label>` is active, the Arduino emits a CSV row every 40 ms:

```
TEL,<t_ms>,<label>,<enc1>,<enc2>,<enc3>,<vx_enc>,<ax_raw>,<ax_f>,<vx_imu>,<alpha_x_hat>,<alpha_x_use>,<i_sum_mA>,<i_delta_mA>,<adxl_ax>,<adxl_ay>,<adxl_az>
```

The Python `TelemetryLogger` in `telemetry.py` parses these and writes them
to a timestamped file in `logs/`. You can load them with pandas:

```python
import pandas as pd
df = pd.read_csv("logs/tel_20250101_120000.csv")
```

Running with `--log-tel` saves all packets automatically.

---

## ZED 2i IMU integration (optional)

The ZED SDK is not a normal pip package.  After installing it:

```bash
# Install the ZED SDK from https://www.stereolabs.com/developers/release
bash ZED_SDK_*.run
# Install the Python bindings into the active conda env:
conda activate 3wd
python /usr/local/zed/get_python_api.py
```

Then run with:

```bash
python jetson/main.py --yaw-src zed
```

Or in Python:

```python
from yaw_provider import build_yaw_provider
yaw = build_yaw_provider("zed")
print(yaw.get_yaw_deg())
yaw.close()
```

If the ZED SDK is not installed the code falls back to `DummyYawProvider`
automatically when `build_yaw_provider("zed")` is called.

---

## Extending the codebase

| Goal | Where to start |
|---|---|
| Autonomous waypoint following | Add a loop in `controller.py` calling `goto()` + `wait_done()` |
| Surface classification | Subscribe to `TelemetryLogger` callback for `i_delta_mA` + `adxl_*` |
| Slip detection | Monitor `alpha_x_use` / `alpha_y_use` from telemetry |
| External yaw fusion (ZED + BNO) | Create a new `YawProvider` subclass in `yaw_provider.py` |
| ROS 2 bridge | Wrap `Controller` in a ROS 2 node, keep serial logic untouched |

---

## Debugging tips

**Nothing received after connect**
- Check that the correct port is selected (`ls /dev/ttyACM*`).
- Verify 115200 baud on both sides.
- The Arduino resets on serial open; wait ~1 s for the boot message.

**`[serial error] ...`  appears in the CLI**
- The USB cable was unplugged or the Arduino reset mid-session.
- Quit and restart `main.py`.

**Commands seem to have no effect**
- Check that the Arduino firmware compiled without errors.
- Open the Arduino Serial Monitor at 115200 baud and type `s` — if it
  responds, the firmware is running.  If not, re-upload.

**`Permission denied: '/dev/ttyACM0'`**
- Run `sudo chmod a+rw /dev/ttyACM0` or add yourself to `dialout` (see above).

**BNO08x / INA260 / ADXL345 not detected**
- Check I2C wiring and addresses.  The Arduino will print an error and halt
  in `setup()` if a required sensor is missing.
- Use an I2C scanner sketch to confirm addresses.
