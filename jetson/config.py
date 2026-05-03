"""
Central configuration for the Jetson serial control package.
Edit these values to match your hardware setup.
"""

# ── Serial connection ─────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyACM0"   # check with: ls /dev/ttyACM* /dev/ttyUSB*
BAUD_RATE    = 115200
READ_TIMEOUT = 0.01             # serial read timeout (seconds), keep small

# ── Telemetry ─────────────────────────────────────────────────
# Arduino emits "TEL,<t_ms>,<label>,<enc1>,...\r\n" when log is on
TEL_PREFIX   = "TEL,"
TEL_COLUMNS  = [
    "t_ms", "label",
    "enc1", "enc2", "enc3",
    "vx_enc_mps", "ax_raw_mps2", "ax_f_mps2", "vx_imu_mps",
    "alpha_x_hat", "alpha_x_use",
    "i_sum_mA", "i_delta_mA",
    "adxl_ax", "adxl_ay", "adxl_az",
]
TEL_NUMERIC = {
    "t_ms":         int,
    "enc1":         int,   "enc2":        int,   "enc3":       int,
    "vx_enc_mps":   float, "ax_raw_mps2": float, "ax_f_mps2":  float,
    "vx_imu_mps":   float,
    "alpha_x_hat":  float, "alpha_x_use": float,
    "i_sum_mA":     float, "i_delta_mA":  float,
    "adxl_ax":      int,   "adxl_ay":     int,   "adxl_az":    int,
}

# ── Paths ─────────────────────────────────────────────────────
import os
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

# ── Motion ────────────────────────────────────────────────────
GOTO_DONE_TIMEOUT_S = 60.0   # max time to wait for [done] after a goto
