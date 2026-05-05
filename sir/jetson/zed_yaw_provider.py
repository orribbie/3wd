"""
ZED 2i IMU yaw provider.

This module is intentionally isolated so that the rest of the codebase
imports cleanly even when the ZED SDK is not installed.

Requirements:
  - ZED SDK 4.x installed via the official ZED installer
  - pyzed Python bindings (installed as part of the SDK)

Activation:
  >>> from zed_yaw_provider import ZedYawProvider
  >>> yaw = ZedYawProvider()
  >>> print(yaw.get_yaw_deg())

If the ZED SDK is missing you will get an ImportError here — that is expected
and is caught by yaw_provider.build_yaw_provider('zed').
"""

import math
import threading
import time

try:
    import pyzed.sl as sl
except ImportError as _exc:
    raise ImportError(
        "pyzed not found. Install the ZED SDK from https://www.stereolabs.com/developers "
        "and run: python /usr/local/zed/get_python_api.py"
    ) from _exc

from yaw_provider import YawProvider


class ZedYawProvider(YawProvider):
    """
    Opens the ZED 2i, enables IMU data, and continuously updates a cached
    yaw value. The camera is opened in DEPTH_MODE.NONE to minimise GPU use
    (we only need the IMU, not depth).

    Parameters
    ----------
    resolution  : sl.RESOLUTION constant (default SVGA = 720p @ 60 fps)
    update_hz   : how often to poll the IMU (default 200 Hz)
    """

    def __init__(
        self,
        resolution=None,
        update_hz: float = 200.0,
    ):
        self._zed = sl.Camera()
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NONE  # no depth needed
        init.camera_resolution = resolution or sl.RESOLUTION.SVGA
        init.sensors_required = True

        err = self._zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            self._zed.close()
            raise RuntimeError(f"ZED open failed: {err}")

        self._sensors_data = sl.SensorsData()
        self._yaw_deg: float = 0.0
        self._lock = threading.Lock()
        self._interval = 1.0 / update_hz
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    # ── YawProvider interface ─────────────────────────────────

    def get_yaw_deg(self) -> float:
        with self._lock:
            return self._yaw_deg

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._zed.close()

    # ── Background polling ────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            if self._zed.get_sensors_data(self._sensors_data, sl.TIME_REFERENCE.CURRENT) \
                    == sl.ERROR_CODE.SUCCESS:
                imu_pose = self._sensors_data.get_imu_data().get_pose()
                # get_euler_angles returns (roll, pitch, yaw) in degrees
                euler = imu_pose.get_euler_angles(radian=False)
                yaw = float(euler[2])
                # Normalise to -180 … +180
                while yaw >  180.0: yaw -= 360.0
                while yaw < -180.0: yaw += 360.0
                with self._lock:
                    self._yaw_deg = yaw
            elapsed = time.monotonic() - t0
            sleep = self._interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
