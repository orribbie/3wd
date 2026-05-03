"""
YawProvider abstraction — decouple yaw source from motion logic.

Three implementations are included here:
  DummyYawProvider   — always returns 0 (useful for testing without hardware)
  ArduinoYawProvider — polls the Arduino BNO055 via the 'y' serial command

The ZED 2i implementation lives in zed_yaw_provider.py to keep the ZED SDK
import isolated (see that file for usage).

Usage:
    from yaw_provider import ArduinoYawProvider
    from controller import Controller

    yaw = ArduinoYawProvider(ctrl)
    print(yaw.get_yaw_deg())   # -180 … +180
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class YawProvider(ABC):
    """Abstract interface: return current yaw in degrees (-180 … +180)."""

    @abstractmethod
    def get_yaw_deg(self) -> float:
        ...

    def close(self) -> None:
        """Release any resources. No-op by default."""


# ── Dummy ─────────────────────────────────────────────────────

class DummyYawProvider(YawProvider):
    """Returns a constant or linearly ramping yaw. Useful for CI / unit tests."""

    def __init__(self, constant_deg: float = 0.0):
        self._value = constant_deg

    def get_yaw_deg(self) -> float:
        return self._value


# ── Arduino / BNO055 ─────────────────────────────────────────

class ArduinoYawProvider(YawProvider):
    """
    Queries the Arduino 'y' command over serial.

    Caches the last reading and only re-queries after `poll_interval_s`
    to avoid flooding the serial port.
    """

    def __init__(self, controller, poll_interval_s: float = 0.1):
        self._ctrl = controller
        self._interval = poll_interval_s
        self._last_yaw: float = 0.0
        self._last_poll: float = 0.0

    def get_yaw_deg(self) -> float:
        now = time.monotonic()
        if now - self._last_poll >= self._interval:
            result = self._ctrl.yaw()
            if result is not None:
                self._last_yaw = float(result)
            self._last_poll = now
        return self._last_yaw


# ── Factory ───────────────────────────────────────────────────

def build_yaw_provider(source: str, **kwargs) -> YawProvider:
    """
    Convenience factory.

    source: "bno" | "zed" | "dummy"  ("arduino" is accepted as alias for "bno")
    kwargs: passed to the provider constructor.

    Falls back to DummyYawProvider if the requested source is unavailable.
    """
    if source == "dummy":
        return DummyYawProvider(**kwargs)

    if source in ("bno", "arduino"):
        ctrl = kwargs.get("controller")
        if ctrl is None:
            raise ValueError("build_yaw_provider('bno') requires controller=<Controller>")
        return ArduinoYawProvider(ctrl, poll_interval_s=kwargs.get("poll_interval_s", 0.1))

    if source == "zed":
        try:
            from zed_yaw_provider import ZedYawProvider
            return ZedYawProvider(**kwargs)
        except ImportError as exc:
            print(f"[yaw] ZED SDK unavailable ({exc}), falling back to DummyYawProvider")
            return DummyYawProvider()

    raise ValueError(f"Unknown yaw source: {source!r}. Choose 'bno', 'zed', or 'dummy'.")
