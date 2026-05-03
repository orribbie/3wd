"""
High-level command interface for the 3WD base.

Wraps SerialLink with named methods that match the Arduino command set.
Also handles waiting for motion completion and optional yaw override.

Usage:
    from serial_link import SerialLink
    from controller import Controller

    link = SerialLink("/dev/ttyACM0")
    link.connect()
    ctrl = Controller(link)

    ctrl.zero()
    ctrl.goto(1.0, 0.0, 0.0)
    ctrl.wait_done(timeout=30)
    ctrl.stop()
    link.disconnect()
"""

import time

from serial_link import SerialLink
from config import GOTO_DONE_TIMEOUT_S


class Controller:
    def __init__(self, link: SerialLink):
        self._link = link

    # ── Motion commands ───────────────────────────────────────

    def goto(self, x: float, y: float, theta: float) -> str:
        """
        Send goto command. x, y in grid units (1 unit = 100 mm by default).
        theta is goal heading in degrees.
        Returns the Arduino acknowledgement line.
        """
        self._link.send(f"g {x} {y} {theta}")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    def stop(self) -> str:
        """Halt all motors immediately."""
        self._link.send("stop")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    def zero(self) -> str:
        """Zero pose and yaw reference on the Arduino."""
        self._link.send("z")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    # ── Status / query ────────────────────────────────────────

    def status(self) -> str:
        """Request a status line from the Arduino."""
        self._link.send("s")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    def yaw(self) -> int | None:
        """Query the current BNO08x yaw (degrees) from the Arduino."""
        self._link.send("y")
        resp = self._link.get_response(timeout=2.0)
        # Expected: "[yaw] 45"
        if resp and "[yaw]" in resp:
            try:
                return int(resp.split("[yaw]")[1].strip())
            except (IndexError, ValueError):
                pass
        return None

    # ── Telemetry ─────────────────────────────────────────────

    def log_start(self, label: str = "data") -> str:
        """Ask the Arduino to start emitting TEL rows with the given label."""
        self._link.send(f"log 1 {label}")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    def log_stop(self) -> str:
        """Stop the Arduino telemetry stream."""
        self._link.send("log 0")
        resp = self._link.get_response(timeout=2.0)
        return resp or ""

    # ── Motion completion ─────────────────────────────────────

    def wait_done(self, timeout: float = GOTO_DONE_TIMEOUT_S) -> bool:
        """
        Block until the Arduino sends '[done]' or timeout expires.
        Returns True if '[done]' was received, False on timeout.

        Intermediate responses (status lines, TEL rows) are silently
        consumed; they remain available via link.get_telemetry().
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            resp = self._link.get_response(timeout=min(remaining, 0.5))
            if resp is None:
                continue
            if "[done]" in resp:
                return True
            # '[stopped]' or '[reset]' also ends the motion
            if "[stopped]" in resp or "[reset]" in resp:
                return False
        return False

    # ── Raw access ────────────────────────────────────────────

    def send_raw(self, cmd: str) -> None:
        """Send an arbitrary command string (for debugging / extension)."""
        self._link.send(cmd)
