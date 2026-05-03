"""
Low-level serial link between Jetson and Arduino.

A background reader thread accumulates incoming bytes and routes them to:
  - response_queue : normal text lines (non-TEL Arduino output)
  - telemetry_queue: parsed telemetry dicts (lines starting with "TEL,")

Usage:
    link = SerialLink("/dev/ttyACM0", 115200)
    link.connect()
    link.send("s")
    resp = link.get_response(timeout=2.0)
    link.disconnect()
"""

import queue
import threading
import time

import serial

from config import SERIAL_PORT, BAUD_RATE, READ_TIMEOUT, TEL_PREFIX, TEL_COLUMNS, TEL_NUMERIC


class SerialLink:
    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._response_q: queue.Queue[str] = queue.Queue()
        self._telemetry_q: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    # ── Connection ────────────────────────────────────────────

    def connect(self, timeout: float = 5.0) -> None:
        """Open serial port and start the reader thread."""
        self._ser = serial.Serial(self.port, self.baud, timeout=READ_TIMEOUT)
        # Flush stale data from Arduino reset
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._ser.in_waiting:
                self._ser.read(self._ser.in_waiting)
            else:
                break
            time.sleep(0.05)
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        """Stop the reader thread and close the serial port."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser and self._ser.is_open:
            self._ser.close()

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open and self._running

    # ── Sending ───────────────────────────────────────────────

    def send(self, cmd: str) -> None:
        """Send a command string followed by newline."""
        if not self.is_connected:
            raise RuntimeError("SerialLink not connected")
        line = cmd.strip() + "\n"
        self._ser.write(line.encode())

    # ── Receiving ─────────────────────────────────────────────

    def get_response(self, timeout: float = 2.0) -> str | None:
        """Block until a text response arrives or timeout expires."""
        try:
            return self._response_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_telemetry(self, block: bool = False, timeout: float = 0.1) -> dict | None:
        """Return next telemetry packet or None."""
        try:
            return self._telemetry_q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def drain_responses(self) -> list[str]:
        """Return all queued text responses without blocking."""
        lines = []
        while True:
            try:
                lines.append(self._response_q.get_nowait())
            except queue.Empty:
                break
        return lines

    # ── Background reader ─────────────────────────────────────

    def _reader(self) -> None:
        """Reads bytes from serial, splits into text lines and TEL rows."""
        buf = b""
        while self._running:
            try:
                chunk = self._ser.read(256)
            except serial.SerialException as exc:
                self._response_q.put(f"[serial error] {exc}")
                self._running = False
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode(errors="replace").strip()
                if not line:
                    continue
                if line.startswith(TEL_PREFIX):
                    pkt = _parse_telemetry(line)
                    if pkt:
                        self._telemetry_q.put(pkt)
                else:
                    self._response_q.put(line)


# ── Telemetry line parser ──────────────────────────────────────

def _parse_telemetry(line: str) -> dict | None:
    """Parse a 'TEL,...' CSV line into a dict. Returns None on parse error."""
    parts = line.split(",")
    # First token is "TEL", remaining are values
    values = parts[1:]
    if len(values) != len(TEL_COLUMNS):
        return None
    pkt = {}
    for col, raw in zip(TEL_COLUMNS, values):
        cast = TEL_NUMERIC.get(col, str)
        try:
            pkt[col] = cast(raw.strip())
        except (ValueError, TypeError):
            pkt[col] = None
    return pkt
