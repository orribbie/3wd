"""
Telemetry consumer: reads parsed TEL dicts from SerialLink and optionally
writes them to a timestamped CSV file.

Usage:
    from serial_link import SerialLink
    from telemetry import TelemetryLogger

    link = SerialLink(...)
    link.connect()

    with TelemetryLogger(link, auto_open=True) as logger:
        # run your motion
        pass  # logger stops on __exit__
"""

import csv
import os
import queue
import threading
import time
from dataclasses import dataclass, fields, asdict
from typing import Callable

from config import LOG_DIR, TEL_COLUMNS


# ── Telemetry record ──────────────────────────────────────────

@dataclass
class TelemetryRecord:
    t_ms:         int   = 0
    label:        str   = ""
    enc1:         int   = 0
    enc2:         int   = 0
    enc3:         int   = 0
    vx_enc_mps:   float = 0.0
    ax_raw_mps2:  float = 0.0
    ax_f_mps2:    float = 0.0
    vx_imu_mps:   float = 0.0
    alpha_x_hat:  float = 1.0
    alpha_x_use:  float = 1.0
    i_sum_mA:     float = 0.0
    i_delta_mA:   float = 0.0
    adxl_ax:      int   = 0
    adxl_ay:      int   = 0
    adxl_az:      int   = 0
    # Appended on the Jetson side
    wall_time_s:  float = 0.0

    @staticmethod
    def from_dict(d: dict) -> "TelemetryRecord":
        rec = TelemetryRecord()
        for f in fields(rec):
            if f.name in d and d[f.name] is not None:
                setattr(rec, f.name, d[f.name])
        rec.wall_time_s = time.time()
        return rec

    def csv_header(self) -> list[str]:
        return [f.name for f in fields(self)]

    def csv_row(self) -> list:
        return list(asdict(self).values())


# ── Logger ────────────────────────────────────────────────────

class TelemetryLogger:
    """
    Reads telemetry dicts from a SerialLink in a background thread and
    writes them to a CSV file.

    Parameters
    ----------
    link       : SerialLink instance (already connected)
    filename   : explicit CSV path; if None, auto-generates under LOG_DIR
    auto_open  : open the CSV file immediately on construction
    callback   : optional callable(TelemetryRecord) called for each packet
    """

    def __init__(
        self,
        link,
        filename: str | None = None,
        auto_open: bool = False,
        callback: Callable[[TelemetryRecord], None] | None = None,
    ):
        self._link = link
        self._filename = filename
        self._callback = callback
        self._file = None
        self._writer = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._count = 0

        if auto_open:
            self.open()

    # ── File management ───────────────────────────────────────

    def open(self, filename: str | None = None) -> str:
        """Open (or create) the CSV file and start the consumer thread."""
        if self._running:
            return self._filename  # already open

        path = filename or self._filename or _auto_filename()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._filename = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._count = 0
        self._running = True
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        return path

    def close(self) -> int:
        """Stop the consumer thread, flush and close the CSV. Returns row count."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        return self._count

    @property
    def filename(self) -> str | None:
        return self._filename

    @property
    def count(self) -> int:
        return self._count

    # ── Context manager ───────────────────────────────────────

    def __enter__(self):
        if not self._running:
            self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Background consumer ───────────────────────────────────

    def _consume(self) -> None:
        header_written = False
        while self._running:
            pkt = self._link.get_telemetry(block=True, timeout=0.2)
            if pkt is None:
                continue
            rec = TelemetryRecord.from_dict(pkt)
            if self._writer:
                if not header_written:
                    self._writer.writerow(rec.csv_header())
                    header_written = True
                self._writer.writerow(rec.csv_row())
                self._count += 1
                if self._count % 50 == 0:
                    self._file.flush()
            if self._callback:
                try:
                    self._callback(rec)
                except Exception:
                    pass


# ── Helpers ───────────────────────────────────────────────────

def _auto_filename() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.abspath(LOG_DIR), f"tel_{stamp}.csv")
