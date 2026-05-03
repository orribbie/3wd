"""
Jetson interactive CLI for the SIR 3WD base.

Run:
    conda activate 3wd
    python jetson/main.py
    python jetson/main.py --port /dev/ttyACM1
    python jetson/main.py --port /dev/ttyACM0 --log-tel  # auto-log all TEL to CSV

Commands (same as Arduino serial interface):
    g x y theta      — goto position (grid units, degrees)
    stop             — halt motors
    z                — zero pose + yaw
    s                — status
    y                — query yaw
    log 1 <label>    — start Arduino telemetry stream
    log 0            — stop  Arduino telemetry stream
    help             — show this list
    quit / exit      — disconnect and exit
"""

import argparse
import os
import sys
import time

# ── Optional rich terminal ────────────────────────────────────
try:
    from rich.console import Console
    from rich.prompt import Prompt
    console = Console()
    def _print(msg: str, style: str = "") -> None:
        console.print(msg, style=style or None)
    def _input(prompt: str) -> str:
        return Prompt.ask(prompt)
except ImportError:
    def _print(msg: str, style: str = "") -> None:  # type: ignore[misc]
        print(msg)
    def _input(prompt: str) -> str:  # type: ignore[misc]
        return input(prompt)

# ── readline history (Linux/macOS) ────────────────────────────
try:
    import readline
    _HIST_FILE = os.path.expanduser("~/.sir_history")
    try:
        readline.read_history_file(_HIST_FILE)
    except FileNotFoundError:
        pass
    import atexit
    atexit.register(readline.write_history_file, _HIST_FILE)
except ImportError:
    pass

# ── Local imports ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import SERIAL_PORT, BAUD_RATE, GOTO_DONE_TIMEOUT_S
from serial_link import SerialLink
from controller import Controller
from telemetry import TelemetryLogger


HELP_TEXT = """
Commands
────────────────────────────────────
  g <x> <y> <theta>   goto position
  stop                halt motors
  z                   zero pose + yaw
  s                   status
  y                   query yaw
  log 1 <label>       start telemetry
  log 0               stop telemetry
  help                this help text
  quit / exit         disconnect & exit
────────────────────────────────────
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SIR 3WD Jetson CLI")
    p.add_argument("--port",    default=SERIAL_PORT, help="Serial port (default: %(default)s)")
    p.add_argument("--baud",    default=BAUD_RATE,   type=int, help="Baud rate")
    p.add_argument("--log-tel", action="store_true",
                   help="Automatically write all incoming TEL packets to a CSV file")
    p.add_argument("--yaw-src", choices=["bno", "zed", "dummy", "arduino"], default=None,
                   help="Yaw source: 'bno' (Arduino BNO055) or 'zed' (ZED 2i IMU)")
    return p.parse_args()


def _flush_responses(link: SerialLink) -> None:
    """Print any buffered Arduino responses that arrived after a send."""
    time.sleep(0.15)
    for line in link.drain_responses():
        _print(f"  {line}", style="cyan")


def main() -> None:
    args = _parse_args()

    _print(f"\n[bold]SIR 3WD CLI[/bold] — connecting to {args.port} @ {args.baud} baud...",
           style="bold green")

    link = SerialLink(args.port, args.baud)
    try:
        link.connect()
    except Exception as exc:
        _print(f"[error] Could not open {args.port}: {exc}", style="bold red")
        _print("  Tip: check permissions → sudo usermod -aG dialout $USER  (re-login after)", style="yellow")
        sys.exit(1)

    _print("Connected. Type 'help' for commands.\n", style="green")

    # Drain the Arduino boot message
    time.sleep(1.5)
    for line in link.drain_responses():
        _print(f"  {line}", style="dim")

    ctrl = Controller(link)

    # Optional: start CSV telemetry logger
    tel_logger: TelemetryLogger | None = None
    if args.log_tel:
        tel_logger = TelemetryLogger(link, auto_open=True)
        _print(f"  Telemetry CSV → {tel_logger.filename}", style="cyan")

    # Optional: yaw provider
    yaw_provider = None
    if args.yaw_src:
        from yaw_provider import build_yaw_provider
        yaw_provider = build_yaw_provider(args.yaw_src, controller=ctrl)
        _print(f"  Yaw source: {args.yaw_src}", style="cyan")

    # ── Command loop ──────────────────────────────────────────
    try:
        while True:
            # Print any async responses that arrived between prompts
            for line in link.drain_responses():
                _print(f"  << {line}", style="dim cyan")

            try:
                raw = _input("[bold yellow]sir>[/bold yellow] ")
            except (EOFError, KeyboardInterrupt):
                _print("\nExiting...", style="bold")
                break

            cmd = raw.strip()
            if not cmd:
                continue

            # ── Local commands ────────────────────────────────
            if cmd in ("quit", "exit"):
                break

            if cmd == "help":
                _print(HELP_TEXT)
                continue

            # ── Pass-through to Arduino ───────────────────────
            link.send(cmd)
            _flush_responses(link)


    finally:
        if tel_logger:
            rows = tel_logger.close()
            _print(f"\n  Telemetry CSV closed ({rows} rows) → {tel_logger.filename}", style="cyan")
        if yaw_provider:
            yaw_provider.close()
        link.disconnect()
        _print("Disconnected.", style="bold")




if __name__ == "__main__":
    main()
