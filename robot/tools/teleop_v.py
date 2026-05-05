#!/usr/bin/env python3
"""
teleop_v.py — Velocity-mode keyboard teleop for SIR 3WD (v fwd turn command).

  UP    : Forward
  LEFT  : Rotate CCW (anticlockwise)
  RIGHT : Rotate CW  (clockwise)
  SPACE : Stop
  +/-   : Increase / decrease speed
  Q     : Quit

Sends 'v fwd turn' at 10 Hz while a key is held.
Auto-stops if no key held (sends v 0 0).

Run: conda activate slam && python tools/teleop_v.py
"""

import os, sys, time, termios, tty, select

_SIR_JETSON = os.path.expanduser("~/sir/jetson")
if _SIR_JETSON not in sys.path:
    sys.path.insert(0, _SIR_JETSON)

from serial_link import SerialLink

# ── Config ────────────────────────────────────────────────────────────────────
PORT      = "/dev/ttyACM0"
BAUD      = 115200
CTRL_HZ   = 20          # command rate — 20 Hz feels snappy, < 500ms watchdog
FWD_PWM   = 60          # default forward PWM (0-100)
TURN_PWM  = 45          # default turn PWM   (0-100)
PWM_STEP  = 5           # +/- increment
PWM_MIN   = 20
PWM_MAX   = 100

# ── Key codes ─────────────────────────────────────────────────────────────────
KEY_UP    = "\x1b[A"
KEY_DOWN  = "\x1b[B"
KEY_RIGHT = "\x1b[C"
KEY_LEFT  = "\x1b[D"
KEY_SPACE = " "
KEY_QUIT  = "q"
KEY_PLUS  = "+"
KEY_MINUS = "-"

# ── Non-blocking key reader ───────────────────────────────────────────────────

def _getch_nowait():
    """Return a key string if one is available, else None. Non-blocking."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # Read escape sequence (up to 2 more bytes)
        rest = ""
        for _ in range(2):
            if select.select([sys.stdin], [], [], 0.02)[0]:
                rest += sys.stdin.read(1)
        return ch + rest
    return ch


def main():
    fwd_pwm  = FWD_PWM
    turn_pwm = TURN_PWM

    link = SerialLink(port=PORT, baud=BAUD)
    print(f"Connecting to {PORT} …")
    try:
        link.connect()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"""
{'='*48}
  SIR 3WD Velocity Teleop  (v command mode)
{'='*48}
  UP     : Forward          fwd={fwd_pwm}
  LEFT   : CCW rotation     turn={turn_pwm}
  RIGHT  : CW  rotation
  SPACE  : Stop
  +/-    : Speed up/down
  Q      : Quit
{'='*48}
Hold a key to keep moving. Release to stop.
""")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    dt = 1.0 / CTRL_HZ

    try:
        tty.setraw(fd)
        cur_fwd  = 0
        cur_turn = 0
        last_key_time = 0.0

        while True:
            t0 = time.time()

            key = _getch_nowait()

            if key is not None:
                last_key_time = t0

                if key.lower() == KEY_QUIT:
                    break
                elif key == KEY_SPACE:
                    cur_fwd = cur_turn = 0
                elif key == KEY_UP:
                    cur_fwd  =  fwd_pwm
                    cur_turn =  0
                elif key == KEY_LEFT:
                    cur_fwd  =  0
                    cur_turn =  turn_pwm
                elif key == KEY_RIGHT:
                    cur_fwd  =  0
                    cur_turn = -turn_pwm
                elif key == KEY_PLUS:
                    fwd_pwm  = min(PWM_MAX, fwd_pwm  + PWM_STEP)
                    turn_pwm = min(PWM_MAX, turn_pwm + PWM_STEP)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write(f"\r  Speed: fwd={fwd_pwm}  turn={turn_pwm}   \n")
                    sys.stdout.flush()
                    tty.setraw(fd)
                elif key == KEY_MINUS:
                    fwd_pwm  = max(PWM_MIN, fwd_pwm  - PWM_STEP)
                    turn_pwm = max(PWM_MIN, turn_pwm - PWM_STEP)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write(f"\r  Speed: fwd={fwd_pwm}  turn={turn_pwm}   \n")
                    sys.stdout.flush()
                    tty.setraw(fd)

            # Auto-stop if no key pressed recently (> 1 control period)
            if t0 - last_key_time > dt * 1.5:
                cur_fwd = cur_turn = 0

            link.send(f"v {cur_fwd} {cur_turn}")


            elapsed = time.time() - t0
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        link.send("v 0 0")
        time.sleep(0.1)
        link.send("stop")
        link.disconnect()
        print("\nStopped.")


if __name__ == "__main__":
    main()
