#!/usr/bin/env python3
import sys
import os
import termios
import tty
import math
import time

# Add sir/jetson to path to use SerialLink
_SIR_JETSON = os.path.expanduser("~/sir/jetson")
if _SIR_JETSON not in sys.path:
    sys.path.insert(0, _SIR_JETSON)

from serial_link import SerialLink

# Teleop Settings
MOVE_STEP = 2.0   # grid units (20cm) per arrow press
ROT_STEP  = 15.0  # degrees per arrow press

def get_key():
    """Read a single keypress from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    print("Connecting to Arduino on /dev/ttyACM0...")
    link = SerialLink(port="/dev/ttyACM0", baud=115200)
    try:
        link.connect()
    except Exception as e:
        print(f"Error: Could not connect to serial port. {e}")
        return

    print("\n" + "="*40)
    print("  SIR 3WD TELEOP READY")
    print("="*40)
    print("  UP        : Move Forward")
    print("  DOWN      : Move Backward")
    print("  LEFT      : Rotate CCW (Anticlockwise)")
    print("  RIGHT     : Rotate CW (Clockwise)")
    print("  SPACE     : Emergency Stop")
    print("  Q         : Quit")
    print("="*40)

    # Initialise state relative to startup origin
    # All commands are absolute from the 'z' reset point.
    link.send("z")
    curr_x, curr_y, curr_th = 0.0, 0.0, 0.0
    
    try:
        while True:
            key = get_key()
            
            if key.lower() == 'q':
                break
            elif key == ' ':
                print("\n[STOP]")
                link.send("stop")
                continue
            
            # Escape sequences for arrows
            if key == '\x1b[A': # UP
                rad = math.radians(curr_th)
                curr_x += MOVE_STEP * math.cos(rad)
                curr_y += MOVE_STEP * math.sin(rad)
                label = "FORWARD"
            elif key == '\x1b[B': # DOWN
                rad = math.radians(curr_th)
                curr_x -= MOVE_STEP * math.cos(rad)
                curr_y -= MOVE_STEP * math.sin(rad)
                label = "BACKWARD"
            elif key == '\x1b[C': # RIGHT
                curr_th -= ROT_STEP
                label = "ROT CW"
            elif key == '\x1b[D': # LEFT
                curr_th += ROT_STEP
                label = "ROT CCW"
            else:
                continue

            # Send the absolute goal to Arduino
            # The Arduino will rotate to face the point if distance > 1 unit,
            # then drive, then rotate to the final curr_th.
            cmd = f"g {curr_x:.2f} {curr_y:.2f} {curr_th:.1f}"
            print(f"\r{label:<10} | {cmd:<25}", end="")
            link.send(cmd)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("\nDisconnecting...")
        link.send("stop")
        link.disconnect()

if __name__ == "__main__":
    main()
