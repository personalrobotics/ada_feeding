#!/usr/bin/env python3
# Adapted from https://github.com/turtlebot/turtlebot/blob/melodic/turtlebot_teleop/scripts/turtlebot_teleop_key
"""
"""

# Standard imports
import termios
import time
import tty
import select
import sys

# Third-party imports
from geometry_msgs.msg import TwistStamped

# Features to consider adding: cartesian angular, joint control, modifying linear/angular speed.
msg = """
Control the ADA arm!
---------------------------
Cartesian control:
  w/s: forward/backwards
  a/d: left/right
  q/e: up/down

Joint control (UNIMPLEMENTED):
  1-6: joint 1-6
  r: reverse the direction of joint movement

CTRL-C to quit
"""

def getKey():
    """
    Read a key from stdin without writing it to terminal.
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

cartesian_control_bindings = {
    'w': (1, 0, 0), # forward
    's': (-1, 0, 0), # backwards
    'a': (0, 1, 0), # left
    'd': (0, -1, 0), # right
    'q': (0, 0, 1), # up
    'e': (0, 0, -1), # down
}

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)
    try:
        while(1):
            key = getKey()
            print(repr(key))

            # Ctrl+C interrupt
            if (key == '\x03'):
                break
    except Exception as e:
        print(repr(e))

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)