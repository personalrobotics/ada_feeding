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
import rclpy

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

def getKey(settings):
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
    'w': ( 1.0,  0.0,  0.0), # forward
    's': (-1.0,  0.0,  0.0), # backwards
    'a': ( 0.0,  1.0,  0.0), # left
    'd': ( 0.0, -1.0,  0.0), # right
    'q': ( 0.0,  0.0,  1.0), # up
    'e': ( 0.0,  0.0, -1.0), # down
}

def main(args=None):
    """
    Launch the ROS node and spin.
    """
    settings = termios.tcgetattr(sys.stdin)

    # Initialize the ROS context
    rclpy.init(args=args)
    node = rclpy.create_node('ada_keyboard_teleop')
    pub = node.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 1)

    # Create the message
    twist_msg = TwistStamped()
    twist_msg.header.frame_id = "j2n6s200_link_base"
    
    prev_key = ''

    try:
        node.get_logger().info(msg)
        while(1):
            rclpy.spin_once(node, timeout_sec=0)
            key = getKey(settings)

            if key in cartesian_control_bindings.keys():
                # Due to keyboard delay before repeat, when the user holds down a
                # key we will read it as the key, followed by some number of empty
                # readings, followed by the key consecutively. To account for this,
                # we require two consecutive readings of the same key before
                # publishing the velcoity commands.
                if prev_key == key:
                    x, y, z = cartesian_control_bindings[key]
                    twist_msg.twist.linear.x = x
                    twist_msg.twist.linear.y = y
                    twist_msg.twist.linear.z = z
            else:
                twist_msg.twist.linear.x = 0.0
                twist_msg.twist.linear.y = 0.0
                twist_msg.twist.linear.z = 0.0

                # Ctrl+C Interrupt
                if (key == '\x03'):
                    break

            # Publish the message
            twist_msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(twist_msg)

            prev_key = key
    except Exception as e:
        print(repr(e))

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__=="__main__":
    main()