#!/usr/bin/env python3
# Adapted from:
# https://github.com/turtlebot/turtlebot/blob/melodic/turtlebot_teleop/scripts/turtlebot_teleop_key
"""
This module contains a ROS2 node to allow the user to teleoperate the ADA arm
using the keyboard. Specifically, this node allows users to send linear cartesian
velocities in the base frame, angular cartesian velocities in the end-effector
frame, or joint velocities to the robot via MoveIt Servo.
"""

# Standard imports
import termios
import tty
import select
import sys

# Third-party imports
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.time import Time
from tf2_geometry_msgs import Vector3Stamped  # pylint: disable=unused-import
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

INSTRUCTION_MSG = """
Control the ADA arm!
---------------------------
Cartesian control (linear):
  w/s: forward/backwards
  a/d: left/right
  q/e: up/down

Cartesian control (angular):
  i/k: +pitch/-pitch
  j/l: +yaw/-yaw
  u/o: +roll/-roll

Joint control:
  1-6: joint 1-6
  r: reverse the direction of joint movement

CTRL-C to quit
"""
BASE_FRAME = "j2n6s200_link_base"
EE_FRAME = "forkTip"


def get_key(settings):
    """
    Read a key from stdin without writing it to terminal.
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


cartesian_control_linear_bindings = {
    "w": (0.0, -1.0, 0.0),  # forward
    "s": (0.0, 1.0, 0.0),  # backwards
    "a": (1.0, 0.0, 0.0),  # left
    "d": (-1.0, 0.0, 0.0),  # right
    "q": (0.0, 0.0, 1.0),  # up
    "e": (0.0, 0.0, -1.0),  # down
}
cartesian_control_angular_bindings = {
    "i": (1.0, 0.0, 0.0),  # +pitch
    "k": (-1.0, 0.0, 0.0),  # -pitch
    "j": (0.0, 1.0, 0.0),  # +yaw
    "l": (0.0, -1.0, 0.0),  # -yaw
    "u": (0.0, 0.0, 1.0),  # +roll
    "o": (0.0, 0.0, -1.0),  # -roll
}
joint_control_bindings = {
    "1": "j2n6s200_joint_1",
    "2": "j2n6s200_joint_2",
    "3": "j2n6s200_joint_3",
    "4": "j2n6s200_joint_4",
    "5": "j2n6s200_joint_5",
    "6": "j2n6s200_joint_6",
}
reverse_joint_direction_key = "r"  # pylint: disable=invalid-name


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    # pylint: disable=too-many-nested-blocks
    # This function does the entire work of teleoperation, so it is expected
    # to be somewhat complex.

    settings = termios.tcgetattr(sys.stdin)

    # Initialize the ROS context
    rclpy.init(args=args)
    node = rclpy.create_node("ada_keyboard_teleop")
    twist_pub = node.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 1)
    joint_pub = node.create_publisher(JointJog, "/servo_node/delta_joint_cmds", 1)

    # Initialize the tf2 buffer and listener
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)  # pylint: disable=unused-variable

    # Create the cartesian control messages
    # The linear velocity is always in the base frame
    linear_msg = Vector3Stamped()
    linear_msg.header.stamp = Time().to_msg()  # use latest time
    linear_msg.header.frame_id = BASE_FRAME
    # The angular velocity is always in the end effector frame
    angular_msg = Vector3Stamped()
    angular_msg.header.stamp = Time().to_msg()  # use latest time
    angular_msg.header.frame_id = EE_FRAME
    # The final message should be either in the base or end effector frame.
    # It should match the `robot_link_command_frame`` servo param.
    twist_msg = TwistStamped()
    twist_msg.header.frame_id = BASE_FRAME

    # Create the joint control message
    joint_msg = JointJog()
    joint_msg.header.frame_id = BASE_FRAME
    joint_velocity_command = 1.0  # rad/s

    prev_key = ""

    try:
        node.get_logger().info(INSTRUCTION_MSG)
        while 1:
            rclpy.spin_once(node, timeout_sec=0)

            publish_joint_msg = False

            key = get_key(settings)
            if key in cartesian_control_linear_bindings:
                # Due to keyboard delay before repeat, when the user holds down a
                # key we will read it as the key, followed by some number of empty
                # readings, followed by the key consecutively. To account for this,
                # we require two consecutive readings of the same key before
                # publishing the velcoity commands.
                if prev_key == key:
                    x, y, z = cartesian_control_linear_bindings[key]
                    linear_msg.vector.x = x
                    linear_msg.vector.y = y
                    linear_msg.vector.z = z

                    # Transform the linear message to the overall twist message frame
                    twist_msg.twist.linear = linear_msg.vector
                    if linear_msg.header.frame_id != twist_msg.header.frame_id:
                        try:
                            linear_transformed = tf_buffer.transform(
                                linear_msg, twist_msg.header.frame_id
                            )
                            twist_msg.twist.linear = linear_transformed.vector
                        except tf2.ExtrapolationException as exc:
                            node.get_logger().warning(
                                f"Transform from {linear_msg.header.frame_id} to "
                                f"{twist_msg.header.frame_id} failed: {type(exc)}: {exc}\n"
                                f"Interpreting the linear velocity in {twist_msg.header.frame_id} "
                                "without transforming."
                            )
            elif key in cartesian_control_angular_bindings:
                if prev_key == key:
                    x, y, z = cartesian_control_angular_bindings[key]
                    angular_msg.vector.x = x
                    angular_msg.vector.y = y
                    angular_msg.vector.z = z

                    # Transform the angular message to the overall twist message frame
                    twist_msg.twist.angular = angular_msg.vector
                    if angular_msg.header.frame_id != twist_msg.header.frame_id:
                        try:
                            angular_transformed = tf_buffer.transform(
                                angular_msg, twist_msg.header.frame_id
                            )
                            twist_msg.twist.angular = angular_transformed.vector
                        except tf2.ExtrapolationException as exc:
                            node.get_logger().warning(
                                f"Transform from {angular_msg.header.frame_id} to "
                                f"{twist_msg.header.frame_id} failed: {type(exc)}: {exc}\n"
                                f"Interpreting the angular velocity in {twist_msg.header.frame_id}"
                                " without transforming."
                            )
            elif key in joint_control_bindings:
                if prev_key == key:
                    joint_msg.joint_names = [joint_control_bindings[key]]
                    joint_msg.velocities = [joint_velocity_command]
                    publish_joint_msg = True
            elif key == reverse_joint_direction_key:
                joint_velocity_command *= -1.0
            else:
                twist_msg.twist.linear.x = 0.0
                twist_msg.twist.linear.y = 0.0
                twist_msg.twist.linear.z = 0.0
                twist_msg.twist.angular.x = 0.0
                twist_msg.twist.angular.y = 0.0
                twist_msg.twist.angular.z = 0.0

                # Ctrl+C Interrupt
                if key == "\x03":
                    break

            # Publish the message
            if publish_joint_msg:
                joint_msg.header.stamp = node.get_clock().now().to_msg()
                joint_pub.publish(joint_msg)
            else:
                twist_msg.header.stamp = node.get_clock().now().to_msg()
                twist_pub.publish(twist_msg)

            prev_key = key
    except Exception as exc:  # pylint: disable=broad-except
        print(repr(exc))

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    # Terminate this node
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
