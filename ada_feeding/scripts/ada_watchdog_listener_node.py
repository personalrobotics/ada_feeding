#!/usr/bin/env python3
"""
This module contains a node, ADAWatchdogListener, which listens to the
watchdog topic and kills itself if the watchdog fails. This is useful
because launchfiles allow us to perform an action when a node exits,
so we can use a node like this to terminate the entire launchfile when
the watchdog fails.
"""

# Standard imports
import threading
from typing import List

# Third-party imports
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor

# Local imports
from ada_feeding import ADAWatchdogListener

def main(args: List = None) -> None:
    """
    Create the ROS2 node to listen to the watchdog.
    """
    rclpy.init(args=args)

    # Create a node
    node = rclpy.create_node("ada_watchdog_listener")

    # Read the parameter for the rate at which to check the watchdog
    watchdog_check_hz = node.declare_parameter(
        "watchdog_check_hz",
        60.0,  # default value
        ParameterDescriptor(
            name="watchdog_check_hz",
            type=ParameterType.PARAMETER_DOUBLE,
            description=(
                "The rate (Hz) at which to check the whether the watchdog has failed."
            ),
            read_only=True,
        ),
    )
    rate = node.create_rate(watchdog_check_hz.value)

    # Create the watchdog listener
    ada_watchdog_listener = ADAWatchdogListener(node)

    # Use a MultiThreadedExecutor
    executor = MultiThreadedExecutor()

    # Spin in the background so that messages and rates get processed
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # As long as ROS2 is running, keep checking the watchdog
    while rclpy.ok():
        node.get_logger().info("Checking watchdog...")

        # Check if the watchdog has failed
        if not ada_watchdog_listener.ok():
            # If so, kill the node
            node.get_logger().error("Watchdog failed. Killing node.")
            node.destroy_node()
            rclpy.shutdown()
            # Join the spin thread
            spin_thread.join()
            return

        # Sleep for the specified amount of time
        rate.sleep()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()
    # Join the spin thread
    spin_thread.join()


if __name__ == "__main__":
    main()