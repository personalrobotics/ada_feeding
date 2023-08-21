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
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding import ADAWatchdogListener


class ADAWatchdogListenerNode(Node):
    """
    A ROS2 node that listens to the watchdog topic and kills itself
    if the watchdog fails.
    """

    def __init__(self) -> None:
        """
        Initialize the node.
        """
        super().__init__("ada_watchdog_listener")

        # Read the parameter for the rate at which to check the watchdog
        watchdog_check_hz = self.declare_parameter(
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

        # Create a service to toggle this node on-and-off
        self.toggle_service = self.create_service(
            SetBool,
            "~/toggle_watchdog_listener",
            self.toggle_callback,
        )
        self.is_on = False
        self.is_on_lock = threading.Lock()

        # Create the watchdog listener
        self.ada_watchdog_listener = ADAWatchdogListener(self)

        # Check the watchdog at the specified rate
        timer_period = 1.0 / watchdog_check_hz.value  # seconds
        self.timer = self.create_timer(timer_period, self.check_watchdog)

    def toggle_callback(self, request: SetBool.Request, response: SetBool.Response):
        """
        Callback for the toggle service.
        """
        response.success = False
        response.message = f"Failed to set is_on to {request.data}"
        with self.is_on_lock:
            self.is_on = request.data
            response.success = True
            response.message = f"Successfully set is_on to {request.data}"
        return response

    def check_watchdog(self):
        """
        If the watchdog listener is on and the watchdog fails, kill this node.
        """
        # Check if the node is on
        with self.is_on_lock:
            is_on = self.is_on

        # If it's on, check if the watchdog has failed
        if is_on and not self.ada_watchdog_listener.ok():
            # If the watchdog has failed, kill this node
            self.get_logger().error("Watchdog failed. Killing node.")
            self.destroy_node()
            rclpy.shutdown()


def main(args: List = None) -> None:
    """
    Create the ROS2 node to listen to the watchdog.
    """
    rclpy.init(args=args)

    ada_watchdog_listener_node = ADAWatchdogListenerNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(ada_watchdog_listener_node, executor=executor)


if __name__ == "__main__":
    main()
