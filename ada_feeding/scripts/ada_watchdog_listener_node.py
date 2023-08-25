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

        # Create a service to toggle this node on-and-off
        self.toggle_service = self.create_service(
            SetBool,
            "~/toggle_watchdog_listener",
            self.toggle_callback,
        )
        self.is_on = True
        self.is_on_lock = threading.Lock()

        # Create the watchdog listener
        self.ada_watchdog_listener = ADAWatchdogListener(
            self, callback_fn=self.watchdog_status_callback
        )

        self.get_logger().info("Initialized!")

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
            if request.data:
                self.get_logger().info("Succesfully turned the watchdog listener on.")
            else:
                self.get_logger().warn(
                    "WARNING: You have turned the watchdog listener off. "
                    "This can be dangerous, because typically the launchfile "
                    "that creates the controllers will terminate if this node "
                    "dies, thereby terminating all robot motion. With this node "
                    "off, there is no guard to stop robot motion if the watchdog "
                    "fails."
                )
        return response

    def watchdog_status_callback(self, new_status: bool) -> None:
        """
        Callback for when the watchdog's status changes. If this node `is_on`
        and the watchdog status changes to false, kill this node.

        Parameters
        ----------
        new_status: bool
            The new status of the watchdog.
        """
        # If the watchdog failed
        if not new_status:
            # If the node is on, kill the node.
            with self.is_on_lock:
                is_on = self.is_on
            if is_on:
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
