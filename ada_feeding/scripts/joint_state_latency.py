#!/usr/bin/env python3
"""
This module defines a barebones ROS2 node that subscribes to the /joint_states
topic and logs the latency between when the message was published and when it was
received.
"""

# Standard imports

# Third-party imports
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState

## TODO: Document when this would be useful

class JointStateLatency(Node):
    """
    The JointStateLatency class defines a barebones ROS2 node that subscribes to
    the /joint_states topic and logs the latency between when the message was
    published and when it was received.
    """

    def __init__(self) -> None:
        """
        Initialize the JointStateLatency node.
        """
        super().__init__("joint_state_latency")
        self._logger = self.get_logger()

        # Initialize the subscriber
        self._sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._callback,
            1,
            callback_group=self._default_callback_group,
        )

        self.latencies_ns = []

    def _callback(self, msg: JointState) -> None:
        """
        Log the latency between when the message was published and when it was
        received.

        Parameters
        ----------
        msg: the received message
        """
        now = self.get_clock().now()
        latency_ns = (now - Time.from_msg(msg.header.stamp)).nanoseconds
        self.latencies_ns.append(latency_ns)
        self._logger.info(f"Latency: {latency_ns / 10**9.0} sec")
        self._logger.info(f"Mean latency: {np.mean(self.latencies_ns) / 10**9.0} sec")


def main(args=None) -> None:
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)
    node = JointStateLatency()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
