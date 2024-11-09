#!/usr/bin/env python3
"""
This module contains a node, SenderCompressedImageNode, which subscribes to topics as specified
by its parameters and publishes it to a single topic.
"""
# pylint: disable=duplicate-code
# TODO: Create a generic way to merge sender, sender_compressed_image, and other variants
# of sender nodes.

# Standard imports
from functools import partial

# Third-party imports
from sensor_msgs.msg import CompressedImage as CompressedImageInput
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

# Local imports
from nano_bridge.msg import CompressedImage as CompressedImageOutput


class SenderCompressedImageNode(Node):
    """
    The SenderCompressedImageNode class subscribes to topics as specified by its parameters and
    publishes it to a single topic.
    """

    def __init__(self) -> None:
        """
        Initialize the sender node.
        """
        super().__init__("sender_compressed_image")

        # Load the parameters
        self.__topic_names: list[str] = []
        self.__load_parameters()

        # Create the publisher
        self.__pub = self.create_publisher(
            msg_type=CompressedImageOutput,
            topic="~/data",
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Subscribe to the topics
        self.__msg_recv_time: dict[str, Time] = {}
        self.__msg_count: dict[str, int] = {}
        self.__subs = {}
        for topic_name in self.__topic_names:
            self.__msg_count[topic_name] = 0
            self.__subs[topic_name] = self.create_subscription(
                msg_type=CompressedImageInput,
                topic=topic_name,
                callback=partial(
                    self.__callback,
                    topic_name=topic_name,
                ),
                callback_group=MutuallyExclusiveCallbackGroup(),
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

    def __load_parameters(self) -> None:
        """
        Load the parameters.
        """
        # Topic Names
        topic_names = self.declare_parameter(
            "topic_names",
            descriptor=ParameterDescriptor(
                name="topic_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=("List of topic names to subscribe to."),
                read_only=True,
            ),
        )
        self.__topic_names = topic_names.value

    def __callback(
        self, msg_in: CompressedImageInput, topic_name: str, debug: bool = False
    ) -> None:
        """
        Callback function for the subscriber.
        """
        if debug:
            start_time = self.get_clock().now()
            if topic_name not in self.__msg_recv_time:
                self.__msg_recv_time[topic_name] = start_time
            self.__msg_count[topic_name] += 1

        # Create the message
        msg_out = CompressedImageOutput(
            topic=topic_name,
            data=msg_in,
        )

        # Publish the message
        self.__pub.publish(msg_out)
        if debug:
            elapsed_sec = (
                self.get_clock().now() - self.__msg_recv_time[topic_name]
            ).nanoseconds / 1.0e9
            self.get_logger().info(
                f"Published message from {topic_name} in "
                f"{(self.get_clock().now() - start_time).nanoseconds / 1.0e9} seconds. "
                f"Total messages: {self.__msg_count[topic_name]}."
                f"Rate: {self.__msg_count[topic_name] / elapsed_sec} Hz."
            )


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    # Create the node
    sender = SenderCompressedImageNode()

    # Spin the node
    executor = MultiThreadedExecutor()
    rclpy.spin(sender, executor=executor)

    # Terminate this node
    sender.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
