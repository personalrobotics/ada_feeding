#!/usr/bin/env python3
"""
This module contains a node, SenderNode, which subscribes to topics as specified
by its parameters and publishes it to a single topic of type ByteMultiArray. We
abuse the ByteMultiArray message type for this purpose, by using
msg.layout.dim[0].label for the topic name, msg.layout.dim[1].label for the topic
type, and msg.data for the pkl-serialized message.
"""
# pylint: disable=duplicate-code
# TODO: Create a generic way to merge sender, sender_compressed_image, and other variants
# of sender nodes.

# Standard imports
from functools import partial
import pickle as pkl
from typing import Any

# Third-party imports
from std_msgs.msg import ByteMultiArray, MultiArrayDimension
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

# Local imports
from .helpers import import_from_string


class SenderNode(Node):
    """
    The SenderNode class subscribes to topics as specified by its parameters and
    publishes it to a single topic of type ByteMultiArray. We abuse the ByteMultiArray
    message type for this purpose, by using msg.layout.dim[0].label for the topic
    name, msg.layout.dim[1].label for the topic type, and msg.data for the pkl-serialized
    message.
    """

    def __init__(self) -> None:
        """
        Initialize the sender node.
        """
        super().__init__("sender")

        # Load the parameters
        self.__topics: list[tuple[str, str]] = []
        self.__load_parameters()

        # Import the message types
        self.__types = {
            topic_type: import_from_string(topic_type)
            for _, topic_type in self.__topics
        }

        # Create the publisher
        self.__pub = self.create_publisher(
            msg_type=ByteMultiArray,
            topic="~/data",
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Subscribe to the topics
        self.__msg_recv_time: dict[str, Time] = {}
        self.__msg_count: dict[str, int] = {}
        self.__subs = {}
        for topic_name, topic_type in self.__topics:
            self.__msg_count[topic_name] = 0
            self.__subs[topic_name] = self.create_subscription(
                msg_type=self.__types[topic_type],
                topic=topic_name,
                callback=partial(
                    self.__callback,
                    topic_name=topic_name,
                    topic_type=topic_type,
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
        topic_names_list = topic_names.value

        # Topic Types
        topic_types = self.declare_parameter(
            "topic_types",
            descriptor=ParameterDescriptor(
                name="topic_types",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=("List of topic types to subscribe to."),
                read_only=True,
            ),
        )
        topic_types_list = topic_types.value

        # Combine them
        n = min(len(topic_names_list), len(topic_types_list))
        self.__topics = list(zip(topic_names_list[:n], topic_types_list[:n]))

    def __callback(
        self, msg: Any, topic_name: str, topic_type: str, debug: bool = False
    ) -> None:
        """
        Callback function for the subscriber.
        """
        if debug:
            start_time = self.get_clock().now()
            if topic_name not in self.__msg_recv_time:
                self.__msg_recv_time[topic_name] = start_time
            self.__msg_count[topic_name] += 1
        # Serialize the message. Note: This takes ~0.2s on `nano`, which means
        # that the rate is limited to ~5Hz.
        data = pkl.dumps(msg)

        # Create the ByteMultiArray message
        msg = ByteMultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label=topic_name),
            MultiArrayDimension(label=topic_type),
        ]
        msg.data = [bytes([x]) for x in data]

        # Publish the message
        self.__pub.publish(msg)
        if debug:
            elapsed_sec = (
                self.get_clock().now() - self.__msg_recv_time[topic_name]
            ).nanoseconds / 1.0e9
            self.get_logger().info(
                f"Published message from {topic_name} ({topic_type}) in "
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
    sender = SenderNode()

    # Spin the node
    executor = MultiThreadedExecutor()
    rclpy.spin(sender, executor=executor)

    # Terminate this node
    sender.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
