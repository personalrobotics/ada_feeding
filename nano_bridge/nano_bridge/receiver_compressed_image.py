#!/usr/bin/env python3
"""
This module contains a node, ReceiverCompressedImageNode, which subscribes to a
topic published by SenderCompressedImageNode and republishes the messages to the original topics,
prepended with a parameter-specified prefix.
"""
# pylint: disable=duplicate-code
# TODO: Create a generic way to merge receiver, receiver_compressed_image, and other variants
# of receiver nodes.

# Standard imports
import os

# Third-party imports
from sensor_msgs.msg import CompressedImage as CompressedImageOutput
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.publisher import Publisher

# Local imports
from nano_bridge.msg import CompressedImage as CompressedImageInput


class ReceiverCompressedImageNode(Node):
    """
    The ReceiverCompressedImageNode class subscribes to a CompressedImage topic published by
    SenderCompressedImageNode and republishes the messages to the original topics, prepended
    with a parameter-specified prefix.
    """

    def __init__(self) -> None:
        """
        Initialize the sender node.
        """
        super().__init__("receiver")

        # Load the parameters
        self.__prefix = ""
        self.__load_parameters()

        # Create the publishers
        self.__pubs: dict[str, Publisher] = {}

        # Create the subscriber
        # pylint: disable=unused-private-member
        self.__sub = self.create_subscription(
            msg_type=CompressedImageInput,
            topic="~/data",
            callback=self.__callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

    def __load_parameters(self) -> None:
        """
        Load the parameters.
        """
        # Prefix
        prefix = self.declare_parameter(
            "prefix",
            descriptor=ParameterDescriptor(
                name="prefix",
                type=ParameterType.PARAMETER_STRING,
                description=("The prefix to append to topic names."),
                read_only=True,
            ),
        )
        self.__prefix = prefix.value

    def __callback(self, msg: CompressedImageInput) -> None:
        """
        Callback function for the subscriber.
        """
        # Get the topic name
        topic_name = msg.topic

        # Create the publisher
        if topic_name not in self.__pubs:
            repub_topic_name = os.path.join(
                "/",
                self.__prefix.lstrip("/"),
                topic_name.lstrip("/"),
            )
            self.__pubs[topic_name] = self.create_publisher(
                msg_type=CompressedImageOutput,
                topic=repub_topic_name,
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )
            self.get_logger().info(f"Created publisher for {repub_topic_name}.")

        # Publish the message
        self.__pubs[topic_name].publish(msg.data)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    # Create the node
    receiver = ReceiverCompressedImageNode()

    # Spin the node
    executor = MultiThreadedExecutor()
    rclpy.spin(receiver, executor=executor)

    # Terminate this node
    receiver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
