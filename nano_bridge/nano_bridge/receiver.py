#!/usr/bin/env python3
"""
This module contains a node, ReceiverNode, which subscribes to a ByteMultiArray
topic published by SenderNode and republishes the messages to the original topics,
prepended with a parameter-specified prefix.
"""

# Standard imports
import os
import pickle as pkl
import traceback
from typing import Type

# Third-party imports
from std_msgs.msg import ByteMultiArray
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.publisher import Publisher

# Local imports
from .helpers import import_from_string


class ReceiverNode(Node):
    """
    The ReceiverNode class subscribes to a ByteMultiArray topic published by
    SenderNode and republishes the messages to the original topics, prepended
    with a parameter-specified prefix.
    """

    def __init__(self) -> None:
        """
        Initialize the sender node.
        """
        super().__init__("nano_bridge_receiver")

        # Load the parameters
        self.__prefix = ""
        self.__load_parameters()

        # Create the publishers
        self.__types: dict[str, Type] = {}
        self.__pubs: dict[str, Publisher] = {}

        # Create the subscriber
        # pylint: disable=unused-private-member
        self.__sub = self.create_subscription(
            msg_type=ByteMultiArray,
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

    def __callback(self, msg: ByteMultiArray) -> None:
        """
        Callback function for the subscriber.
        """
        # Get the topic name and type
        if len(msg.layout.dim) < 2:
            self.get_logger().error("Invalid message layout.")
            return
        topic_name = msg.layout.dim[0].label
        topic_type = msg.layout.dim[1].label

        # Import the topic type
        if topic_type not in self.__types:
            self.__types[topic_type] = import_from_string(topic_type)

        # Create the publisher
        if topic_name not in self.__pubs:
            repub_topic_name = os.path.join(
                "/",
                self.__prefix.lstrip("/"),
                topic_name.lstrip("/"),
            )
            self.__pubs[topic_name] = self.create_publisher(
                msg_type=self.__types[topic_type],
                topic=repub_topic_name,
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )
            self.get_logger().info(f"Created publisher for {repub_topic_name}.")

        # Unpickle the message
        try:
            data = b"".join(msg.data)
            msg_decoded = pkl.loads(data)
        except pkl.PickleError as exc:
            self.get_logger().error(f"{traceback.format_exc()}")
            self.get_logger().error(f"Error unpickling message: {exc}")
            return

        # Publish the message
        self.__pubs[topic_name].publish(msg_decoded)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    # Create the node
    receiver = ReceiverNode()

    # Spin the node
    executor = MultiThreadedExecutor()
    rclpy.spin(receiver, executor=executor)

    # Terminate this node
    receiver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
