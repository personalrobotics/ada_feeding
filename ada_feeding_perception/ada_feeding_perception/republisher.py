#!/usr/bin/env python3
"""
This module defines the Republisher node, which takes in parameters for `from_topics`
and `to_topics` and republishes the messages from the `from_topics` to the `to_topics`.

This node is intended to address the issue where, when subscribing to images on a
different topic to the one you are publishing them on, the rate slows down a lot if
you have >3 subscriptions.
"""

# Standard imports
from typing import Any, Callable, List, Tuple

# Third-party imports
from ada_feeding.helpers import import_from_string
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Local imports


class Republisher(Node):
    """
    A node that takes in parameters for `from_topics` and `to_topics` and republishes
    the messages from the `from_topics` to the `to_topics`.

    TODO: As opposed to always subscribing to the `from_topic`, this node should
    only do so once there is a subscriber on the `to_topic`.
    """

    def __init__(self) -> None:
        """
        Initialize the node.
        """
        super().__init__("republisher")

        # Load the parameters
        self.from_topics, self.topic_type_strs, self.to_topics = self.load_parameters()

        # Import the topic types
        self.topic_types = []
        for topic_type_str in self.topic_type_strs:
            self.topic_types.append(import_from_string(topic_type_str))

        # For each topic, create a callback, publisher, and subscriber
        num_topics = min(
            len(self.from_topics), len(self.topic_types), len(self.to_topics)
        )
        self.callbacks = []
        self.pubs = []
        self.subs = []
        for i in range(num_topics):
            # Create the callback
            callback = self.create_callback(i)
            self.callbacks.append(callback)

            # Create the publisher
            publisher = self.create_publisher(
                msg_type=self.topic_types[i],
                topic=self.to_topics[i],
                qos_profile=1,  # TODO: we should mirror the QOS profile of the from_topic
            )
            self.pubs.append(publisher)

            # Create the subscriber
            subscriber = self.create_subscription(
                msg_type=self.topic_types[i],
                topic=self.from_topics[i],
                callback=callback,
                qos_profile=1,  # TODO: we should mirror the QOS profile of the from_topic
            )
            self.subs.append(subscriber)

    def load_parameters(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Load the parameters for the republisher.

        Returns
        -------
        from_topics : List[str]
            The topics to subscribe to.
        topic_types : List[str]
            The types of the topics to subscribe to in format, e.g., `std_msgs.msg.String`.
        to_topics : List[str]
            The topics to publish to.
        """
        # Read the from topics
        from_topics = self.declare_parameter(
            "from_topics",
            descriptor=ParameterDescriptor(
                name="from_topics",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="List of the topics to subscribe to.",
                read_only=True,
            ),
        )

        # Read the topic types
        topic_types = self.declare_parameter(
            "topic_types",
            descriptor=ParameterDescriptor(
                name="topic_types",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of the types of the topics to subscribe to  in format, "
                    "e.g., `std_msgs.msg.String`."
                ),
                read_only=True,
            ),
        )

        # Read the to topics
        to_topics = self.declare_parameter(
            "to_topics",
            descriptor=ParameterDescriptor(
                name="to_topics",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="List of the topics to publish to.",
                read_only=True,
            ),
        )

        return from_topics.value, topic_types.value, to_topics.value

    def create_callback(self, i: int) -> Callable:
        """
        Create the callback for the subscriber.

        Parameters
        ----------
        i : int
            The index of the callback.

        Returns
        -------
        callback : Callable
            The callback for the subscriber.
        """

        def callback(msg: Any):
            """
            The callback for the subscriber.

            Parameters
            ----------
            msg : Any
                The message from the subscriber.
            """
            self.get_logger().info(
                f"Received message on topic {i} {self.from_topics[i]}"
            )
            self.pubs[i].publish(msg)

        return callback


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    republisher = Republisher()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(republisher, executor=executor)


if __name__ == "__main__":
    main()
