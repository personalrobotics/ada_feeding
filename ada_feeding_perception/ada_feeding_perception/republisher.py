#!/usr/bin/env python3
"""
This module defines the Republisher node, which takes in parameters for `from_topics`
and `republished_namespace` and republishes the messages from the `from_topics` within
the specified namespace.

This node is intended to address the issue where, when subscribing to images on a
different machine to the one you are publishing them on, the rate slows down a lot if
you have >3 subscriptions.
"""

# Standard imports
import os
from typing import Any, Callable, List, Tuple

# Third-party imports
from ament_index_python.packages import get_package_share_directory
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

# Local imports
from ada_feeding.helpers import import_from_string


class Republisher(Node):
    """
    A node that takes in parameters for `from_topics` and `republished_namespace` and
    republishes the messages from the `from_topics` within the specified namespace.
    """

    def __init__(self) -> None:
        """
        Initialize the node.
        """
        super().__init__("republisher")

        self._default_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # Configure the post-processing functions
        identity_post_processor_str = "none"
        mask_post_processosr_str = "mask"
        self.post_processors = {
            identity_post_processor_str: lambda msg: msg,
            mask_post_processosr_str: self.mask,
        }

        # Load the parameters
        (
            self.from_topics,
            topic_type_strs,
            republished_namespace,
            post_processor_strs,
            mask_relative_path,
        ) = self.load_parameters()

        # Import the topic types
        self.topic_types = []
        for topic_type_str in topic_type_strs:
            self.topic_types.append(import_from_string(topic_type_str))

        # If at least one post-processor is "mask", load the mask
        self.mask_img = None
        for i, post_processor_str in enumerate(post_processor_strs):
            if post_processor_str == mask_post_processosr_str:
                if mask_relative_path is None:
                    self.get_logger().warn(
                        "Must specify `mask_relative_path` to use post-processor "
                        f"`{mask_post_processosr_str}`. Replacing with post-processor "
                        f"{identity_post_processor_str}."
                    )
                    post_processor_strs[i] = identity_post_processor_str
                    continue

                # Get the mask path
                mask_path = os.path.join(
                    get_package_share_directory("ada_feeding_perception"),
                    mask_relative_path,
                )

                # Load the image as a binary mask
                self.mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

                # Load the CV Bridge
                self.bridge = CvBridge()
                break

        # For each topic, create a callback, publisher, and subscriber
        num_topics = min(len(self.from_topics), len(self.topic_types))
        self.callbacks = []
        self.pubs = []
        self.subs = []
        for i in range(num_topics):
            # Get the post-processor
            if (
                i < len(post_processor_strs)
                and post_processor_strs[i] in self.post_processors
            ):
                post_processor_fn = self.post_processors[post_processor_strs[i]]
            else:
                self.get_logger().warn(
                    f"No valid post-processor for topic at index {i}. Using "
                    f"'{identity_post_processor_str}' instead."
                )
                post_processor_fn = self.post_processors[identity_post_processor_str]

            # Check the type of the post_processor
            if "msg" in post_processor_fn.__annotations__:
                if (
                    post_processor_fn.__annotations__["msg"] != Any
                    and post_processor_fn.__annotations__["msg"] != self.topic_types[i]
                ):
                    self.get_logger().warn(
                        f"Type mismatch for post-processor for topic at index {i}. "
                        f"Using '{identity_post_processor_str}' instead."
                    )
                    post_processor_fn = self.post_processors[
                        identity_post_processor_str
                    ]

            # Create the callback
            callback = self.create_callback(i, post_processor_fn)
            self.callbacks.append(callback)

            # Create the publisher
            to_topic = "/".join(
                [
                    "",
                    republished_namespace.lstrip("/"),
                    self.from_topics[i].lstrip("/"),
                ]
            )
            publisher = self.create_publisher(
                msg_type=self.topic_types[i],
                topic=to_topic,
                qos_profile=1,  # TODO: we should get and mirror the QOS profile of the from_topic
            )
            self.pubs.append(publisher)

            # Create the subscriber
            subscriber = self.create_subscription(
                msg_type=self.topic_types[i],
                topic=self.from_topics[i],
                callback=callback,
                qos_profile=1,  # TODO: we should get and mirror the QOS profile of the from_topic
                callback_group=MutuallyExclusiveCallbackGroup(),
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
        post_processors : List[str]
            The post-processing functions to apply to the messages before republishing.
        republished_namespace : str
            The namespace to republish topics under.
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

        # Read the namespace to republish topics to
        republished_namespace = self.declare_parameter(
            "republished_namespace",
            "/local",
            descriptor=ParameterDescriptor(
                name="republished_namespace",
                type=ParameterType.PARAMETER_STRING,
                description="The namespace to republish topics under.",
                read_only=True,
            ),
        )

        # Read the post-processing functions
        post_processors = self.declare_parameter(
            "post_processors",
            descriptor=ParameterDescriptor(
                name="post_processors",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of the post-processing functions to apply to the messages "
                    "before republishing."
                ),
                read_only=True,
            ),
        )

        # Get the mask's relative path to use with the "mask" post-processor
        mask_relative_path = self.declare_parameter(
            "mask_relative_path",
            None,
            descriptor=ParameterDescriptor(
                name="mask_relative_path",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The path of the binary mask to be used with the post-processor, "
                    "relative to the ada_feeding_perception share directory."
                ),
                read_only=True,
            ),
        )

        # Replace unset parameters with empty list
        from_topics_retval = from_topics.value
        if from_topics_retval is None:
            from_topics_retval = []
        topic_types_retval = topic_types.value
        if topic_types_retval is None:
            topic_types_retval = []
        post_processors_retval = post_processors.value
        if post_processors_retval is None:
            post_processors_retval = []

        return (
            from_topics_retval,
            topic_types_retval,
            republished_namespace.value,
            post_processors_retval,
            mask_relative_path.value,
        )

    def create_callback(self, i: int, post_processor: Callable[[Any], Any]) -> Callable:
        """
        Create the callback for the subscriber.

        Parameters
        ----------
        i : int
            The index of the callback.
        post_processor : Callable[[Any], Any]
            The post-processing function to apply to the message before republishing.
            This must take in a message and return a message of the same type.

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
            # self.get_logger().info(
            #     f"Received message on topic {i} {self.from_topics[i]}"
            # )
            self.pubs[i].publish(post_processor(msg))

        return callback

    def mask(self, msg: Image) -> Image:
        """
        Applies a fixed mask to the image. Scales the mask to the image.

        Parameters
        ----------
        msg : Image
            The image to mask.

        Returns
        -------
        Image
            The masked image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = self.bridge.imgmsg_to_cv2(msg)
        # self.get_logger().info(
        #     f"img {img}"
        # )
        # scale_img = lambda img: 255*(img-img.min())/(img.max()-img.min())
        # cv.imshow("img", scale_img(img))

        # Scale the mask to be the size of the img
        mask = cv.resize(self.mask_img, img.shape[:2][::-1])

        # Apply the mask to the img
        masked_img = cv.bitwise_and(img, img, mask=mask)
        # self.get_logger().info(
        #     f"masked_img {masked_img}"
        # )
        # cv.imshow("masked_img", scale_img(masked_img))

        # cv.imshow("mask", mask)
        # cv.waitKey(0)

        # Get the new img message
        masked_msg = self.bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    republisher = Republisher()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=len(republisher.subs))

    rclpy.spin(republisher, executor=executor)


if __name__ == "__main__":
    main()
