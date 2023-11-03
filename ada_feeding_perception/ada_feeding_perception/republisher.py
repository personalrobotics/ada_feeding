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
        mask_post_processor_str = "mask"
        temporal_post_processor_str = "temporal"
        spatial_post_processor_str = "spatial"
        threshold_post_processor_str = "threshold"
        self.create_post_processors = {
            identity_post_processor_str: lambda: lambda msg: msg,
            mask_post_processor_str: lambda: self.mask_post_processor,
            temporal_post_processor_str: self.create_temporal_post_processor,
            spatial_post_processor_str: lambda: self.spatial_post_processor,
            threshold_post_processor_str: lambda: self.threshold_post_processor,
        }

        # Load the parameters
        (
            self.from_topics,
            topic_type_strs,
            republished_namespace,
            post_processors_strs,
            mask_relative_path,
            self.temporal_window_size,
            self.spatial_num_pixels,
            self.threshold_min,
            self.threshold_max,
        ) = self.load_parameters()

        # Import the topic types
        self.topic_types = []
        for topic_type_str in topic_type_strs:
            self.topic_types.append(import_from_string(topic_type_str))

        # Configure the post-processors
        self.bridge = CvBridge()
        self.mask_img = None
        for i, post_processors_str in enumerate(post_processors_strs):
            for j, post_processor_str in enumerate(post_processors_str):
                # If at least one post-processor is "mask", load the mask
                if (
                    post_processor_str == mask_post_processor_str
                    and self.mask_img is None
                ):
                    if mask_relative_path is None:
                        self.get_logger().warn(
                            "Must specify `mask_relative_path` to use post-processor "
                            f"`{mask_post_processor_str}`. Replacing with post-processor "
                            f"{identity_post_processor_str}."
                        )
                        post_processors_strs[i][j] = identity_post_processor_str
                        continue

                    # Get the mask path
                    mask_path = os.path.join(
                        get_package_share_directory("ada_feeding_perception"),
                        mask_relative_path,
                    )

                    # Load the image as a binary mask
                    self.mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        # For each topic, create a callback, publisher, and subscriber
        num_topics = min(len(self.from_topics), len(self.topic_types))
        self.callbacks = []
        self.pubs = []
        self.subs = []
        for i in range(num_topics):
            # Get the post-processors
            post_processor_fns = []
            if i < len(post_processors_strs):
                for post_processor_str in post_processors_strs[i]:
                    if post_processor_str in self.create_post_processors:
                        post_processor_fn = self.create_post_processors[
                            post_processor_str
                        ]()
                        # Check the type of the post-processor
                        if "msg" in post_processor_fn.__annotations__:
                            if (
                                post_processor_fn.__annotations__["msg"] == Any
                                or post_processor_fn.__annotations__["msg"]
                                == self.topic_types[i]
                            ):
                                post_processor_fns.append(post_processor_fn)
                            else:
                                self.get_logger().warn(
                                    f"Type mismatch for post-processor {post_processor_str} for topic at index {i}. "
                                    f"Will not post-process."
                                )
            else:
                self.get_logger().warn(
                    f"No valid post-processor for topic at index {i}. Will not post-process."
                )

            # Create the callback
            callback = self.create_callback(i, post_processor_fns)
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

    def load_parameters(
        self,
    ) -> Tuple[List[str], List[str], List[List[str]], str, str, int, int, int, int]:
        """
        Load the parameters for the republisher.

        Returns
        -------
        from_topics : List[str]
            The topics to subscribe to.
        topic_types : List[str]
            The types of the topics to subscribe to in format, e.g., `std_msgs.msg.String`.
        post_processors : List[List[str]]
            The post-processing functions to apply to the messages before republishing.
        republished_namespace : str
            The namespace to republish topics under.
        mask_relative_path : str
            The path of the binary mask to be used with the post-processor, relative to
            the ada_feeding_perception share directory.
        temporal_window_size : int
            The size of the window (num frames) to use for the temporal post-processor.
        spatial_num_pixels : int
            The number of pixels to use for the spatial post-processor.
        threshold_min : int
            The minimum value to use for the threshold post-processor.
        threshold_max : int
            The maximum value to use for the threshold post-processor.
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
                    "before republishing, as a comma-separated list."
                ),
                read_only=True,
            ),
        )
        # Split the post-processors
        post_processors_retval = post_processors.value
        if post_processors_retval is None:
            post_processors_retval = []
        else:
            post_processors_retval = [
                [
                    single_post_processor.strip()
                    for single_post_processor in post_processor.split(",")
                ]
                for post_processor in post_processors_retval
            ]

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

        # Get the window size to use with the temporal post-processor
        temporal_window_size = self.declare_parameter(
            "temporal_window_size",
            5,
            descriptor=ParameterDescriptor(
                name="temporal_window_size",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The size of the window (num frames) to use for the temporal post-processor. "
                    "Default: 5"
                ),
                read_only=True,
            ),
        )

        # Get the number of pixels to use with the spatial post-processor
        spatial_num_pixels = self.declare_parameter(
            "spatial_num_pixels",
            10,
            descriptor=ParameterDescriptor(
                name="spatial_num_pixels",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The number of pixels to use for the spatial post-processor. "
                    "Default: 10"
                ),
                read_only=True,
            ),
        )

        # The min/max threshold values for the threshold post-processor
        threshold_min = self.declare_parameter(
            "threshold_min",
            0,
            descriptor=ParameterDescriptor(
                name="threshold_min",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The minimum value to use for the threshold post-processor. "
                    "Default: 0"
                ),
                read_only=True,
            ),
        )
        threshold_max = self.declare_parameter(
            "threshold_max",
            20000,
            descriptor=ParameterDescriptor(
                name="threshold_max",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The maximum value to use for the threshold post-processor. "
                    "Default: 20000"
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

        return (
            from_topics_retval,
            topic_types_retval,
            republished_namespace.value,
            post_processors_retval,
            mask_relative_path.value,
            temporal_window_size.value,
            spatial_num_pixels.value,
            threshold_min.value,
            threshold_max.value,
        )

    def create_callback(
        self, i: int, post_processors: List[Callable[[Any], Any]]
    ) -> Callable:
        """
        Create the callback for the subscriber.

        Parameters
        ----------
        i : int
            The index of the callback.
        post_processor : List[Callable[[Any], Any]]
            The post-processing functions to apply to the message before republishing.
            Each must take in a message and return a message of the same type.

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
            for post_processor in post_processors:
                msg = post_processor(msg)
            self.pubs[i].publish(msg)

        return callback

    def mask_post_processor(self, msg: Image) -> Image:
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

    def create_temporal_post_processor(self) -> Callable[[Image], Image]:
        """
        Creates the temporal post-processor function, with a dedicated window.
        """

        temporal_window = []

        def temporal_post_processor(msg: Image) -> Image:
            """
            The temporal post-processor stores the last `temporal_window_size` images.
            It returns the most recent image, but only the pixels in that image that
            are non-zero across all images in the window.

            Parameters
            ----------
            msg : Image
                The image to process.

            Returns
            -------
            Image
                The processed image. All other attributes of the message remain the same.
            """
            # Read the ROS msg as a CV image
            img = self.bridge.imgmsg_to_cv2(msg)

            # Add it to the window
            temporal_window.append(img)

            # If the window is full, remove the oldest image
            if len(temporal_window) > self.temporal_window_size:
                temporal_window.pop(0)

            # Get the mask
            mask = (img > 0).astype(np.uint8)
            for i in range(0, len(temporal_window) - 1):
                mask = np.bitwise_and(mask, (temporal_window[i] > 0).astype(np.uint8))

            # Mask the latest image
            masked_img = cv.bitwise_and(img, img, mask=mask)

            # Get the new img message
            masked_msg = self.bridge.cv2_to_imgmsg(masked_img)
            masked_msg.header = msg.header

            return masked_msg

        return temporal_post_processor

    def spatial_post_processor(self, msg: Image) -> Image:
        """
        Applies the `opening` morpholical transformation to the image.

        Parameters
        ----------
        msg : Image
            The image to process.

        Returns
        -------
        Image
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = self.bridge.imgmsg_to_cv2(msg)

        # Apply the opening morphological transformation
        mask = (img > 0).astype(np.uint8)
        kernel = np.ones((self.spatial_num_pixels, self.spatial_num_pixels), np.uint8)
        opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        masked_img = cv.bitwise_and(img, img, mask=opened_mask)

        # Get the new img message
        masked_msg = self.bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg

    def threshold_post_processor(self, msg: Image) -> Image:
        """
        Applies a threshold to the image.

        Parameters
        ----------
        msg : Image
            The image to process.

        Returns
        -------
        Image
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = self.bridge.imgmsg_to_cv2(msg)

        # Apply the threshold
        mask = cv.inRange(img, self.threshold_min, self.threshold_max)
        masked_img = cv.bitwise_and(img, img, mask=mask)

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
