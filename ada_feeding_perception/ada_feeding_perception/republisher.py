#!/usr/bin/env python3
"""
This module defines the Republisher node, which republishes the messages from one
topic to another, with user-configureable post-processors.

This node can be used to address several issues:
    1. Network latency with too many subscribers to a topic that is being published
       on a different machine.
    2. Republishing a topic with some modifications (e.g., filtering, throttling, etc.).
"""

# Standard imports
import math
import os
import time
from typing import Any, Callable, List, Tuple

# Third-party imports
from ament_index_python.packages import get_package_share_directory
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Local imports
from ada_feeding.helpers import import_from_string
from .depth_post_processors import (
    create_identity_post_processor,
    create_mask_post_processor,
    create_spatial_post_processor,
    create_temporal_post_processor,
    create_threshold_post_processor,
    post_processor_chain,
)


class Republisher(Node):
    """
    A node that republishes the messages, with optional post-processing.
    """

    # pylint: disable=too-many-instance-attributes
    # One over is fine.

    def __init__(self) -> None:
        """
        Initialize the node.
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-nested-blocks, too-many-statements
        # Necessary because we are handling several post-processors and topics.
        super().__init__("republisher")

        self._default_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

        # Load the parameters
        (
            self.from_topics,
            in_topic_type_strs,
            out_topic_type_strs,
            to_topics,
            target_rates,
            post_processors_strs,
            mask_relative_path,
            temporal_window_size,
            spatial_num_pixels,
            threshold_min,
            threshold_max,
        ) = self.load_parameters()

        # Configure the post-processors.
        identity_post_processor_str = "none"
        mask_post_processor_str = "mask"
        temporal_post_processor_str = "temporal"
        spatial_post_processor_str = "spatial"
        threshold_post_processor_str = "threshold"
        # If at least one post-processor is "mask", load the mask
        bridge = CvBridge()
        mask_img = None
        for i, post_processors_str in enumerate(post_processors_strs):
            break_loop = False
            for j, post_processor_str in enumerate(post_processors_str):
                if post_processor_str == mask_post_processor_str and mask_img is None:
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
                    mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                    break_loop = True
                    break
            if break_loop:
                break

        # Configure the post-processing functions. The values are a tuple of the function and its kwargs
        self.create_post_processors = {
            identity_post_processor_str: (
                create_identity_post_processor,
                {"bridge": bridge},
            ),
            mask_post_processor_str: (
                create_mask_post_processor,
                {"mask_img": mask_img, "bridge": bridge},
            ),
            temporal_post_processor_str: (
                create_temporal_post_processor,
                {"temporal_window_size": temporal_window_size, "bridge": bridge},
            ),
            spatial_post_processor_str: (
                create_spatial_post_processor,
                {"spatial_num_pixels": spatial_num_pixels, "bridge": bridge},
            ),
            threshold_post_processor_str: (
                create_threshold_post_processor,
                {
                    "threshold_min": threshold_min,
                    "threshold_max": threshold_max,
                    "bridge": bridge,
                },
            ),
        }

        # Import the topic types
        self.in_topic_types = []
        for in_topic_type_str in in_topic_type_strs:
            self.in_topic_types.append(import_from_string(in_topic_type_str))
        self.out_topic_types = []
        for out_topic_type_str in out_topic_type_strs:
            if len(out_topic_type_str) > 0:
                if out_topic_type_str.strip() not in [
                    "sensor_msgs.msg.CompressedImage",
                    "sensor_msgs.msg.Image",
                ]:
                    self.get_logger().warn(
                        "out_topic_types must be either sensor_msgs.msg.Image or "
                        "sensor_msgs.msg.CompressedImage. Returning same type as input."
                    )
                    self.out_topic_types.append(None)
                else:
                    self.out_topic_types.append(import_from_string(out_topic_type_str))
            else:
                self.out_topic_types.append(None)

        # For each topic, create a callback, publisher, and subscriber
        num_topics = min(
            len(self.from_topics),
            len(self.in_topic_types),
            len(self.out_topic_types),
            len(to_topics),
            len(target_rates),
        )
        self.callbacks = []
        self.pubs = []
        self.subs = []
        for i in range(num_topics):
            # Get the post-processors
            fns, kwargs = [], []
            if i < len(post_processors_strs):
                for post_processor_str in post_processors_strs[i]:
                    if post_processor_str in self.create_post_processors:
                        (
                            create_post_processor_fn,
                            create_post_processor_kwargs,
                        ) = self.create_post_processors[post_processor_str]
                        fns.append(create_post_processor_fn)
                        kwargs.append(create_post_processor_kwargs)
            else:
                self.get_logger().warn(
                    f"No valid post-processor for topic at index {i}. Will not post-process."
                )

            # Create the callback
            if self.out_topic_types[i] is None:
                compress = None
            else:
                compress = self.out_topic_types[i] == CompressedImage
            post_processor = post_processor_chain(fns, kwargs, compress)
            callback = self.create_callback(i, post_processor, target_rates[i])
            self.callbacks.append(callback)

            # Create the publisher
            to_topic = to_topics[i]
            publisher = self.create_publisher(
                msg_type=self.in_topic_types[i]
                if self.out_topic_types[i] is None
                else self.out_topic_types[i],
                topic=to_topic,
                qos_profile=1,  # TODO: we should get and mirror the QOS profile of the from_topic
            )
            self.pubs.append(publisher)

            # Create the subscriber
            subscriber = self.create_subscription(
                msg_type=self.in_topic_types[i],
                topic=self.from_topics[i],
                callback=callback,
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),  # TODO: we should get and mirror the QOS profile of the from_topic
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.subs.append(subscriber)

    def load_parameters(
        self,
    ) -> Tuple[
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[List[str]],
        str,
        int,
        int,
        int,
        int,
    ]:
        """
        Load the parameters for the republisher.

        Returns
        -------
        from_topics : List[str]
            The topics to subscribe to.
        in_topic_types : List[str]
            The types of the topics to subscribe to in format, e.g., `std_msgs.msg.String`.
        out_topic_types : List[str]
            The types of the topics to publish to in format, e.g., `std_msgs.msg.String`.
        to_topics : List[str]
            The topics to republish to.
        post_processors : List[List[str]]
            The post-processing functions to apply to the messages before republishing.
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

        # pylint: disable=too-many-locals
        # This needs many parameters given how flexible it is intended to be.

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

        # Read the input topic types
        in_topic_types = self.declare_parameter(
            "in_topic_types",
            descriptor=ParameterDescriptor(
                name="in_topic_types",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of the types of the topics to subscribe to in format, "
                    "e.g., `std_msgs.msg.String`."
                ),
                read_only=True,
            ),
        )

        # Read the output topic types
        out_topic_types = self.declare_parameter(
            "out_topic_types",
            descriptor=ParameterDescriptor(
                name="out_topic_types",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of the types of the topics to publish to in format, "
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
                description="List of the topics to republish to.",
                read_only=True,
            ),
        )

        # Read the target rates
        target_rates = self.declare_parameter(
            "target_rates",
            descriptor=ParameterDescriptor(
                name="target_rates",
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description="Target rates (hz) for the republications.",
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
        in_topic_types_retval = in_topic_types.value
        if in_topic_types_retval is None:
            in_topic_types_retval = []
        out_topic_types_retval = out_topic_types.value
        if out_topic_types_retval is None:
            out_topic_types_retval = []
        to_topics_retval = to_topics.value
        if to_topics_retval is None:
            to_topics_retval = []
        target_rates_retval = target_rates.value
        if target_rates_retval is None:
            target_rates_retval = []

        return (
            from_topics_retval,
            in_topic_types_retval,
            out_topic_types_retval,
            to_topics.value,
            target_rates_retval,
            post_processors_retval,
            mask_relative_path.value,
            temporal_window_size.value,
            spatial_num_pixels.value,
            threshold_min.value,
            threshold_max.value,
        )

    def create_callback(
        self,
        i: int,
        post_processor: Callable[[Any], Any],
        target_rate: float,
    ) -> Callable:
        """
        Create the callback for the subscriber.

        Parameters
        ----------
        i : int
            The index of the callback.
        post_processor : Callable[[Any], Any]
            The chained post-processing function to apply to the message before republishing.
        target_rate : float
            the target rate for the publication

        Returns
        -------
        callback : Callable
            The callback for the subscriber.
        """
        if target_rate <= 0:
            interval = -math.inf
        else:
            interval = 1.0 / target_rate
        last_published_time = None

        def callback(msg: Any):
            """
            The callback for the subscriber.

            Parameters
            ----------
            msg : Any
                The message from the subscriber.
            """
            nonlocal last_published_time
            # self.get_logger().info(
            #     f"Received message on topic {i} {self.from_topics[i]}"
            # )
            msg = post_processor(msg)
            if (
                last_published_time is None
                or time.time() - last_published_time >= interval
            ):
                self.pubs[i].publish(msg)
                last_published_time = time.time()

        return callback


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
