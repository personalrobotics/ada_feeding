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
from typing import Optional

# Third-party imports
from sensor_msgs.msg import CameraInfo, CompressedImage as CompressedImageOutput
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
        self.__sync_camera_info_with_topic: Optional[str] = None
        self.__camera_info_pub_topic: Optional[str] = None
        self.__camera_info_msg = None
        self.__load_parameters()

        # Create the publishers
        self.__pubs: dict[str, Publisher] = {}
        if self.__sync_camera_info_with_topic is not None:
            self.__pub_camera_info = self.create_publisher(
                msg_type=CameraInfo,
                topic=self.__camera_info_pub_topic,
                qos_profile=QoSProfile(
                    depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
                ),
            )

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

        # Camera Info
        sync_camera_info_with_topic = self.declare_parameter(
            "sync_camera_info_with_topic",
            None,
            descriptor=ParameterDescriptor(
                name="sync_camera_info_with_topic",
                type=ParameterType.PARAMETER_STRING,
                description=("Whether to sync camera info with topic."),
                read_only=True,
            ),
        )
        self.__sync_camera_info_with_topic = sync_camera_info_with_topic.value

        camera_info_pub_topic = self.declare_parameter(
            "camera_info_pub_topic",
            None,
            descriptor=ParameterDescriptor(
                name="camera_info_pub_topic",
                type=ParameterType.PARAMETER_STRING,
                description=("The topic to publish camera info."),
                read_only=True,
            ),
        )
        self.__camera_info_pub_topic = camera_info_pub_topic.value
        if (
            self.__sync_camera_info_with_topic is not None
            and self.__camera_info_pub_topic is None
        ):
            raise ValueError(
                "If sync_camera_info_with_topic is set, camera_info_pub_topic must be set."
            )

        if self.__sync_camera_info_with_topic is not None:
            self.__camera_info_msg = CameraInfo()
            frame_id = self.declare_parameter(
                "camera_info.frame_id",
                "camera_color_optical_frame",
                descriptor=ParameterDescriptor(
                    name="camera_info.frame_id",
                    type=ParameterType.PARAMETER_STRING,
                    description=("The frame ID of the camera."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.header.frame_id = frame_id.value
            height = self.declare_parameter(
                "camera_info.height",
                480,
                descriptor=ParameterDescriptor(
                    name="camera_info.height",
                    type=ParameterType.PARAMETER_INTEGER,
                    description=("The height of the image."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.height = height.value
            width = self.declare_parameter(
                "camera_info.width",
                640,
                descriptor=ParameterDescriptor(
                    name="camera_info.width",
                    type=ParameterType.PARAMETER_INTEGER,
                    description=("The width of the image."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.width = width.value
            distortion_model = self.declare_parameter(
                "camera_info.distortion_model",
                "plumb_bob",
                descriptor=ParameterDescriptor(
                    name="camera_info.distortion_model",
                    type=ParameterType.PARAMETER_STRING,
                    description=("The distortion model of the camera."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.distortion_model = distortion_model.value
            d = self.declare_parameter(
                "camera_info.d",
                [0.0, 0.0, 0.0, 0.0, 0.0],
                descriptor=ParameterDescriptor(
                    name="camera_info.d",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=("The distortion coefficients."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.d = d.value
            k = self.declare_parameter(
                "camera_info.k",
                [
                    614.5933227539062,
                    0.0,
                    312.1358947753906,
                    0.0,
                    614.6914672851562,
                    223.70831298828125,
                    0.0,
                    0.0,
                    1.0,
                ],
                descriptor=ParameterDescriptor(
                    name="camera_info.k",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=("The camera matrix."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.k = k.value
            r = self.declare_parameter(
                "camera_info.r",
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                descriptor=ParameterDescriptor(
                    name="camera_info.r",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=("The rectification matrix."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.r = r.value
            p = self.declare_parameter(
                "camera_info.p",
                [
                    614.5933227539062,
                    0.0,
                    312.1358947753906,
                    0.0,
                    0.0,
                    614.6914672851562,
                    223.70831298828125,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                descriptor=ParameterDescriptor(
                    name="camera_info.p",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=("The projection matrix."),
                    read_only=True,
                ),
            )
            self.__camera_info_msg.p = p.value

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

        # Create the camera info message
        if self.__sync_camera_info_with_topic is not None:
            self.__camera_info_msg.header.stamp = msg.header.stamp
            self.__pub_camera_info.publish(self.__camera_info_msg)

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
