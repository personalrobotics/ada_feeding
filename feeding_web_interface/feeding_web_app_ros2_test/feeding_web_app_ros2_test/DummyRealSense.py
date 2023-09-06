#!/usr/bin/env python3

# Standard imports
import threading

# Third-party imports
from cv_bridge import CvBridge
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class DummyRealSense(Node):
    """
    Reads in a video file and publishes the frames as ROS2 messages.
    """

    def __init__(self):
        super().__init__("dummy_real_sense")

        # Load the rgb_path parameter
        rgb_path = self.declare_parameter(
            "rgb_path",
            None,
            ParameterDescriptor(
                name="rgb_path",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The path to the RGB image/video. A video needs the mp4 extension, "
                    "and an image needs the jpg extension."
                ),
                read_only=True,
            ),
        )
        depth_path = self.declare_parameter(
            "depth_path",
            "",
            ParameterDescriptor(
                name="depth_path",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The path to the depth image. A depth image needs the png extension."
                ),
                read_only=True,
            ),
        )
        self.fps = self.declare_parameter(
            "fps",
            30,
            ParameterDescriptor(
                name="fps",
                type=ParameterType.PARAMETER_INTEGER,
                description="The fps of the video.",
                read_only=True,
            ),
        ).value

        # Read in the RGB image/video
        if rgb_path.value.lower().endswith(".mp4"):
            self.video = cv2.VideoCapture(rgb_path.value)
        elif rgb_path.value.lower().endswith(".jpg"):
            self.video = None
            self.frame = cv2.imread(rgb_path.value)
        else:
            self.video = None
            self.frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Read in the depth image
        if depth_path.value.lower().endswith(".png"):
            self.depth_frame = cv2.imread(depth_path.value, cv2.IMREAD_ANYDEPTH)
        else:
            # All points are 1000mm away
            self.depth_frame = np.ones((480, 640), dtype=np.uint16) * 1000

        # Create the publisher
        self.image_publisher = self.create_publisher(Image, "~/image", 1)
        self.compressed_image_publisher = self.create_publisher(
            CompressedImage, "~/compressed_image", 1
        )
        self.aligned_depth_publisher = self.create_publisher(
            Image, "~/aligned_depth", 1
        )
        if self.video is not None:
            self.num_frames = 0
        self.bridge = CvBridge()

        # Launch the publisher in a separate thread
        self.thread = threading.Thread(target=self.publish_frames, daemon=True)
        self.thread.start()

        # Log that the node initialized
        self.get_logger().info("DummyRealSense node ready")

    def publish_frames(self):
        # Maintain the rate
        rate = self.create_rate(self.fps)

        while True:
            # Get the RGB frame
            if self.video is None:
                frame = self.frame
            else:
                ret, frame = self.video.read()
                self.num_frames += 1
                # If the last frame is reached, reset the capture and the frame_counter
                if self.num_frames == self.video.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.num_frames = 0
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, self.num_frames)

            # Publish the RGB Image message
            frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            frame_msg.header.frame_id = "camera_color_optical_frame"
            frame_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_publisher.publish(frame_msg)

            # Publish the RGB CompressedImage message
            compressed_frame_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
            compressed_frame_msg.header.frame_id = "camera_color_optical_frame"
            compressed_frame_msg.header.stamp = self.get_clock().now().to_msg()
            self.compressed_image_publisher.publish(compressed_frame_msg)

            # Publish the depth Image message
            depth_frame_msg = self.bridge.cv2_to_imgmsg(self.depth_frame, "passthrough")
            depth_frame_msg.header.frame_id = "camera_color_optical_frame"
            depth_frame_msg.header.stamp = self.get_clock().now().to_msg()
            self.aligned_depth_publisher.publish(depth_frame_msg)

            rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    dummy_real_sense = DummyRealSense()

    rclpy.spin(dummy_real_sense)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    dummy_real_sense.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
