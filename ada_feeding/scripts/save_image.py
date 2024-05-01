#!/usr/bin/env python3
"""
This module contains a node that exposes a ROS service that saves the latest
RGB image and depth image from the RealSense, at the parameter-specified filepath.
"""

# Standard imports
import threading
from typing import Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import SetBool

# Local imports
from ada_feeding_perception.helpers import ros_msg_to_cv2_image


class SaveImage(Node):
    """
    The SaveImage class exposes a ROS service that saves the latest RGB image
    and depth image from the RealSense, at the parameter-specified filepath.
    """

    def __init__(self) -> None:
        """
        Initialize the SaveImage node.
        """
        super().__init__("save_image")

        # Create a CvBridge to convert ROS messages to OpenCV images
        self.bridge = CvBridge()

        # Get the filepath to save the images to
        self.filepath = self.declare_parameter(
            "filepath",
            None,
            ParameterDescriptor(
                name="filepath",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The filepath to save the images to. This should be an absolute path, "
                    "including the filename but excluding the extension. `_rgb.jpg` and "
                    "`_depth.png` will be appended to the filepath to save the RGB image "
                    "and depth image, respectively."
                ),
                read_only=True,
            ),
        )

        # Create a service that saves the latest RGB image and depth image
        # from the RealSense, at the parameter-specified filepath
        self.create_service(SetBool, "save_image", self.save_image_callback)

        # Add subscribers to RealSense's compressed color image and aligned depth
        # image topics
        self.latest_color_image = None
        self.latest_color_image_lock = threading.Lock()
        self.create_subscription(
            CompressedImage,
            "/local/camera/color/image_raw/compressed",
            self.color_image_callback,
            1,
        )
        self.latest_depth_image = None
        self.latest_depth_image_lock = threading.Lock()
        self.create_subscription(
            CompressedImage,
            "/local/camera/aligned_depth_to_color/image_raw/compressedDepth",
            self.depth_image_callback,
            1,
        )

        # Log that the node is ready
        self.get_logger().info("SaveImage node ready")

    def color_image_callback(self, msg: Union[CompressedImage, Image]) -> None:
        """
        Callback function for the RealSense's compressed color image topic.
        """
        with self.latest_color_image_lock:
            self.latest_color_image = msg

    def depth_image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the RealSense's aligned depth image topic.
        """
        with self.latest_depth_image_lock:
            self.latest_depth_image = msg

    def save_image_callback(
        self, _: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """
        Callback function for the save_image ROS service.
        """
        response.message = ""
        # Check if the filepath parameter is set
        if self.filepath.value is None:
            response.success = False
            response.message += ", The filepath parameter is not set"
            return response

        # Check if the latest color image is available
        with self.latest_color_image_lock:
            if self.latest_color_image is None:
                response.success = False
                response.message += ", The latest color image is not available"
            else:
                # Save the latest color image
                color_image = ros_msg_to_cv2_image(self.latest_color_image, self.bridge)
                color_image_filepath = self.filepath.value + "_rgb.jpg"
                self.get_logger().info(f"Saving color image to {color_image_filepath}")
                cv2.imwrite(color_image_filepath, color_image)

        # Check if the latest depth image is available
        with self.latest_depth_image_lock:
            if self.latest_depth_image is None:
                response.success = False
                response.message += ", The latest depth image is not available"
            else:
                # Save the latest depth image
                depth_image = ros_msg_to_cv2_image(self.latest_depth_image, self.bridge)
                depth_image_filepath = self.filepath.value + "_depth.png"
                self.get_logger().info(f"Saving depth image to {depth_image_filepath}")
                cv2.imwrite(depth_image_filepath, depth_image)

        # Return a success response
        response.success = len(response.message) == 0
        response.message = "Successfully saved the latest color image and depth image"
        return response


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    # Initialize the ROS context
    rclpy.init(args=args)

    # Create the SaveImage node
    save_image = SaveImage()

    # Spin the node
    rclpy.spin(save_image)

    # Destroy the node
    save_image.destroy_node()

    # Shutdown the ROS context
    rclpy.shutdown()


if __name__ == "__main__":
    main()
