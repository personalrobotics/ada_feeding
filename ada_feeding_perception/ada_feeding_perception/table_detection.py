# Standard imports
import os
import threading
from typing import Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, RegionOfInterest
import torch
import pyrealsense2
from std_srvs.srv import Trigger


# Local imports
from ada_feeding_perception.helpers import (
    bbox_from_mask,
    crop_image_mask_and_point,
    cv2_image_to_ros_msg,
    download_checkpoint,
    get_connected_component,
    get_img_msg_type,
    ros_msg_to_cv2_image,
)

class TableDetectionNode(Node):
	"""
    This node fits a table to the received depth image and publishes a 3d PoseStamped location
	of a specific point on the table with respect to the camera frame.
	"""
	
	def __init__(self):
		"""
        Initialize the TableDetectionNode.
        """
		
		super().__init__("table_detection")

        self.get_logger().info(f"Table detection node is running! -------------------------------")
		
		# Create the service
        self.srv = self.create_service(
            Trigger,
			"~/fit_table",
            self.table_detection_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
		
		# Subscribe to the camera info topic, to get the camera intrinsics
        self.camera_info = None
        self.camera_info_lock = threading.Lock()
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the aligned depth image topic, to store the latest depth image        self.latest_depth_img_msg = None
        self.latest_depth_img_msg_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(aligned_depth_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to Image. {err}"
            )
            aligned_depth_type = Image
        self.depth_image_subscriber = self.create_subscription(
            aligned_depth_type,
            aligned_depth_topic,
            self.depth_image_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the RGB image topic, to store the latest image
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()
        image_topic = "~/image"
        try:
            image_type = get_img_msg_type(image_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of image topic. Defaulting to CompressedImage. {err}"
            )
            image_type = CompressedImage
        self.image_subscriber = self.create_subscription(
            image_type,
            image_topic,
            self.image_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
    
    def fit_table(
        self, 
        image: npt.NDArray, 
        image_depth: npt.NDArray, 
        target_u: int,
        target_v: int, 
        table_buffer: float = 50
    ):
        """
            Parameters
            ----------
            Find table plane.
            image(image matrix): RGB image of plate
            image_depth(image matrix): depth image of plate
            target_u: u coordinate to obtain table depth 
            target_v: v coordinate to obtain table depth 
            table_buffer: buffer for plate radius 

            Returns
            ----------
            table[target_u][target_v]: the depth to the table at coordinates (target_u, target_v)
        """

        # Grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # Detect Largest Circle (Plate)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self._hough_accum, self._hough_min_dist,
            param1=self._hough_param1, param2=self._hough_param2, minRadius=self._hough_min, maxRadius=self._hough_max)
        if circles is None:
            return None, None, None
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        for (x,y,r) in circles:
            print("Radius: " + str(r))
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r

        # Create Mask for Depth Image
        plate_mask = np.zeros(image_depth.shape)
        cv2.circle(plate_mask, plate_uv, plate_r + table_buffer, 1.0, -1)
        cv2.circle(plate_mask, plate_uv, plate_r, 0.0, -1)
        depth_mask = (image_depth * (plate_mask).astype("uint16")).astype(float)

        # Noise removal
        kernel = np.ones((6,6), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations = 3)

        # Remove Outliers
        depth_var = np.abs(depth_mask - np.mean(depth_mask[depth_mask > 0]))
        depth_std = np.std(depth_mask[depth_mask > 0])
        depth_mask[depth_var > 2.0*depth_std] = 0

        # Fit Plane: depth = a*u + b*v + c
        d_idx = np.where(depth_mask > 0)
        d = depth_mask[d_idx].astype(float)
        coeffs = np.hstack((np.vstack(d_idx).T, np.ones((len(d_idx[0]), 1)))).astype(float)
        b, a, c = np.linalg.lstsq(coeffs, d)[0]

        # Create Table Depth Image
        u = np.linspace(0, image_depth.shape[1], image_depth.shape[1], False)
        v = np.linspace(0, image_depth.shape[0], image_depth.shape[0], False)
        U, V = np.meshgrid(u, v)
        table = a*U + b*V + c
        table = table.astype("uint16")

        return table[target_u][target_v]

    def table_detection_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ):
        # Get the latest RGB image message
            with self.latest_img_msg_lock:
                rgb_msg = self.latest_img_msg
            if rgb_msg is None:
                self.get_logger().warn(
                    "No RGB image message received.", throttle_duration_sec=1
                )
                continue
        
        # Get the latest depth image
        with self.latest_depth_img_msg_lock:
            depth_img_msg = self.latest_depth_img_msg

        # Get table depth from camera at the (u, v) coordinate (320, 240)
        table_depth = self.fit_table(
            self, rgb_msg, depth_img_msg, 320, 240
        )

        
        self.get_logger().info(f"table depth, {table_depth}, {type(table_depth)}")

        # Get the camera matrix
        with self.camera_info_lock:
            camera_matrix = self.camera_matrix
        
        # To be replaced with PoseStamped message
        response.success = True
        response.message = f"To be replaced with PoseStamped message"
        
        return response

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Store the latest camera info message.

        Parameters
        ----------
        msg: The camera info message.
        """
        with self.camera_info_lock:
            self.camera_info = msg

    def depth_image_callback(self, msg: Union[Image, CompressedImage]) -> None:
        """
        Store the latest depth image message.

        Parameters
        ----------
        msg: The depth image message.
        """
        with self.latest_depth_img_msg_lock:
            self.latest_depth_img_msg = msg

    def image_callback(self, msg: Union[Image, CompressedImage]) -> None:
        """
        Store the latest image message.

        Parameters
        ----------
        msg: The image message.
        """
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg
        

def main(args=None):
    """
    Launch the ROS node and spin.
    """

    asdfasd 
    
    self.get_logger().info(f"TableDetection node is running! -------------------------------")

    rclpy.init(args=args)

    table_detection = TableDetectionNode()

    rclpy.spin(table_detection)


if __name__ == "__main__":
    main()

		
