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
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import torch
import pyrealsense2
from std_srvs.srv import Trigger


# Local imports
from ada_feeding_msgs.msg import Mask
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
        self.get_logger().info("Entering table detection node!")

        # Table Plane Params
        self._hough_accum   = 1.5
        self._hough_min_dist= 100
        self._hough_param1  = 100 # Larger is more selective
        self._hough_param2  = 125 # Larger is more selective/decreases chance of false positives
        self._hough_min     = 75
        self._hough_max     = 200
        self._table_buffer  = 50 # Extra radius around plate to use for table

        # Create the service
        self.srv = self.create_service(
            Trigger,
            'fit_to_table',
            self.fit_to_table_callback,
        )

        # Approximate values of current camera intrinsics matrix
        # (updated with subscription)
        self.camera_matrix = [614, 0, 312, 0, 614, 223, 0, 0, 1]
        self.camera_info = None
        # Subscribe to the camera info topic, to get the camera intrinsics
        self.camera_info_lock = threading.Lock()
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the aligned depth image topic, to store the latest depth image
        self.latest_depth_img_msg = None
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
        image_msg: Image, 
        image_depth_msg: Image, 
    ):
        """
            Parameters
            ----------
            Find table plane.
            image(image matrix): RGB image of plate
            image_depth(image matrix): depth image of plate
            target_u: u coordinate to obtain table depth 
            target_v: v coordinate to obtain table depth 

            Returns
            ----------
            table: depth array of the table
        """

        # Convert ROS images to CV images
        image = ros_msg_to_cv2_image(image_msg, self.bridge)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        image_depth = ros_msg_to_cv2_image(image_depth_msg, self.bridge)

        # Detect Largest Circle (Plate)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self._hough_accum, self._hough_min_dist,
            param1=self._hough_param1, param2=self._hough_param2, minRadius=self._hough_min, maxRadius=self._hough_max)
        if circles is None:
            return None, None, None
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        print(len(circles))
        for (x,y,r) in circles:
            print("Radius: " + str(r))
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r
        
        # Testing - Draw out the circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x,y,r) in circles:
                center = (x, y)
                # circle center
                cv2.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = r
                cv2.circle(gray, center, radius, (255, 0, 255), 3)
                cv2.circle(gray, center, radius + self._table_buffer, (255, 0, 255), 3)
            cv2.imshow("detected circles", gray)
            cv2.waitKey(0)

        # Create Mask for Depth Image
        plate_mask = np.zeros(image_depth.shape)
        cv2.circle(plate_mask, plate_uv, plate_r + self._table_buffer, 1.0, -1)
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

        # Get Median of Mask
        # d_idx = np.where(depth_mask > 0)
        # d = depth_mask[d_idx].astype(float)
        # depth_median = np.median(d).astype("uint16")
        # self.get_logger().info(f"depth median, {depth_median}")
        
        # Fit Plane: depth = a*u + b*v + c
        d_idx = np.where(depth_mask > 0)
        d = depth_mask[d_idx].astype(float)
        # d = np.full(len(d_idx[0]), depth_median)
        # self.get_logger().info(f"d, {d}, {d.shape}")
        coeffs = np.hstack((np.vstack(d_idx).T, np.ones((len(d_idx[0]), 1)))).astype(float)
        b, a, c = np.linalg.lstsq(coeffs, d)[0] # coefficients b and a are reversed because of matrix row/col structure and its correspondence to x/y

        # TODO: add a comment describing flipped coefficients a & b
        
        # Create Table Depth Image
        u = np.linspace(0, depth_mask.shape[1], depth_mask.shape[1], False)
        v = np.linspace(0, depth_mask.shape[0], depth_mask.shape[0], False)
        U, V = np.meshgrid(u, v)
        table = a*U + b*V + c
        table = table.astype("uint16")
        # table_depth = a*target_u + b*target_v + c
        # table_depth = table_depth.astype("uint16")

        return table

    def get_orientation(
        self, 
        camera_info: CameraInfo, 
        table_depth: Image, 
        target_u: int,
        target_v: int,
    ):
        # Construct camera intrinsics
        intrinsics = pyrealsense2.intrinsics()
        intrinsics.width = camera_info.width
        intrinsics.height = camera_info.height
        intrinsics.ppx = camera_info.k[2]
        intrinsics.ppy = camera_info.k[5]
        intrinsics.fx = camera_info.k[0]
        intrinsics.fy = camera_info.k[4]
        if camera_info.distortion_model == "plumb_bob":
            intrinsics.model = pyrealsense2.distortion.brown_conrady
            intrinsics.coeffs = list(camera_info.d)
        elif camera_info.distortion_model == "equidistant":
            intrinsics.model = pyrealsense2.distortion.kannala_brandt4
            intrinsics.coeffs = list(camera_info.d)
        else:
            logger.warning(
                f"Unsupported camera distortion model: {camera_info.distortion_model}"
            )
            intrinsics.model = pyrealsense2.distortion.none
        
        # Calculate real world 3D coordinates of the center pixel of table depth image
        center = pyrealsense2.rs2_deproject_pixel_to_point(
            intrinsics, [target_u, target_v], table_depth[target_v][target_u] / 1000 
        )

        self.get_logger().info(f"table center, {center}")

        return 0
        
    def fit_to_table_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ):
        self.get_logger().info("Entering fit_table callback!")
        # Get the latest RGB image message
        rgb_msg = None
        with self.latest_img_msg_lock:
            rgb_msg = self.latest_img_msg
        
        # Get the latest depth image
        with self.latest_depth_img_msg_lock:
            depth_img_msg = self.latest_depth_img_msg

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Get table depth from camera at the (u, v) coordinate (320, 240)
        table_depth = self.fit_table(
            rgb_msg, depth_img_msg
        )
        
        self.get_logger().info(f"table depth, {table_depth}, {type(table_depth)}")
        
        # Get the camera matrix
        cam_info = None
        with self.camera_info_lock:
            camera_matrix = self.camera_matrix
            if self.camera_info is not None:
                cam_info = self.camera_info
            else:
                self.get_logger().warn(
                    "Camera info not received, not including in result message"
                )
        
        orientation = self.get_orientation(
            cam_info, table_depth, 320, 240
        )

        self.get_logger().info(f"orientation, {orientation}, {type(orientation)}")

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
            self.camera_matrix = msg.k
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

    rclpy.init()

    table_detection = TableDetectionNode()

    rclpy.spin(table_detection)
    
if __name__ == "__main__":
    main()

		
