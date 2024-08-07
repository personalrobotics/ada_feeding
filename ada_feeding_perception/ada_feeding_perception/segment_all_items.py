"""
This file defines the SegmentAllItems class, which launches an action server that
segments all food items in the latest image and defines each segmentation with a semantic
label using GPT-4V, GroundingDINO, and Segment Anything.
"""

# Standard imports
import os
import threading
from typing import Optional, Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
from efficient_sam.efficient_sam import build_efficient_sam
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
from segment_anything import sam_model_registry, SamPredictor
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, RegionOfInterest
import torch
from torchvision import transforms

# Local imports
from ada_feeding_perception.action import SegmentAllItems
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

class SegmentAllItemsNode(Node):
    """
    The SegmentAllItemsNode launches an action server that segments all food 
    items in the latest image and defines each segmentation with a semantic
    label using GPT-4V, GroundingDINO, and Segment Anything.
    """

    def __init__(self):
        """
        Initialize the SegmentAllItemsNode.
        """

        super().__init__("segment_all_items")

        # Check if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the parameters'

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

        # Subscribe to the aligned depth image topic, to store the latest depth image
        # NOTE: We assume this is in the same frame as the RGB image
        self.latest_depth_img_msg = None
        self.latest_depth_img_msg_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(aligned_depth_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to CompressedImage. {err}"
            )
            aligned_depth_type = CompressedImage
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

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the Action Server.
        # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
        self._action_server = ActionServer(
            self,
            SegmentAllItems,
            "SegmentAllItems",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        

