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
from ada_feeding_perception.segment_from_point import (
    initialize_sam,
    initialize_efficient_sam,
    generate_mask_msg,
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

    def read_params(
        self,
    ) -> Tuple[Parameter, Parameter, Parameter, Parameter, Parameter]:
        """
        Read the parameters for this node.

        Returns
        -------
        """
        (
            sam_model_name,
            sam_model_base_url,
            efficient_sam_model_name,
            efficient_sam_model_base_url,
            groundingdino_model_path,
            model_dir,
            use_efficient_sam,
            n_contender_masks,
            rate_hz,
            min_depth_mm,
            max_depth_mm,
        ) = self.declare_parameters(
            "",
            [
                (
                    "sam_model_name",
                    None,
                    ParameterDescriptor(
                        name="sam_model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the model checkpoint to use for SAM",
                        read_only=True,
                    ),
                ),
                (
                    "sam_model_base_url",
                    None,
                    ParameterDescriptor(
                        name="sam_model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The URL to download the model checkpoint from if "
                            "it is not already downloaded for SAM"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "efficient_sam_model_name",
                    None,
                    ParameterDescriptor(
                        name="efficient_sam_model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the model checkpoint to use for EfficientSAM",
                        read_only=True,
                    ),
                ),
                (
                    "efficient_sam_model_base_url",
                    None,
                    ParameterDescriptor(
                        name="efficient_sam_model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The URL to download the model checkpoint from if "
                            "it is not already downloaded for EfficientSAM"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "groundingdino_model_path",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_model_path",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the model checkpoint to use for Open-GroundingDINO",
                        read_only=True,
                    ),
                ),
                (
                    "model_dir",
                    None,
                    ParameterDescriptor(
                        name="model_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The location of the directory where the model "
                            "checkpoint is / should be stored"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "use_efficient_sam",
                    True,
                    ParameterDescriptor(
                        name="use_efficient_sam",
                        type=ParameterType.PARAMETER_BOOL,
                        description=("Whether to use EfficientSAM or SAM"),
                        read_only=True,
                    ),
                ),
                (
                    "n_contender_masks",
                    3,
                    ParameterDescriptor(
                        name="n_contender_masks",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="The number of contender masks to return per point.",
                        read_only=True,
                    ),
                ),
                (
                    "rate_hz",
                    10.0,
                    ParameterDescriptor(
                        name="rate_hz",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description="The rate at which to return feedback.",
                        read_only=True,
                    ),
                ),
                (
                    "min_depth_mm",
                    330,
                    ParameterDescriptor(
                        name="min_depth_mm",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="The minimum depth in mm to consider in a mask.",
                        read_only=True,
                    ),
                ),
                (
                    "max_depth_mm",
                    10150000,
                    ParameterDescriptor(
                        name="max_depth_mm",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="The maximum depth in mm to consider in a mask.",
                        read_only=True,
                    ),
                ),
            ],
        )

        if use_efficient_sam.value:
            seg_model_name = efficient_sam_model_name.value
            seg_model_base_url = efficient_sam_model_base_url.value
        else:
            seg_model_name = sam_model_name.value
            seg_model_base_url = sam_model_base_url.value

        return (
            seg_model_name,
            seg_model_base_url,
            groundingdino_model_path.value,
            model_dir.value,
            use_efficient_sam.value,
            n_contender_masks.value,
            rate_hz.value,
            min_depth_mm.value,
            max_depth_mm.value,
        )
    
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
    rclpy.init(args=args)

    segment_all_items = SegmentAllItemsNode()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=5)

    rclpy.spin(segment_all_items, executor=executor)


if __name__ == "__main__":
    main()


        

