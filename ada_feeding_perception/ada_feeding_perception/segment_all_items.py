"""
This file defines the SegmentAllItems class, which launches an action server that
takes a list of labels describing food items on a plate and returns segmentation masks
of all food items in the latest image and defines each segmentation with a semantic label
using a pipeline of foundation models including GPT-4o, GroundingDINO, and SegmentAnything.
"""

# Standard imports
import os
import threading
from typing import Optional, Tuple, Union, Dict, List

# Third-party imports
import cv2
import time
import random
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap, clean_state_dict
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, RegionOfInterest
import torch
from torchvision import transforms
from PIL import Image as ImagePIL
from copy import deepcopy
import base64
from openai import OpenAI
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
import base64
import yaml
import json
from rosidl_runtime_py.convert import message_to_ordereddict
# from rosidl_runtime_py.utilities import convert_message_to_ordereddict  # requires ROS2

# Local imports
from ada_feeding_msgs.action import SegmentAllItems, GenerateCaption, SegmentFromBox, AcquireFood 
from ada_feeding_msgs.msg import Mask
from ada_feeding_perception.helpers import (
    BoundingBox,
    crop_image_mask_and_point,
    cv2_image_to_ros_msg,
    download_checkpoint,
    get_connected_component,
    get_img_msg_type,
    ros_msg_to_cv2_image,
)
from ada_feeding_perception.ada_feeding_perception_node import ADAFeedingPerceptionNode


class SegmentAllItemsNode(Node):
    """
    The SegmentAllItemsNode launches an action server that segments all food
    items in the latest image and defines each segmentation with a semantic
    label using GPT-4o, GroundingDINO, and SegmentAnything.
    """

    def __init__(self, node: ADAFeedingPerceptionNode):
        """
        Initialize the SegmentAllItemsNode.

        Parameters
        ----------
        node: The ADAFeedingPerceptionNode.
              The node that contains all functionality to get camera images (RGB and depth)
              and camera info.
        """
        self._node = node

        # Check if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load environment variables from the .env file
        load_dotenv()

        # Load the parameters'
        (
            seg_model_name,
            seg_model_base_url,
            groundingdino_config_name,
            groundingdino_model_name,
            groundingdino_model_base_url,
            model_dir,
            self.use_efficient_sam,
            self.rate_hz,
            self.box_threshold,
            self.text_threshold,
            self.min_depth_mm,
            self.max_depth_mm,
            self.viz_groundingdino,
        ) = self.read_params()

        # Download the checkpoint for SAM/EfficientSAM if it doesn't exist
        seg_model_path = os.path.join(model_dir, seg_model_name)
        if not os.path.isfile(seg_model_path):
            self._node.get_logger().info(
                "Model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(seg_model_name, model_dir, seg_model_base_url)
            self._node.get_logger().info(
                f"Model checkpoint downloaded {seg_model_path}."
            )

        # Download the checkpoint for GroundingDINO if it doesn't exist
        groundingdino_model_path = os.path.join(model_dir, groundingdino_model_name)
        if not os.path.isfile(groundingdino_model_path):
            self._node.get_logger().info(
                "Model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(
                groundingdino_model_name, model_dir, groundingdino_model_base_url
            )
            self._node.get_logger().info(
                f"Model checkpoint downloaded {groundingdino_model_path}."
            )

        # Set the path to the GroundingDINO configurations file in the model directory
        groundingdino_config_path = os.path.join(model_dir, groundingdino_config_name)

        # Subscribe to the camera info topic, to get the camera intrinsics
        self.camera_info_topic = "~/camera_info"
        self.camera_info = None
        self._node.add_subscription(
            CameraInfo,
            self.camera_info_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the aligned depth image topic, to store the latest depth image
        # NOTE: We assume this is in the same frame as the RGB image
        self.aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(self.aligned_depth_topic, self._node)
        except ValueError as err:
            self._node.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to Image. {err}"
            )
            aligned_depth_type = Image
        # Subscribe to the depth image
        self._node.add_subscription(
            aligned_depth_type,
            self.aligned_depth_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the RGB image topic, to store the latest image
        self.rgb_image_topic = "~/image"
        try:
            image_type = get_img_msg_type(self.rgb_image_topic, self._node)
        except ValueError as err:
            self._node.get_logger().error(
                f"Error getting type of image topic. Defaulting to CompressedImage. {err}"
            )
            image_type = CompressedImage
        self._node.add_subscription(
            image_type,
            self.rgb_image_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Initialize GroundingDINO
        self.initialize_grounding_dino(
            groundingdino_config_path, groundingdino_model_path
        )

        # Initialize Segment Anything
        if self.use_efficient_sam:
            self.initialize_efficient_sam(seg_model_name, seg_model_path)
        else:
            self.initialize_sam(seg_model_name, seg_model_path)

        # Initialize the labels list to store the input labels
        self.labels_list: list[str] = []

        # Initialize BERT for label embedding
        self.bert_tokenizer, self.bert_model = self.initialize_bert()

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the shared resource to ensure that the action server rejects all
        # goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the Action Server to invoke GPT-4o
        # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
        self._gpt4o_action_server = ActionServer(
            self._node,
            GenerateCaption,
            "GenerateCaption",
            execute_callback=self.invoke_gpt4o_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the Action Server.
        # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
        self._action_server = ActionServer(
            self._node,
            SegmentAllItems,
            "SegmentAllItems",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the Action Server to perform segmentation from a bounding box
        # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
        self._segment_action_server = ActionServer(
            self._node,
            SegmentFromBox,
            "SegmentFromBox",
            execute_callback=self.segmentation_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # If the GroundingDINO results visualization flage is set, then a publisher
        # is created to visualize the bounding box predictions of GroundingDINO
        if self.viz_groundingdino:
            self.viz_groundingdino_pub = self._node.create_publisher(
                Image, "~/groundingdino_detection", 1
            )

        # Initialize the OpenAI API and load environment variables
        API_KEY = ""
        self.openai = OpenAI(api_key=API_KEY)

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
            groundingdino_config_name,
            groundingdino_model_name,
            groundingdino_model_base_url,
            model_dir,
            use_efficient_sam,
            rate_hz,
            box_threshold,
            text_threshold,
            min_depth_mm,
            max_depth_mm,
            viz_groundingdino,
        ) = self._node.declare_parameters(
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
                    "groundingdino_config_name",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_config_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the configuration file to use for GroundingDINO",
                        read_only=True,
                    ),
                ),
                (
                    "groundingdino_model_name",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the model checkpoint to use for GroundingDINO",
                        read_only=True,
                    ),
                ),
                (
                    "groundingdino_model_base_url",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The URL to download the model checkpoint from if "
                            "it is not already downloaded for GroundingDINO"
                        ),
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
                    "box_threshold",
                    0.30,
                    ParameterDescriptor(
                        name="box_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description="The lower threshold for the bounding box detections"
                        + "by GroundingDINO.",
                        read_only=True,
                    ),
                ),
                (
                    "text_threshold",
                    0.25,
                    ParameterDescriptor(
                        name="text_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description="The lower threshold for the text detections"
                        + "by GroundingDINO.",
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
                (
                    "viz_groundingdino",
                    True,
                    ParameterDescriptor(
                        name="viz_groundingdino",
                        type=ParameterType.PARAMETER_BOOL,
                        description="Whether to visualize the bounding box"
                        + "predictions of GroundingDINO.",
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
            groundingdino_config_name.value,
            groundingdino_model_name.value,
            groundingdino_model_base_url.value,
            model_dir.value,
            use_efficient_sam.value,
            rate_hz.value,
            box_threshold.value,
            text_threshold.value,
            min_depth_mm.value,
            max_depth_mm.value,
            viz_groundingdino.value,
        )

    def initialize_grounding_dino(
        self, groundingdino_config_path: str, groundingdino_model_path: str
    ) -> None:
        """
        Initialize the GroundingDINO model.

        Parameters
        ----------
        groundingdino_config_path: The path to the GroundingDINO configuration file.
        groundingdino_model_path: The path to the GroundingDINO model checkpoint.
        """
        self._node.get_logger().info("Initializing GroundingDINO...")

        # Get model configuration arguments from the configuration file
        config_args = SLConfig.fromfile(groundingdino_config_path)
        config_args.device = self.device
        groundingdino = build_model(config_args)

        # Load the GroundingDINO model checkpoint
        checkpoint = torch.load(groundingdino_model_path, map_location=self.device)
        load_log = groundingdino.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        self._node.get_logger().info(f"Loaded model checkpoint: {load_log}")
        _ = groundingdino.eval()
        self.groundingdino = groundingdino.to(device=self.device)

        self._node.get_logger().info("...Done!")

    def initialize_sam(self, model_name: str, model_path: str) -> None:
        """
        Initialize all attributes needed for food segmentation with SAM.

        This includes loading the SAM, launching the action
        server, and more. Note that we are guarenteed the model exists since
        it was downloaded in the __init__ function of this class.

        Parameters
        ----------
        model_name: The name of the model to load.
        model_path: The path to the model checkpoint to load.

        Raises
        ------
        ValueError if the model name does not contain vit_h, vit_l, or vit_b
        """
        self._node.get_logger().info("Initializing SAM...")
        # Load the model and move it to the specified device
        if "vit_b" in model_name:  # base model
            model_type = "vit_b"
        elif "vit_l" in model_name:  # large model
            model_type = "vit_l"
        elif "vit_h" in model_name:  # huge model
            model_type = "vit_h"
        else:
            raise ValueError(f"Unknown model type {model_name}")
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=self.device)

        # Create the predictor
        # NOTE: If we allow for concurrent goals, this should be protected by
        # a lock.
        self.sam = SamPredictor(sam)

        self._node.get_logger().info("...Done!")

    def initialize_efficient_sam(self, model_name: str, model_path: str) -> None:
        """
        Initialize all attributes needed for food segmentation with EfficientSAM.

        This includes loading the EfficientSAM model, launching the action
        server, and more. Note that we are guarenteed the model exists since
        it was downloaded in the __init__ function of this class.

        Parameters
        ----------
        model_name: The name of the model to load.
        model_path: The path to the model checkpoint to load.

        Raises
        ------
        ValueError if the model name does not contain efficient_sam
        """
        self._node.get_logger().info("Initializing EfficientSAM...")
        # Hardcoded from https://github.com/yformer/EfficientSAM/blob/main/efficient_sam/build_efficient_sam.py
        if "vits" in model_name:
            encoder_patch_embed_dim = 384
            encoder_num_heads = 6
        elif "vitt" in model_name:
            encoder_patch_embed_dim = 192
            encoder_num_heads = 3
        else:
            raise ValueError(f"Unknown model type {model_name}")
        self.efficient_sam = build_efficient_sam(
            encoder_patch_embed_dim=encoder_patch_embed_dim,
            encoder_num_heads=encoder_num_heads,
            checkpoint=model_path,
        ).eval()
        self.efficient_sam.to(device=self.device)

        self._node.get_logger().info("...Done!")

    def initialize_bert(self) -> Tuple[BertTokenizer, BertModel]:
        """
        Load the BERT model for label embedding.

        Returns
        -------
        tokenizer (BertTokenizer): The BERT tokenizer.
        model (BertModel): The BERT model.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        return tokenizer, model

    def goal_callback(self, goal_request: SegmentAllItems.Goal) -> GoalResponse:
        """
        Accept or reject the goal request based on the availability of the latest
        RGB and depth images.

        Parameters
        ----------
        goal_request: The goal request.
        """
        # If no RGB image is received, reject the goal request
        self._node.get_logger().info("Received goal request...")
        latest_rgb_img_msg = self._node.get_latest_msg(self.rgb_image_topic)
        if latest_rgb_img_msg is None:
            self._node.get_logger().info(
                "Rejecting goal request because no color image was received"
            )
            return GoalResponse.REJECT

        # If no depth image is received, reject the goal request
        latest_depth_img_msg = self._node.get_latest_msg(self.aligned_depth_topic)
        if latest_depth_img_msg is None:
            self._node.get_logger().info(
                "Rejecting goal request because no depth image was received"
            )
            return GoalResponse.REJECT

        # Accept the goal request is there isn't already an active one,
        # otherwise reject it
        with self.active_goal_request_lock:
            if self.active_goal_request is None:
                self._node.get_logger().info("Accepting goal request")
                self.active_goal_request = goal_request
                return GoalResponse.ACCEPT
            self._node.get_logger().info(
                "Rejecting goal request because there is already an active one"
            )
            return GoalResponse.REJECT

    def cancel_callback(self, _: ServerGoalHandle) -> CancelResponse:
        """
        Always accept the cancel request, however, 'execute_callback'
        will wait for segmentation to complete and not interrupt the process
        in response to a cancel request.

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self._node.get_logger().info("Cancelling the goal request...")
        return CancelResponse.ACCEPT

    def run_gpt4o(self, image_msg: Image, labels_list: list[str]):
        """
        Run GPT-4o on the image.

        Parameters
        ----------
        image_msg: An above the plate image (as a ROS image message) that is used as
                reference for GPT-4o to generate a caption input for GroundingDINO.
        labels_list: The list of food items to compile into a sentence prompt.

        Returns
        -------
        vlm_query: The caption generated by GPT-4o that is used as text input for
                   GroundingDINO.
        """
        self._node.get_logger().info("Running GPT-4o...")

        # Convert the image message to a CV2 image
        image = ros_msg_to_cv2_image(image_msg)

        # Encode the image to JPEG format
        _, buffer = cv2.imencode(".jpg", image)

        # Convert the buffer to the base 64 encoded image for GPT-4o
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Write the system and user prompts for GPT-4o
        system_query = f"""
            You are a prompt engineer that is assigned to describe items
            in an image you have been queried so that a vision language model
            can take in your prompt as a query and use it to for classification
            tasks. You respond in string format and do not provide any explanation
            for your responses.
        """
        user_query = f"""
            Your objective is to generate a sentence prompt that describes the food
            items on a plate in an image.
            You are given an image of a plate with food items and a list of the food items
            on the plate.
            Please compile the inputs from the list into a sentence prompt that effectively
            lists the food items on the plate.
            Add qualifiers to the prompt to better visually describe the food for the VLM
            to identify. Don't add any irrelevant qualifiers.
            Ensure that the words in the input list are included in the prompt and do not
            get altered in the sentence. For instance the word "apple" should not be changed
            to "apples" if "apple" is provided in the input list.

            Here is the input list of food items to compile into a string: {labels_list}

            Here are some sample responses that convey how you should format your responses:

            Food items including red strawberry fruit, green kiwi slices, yellow banana slices,
            and orange carrot sticks on a plate.

            Food items including baked chicken pieces, black olive pieces, bell pepper slices,
            and artichoke on a plate.
        """

        # Run GPT-4o to generate a sentence caption given the user query, system query,
        # and image prompt
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_query},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
        )

        # Get the caption generated by GPT-4o
        vlm_query = response.choices[0].message.content

        # Define the result and create result message header
        result = GenerateCaption.Result()
        result.header = image_msg.header
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            result.camera_info = self.camera_info
        else:
            self._node.get_logger().warn(
                "Camera info not received, not including in result message"
            )

        # Set the caption generated by GPT-4o as the result caption
        result.caption = vlm_query

        return result

    def run_sam(
        self,
        image: npt.NDArray,
        seed_point: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        prompt: int,
    ):
        """
        Run SAM on the image.

        Parameters
        ----------
        image: The image to perform segmentation on.
        seed_point: The seed point for SAM to segment from.
        bbox: The bounding box prompt for SAM to segment.
        prompt: The prompt to use for SAM. If 0, use the seed point prompt.
                If 1, use the bounding box prompt.

        Returns
        -------
        masks: The masks for each segmentation.
        scores: The confidence scores for each segmentation.
        """
        self._node.get_logger().info("Segmenting image with SAM...")

        # Convert image from BGR to RGB for Segment Anything
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run SAM on the image using the input prompt
        self.sam.set_image(image)
        if prompt == 0:
            masks, scores, _ = self.sam.set_seed_point(
                point_coords=seed_point,
                point_labels=[1],
                multimask_output=True,
            )
        else:
            masks, scores, _ = self.sam.set_bbox(
                box=bbox,
                multimask_output=True,
            )

        return masks, scores

    def run_efficient_sam(
        self,
        image: npt.NDArray,
        seed_point: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        prompt: int,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Run EfficientSAM on the image.

        Parameters
        ----------
        image: The image to perform segmentation on.
        seed_point: The seed point for EfficientSAM to segment from.
        bbox: The bounding box prompt for EfficientSAM to segment.
        prompt: The prompt to use for EfficientSAM. If 0, use the seed point prompt.
                If 1, use the bounding box prompt.

        Returns
        -------
        masks: The masks for each segmentation.
        scores: The confidence scores for each segmentation.
        """
        self._node.get_logger().info("Segmenting image with EfficientSAM...")

        # Convert image from BGR to RGB for Segment Anything
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a tensor
        image_tensor = transforms.ToTensor()(image).to(self.device)

        # Convert input prompt (seed point or bounding box) to a tensor
        if prompt == 0:
            prompt_tensor = torch.tensor(np.array(seed_point).reshape((1, 1, 1, 2))).to(
                device=self.device
            )

            # Define the labels for the input prompt
            prompt_labels = torch.tensor([[[1]]]).to(device=self.device)
        else:
            prompt_tensor = torch.reshape(torch.tensor(bbox), [1, 1, 2, 2]).to(
                self.device
            )

            # Define the labels for the input prompt
            prompt_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2]).to(
                self.device
            )

        # Run EfficientSAM on the image using the input prompt
        predicted_logits, predicted_iou = self.efficient_sam(
            image_tensor[None, ...],
            prompt_tensor,
            prompt_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        masks = torch.ge(predicted_logits[0, 0, :, :, :], 0).cpu().detach().numpy()
        scores = predicted_iou[0, 0, :].cpu().detach().numpy()

        return masks, scores
    
    def non_max_suppression(self, predictions, iou_threshold: float = 0.5):
        boxes = [p["box"] for p in predictions]
        conf_scores = torch.tensor([p["confidence"] for p in predictions])
        phrases = [p["phrase"] for p in predictions]

        # Extract coordinates from predicted boxes
        x0 = torch.tensor([box[0] for box in boxes])
        y0 = torch.tensor([box[1] for box in boxes])
        x1 = torch.tensor([box[2] for box in boxes])
        y1 = torch.tensor([box[3] for box in boxes])

        # Calculate areas of predicted boxes and sort by confidence 
        areas = (x1 - x0) * (y1 - y0)
        indices = conf_scores.argsort()

        # Initialize a list to hold the selected boxes
        selected_boxes = []

        while len(indices) > 0:
            # Select the box with the highest confidence score
            current_index = indices[-1]
            selected_boxes.append({"box": boxes[current_index], "phrase": phrases[current_index], "confidence": conf_scores[current_index]})

            indices = indices[:-1]

            # Select coordinates of boxes to compare with the current box
            x0_coords = torch.index_select(x0, 0, indices)
            y0_coords = torch.index_select(y0, 0, indices)
            x1_coords = torch.index_select(x1, 0, indices)
            y1_coords = torch.index_select(y1, 0, indices)

            # Determine coordinates of intersection boxes
            inter_x0 = torch.max(x0[current_index], x0_coords)
            inter_y0 = torch.max(y0[current_index], y0_coords)
            inter_x1 = torch.min(x1[current_index], x1_coords)
            inter_y1 = torch.min(y1[current_index], y1_coords)

            # Calculate heights and widths of the intersection boxes
            widths = inter_x1 - inter_x0
            heights = inter_y1 - inter_y0

            # Clamp the heights and width to avoid negative values
            widths = torch.clamp(widths, min=0.0)
            heights = torch.clamp(heights, min=0.0)

            # Calculate the intersection area
            inter_area = widths * heights

            # Calculate the regular areas of the remaining boxes
            areas_remaining = torch.index_select(areas, 0, indices)

            # Find the union ares of the predicted boxes in the current iteration
            # with the selected box
            union_area = areas_remaining + areas[current_index] - inter_area

            # Find IoU of predicted boxes with the selected box
            iou = inter_area / union_area

            # Remove boxes with IoU below the threshold
            indices = indices[iou < iou_threshold]

        return selected_boxes
    
    def get_label_embedding(self, label: str, tokenizer: BertTokenizer, model: BertModel) -> torch.Tensor:
        """
        Get the label embedding for a given label using BERT.

        Parameters
        ----------
        label (str): The label for which to get the embedding.
        tokenizer (BertTokenizer): The BERT tokenizer.
        model (BertModel): The BERT model.

        Returns
        -------
        embedding (torch.Tensor): The BERT embedding of the label.
        """
        # Tokenize and encode the label
        inputs = tokenizer(label, return_tensors="pt")
        
        # Get the embeddings from BERT
        with torch.no_grad():
            outputs = model(**inputs)
        
        final_hidden_state = outputs.last_hidden_state
        label_embedding = torch.tensor(final_hidden_state.mean(dim=1))
        
        # Return the last hidden state as the embedding
        return label_embedding
    
    def nearest_label_interpolation(self, predictions: Dict, gt_labels: List[str], tokenizer: BertTokenizer, model: BertModel) -> Dict:
        """
        Interpolate predicted labels to match ground truth labels by matching
        labels together based on their embedding similarity. Then, create a new
        prediction dictionary (label to masks) with the interpolated labels as keys. 
        
        Parameters
        ----------
        predictions (Dict): Dictionary of predicted masks.
        gt_labels (List[str]): List of ground truth labels.

        Returns
        -------
        interpolated_predictions (Dict): Dictionary of interpolated predictions.
        """
        if predictions is None or len(predictions) == 0:
            print("No predictions found.")
            return {}
        
        # Initialize the interpolated predictions dictionary
        interpolated_predictions = {}

        # Convert predicted labels to BERT embeddings
        pred_labels = list(predictions.keys())
        #print(f"Predicted labels: {pred_labels}")
        pred_embeddings = [self.get_label_embedding(label, tokenizer, model) for label in pred_labels]
        pred_embeddings = torch.stack(pred_embeddings)

        # Convert ground truth labels to BERT embeddings
        gt_embeddings = [self.get_label_embedding(label, tokenizer, model) for label in gt_labels]
        gt_embeddings = torch.stack(gt_embeddings)

        # Calculate the cosine similarity between predicted and ground truth embeddings
        similarity_matrix = F.cosine_similarity(pred_embeddings.unsqueeze(1), gt_embeddings.unsqueeze(0), dim=-1)
        #print(f"Similarity matrix shape: {similarity_matrix.shape}")

        # Find the nearest ground truth label for each predicted label
        nearest_labels = torch.max(similarity_matrix, dim=1)[1]
        old_to_new = {pred_labels[i]: gt_labels[nearest_labels[i]] for i in range(len(pred_labels))}

        # Create the new prediction dictionary with interpolated labels
        for pred_label, boxes in predictions.items():
            new_label = old_to_new[pred_label]
            if new_label not in interpolated_predictions:
                interpolated_predictions[new_label] = []
            interpolated_predictions[new_label].extend(boxes)

        return interpolated_predictions

    def run_grounding_dino(
        self,
        image: npt.NDArray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ):
        """
        Run GroundingDINO on the image.

        Parameters
        ----------
        image: The CV2 above the plate image to retrieve semantically labeled bounding
               boxes from.
        caption: The caption to use as text input for GroundingDINO.
        box_threshold: The threshold for the bounding box.
        text_threshold: The threshold for the text.

        Returns
        -------
        bbox_predictions: A dictionary containing the bounding boxes for each food item label
                        detected from the image.
        """
        self._node.get_logger().info("Running GroundingDINO...")

        # Set the initial time to measure the elapsed time running GroundingDINO on the
        # desired image and text prompts.
        inference_time = time.time()

        # Preprocess the image for GroundingDINO detection
        _, image_transformed = self.load_image(image)

        # Lowercase and strip the caption
        caption = caption.lower().strip()

        # Run GroundingDINO on the image using the input caption
        image_transformed = image_transformed.to(device=self.device)
        with torch.no_grad():
            outputs = self.groundingdino(
                image_transformed[None],
                captions=[caption],
            )
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        # Filter the output based on the box and text thresholds
        boxes_cxcywh = {}
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_thresh_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_thresh_mask]
        boxes_filt = boxes_filt[filt_thresh_mask]

        # Tokenize the caption
        tokenizer = self.groundingdino.tokenizer
        caption_tokens = tokenizer(caption)
            
        # Build the dictionary of bounding boxes for each food item label detected
        for logit, box in zip(logits_filt, boxes_filt):
            # Predict phrases based on the bounding boxes and the text threshold
            phrase = get_phrases_from_posmap(logit > text_threshold, caption_tokens, tokenizer)
            confidence_score = logit.max().item()

            #print(f"{phrase}")
            if phrase not in boxes_cxcywh:
                boxes_cxcywh[phrase] = []
            #boxes_cxcywh[phrase].append(box.cpu().numpy())
            boxes_cxcywh[phrase].append({"box": box.cpu().numpy(), "confidence": confidence_score})

        # Define height and width of image
        if isinstance(image, torch.Tensor):
            _, height, width = image.shape
        else:
            height, width, _ = image.shape

        # Convert the bounding boxes outputted by GroundingDINO to the following format
        # [top left x-value, top left y-value, bottom right x-value, bottom right y-value]
        # and unnormalize the bounding box coordinate values
        # boxes_xyxy = {}
        box_list = []
        for phrase, boxes in boxes_cxcywh.items():
            # boxes_xyxy[phrase] = []
            for box in boxes:
                # Scale the box from percentage values to pixel values
                scaled_box = np.multiply(box["box"], np.array([width, height, width, height]))
                logit = box["confidence"]
                center_x, center_y, w, h = scaled_box

                # Get the bottom left and top right coordinates of the box
                x0 = center_x - (w / 2)
                y0 = center_y - (h / 2)
                x1 = x0 + w
                y1 = y0 + h
                # boxes_xyxy[phrase].append([x0, y0, x1, y1])
                box_list.append({"phrase": phrase, "box": [x0, y0, x1, y1], "confidence": logit})

        # Measure the elapsed time running GroundingDINO on the image prompt
        inference_time = int(round((time.time() - inference_time) * 1000))
        self._node.get_logger().info(f"GroundingDINO inference time: {inference_time} ms")
        start_time = time.time()

        # Perform non-maximum suppression to remove overlapping bounding boxes
        # and select the most confident bounding boxes
        selected_boxes = self.non_max_suppression(box_list, iou_threshold=0.8)
        pre_nms_num_boxes = len(box_list)

        predictions = {}
        for box in selected_boxes:
            phrase = box["phrase"]
            box_coords = box["box"]
            if phrase not in predictions:
                predictions[phrase] = []
            predictions[phrase].append(box_coords)
        post_nms_num_boxes = len(selected_boxes)

        self._node.get_logger().info(f"Pre-NMS: {pre_nms_num_boxes}, Post-NMS: {post_nms_num_boxes}")

        # Measure elapsed time of running non-maximum suppression
        nms_time = int(round((time.time() - start_time) * 1000))
        self._node.get_logger().info(f"Non-maximum suppression time: {nms_time} ms")

        # Interpolate the predictions for each image to the nearest label
        # interpolated_predictions = {}
        # interpolated = self.nearest_label_interpolation(predictions, self.labels_list, self.bert_tokenizer, self.bert_model)
        # interpolated_predictions = interpolated

        return predictions

    def load_image(self, image_array: npt.NDArray):
        """
        Load the image and apply transformations to it.

        Parameters
        ----------
        image_array: The image to load.

        Returns
        -------
        image_pil: The image in Image pillow format.
        image: The image in tensor format.
        """
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Convert image to image pillow to apply transformation
        image_pil = ImagePIL.fromarray(image_array, mode="RGB")
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w

        return image_pil, image

    def generate_mask_msg(
        self,
        item_id: str,
        object_id: str,
        score: float,
        mask: npt.NDArray[np.bool_],
        image: npt.NDArray,
        depth_img: npt.NDArray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Mask]:
        """
        Convert a mask detected by EfficientSAM or SAM into a ROS Mask message.

        Parameters
        ----------
        item_id: The item ID of the mask.
        object_id: The object ID of the mask.
        score: The confidence score outputed by the segmentation model for the mask.
        mask: The pixel-wise mask detected.
        image: The image the mask was detected on.
        depth_img: The most recent depth image.
        bbox: The bounding box from GroundingDINO.
        """
        # Calculate center of the bounding box and use it as the seed point for
        # getting the connected component of the mask
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        center_x = int(center_x)
        center_y = int(center_y)
        seed_point = (center_x, center_y)

        # Use the mask to get a connected component containing the seed point
        cleaned_mask = get_connected_component(mask, seed_point)

        # Use the cleaned mask to calculate the median depth over the mask
        masked_depth = depth_img[cleaned_mask]
        median_depth_mm = np.median(
            masked_depth[
                np.logical_and(
                    masked_depth >= self.min_depth_mm, masked_depth <= self.max_depth_mm
                )
            ]
        )
        # If the depth is invalid, skip this mask and return None
        if np.isnan(median_depth_mm):
            self._node.get_logger().warn(
                f"No depth points within [{self.min_depth_mm}, {self.max_depth_mm}] mm range "
                f"for mask {item_id}. Skipping mask."
            )
            return None

        # Convert the bounding box from a Python Tuple into the BoundingBox class
        # from helpers.py to call the crop_image_mask_and_point
        bbox_converted = BoundingBox(
            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        )

        # Crop the image and the mask
        cropped_image, cropped_mask, _ = crop_image_mask_and_point(
            image, cleaned_mask, seed_point, bbox_converted
        )
        cropped_depth, _, _ = crop_image_mask_and_point(
            depth_img, cleaned_mask, seed_point, bbox_converted
        )

        # Convert the mask to an image
        mask_img = np.where(cropped_mask, 255, 0).astype(np.uint8)

        # Create the ROS mask message
        mask_msg = Mask()
        mask_msg.roi = RegionOfInterest(
            x_offset=int(bbox[0]),
            y_offset=int(bbox[1]),
            height=int(bbox[3] - bbox[1]),
            width=int(bbox[2] - bbox[0]),
            do_rectify=False,
        )
        mask_msg.mask = cv2_image_to_ros_msg(mask_img, compress=True)
        mask_msg.rgb_image = cv2_image_to_ros_msg(cropped_image, compress=True)
        mask_msg.depth_image = cv2_image_to_ros_msg(cropped_depth, compress=False)
        mask_msg.average_depth = median_depth_mm / 1000.0
        mask_msg.item_id = item_id
        mask_msg.object_id = object_id
        mask_msg.confidence = float(score)

        return mask_msg
    
    def display_mask(self, image: npt.NDArray, mask: npt.NDArray, item_label: str):
        """
        Display the masks on the image.

        Parameters
        ----------
        image: The image to display the masks on.
        masks: The masks to display on the image.
        """
        self._node.get_logger().info("Displaying masks...")

        # Create a deep copy of the image to visualize
        image_copy = deepcopy(image)

        # Display the mask on the image
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        color_dims = 3
        mask = np.stack([mask] * color_dims, axis=-1)
        color_scalar = random.randint(80, 255)
        mask = np.multiply(mask, color_scalar)
        cv2.imshow(item_label, mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_groundingdino_results(self, image: Image, predictions: dict):
        """
        Visualizes the bounding box predictions of GroundingDINO and then
        publishes the image as a ROS message.

        Parameters
        ----------
        image: The image to visualize.
        predictions: The bounding box predictions of GroundingDINO.
        """
        # Create a deep copy of the image to visualize
        image_copy = deepcopy(image)

        # Define height of image
        height, _, _ = image_copy.shape

        # Draw bounding boxes and text labels for each prediction on the image
        for phrase, boxes in predictions.items():
            for box in boxes:
                x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                color = (0, 255, 0)
                thickness = 6
                image_copy = cv2.rectangle(
                    image_copy, (x0, y0), (x1, y1), color, thickness
                )

                # Display text label below bounding box
                image_copy = cv2.putText(
                    image_copy,
                    phrase,
                    (x0, y0 - 12),
                    0,
                    1e-3 * height,
                    color,
                    thickness // 3,
                )

        cv2.imshow("GroundingDINO Predictions", image_copy)
        cv2.waitKey(0)

        # Publish the image as a ROS message
        self.viz_groundingdino_pub.publish(
            cv2_image_to_ros_msg(image_copy, compress=False, bridge=self.bridge, encoding="bgr8")
        )

    async def run_vision_pipeline(self, image_msg: Image, caption: str):
        """
        Run the vision pipeline consisting of two foundation models, GroundingDINO and
        EfficientSAM, on the image. GroundingDINO is prompted with a caption and the latest image,
        and outputs bounding boxes for each semantic label in the caption detected in the
        image. The detected bounding boxes + semantic label pairs are then segmented by passing
        in the bounding box detections into EfficientSAM which outputs pixel-wise masks for each
        bounding box. The top masks are then returned along with the semantic label for each mask
        as a dictionary.

        Parameters
        ----------
        image_msg: The image to segment, as a ROS image message.
        caption: The caption to use for GroundingDINO containing all the food items
                 detected in the image.

        Returns
        -------
        result: The result message containing masks for all food items detected in the image
                paired with semantic labels.
        """
        self._node.get_logger().info("Running the vision pipeline...")

        # Set the initial time to measure the elapsed time running GroundingDINO on the
        # desired image and text prompts.
        inference_time = time.time()

        # Define the result and create result message header
        result = SegmentAllItems.Result()
        result.header = image_msg.header
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            result.camera_info = self.camera_info
        else:
            self._node.get_logger().warn(
                "Camera info not received, not including in result message"
            )

        # Get the latest depth image and convert the depth image to OpenCV format
        depth_img_msg = self._node.get_latest_msg(self.aligned_depth_topic)
        depth_img = ros_msg_to_cv2_image(depth_img_msg, self.bridge)

        # Convert the image to OpenCV format
        image = ros_msg_to_cv2_image(image_msg, self.bridge)

        # Run GroundingDINO on the image
        bbox_predictions = self.run_grounding_dino(
            image, caption, self.box_threshold, self.text_threshold
        )

        # Publish a visualization of the GroundingDINO predictions, if the visualization
        # flag is set to true
        if self.viz_groundingdino:
            self.visualize_groundingdino_results(image, bbox_predictions)

        # Collect the top contender mask for each food item label detected by
        # GroundingDINO using EfficientSAM and create dictionary of mask
        # predictions from the pipeline
        detected_items = []
        item_labels = []
        masks_list = []
        mask_num = 1
        for phrase, boxes in bbox_predictions.items():
            for box in boxes:
                # Convert the bounding box from a tuple into a RegionOfInterest message
                roi_msg = RegionOfInterest(
                    x_offset=int(box[0]),
                    y_offset=int(box[1]),
                    height=int(box[3] - box[1]),
                    width=int(box[2] - box[0]),
                    do_rectify=False,
                )
                detected_items.append(roi_msg)
                item_labels.append(phrase)
                #masks, scores = self.run_efficient_sam(image, None, box, 1)
                #if len(masks) > 0:
                #    masks_list.append(masks[0])
                #    item_id = f"food_id_{mask_num:d}"
                #    mask_num += 1
                #    mask_msg = self.generate_mask_msg(
                #        item_id, phrase, scores[0], masks[0], image, depth_img, box
                #    )
                #    detected_items.append(mask_msg)
                #    item_labels.append(phrase)

        result.detected_items = detected_items
        result.item_labels = item_labels

        # Measure the elapsed time running GroundingDINO on the image prompt
        inference_time = int(round((time.time() - inference_time) * 1000))
        self._node.get_logger().info(
            f"Approximate Vision Pipeline Inference Time: {inference_time}"
        )

        return result
    
    async def segment_from_bbox(
        self, image_msg: Image, label: str, bbox: Tuple[int, int, int, int]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Segment an image using a bounding box.

        Parameters
        ----------
        image: The image to segment.
        bbox: The bounding box to segment.

        Returns
        -------
        masks: The masks for each segmentation.
        scores: The confidence scores for each segmentation.
        """
        # Get the latest depth image and convert the depth image to OpenCV format
        depth_img_msg = self._node.get_latest_msg(self.aligned_depth_topic)
        depth_img = ros_msg_to_cv2_image(depth_img_msg, self.bridge)

        # Convert the image to OpenCV format
        image = ros_msg_to_cv2_image(image_msg, self.bridge)

        # Run EfficientSAM on the image using the input bounding box prompt
        masks, scores = self.run_efficient_sam(image, None, bbox, 1)
        mask_num = 1
        if len(masks) > 0:
            item_id = f"food_id_{mask_num:d}"
            mask_num += 1
            mask_msg = self.generate_mask_msg(
                item_id, label, scores[0], masks[0], image, depth_img, bbox
            )
            self.display_mask(image, masks[0], label) 

        # Define the result and create result message header
        result = SegmentFromBox.Result()
        result.header = image_msg.header
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            result.camera_info = self.camera_info
        else:
            self._node.get_logger().warn(
                "Camera info not received, not including in result message"
            )

        # Set the mask message as the result
        result.detected_item = mask_msg

        # self._node.get_logger().info(
        #     f"Segmented items: {result}"
        # )
        # # Save the mask message to a file
        filename = f"segmentation_result_{item_id}.txt"
        # with open(filename, "w") as f:
        #     base64_str = base64.b64encode(result).decode("utf-8")
        #     f.write(base64_str)
        # self._node.get_logger().info(f"Saved segmentation result to {filename}")

        # Create an AcquireFood Goal message 
        acquire_goal = AcquireFood.Goal()
        acquire_goal.header = image_msg.header
        acquire_goal.camera_info = result.camera_info
        acquire_goal.detected_food = result.detected_item

        result_dict = message_to_ordereddict(acquire_goal)
        # with open(filename.replace(".txt", ".yaml"), "w") as f:
        #     yaml.dump(result_dict, f, sort_keys=False)
        # Convert OrderedDict to plain dict via JSON round-trip
        clean_dict = json.loads(json.dumps(result_dict))

        # Save to YAML
        filename = f"segmentation_result_{item_id}.yaml"
        with open(filename, "w") as f:
            yaml.dump(clean_dict, f, sort_keys=False)

        with open(filename, 'r') as f:
            yaml_obj = yaml.safe_load(f)

        # Convert to compact JSON string (which ROS CLI accepts)
        goal_str = json.dumps(yaml_obj)
        self._node.get_logger().info(f"Goal string: {goal_str}")

        # Save goal_str to a text file
        cmd_filename = f"acquire_food_goal_{item_id}.txt"
        with open(cmd_filename, "w") as f:
            f.write(goal_str)

        return result

    async def invoke_gpt4o_callback(
        self, goal_handle: ServerGoalHandle
    ) -> GenerateCaption.Result:
        """
        Callback function for the GPT-4o service. This function takes in a list
        of string labels describing the foods on an image as input and returns a
        caption generated by GPT-4o that compiles these labels into a descriptive
        sentence used as a query for GroundingDINO.

        Parameters
        ----------
        request: The given request message.
        response: The created response message.

        Returns
        ----------
        response: The updated response message based on the request.
        """
        self._node.get_logger().info("Received a new goal!")
        starting_time = self._node.get_clock().now()

        # Get the latest image and camera info
        latest_img_msg = self._node.get_latest_msg(self.rgb_image_topic)
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            camera_info = self.camera_info
        else:
            camera_info = None

        # Check if the image and camera info are available
        if latest_img_msg is None or camera_info is None:
            self._node.get_logger().error("Image or camera info not available.")
            return GenerateCaption.Result()

        # Convert the image message to a CV2 image
        image = ros_msg_to_cv2_image(latest_img_msg)

        # Get the input labels from the request
        input_labels = goal_handle.request.input_labels

        # Store the input labels list
        self.labels_list = input_labels
        self._node.get_logger().info(f"Input labels: {self.input_labels}")

        # Create a rate object to control the rate at which to return feedback
        rate = self._node.create_rate(self.rate_hz)

        # Define a cleanup function to destroy the rate
        def cleanup():
            self._node.destroy_rate(rate)

        # Start running the GPT-4o inference as a separate thread
        gpt4o_task = self._node.executor.create_task(
            self.run_gpt4o, latest_img_msg, input_labels
        )

        # Keep publishing feedback (elapsed time) while waiting for
        # the GPT-4o inference to finish
        feedback = GenerateCaption.Feedback()
        while (
            rclpy.ok() and not goal_handle.is_cancel_requested and not gpt4o_task.done()
        ):
            feedback.elapsed_time = (
                self._node.get_clock().now() - starting_time
            ).to_msg()
            goal_handle.publish_feedback(feedback)
            rate.sleep()

        # If there is a cancel request, cancel the GPT-4o task
        if goal_handle.is_cancel_requested:
            self._node.get_logger().info("Goal cancelled.")
            goal_handle.canceled()
            response = GenerateCaption.Result()
            response.status = response.STATUS_CANCELED

            # Cleanup the rate
            cleanup()
            return response

        # Set the result after inference is complete
        self._node.get_logger().info("Goal not cancelled.")
        self._node.get_logger().info("GPT-4o inference completed successfully.")
        response = gpt4o_task.result()
        goal_handle.succeed()
        response.status = response.STATUS_SUCCEEDED

        # Clear the active goal
        with self.active_goal_request_lock:
            self.active_goal_request = None

        # Cleanup the rate
        cleanup()
        return response
    
    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> SegmentAllItems.Result:
        """
        Execute the action server callback.

        Parameters
        ----------
        goal_handle: The goal handle for the action server.

        Returns
        -------
        result: The result message containing masks for all food items detected in the image
                paired with semantic labels.
        """
        self._node.get_logger().info("Received a new goal!")
        starting_time = self._node.get_clock().now()

        # Get the latest image and camera info
        latest_img_msg = self._node.get_latest_msg(self.rgb_image_topic)
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            camera_info = self.camera_info
        else:
            camera_info = None

        # Check if the image and camera info are available
        if latest_img_msg is None or camera_info is None:
            self._node.get_logger().error("Image or camera info not available.")
            return SegmentAllItems.Result()

        # Get the caption from the goal request
        caption = goal_handle.request.caption

        # Create a rate object to control the rate of the vision pipeline
        rate = self._node.create_rate(self.rate_hz)

        # Define a cleanup function to destroy the rate
        def cleanup():
            self._node.destroy_rate(rate)

        # Start running the vision pipeline as a separate thread
        vision_pipeline_task = self._node.executor.create_task(
            self.run_vision_pipeline, latest_img_msg, caption
        )

        # Wait for the vision pipeline to finish and keep publishing
        # feedback (elapsed time) while waiting
        feedback = SegmentAllItems.Feedback()
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not vision_pipeline_task.done()
        ):
            feedback.elapsed_time = (
                self._node.get_clock().now() - starting_time
            ).to_msg()
            goal_handle.publish_feedback(feedback)
            rate.sleep()

        # If there is a cancel request, cancel the vision pipeline task
        if goal_handle.is_cancel_requested:
            self._node.get_logger().info("Goal cancelled.")
            goal_handle.canceled()
            result = SegmentAllItems.Result()
            result.status = result.STATUS_CANCELLED

            # Clear the active goal
            with self.active_goal_request_lock:
                self.active_goal_request = None

            # Cleanup the rate
            cleanup()
            return result

        # Set the result after the task has been completed
        self._node.get_logger().info("Goal not cancelled.")
        self._node.get_logger().info("Vision pipeline completed successfully.")
        result = vision_pipeline_task.result()
        goal_handle.succeed()
        result.status = result.STATUS_SUCCEEDED

        # Clear the active goal
        with self.active_goal_request_lock:
            self.active_goal_request = None

        # Cleanup the rate
        cleanup()
        return result
    
    async def segmentation_callback(
        self, goal_handle: ServerGoalHandle
    ) -> SegmentFromBox.Result:
        """
        """
        self._node.get_logger().info("Received a new goal!")
        starting_time = self._node.get_clock().now()

        # Get the latest image and camera info
        latest_img_msg = self._node.get_latest_msg(self.rgb_image_topic)
        if self.camera_info is None:
            self.camera_info = self._node.get_latest_msg(self.camera_info_topic)
        if self.camera_info is not None:
            camera_info = self.camera_info
        else:
            camera_info = None

        # Check if the image and camera info are available
        if latest_img_msg is None or camera_info is None:
            self._node.get_logger().error("Image or camera info not available.")
            return SegmentFromBox.Result()

        # Get the desired bounding box to segment and its semantic label from the goal request
        roi_msg = goal_handle.request.region_of_interest
        label = goal_handle.request.label

        # Store the top left and bottom right coordinates of the bounding box given
        # the RegionOfInterest message as a list for input to the segmentation model
        bbox_xyxy = [
            roi_msg.x_offset, 
            roi_msg.y_offset, 
            roi_msg.x_offset + roi_msg.width, 
            roi_msg.y_offset + roi_msg.height
        ]

        # Create a rate object to control the rate of the segmentation task
        rate = self._node.create_rate(self.rate_hz)

        # Define a cleanup function to destroy the rate
        def cleanup():
            self._node.destroy_rate(rate)

        # Start running the segmentation task as a separate thread
        segment_from_bbox_task = self._node.executor.create_task(
            self.segment_from_bbox, latest_img_msg, label, bbox_xyxy
        )

        # Publish feedback (elapsed time) until segmentation is complete
        feedback = SegmentFromBox.Feedback()
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not segment_from_bbox_task.done()
        ):
            feedback.elapsed_time = (
                self._node.get_clock().now() - starting_time
            ).to_msg()
            goal_handle.publish_feedback(feedback)
            rate.sleep()

        # If there is a cancel request, cancel the vision pipeline task
        if goal_handle.is_cancel_requested:
            self._node.get_logger().info("Goal cancelled.")
            goal_handle.canceled()
            result = SegmentFromBox.Result()
            result.status = result.STATUS_CANCELLED

            # Clear the active goal
            with self.active_goal_request_lock:
                self.active_goal_request = None

            # Cleanup the rate
            cleanup()
            return result

        # Set the result after the task has been completed
        self._node.get_logger().info("Goal not cancelled.")
        self._node.get_logger().info("Segmentation task completed successfully.")
        result = segment_from_bbox_task.result()
        goal_handle.succeed()
        result.status = result.STATUS_SUCCEEDED

        # Clear the active goal
        with self.active_goal_request_lock:
            self.active_goal_request = None

        # Cleanup the rate
        cleanup()
        return result


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    node = ADAFeedingPerceptionNode("segment_all_items")
    segment_all_items = SegmentAllItemsNode(node)

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=5)

    rclpy.spin(node, executor=executor)

    # Destroy the node
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
