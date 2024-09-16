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
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap, clean_state_dict
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, RegionOfInterest
import torch
from torchvision import transforms
from PIL import Image as ImagePIL

# Local imports
from ada_feeding_msgs.action import SegmentAllItems
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
        (
            seg_model_name,
            seg_model_base_url,
            groundingdino_config_name,
            groundingdino_model_name,
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
            self.get_logger().info("Model checkpoint does not exist. Downloading...")
            download_checkpoint(seg_model_name, model_dir, seg_model_base_url)
            self.get_logger().info(f"Model checkpoint downloaded {seg_model_path}.")

        # Set the path to GroundingDINO and its config file in the model directory 
        groundingdino_model_path = os.path.join(model_dir, groundingdino_model_name)
        groundingdino_config_path = os.path.join(model_dir, groundingdino_config_name)

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

        # Initialize GroundingDINO 
        self.initialize_grounding_dino(groundingdino_config_path, groundingdino_model_path)

        # Initialize Segment Anything
        if self.use_efficient_sam:
            self.initialize_efficient_sam(seg_model_name, seg_model_path)
        else:
            self.initialize_sam(seg_model_name, seg_model_path)

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the shared resource to ensure that the action server rejects all
        # goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

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

        # If the GroundingDINO results visualization flage is set, then a publisher
        # is created to visualize the bounding box predictions of GroundingDINO
        if self.viz_groundingdino:
            self.viz_groundingdino_pub = self.create_publisher(
                Image, "~/groundingdino_detection", 1
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
            groundingdino_config_name,
            groundingdino_model_name,
            model_dir,
            use_efficient_sam,
            rate_hz,
            box_threshold,
            text_threshold,
            min_depth_mm,
            max_depth_mm,
            viz_groundingdino,
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
                    "groundingdino_config_name",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_config_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the configuration file to use for Open-GroundingDINO",
                        read_only=True,
                    ),
                ),
                (
                    "groundingdino_model_name",
                    None,
                    ParameterDescriptor(
                        name="groundingdino_model_name",
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
                        description="The lower threshold for the bounding box detections" + 
                                    "by Open-GroundingDINO.",
                        read_only=True,
                    ),
                ),
                (
                    "text_threshold",
                    0.25,
                    ParameterDescriptor(
                        name="text_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description="The lower threshold for the text detections" +
                                    "by Open-GroundingDINO.",
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
                )
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
        Initialize the Open-GroundingDINO model.

        Parameters
        ----------
        groundingdino_config_path: The path to the Open-GroundingDINO configuration file.
        groundingdino_model_path: The path to the Open-GroundingDINO model checkpoint.
        """
        self.get_logger().info("Initializing Open-GroundingDINO...")

        # Get model configuration arguments from the configuration file
        config_args = SLConfig.fromfile(groundingdino_config_path)
        config_args.device = self.device
        groundingdino = build_model(config_args)

        # Load the GroundingDINO model checkpoint
        checkpoint = torch.load(groundingdino_model_path, map_location="cpu")
        load_log = groundingdino.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )
        self.get_logger().info(f"Loaded model checkpoint: {load_log}")
        _ = groundingdino.eval()
        self.groundingdino = groundingdino
        self.groundingdino.to(device=self.device)

        self.get_logger().info("...Done!")

    # Move bottom two functions to helpers file and import them for 
    # both segment_all_items.py and segment_from_point.py    
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
        self.get_logger().info("Initializing SAM...")
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

        self.get_logger().info("...Done!")

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
        self.get_logger().info("Initializing EfficientSAM...")
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

        self.get_logger().info("...Done!")
    
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
    
    def goal_callback(self, goal_request: SegmentAllItems.Goal) -> GoalResponse:
        """
        """
        # If no RGB image is received, reject the goal request
        with self.latest_img_msg_lock:
            if self.latest_img_msg is None:
                self.get_logger().info(
                    "Rejecting goal request because no color image was received"
                )
                return GoalResponse.REJECT

        # If no depth image is received, reject the goal request
        with self.latest_depth_img_msg_lock:
            if self.latest_depth_img_msg is None:
                self.get_logger().info(
                    "Rejecting goal request because no depth image was received"
                )
                return GoalResponse.REJECT
        
        # Accept the goal request is there isn't already an active one,
        # otherwise reject it
        with self.active_goal_request_lock:
            if self.active_goal_request is None:
                self.get_logger().info("Accepting goal request")
                self.active_goal_request = goal_request 
                return GoalResponse.ACCEPT
            self.get_logger().info(
                "Rejecting goal request because there is already an active one"
            )
            return GoalResponse.REJECT
    
    def cancel_callback(self) -> CancelResponse:
        """
        """
        return CancelResponse.ACCEPT

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
        image: The image to segment, in BGR.
        seed_point: The seed point for SAM to segment from.
        bbox: The bounding box prompt for SAM to segment. 
        prompt: The prompt to use for SAM. If 0, use the seed point prompt. 
                If 1, use the bounding box prompt.

        Returns
        -------
        A list of tuples containing the confidence and mask for each segmentation.
        """
        self.get_logger().info("Segmenting image with SAM...")

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
        image: The image to segment, in BGR.
        seed_point: The seed point for EfficientSAM to segment from.
        bbox: The bounding box prompt for EfficientSAM to segment. 
        prompt: The prompt to use for EfficientSAM. If 0, use the seed point prompt. 
                If 1, use the bounding box prompt.

        Returns
        -------
        masks: The masks for each segmentation.
        scores: The confidence scores for each segmentation.
        """
        self.get_logger().info("Segmenting image with EfficientSAM...")

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
            prompt_tensor = torch.reshape(torch.tensor(bbox), [1, 1, 2, 2]).to(self.device)

            # Define the labels for the input prompt
            prompt_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2]).to(self.device)

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
    
    def run_grounding_dino(
        self, 
        image: npt.NDArray, 
        caption: str, 
        box_threshold: int, 
        text_threshold: int, 
    ):
        """
        Run Open-GroundingDINO on the image.

        Parameters
        ----------
        image: The image to retrieve semantically labeled bounding boxes from, in BGR.
        caption: The caption to use for Open-GroundingDINO.
        box_threshold: The threshold for the bounding box.
        text_threshold: The threshold for the text.

        Returns
        -------
        bbox_predictions: A dictionary containing the bounding boxes for each food item label 
                        detected from the image.
        """
        self.get_logger().info("Running Open-GroundingDINO...")

        # Convert image to Image pillow
        image_pil = self.load_image(image)

        # Lowercase and strip the caption
        caption = caption.lower().strip()

        # Run Open-GroundingDINO on the image using the input caption
        image.to(device=self.device)
        with torch.no_grad():
            outputs = self.groundingdino(
                image_pil[None],
                captions=[caption],
            )
            logits = outputs["pred_logits"].sigmoid()[0]
            boxes = outputs["pred_boxes"][0]
        
        # Filter the output based on the box and text thresholds
        bbox_predictions = {}
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
            if logit.max() > text_threshold:
                bbox_predictions[phrase].append(box.cpu().numpy())

        return bbox_predictions

    def load_image(image_array: npt.NDArray):
        # Convert image to image pillow to apply transformation
        image_pil = ImagePIL.fromarray(image_array) 

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
        score: float, 
        mask:npt.NDArray[np.bool_], 
        image: npt.NDArray, 
        depth_img: npt.NDArray, 
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Mask]:
        """
        """
        
        mask_msg = Mask()
        return mask_msg
    
    def visualize_groundingdino_results(self, image: Image, predictions: dict):
        """
        Visualizes the bounding box predictions of GroundingDINO and then 
        publishes the image as a ROS message.

        Parameters
        ----------
        image: The image to visualize.
        predictions: The bounding box predictions of GroundingDINO.
        """
        
        for phrase, boxes in predictions.items():
            for box in boxes:
                # Visualize bounding box
                x0, y0, x1, y1 = box
                color = (0, 255, 0)
                thickness = 6
                cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

                # Display text label below bounding box
                height, _, _ = image.shape()
                cv2.putText(image, phrase, (x0, y0 - 12), 0, 1e-3 * height, color, thickness // 3)

        # Publish the image
        self.viz_groundingdino_pub.publish(
            cv2_image_to_ros_msg(image, compress=False, bridge=self.bridge)
        )
    
    async def run_vision_pipeline(self, image_msg: Image, caption: str):
        """
        Run the vision pipeline consisting of GroundingDINO and EfficientSAM on the image.
        The caption and latest image are prompted to GroundingDINO which outputs bounding boxes
        for each food item label in the caption detected in the image. The detected items are then
        segmented by passing in the bounding box detections into EfficientSAM which outputs
        pixel-wise masks for each bounding box. The top masks are then returned along with the 
        food item label for each mask as a dictionary. 

        Parameters
        ----------
        image: The image to segment, in BGR.
        caption: The caption to use for GroundingDINO containing all the food items 
                 detected in the image.

        Returns
        -------
        
        """
        self.get_logger().info("Running the vision pipeline...")
        # Define the result and create result message header
        result = SegmentAllItems.Result()
        result.header = image_msg.header
        with self.camera_info_lock:
            if self.camera_info is not None:
                result.camera_info = self.camera_info
            else:
                self.get_logger().warn(
                    "Camera info not received, not including in result message"
                )

        # Get the latest depth image and convert the depth image to OpenCV format
        with self.latest_depth_img_msg_lock:
            depth_img_msg = self.latest_depth_img_msg
        depth_img = ros_msg_to_cv2_image(depth_img_msg, self.bridge)
        
        # Convert the image to OpenCV format 
        image = ros_msg_to_cv2_image(image_msg, self.bridge)

        # Run Open-GroundingDINO on the image
        bbox_predictions = self.run_grounding_dino(image, caption, self.box_threshold, self.text_threshold)

        # Publish a visualization of the GroundingDINO predictions, if the visualization
        # flag is set to true 
        if self.viz_groundingdino:
            self.visualize_groundingdino_results(image, bbox_predictions)

        # Collect the top contender mask for each food item label detected by 
        # GroundingDINO using EfficientSAM and create dictionary of mask 
        # predictions from the pipeline
        """
        detected_items = []
        item_labels = []
        for phrase, boxes in bbox_predictions.items():
            for box in boxes:
                masks, scores = self.run_efficient_sam(image, None, box, 1)
                if len(masks) > 0:
                    mask_msg = self.generate_mask_msg(
                        phrase, scores[0], masks[0], image, depth_img, box
                    )
                    detected_items.append(mask_msg)
                    item_labels.append(phrase)
                break
        result.detected_items = detected_items
        result.item_labels = item_labels
        """ 
            
        return result 

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
        The result of the action server containing masks for all food items 
        detected in the image.
        """
        starting_time = self.get_clock().now()
        self.get_logger().info("Received a new goal!")

        # Get the latest image and camera info
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg
        with self.camera_info_lock:
            camera_info = self.camera_info

        # Check if the image and camera info are available
        if latest_img_msg is None or camera_info is None:
            self.get_logger().error("Image or camera info not available.")
            return SegmentAllItems.Result()

        # Convert the input label list from goal request to a single string caption
        # for GroundingDINO
        caption = '. '.join(goal_handle.request.input_labels).lower().strip()
        caption += '.'
        self.get_logger().info(f"caption: {caption}")

        # Start running the vision pipeline as a separate thread
        rate = self.create_rate(self.rate_hz)
        vision_pipeline_task = self.executor.create_task(
            self.run_vision_pipeline, latest_img_msg, caption
        )

        # Wait for the vision pipeline to finish and keep publishing 
        # feedback (elapsed time) while waiting
        feedback = SegmentAllItems.Feedback()
        while (
            rclpy.ok() 
            and not goal_handle.is_cancel_requested()
            and not vision_pipeline_task.done()
        ):
            feedback.elapsed_time = ((self.get_clock().now() - starting_time).nanoseconds / 
                                     1e9).to_msg()
            goal_handle.publish_feedback(feedback) 
            rate.sleep()

        # If there is a cancel request, cancel the vision pipeline task
        if goal_handle.is_cancel_requested():
            self.get_logger().info("Goal cancelled.")
            goal_handle.canceled()
            result = SegmentAllItems.Result()
            result.status = result.STATUS_CANCELLED

            # Clear the active goal
            with self.active_goal_request_lock:
                self.active_goal_request = None

            return result
        
        # Set the result after the task has been completed        
        self.get_logger().info("Goal not cancelled.")
        self.get_logger().info("VIsion pipeline completed successfully.")
        result = vision_pipeline_task.result()
        goal_handle.succeed()
        result.status = result.STATUS_SUCCEEDED

        # Clear the active goal
        with self.active_goal_request_lock:
            self.active_goal_request = None

        return result


    
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
