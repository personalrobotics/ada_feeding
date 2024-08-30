#!/usr/bin/env python3
"""
This file defines the SegmentFromPointNode class, which launches an action
server that takes in a seed point, segments the latest image with that seed
point using Segment Anything, and returns the top n contender masks.
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
from rclpy.qos import QoSProfile, ReliabilityPolicy
from segment_anything import sam_model_registry, SamPredictor
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, RegionOfInterest
import torch
from torchvision import transforms

# Local imports
from ada_feeding_msgs.action import SegmentFromPoint
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


class SegmentFromPointNode(Node):
    """
    The SegmentFromPointNode launches an action server that takes in a seed point,
    segments the latest image with that seed point using Segment Anything, and
    returns the top n contender masks.
    """

    # pylint: disable=too-many-instance-attributes
    # Having more than 7 instance attributes is unavoidable here, since for
    # every subscription we need to store the subscription, mutex, and data,
    # and we have 3 subscriptions.

    def __init__(self) -> None:
        """
        Initialize the SegmentFromPointNode.
        """

        super().__init__("segment_from_point")

        # Check if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device {self.device}")

        # Read the parameters
        # NOTE: These parameters are only read once. Any changes after the node
        # is initialized will not be reflected.
        (
            model_name,
            model_base_url,
            model_dir,
            self.use_efficient_sam,
            self.n_contender_masks,
            self.rate_hz,
            self.min_depth_mm,
            self.max_depth_mm,
        ) = self.read_params()

        # Download the checkpoint if it doesn't exist
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            self.get_logger().info("Model checkpoint does not exist. Downloading...")
            download_checkpoint(model_name, model_dir, model_base_url)
            self.get_logger().info(f"Model checkpoint downloaded {model_path}.")

        # Create the shared resource to ensure that the action server rejects all
        # goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Subscribe to the camera info topic, to get the camera intrinsics
        self.camera_info = None
        self.camera_info_lock = threading.Lock()
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
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
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
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
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Initialize Segment Anything
        if self.use_efficient_sam:
            self.initialize_efficient_sam(model_name, model_path)
        else:
            self.initialize_sam(model_name, model_path)

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the Action Server.
        # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
        self._action_server = ActionServer(
            self,
            SegmentFromPoint,
            "SegmentFromPoint",
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
        model_name: The name of the Segment Anything model checkpoint to use
        model_base_url: The URL to download the model checkpoint from if it is
            not already downloaded
        model_dir: The location of the directory where the model checkpoint is / should be stored
        use_efficient_sam: Whether to use EfficientSAM or SAM
        n_contender_masks: The number of contender masks to return per point.
        rate_hz: The rate at which to return feedback.
        min_depth_mm: The minimum depth in mm to consider for a mask.
        max_depth_mm: The maximum depth in mm to consider for a mask.
        """
        (
            sam_model_name,
            sam_model_base_url,
            efficient_sam_model_name,
            efficient_sam_model_base_url,
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
            model_name = efficient_sam_model_name.value
            model_base_url = efficient_sam_model_base_url.value
        else:
            model_name = sam_model_name.value
            model_base_url = sam_model_base_url.value

        return (
            model_name,
            model_base_url,
            model_dir.value,
            use_efficient_sam.value,
            n_contender_masks.value,
            rate_hz.value,
            min_depth_mm.value,
            max_depth_mm.value,
        )

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
        self.predictor = SamPredictor(sam)

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

    def goal_callback(self, goal_request: SegmentFromPoint.Goal) -> GoalResponse:
        """
        Accept a goal if this action does not already have an active goal,
        else reject.

        NOTE: Since each segmenatation takes ~200ms on a machine with a GPU,
        we do not allow the user to cancel a goal and start a new one. It is
        more straightforward to require the first goal to finish.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info("Received goal request")
        with self.latest_img_msg_lock:
            if self.latest_img_msg is None:
                self.get_logger().info(
                    "Rejecting goal request since no color image received"
                )
                return GoalResponse.REJECT
        with self.latest_depth_img_msg_lock:
            if self.latest_depth_img_msg is None:
                self.get_logger().info(
                    "Rejecting goal request since no depth image received"
                )
                return GoalResponse.REJECT
        with self.active_goal_request_lock:
            if self.active_goal_request is None:
                self.get_logger().info("Accepting goal request")
                self.active_goal_request = goal_request
                return GoalResponse.ACCEPT
            self.get_logger().info(
                "Rejecting goal request since there is already an active one"
            )
            return GoalResponse.REJECT

    def cancel_callback(self, _: ServerGoalHandle) -> CancelResponse:
        """
        Always accept client requests to cancel the active goal. However, note
        that in practice `execute_callback` doesn't interrupt the segmentation
        to process the callback, and instead waits for segmentation to complete.

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def run_sam(
        self, image: npt.NDArray, seed_point: Tuple[int, int]
    ) -> list[Tuple[float, npt.NDArray[np.bool_]]]:
        """
        Run the SAM model on the image and seed point.

        Parameters
        ----------
        image: The image to segment, in BGR.
        seed_point: The seed point to segment from.

        Returns
        -------
        masks: The masks outputted by the model.
        scores: The scores outputted by the model.
        """
        self.get_logger().info("Segmenting image with SAM...")
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segment the image with that input point
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([seed_point]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks, scores

    def run_efficient_sam(
        self, image: npt.NDArray, seed_point: Tuple[int, int]
    ) -> list[Tuple[float, npt.NDArray[np.bool_]]]:
        """
        Run the EfficientSAM model on the image and seed point.

        Code adapted from https://github.com/yformer/EfficientSAM/blob/main/EfficientSAM_example.py

        Parameters
        ----------
        image: The image to segment, in BGR.
        seed_point: The seed point to segment from.

        Returns
        -------
        masks: The masks outputted by the model.
        scores: The scores outputted by the model.
        """
        self.get_logger().info("Segmenting image with EfficientSAM...")
        # Convert image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a tensor
        image_tensor = transforms.ToTensor()(image).to(device=self.device)

        # Convert the seed point to a tensor
        input_points = torch.tensor(np.array(seed_point).reshape((1, 1, 1, 2))).to(
            device=self.device
        )

        # Convert the labels to a tensor
        input_labels = torch.tensor([[[1]]]).to(device=self.device)

        # Run the model
        predicted_logits, predicted_iou = self.efficient_sam(
            image_tensor[None, ...],
            input_points,
            input_labels,
        )
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )
        masks = torch.ge(predicted_logits[0, 0, :, :, :], 0).cpu().detach().numpy()
        scores = predicted_iou[0, 0, :].cpu().detach().numpy()
        return masks, scores

    async def segment_image(
        self, seed_point: Tuple[int, int], image_msg: Image
    ) -> SegmentFromPoint.Result:
        """
        Segment image using the SAM model.

        Parameters
        ----------
        seed_point: The seed point to segment from.
        image_msg: The Image message containing the image to segment.

        Returns
        -------
        result: The result message containing the contender masks.
        """
        # pylint: disable=too-many-locals
        # All are necessary.

        self.get_logger().info("Segmenting image...")
        # Create the result
        result = SegmentFromPoint.Result()
        result.header = image_msg.header
        with self.camera_info_lock:
            if self.camera_info is not None:
                result.camera_info = self.camera_info
            else:
                self.get_logger().warn(
                    "Camera info not received, not including in result message"
                )

        # Get the latest depth image
        with self.latest_depth_img_msg_lock:
            depth_img_msg = self.latest_depth_img_msg

        # Convert the image to OpenCV format
        image = ros_msg_to_cv2_image(image_msg, self.bridge)

        # Convert the depth image to OpenCV format. The depth image is a
        # 16-bit image with depth in mm.
        depth_img = ros_msg_to_cv2_image(depth_img_msg, self.bridge)

        # Segment the image
        start_time = self.get_clock().now()
        if self.use_efficient_sam:
            masks, scores = self.run_efficient_sam(image, seed_point)
        else:
            masks, scores = self.run_sam(image, seed_point)
        elpased_time = self.get_clock().now() - start_time
        self.get_logger().info(
            f"Elapsed time Model: {elpased_time.nanoseconds / 10.0**9} secs"
        )

        # Sort the masks from highest to lowest score
        scored_masks_sorted = sorted(
            zip(scores, masks), key=lambda x: x[0], reverse=True
        )

        # Convert the top `self.n_contender_masks` masks to ROS messages
        for mask_num in range(min(len(scored_masks_sorted), self.n_contender_masks)):
            score, mask = scored_masks_sorted[mask_num]
            item_id = f"food_id_{mask_num:d}"
            mask_msg = self.generate_mask_msg(
                item_id, score.item(), mask, image, depth_img, seed_point
            )
            if mask_msg is not None:
                result.detected_items.append(mask_msg)

        # Return the result message
        return result

    def generate_mask_msg(
        self,
        item_id: str,
        score: float,
        mask: npt.NDArray[np.bool_],
        image: npt.NDArray,
        depth_img: npt.NDArray,
        seed_point: Tuple[int, int],
    ) -> Optional[Mask]:
        """
        Convert a mask detected by SegmentAnything to a ROS Mask msg.

        Parameters
        ----------
        item_id: The ID of the mask to include in the message.
        score: The score (confidence) SegmentAnything outputted for the mask.
        mask: The mask.
        image: The image the mask was detected in.
        depth_img: The most recent depth image.
        seed_point: The seed point used to segment the image.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # More arguments is fine here, since the purpose of this function is to consolidate
        # multiple pieces of data into a single message.

        # Clean the mask to only contain the connected component containing
        # the seed point
        # TODO: Thoroughly test this, in case we need to add more mask cleaning!
        #       Erosion/dilation could be useful here.
        cleaned_mask = get_connected_component(mask, seed_point)
        # Get the **median** depth over the mask
        masked_depth = depth_img[cleaned_mask]
        average_depth_mm = np.median(
            masked_depth[
                np.logical_and(
                    masked_depth >= self.min_depth_mm, masked_depth <= self.max_depth_mm
                )
            ]
        )
        if np.isnan(average_depth_mm):
            self.get_logger().warn(
                f"No depth points within [{self.min_depth_mm}, {self.max_depth_mm}] mm range "
                f"for mask {item_id}. Skipping mask."
            )
            return None
        # Compute the bounding box
        bbox = bbox_from_mask(cleaned_mask)
        # Crop the image and the mask
        cropped_image, cropped_mask, _ = crop_image_mask_and_point(
            image, cleaned_mask, seed_point, bbox
        )
        cropped_depth, _, _ = crop_image_mask_and_point(
            depth_img, cleaned_mask, seed_point, bbox
        )
        # Convert the mask to an image
        mask_img = np.where(cropped_mask, 255, 0).astype(np.uint8)

        # Create the message
        mask_msg = Mask()
        mask_msg.roi = RegionOfInterest(
            x_offset=int(bbox.xmin),
            y_offset=int(bbox.ymin),
            height=int(bbox.ymax - bbox.ymin),
            width=int(bbox.xmax - bbox.xmin),
            do_rectify=False,
        )
        mask_msg.mask = cv2_image_to_ros_msg(mask_img, compress=True)
        mask_msg.rgb_image = cv2_image_to_ros_msg(cropped_image, compress=True)
        mask_msg.depth_image = cv2_image_to_ros_msg(cropped_depth, compress=False)
        mask_msg.average_depth = average_depth_mm / 1000.0
        mask_msg.item_id = item_id
        mask_msg.confidence = score

        return mask_msg

    async def execute_callback(
        self, goal_handle: ServerGoalHandle
    ) -> SegmentFromPoint.Result:
        """
        Execute the goal.

        Note that as it stands, this function communicates no feedback to the
        client. This is because each segmentation call takes ~200ms on a machine
        with a GPU. If we want to provide feedback, we'll need to run segmentation
        in a separate process and communicate goals/results with a Pipe, which
        would add latency and introduce complexity.

        Parameters
        ----------
        goal_handle: The goal handle for this action call.

        Returns
        -------
        result: The result message containing the contender masks.
        """
        self.get_logger().info(f"Executing goal...{goal_handle}")
        start_time = self.get_clock().now()

        # Get the latest image
        latest_img_msg = None
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg

        # Start image segmentation as a co-routine
        seed_point = (
            int(goal_handle.request.seed_point.point.x),
            int(goal_handle.request.seed_point.point.y),
        )
        rate = self.create_rate(self.rate_hz)
        segment_image_task = self.executor.create_task(
            self.segment_image, seed_point, latest_img_msg
        )

        # Keep returning feedback until the task is done
        feedback = SegmentFromPoint.Feedback()
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not segment_image_task.done()
        ):
            feedback.elapsed_time = (self.get_clock().now() - start_time).to_msg()
            goal_handle.publish_feedback(feedback)
            rate.sleep()

        # Check if there was a cancel request
        if goal_handle.is_cancel_requested or not rclpy.ok():
            self.get_logger().info("Goal canceled")
            goal_handle.canceled()
            result = SegmentFromPoint.Result()
            result.status = result.STATUS_CANCELED
            with self.active_goal_request_lock:
                self.active_goal_request = None  # Clear the active goal
            return result
        self.get_logger().info("Goal not canceled")

        # Task must be done, given that we broke out of the while loop and
        # did not land in the above conditional
        result = segment_image_task.result()

        # Return the result
        self.get_logger().info("Segmentation succeeded, returning")
        goal_handle.succeed()
        result.status = result.STATUS_SUCCEEDED
        with self.active_goal_request_lock:
            self.active_goal_request = None  # Clear the active goal
        self.get_logger().info(
            "...Done. Got masks with average depth "
            f"{[m.average_depth for m in result.detected_items]} m."
        )
        return result


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    segment_from_point = SegmentFromPointNode()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=5)

    rclpy.spin(segment_from_point, executor=executor)


if __name__ == "__main__":
    main()
