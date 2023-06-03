#!/usr/bin/env python3
"""
This file defines the SegmentFromPointNode class, which launches an action
server that takes in a seed point, segments the latest image with that seed
point using Segment Anything, and returns the top n contender masks.
"""
# Standard imports
import os
import threading
from typing import Tuple

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from segment_anything import sam_model_registry, SamPredictor
from sensor_msgs.msg import CompressedImage, Image, RegionOfInterest
import torch

# Local imports
from ada_feeding_msgs.action import SegmentFromPoint
from ada_feeding_msgs.msg import Mask
from ada_feeding_perception.helpers import (
    bbox_from_mask,
    crop_image_mask_and_point,
    download_checkpoint,
    get_connected_component,
)


class SegmentFromPointNode(Node):
    """
    The SegmentFromPointNode launches an action server that takes in a seed point,
    segments the latest image with that seed point using Segment Anything, and
    returns the top n contender masks.
    """

    def __init__(self) -> None:
        """
        Initialize the SegmentFromPointNode.
        """
        super().__init__("segment_from_point")

        # Check if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Read the parameters
        # NOTE: These parameters are only read once. Any changes after the node
        # is initialized will not be reflected.
        (
            model_name,
            model_base_url,
            model_dir,
            n_contender_masks,
        ) = self.read_params()
        self.model_name = model_name.value
        self.n_contender_masks = n_contender_masks.value

        # Download the checkpoint if it doesn't exist
        self.model_path = os.path.join(model_dir.value, self.model_name)
        if not os.path.isfile(self.model_path):
            self.get_logger().info("Model checkpoint does not exist. Downloading...")
            download_checkpoint(self.model_name, model_dir.value, model_base_url.value)
            self.get_logger().info(
                "Model checkpoint downloaded %s."
                % os.path.join(model_dir.value, self.model_name)
            )

        # Create the shared resource to ensure that the action server rejects all
        # goals while a goal is currently active.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Subscribe to the image topic, to store the latest image
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            1,
        )
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()

    def read_params(
        self,
    ) -> Tuple[Parameter, Parameter, Parameter, Parameter, Parameter]:
        """
        Read the parameters for this node.

        Returns
        -------
        model_name: The name of the Segment Anything model checkpoint to use
        model_base_url: The URL to download the model checkpoint from if it is not already downloaded
        model_dir: The location of the directory where the model checkpoint is / should be stored
        n_contender_masks: The number of contender masks to return per point.
        """
        return self.declare_parameters(
            "",
            [
                (
                    "model_name",
                    None,
                    ParameterDescriptor(
                        name="model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the Segment Anything model checkpoint to use",
                        read_only=True,
                    ),
                ),
                (
                    "model_base_url",
                    None,
                    ParameterDescriptor(
                        name="model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description="The URL to download the model checkpoint from if it is not already downloaded",
                        read_only=True,
                    ),
                ),
                (
                    "model_dir",
                    None,
                    ParameterDescriptor(
                        name="model_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description="The location of the directory where the model checkpoint is / should be stored",
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
            ],
        )

    def initialize_food_segmentation(self) -> None:
        """
        Initialize all attributes needed for food segmentation.

        This includes loading the Segment Anything model, launching the action
        server, and more. Note that we are guarenteed the model exists since
        it was downloaded in the __init__ function of this class.

        Raises
        ------
        ValueError if the model name does not contain vit_h, vit_l, or vit_b
        """
        self.get_logger().info("Initializing food segmentation...")
        # Load the model and move it to the specified device
        if "vit_b" in self.model_name:  # base model
            model_type = "vit_b"
        elif "vit_l" in self.model_name:  # large model
            model_type = "vit_l"
        elif "vit_h" in self.model_name:  # huge model
            model_type = "vit_h"
        else:
            raise ValueError("Unknown model type %s" % self.model_name)
        sam = sam_model_registry[model_type](checkpoint=self.model_path)
        sam.to(device=self.device)

        # Create the predictor
        # NOTE: If we allow for concurrent goals, this should be protected by
        # a lock.
        self.predictor = SamPredictor(sam)

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the action server
        self._action_server = ActionServer(
            self,
            SegmentFromPoint,
            "SegmentFromPoint",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.get_logger().info("...Done!")

    def image_callback(self, msg: Image) -> None:
        """
        Store the latest image message.

        Parameters
        ----------
        msg: The image message.
        """
        with self.latest_img_msg_lock:
            # Only create the action server after we have received at least one
            # image
            if self.latest_img_msg is None:
                self.initialize_food_segmentation()
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
        with self.active_goal_request_lock:
            if self.active_goal_request is None:
                self.get_logger().info("Accepting goal request")
                self.active_goal_request = goal_request
                return GoalResponse.ACCEPT
            self.get_logger().info("Rejecting goal request")
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

    def segment_image(
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
        self.get_logger().info("Segmenting image...")
        # Create the result
        result = SegmentFromPoint.Result()
        result.header = image_msg.header

        # Convert the image to OpenCV format
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Segment the image
        input_point = np.array([seed_point])
        input_label = np.array([1])
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Sort the masks from highest to lowest score
        scored_masks = list(zip(scores, masks))
        scored_masks_sorted = sorted(scored_masks, key=lambda x: x[0], reverse=True)

        # After getting the masks
        mask_num = -1
        for score, mask in scored_masks_sorted[: self.n_contender_masks]:
            mask_num += 1
            # Clean the mask to only contain the connected component containing
            # the seed point
            # TODO: Thorughly test this, in case we need to add more mask cleaning!
            cleaned_mask = get_connected_component(mask, seed_point)
            # Compute the bounding box
            bbox = bbox_from_mask(cleaned_mask)
            # Crop the image and the mask
            _, cropped_mask, _ = crop_image_mask_and_point(
                image, cleaned_mask, seed_point, bbox
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
            mask_msg.mask = CompressedImage(
                format="jpeg",
                data=cv2.imencode(".jpg", mask_img)[1].tostring(),
            )
            mask_msg.item_id = "food_id_%d" % (mask_num)
            mask_msg.confidence = score.item()
            result.detected_items.append(mask_msg)

        # Return the result message
        return result

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
        self.get_logger().info("Executing goal...%s" % (goal_handle,))

        # Get the latest image
        latest_img_msg = None
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg

        # Segment the image
        seed_point = (
            int(goal_handle.request.seed_point.point.x),
            int(goal_handle.request.seed_point.point.y),
        )
        result = self.segment_image(seed_point, latest_img_msg)

        # Check if there was a cancel request
        if goal_handle.is_cancel_requested:
            self.get_logger().info("Goal canceled")
            goal_handle.canceled()
            result = SegmentFromPoint.Result()
            result.status = result.STATUS_CANCELED
            with self.active_goal_request_lock:
                self.active_goal_request = None  # Clear the active goal
            return result
        self.get_logger().info("Goal not canceled")

        # Return the result
        self.get_logger().info("Segmentation succeeded, returning")
        goal_handle.succeed()
        result.status = result.STATUS_SUCCEEDED
        with self.active_goal_request_lock:
            self.active_goal_request = None  # Clear the active goal
        return result


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    segment_from_point = SegmentFromPointNode()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(segment_from_point, executor=executor)


if __name__ == "__main__":
    main()
