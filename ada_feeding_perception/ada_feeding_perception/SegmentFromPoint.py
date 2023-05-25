#!/usr/bin/env python3
from ada_feeding_msgs.action import SegmentFromPoint
from ada_feeding_msgs.msg import Mask
import cv2
from cv_bridge import CvBridge
import math
import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, RegionOfInterest
from shapely.geometry import MultiPoint
import threading
import time

import torch
import matplotlib.pyplot as plt
import sys
import os
from segment_anything import sam_model_registry, SamPredictor
from skimage.measure import regionprops


class SegmentFromPointNode(Node):
    def __init__(self, sleep_time=2.0, send_feedback_hz=10):
        """
        Create a SegmentFromPoint action server. This dummy action will sleep
        for sleep_time seconds before returning a result.

        Parameters
        ----------
        sleep_time: How many seconds this dummy node should sleep before returning a result.
        send_feedback_hz: The target frequency at which to send feedback.
        """
        super().__init__("segment_from_point")

        self.active_goal_request = None
        self.sleep_time = sleep_time
        self.send_feedback_hz = send_feedback_hz

        # Subscribe to the image topic, to store the latest image
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            1,
        )
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()

        # Convert between ROS and CV images
        self.bridge = CvBridge()

    def image_callback(self, msg):
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
                # Create the action server
                self._action_server = ActionServer(
                    self,
                    SegmentFromPoint,
                    "SegmentFromPoint",
                    self.execute_callback,
                    goal_callback=self.goal_callback,
                    cancel_callback=self.cancel_callback,
                )
            self.latest_img_msg = msg

    def goal_callback(self, goal_request):
        """
        Accept a goal if this action does not already have an active goal,
        else reject.

        TODO: Once we integrate this with the SegmentAnything code, we should
        think more carefully about whether we truly want to reject future goals.
        Say the user clicks on a point and then changes their mind. We'd ideally
        want to cancel past goals and accept the new one. But that requires
        being able to interrupt the segmentation thread, which may not be
        easy depending on how SegmentAnything is implemented.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info("Received goal request")
        if self.active_goal_request is None:
            self.get_logger().info("Accepting goal request")
            self.active_goal_request = goal_request
            return GoalResponse.ACCEPT
        self.get_logger().info("Rejecting goal request")
        return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """
        Always accept client requests to cancel the active goal. Note that this
        function should not actually impelement the cancel; that is handled in
        `execute_callback`

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def bbox_from_mask(mask):
        # Find the bounding box coordinates from the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, cmin, rmax, cmax

    def crop_image_and_mask(image, mask, bbox):
        minr, minc, maxr, maxc = bbox
        cropped_image = image[minr:maxr, minc:maxc]
        cropped_mask = mask[minr:maxr, minc:maxc]
        return cropped_image, cropped_mask

    def segment_image(self, seed_point, result, segmentation_success):
        """
        Segment image using the SAM model.

        Parameters
        ----------
        seed_point: The seed point to segment from.
        result: The result to set.
        segmentation_success: The list to append the segmentation success to.
        """
        self.get_logger().info("Segmenting image...")
        # Get the latest image
        latest_img_msg = None
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg
        result.header = latest_img_msg.header
        image = self.bridge.imgmsg_to_cv2(latest_img_msg, desired_encoding="bgr8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_point = np.array(seed_point)

        # Model parameters
        device = 'cuda'
        model_type = "vit_b"
        sam_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'sam_vit_b_01ec64.pth'))
        # if the model can't be found download it
        if not os.path.isfile(sam_checkpoint):
            import urllib.request
            print("SAM model checkpoint not found. Downloading...")
            if not os.path.exists(os.path.dirname(sam_checkpoint)):
                os.makedirs(os.path.dirname(sam_checkpoint))
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
            print("Download complete! Model saved at {}".format(sam_checkpoint))

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)
        predictor.set_image(image)

        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        scored_masks = [(score, mask) for score, mask in zip(scores, masks)]
        scored_masks_sorted = sorted(scored_masks, key=lambda x: x[0], reverse=True)

        # After getting the masks
        for i, (score, mask) in enumerate(scored_masks_sorted):
            # compute the bounding box from the mask
            bbox = self.bbox_from_mask(mask)
            # crop the image and the mask
            cropped_image, cropped_mask = self.crop_image_and_mask(image, mask, bbox)
            # save the cropped mask as an image
            mask_img = np.where(cropped_mask, 255, 0).astype(np.uint8)

            # Create the message
            mask_msg = Mask()
            mask_msg.roi = RegionOfInterest(
                x_offset=int(bbox[1]),
                y_offset=int(bbox[0]),
                height=int(bbox[2] - bbox[0]),
                width=int(bbox[3] - bbox[1]),
                do_rectify=False,
            )
            mask_msg.mask = CompressedImage(
                format="jpeg",
                data=cv2.imencode(".jpg", mask_img)[1].tostring(),
            )
            mask_msg.item_id = "food_id_%d" % (i)
            mask_msg.confidence = score.item()
            result.detected_items.append(mask_msg)

        # Return Success
        segmentation_success[0] = True

    async def execute_callback(self, goal_handle):
        """
        Sleeps for `self.sleep_time` seconds, then returns a result.
        """
        self.get_logger().info("Executing goal...%s" % (goal_handle,))

        # Load the feedback parameters
        feedback_rate = self.create_rate(self.send_feedback_hz)
        feedback_msg = SegmentFromPoint.Feedback()

        # Get the seed point
        seed_point = (
            goal_handle.request.seed_point.point.x,
            goal_handle.request.seed_point.point.y,
        )

        # Start the segmentation thread
        result = SegmentFromPoint.Result()
        segmentation_success = [False]
        segmentation_thread = threading.Thread(
            target=self.segment_image,
            args=(seed_point, result, segmentation_success),
            daemon=True,
        )
        segmentation_thread.start()
        segmentation_start_time = self.get_clock().now()

        # Monitor the segmentation thread, and send feedback
        while rclpy.ok():
            # Check if there is a cancel request
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                result = SegmentFromPoint.Result()
                result.status = result.STATUS_CANCELED
                self.active_goal_request = None  # Clear the active goal
                return result

            # Check if the segmentation thread has finished
            if not segmentation_thread.is_alive():
                if segmentation_success[0]:
                    self.get_logger().info("Segmentation succeeded, returning")
                    # Succeed the goal
                    goal_handle.succeed()
                    result.status = result.STATUS_SUCCEEDED
                    self.active_goal_request = None  # Clear the active goal
                    return result
                else:
                    self.get_logger().info("Segmentation failed, aborting")
                    # Abort the goal
                    goal_handle.abort()
                    result = SegmentFromPoint.Result()
                    result.status = result.STATUS_FAILED
                    self.active_goal_request = None  # Clear the active goal
                    return result

            # Send feedback
            feedback_msg.elapsed_time = (
                self.get_clock().now() - segmentation_start_time
            ).to_msg()
            self.get_logger().info("Feedback: %s" % feedback_msg)
            goal_handle.publish_feedback(feedback_msg)

            # Sleep for the specified feedback rate
            feedback_rate.sleep()

        # If we get here, something went wrong
        self.get_logger().info("Unknown error, aborting")
        goal_handle.abort()
        result = SegmentFromPoint.Result()
        result.status = result.STATUS_UNKNOWN
        self.active_goal_request = None  # Clear the active goal
        return result


def main(args=None):
    rclpy.init(args=args)

    segment_from_point = SegmentFromPointNode()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(segment_from_point, executor=executor)


if __name__ == "__main__":
    main()
