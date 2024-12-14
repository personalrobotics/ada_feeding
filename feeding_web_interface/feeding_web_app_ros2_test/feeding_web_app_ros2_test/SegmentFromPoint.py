#!/usr/bin/env python3
# Copyright (c) 2024, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import array
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
from sensor_msgs.msg import CompressedImage, Image, RegionOfInterest, CameraInfo
from shapely.geometry import MultiPoint
import threading
import time


# The fixed header that ROS2 Humble's compressed depth image transport plugin prepends to
# the data. The exact value was empirically determined, but the below link shows the code
# that prepends additional data:
#
# https://github.com/ros-perception/image_transport_plugins/blob/5ef79d74c4347e6a2d151df63230d5fea1357137/compressed_depth_image_transport/src/codec.cpp#L337
_COMPRESSED_DEPTH_16UC1_HEADER = array.array(
    "B", [0, 0, 0, 0, 46, 32, 133, 4, 192, 24, 60, 78]
)


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
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.image_callback,
            1,
        )
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()

        # Subscribe to the depth topic, to store the latest image
        self.depth_subscriber = self.create_subscription(
            CompressedImage,
            "/camera/aligned_depth_to_color/image_raw/compressedDepth",
            self.depth_callback,
            1,
        )
        self.latest_depth_msg = None
        self.latest_depth_msg_lock = threading.Lock()

        # Subscribe to the camera info
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 1
        )
        self.camera_info = None
        self.camera_info_lock = threading.Lock()

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

    def image_callback(self, msg):
        """
        Store the latest image message.

        Parameters
        ----------
        msg: The image message.
        """
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg

    def depth_callback(self, msg):
        """
        Store the latest depth message.

        Parameters
        ----------
        msg: The depth message.
        """
        with self.latest_depth_msg_lock:
            self.latest_depth_msg = msg

    def camera_info_callback(self, msg):
        """
        Store the latest camera info message.

        Parameters
        ----------
        msg: The camera info message.
        """
        with self.camera_info_lock:
            self.camera_info = msg

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

        # Reject if there is already an active goal
        if self.active_goal_request is not None:
            self.get_logger().info(
                "Rejecting goal request because there is already an active goal"
            )
            return GoalResponse.REJECT

        # Reject if this node has not received an RGB and depth image
        with self.latest_img_msg_lock:
            with self.latest_depth_msg_lock:
                with self.camera_info_lock:
                    if (
                        self.latest_img_msg is None
                        or self.latest_depth_msg is None
                        or self.camera_info is None
                    ):
                        self.get_logger().info(
                            "Rejecting goal request because RGB and depth images "
                            "have not been received"
                        )
                        return GoalResponse.REJECT

        # Otherwise accept
        self.get_logger().info("Accepting goal request")
        self.active_goal_request = goal_request
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Always accept client requests to cancel the active goal. Note that this
        function should not actually implement the cancel; that is handled in
        `execute_callback`

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

    def segment_image(self, seed_point, result, segmentation_success):
        """
        Dummy segmentation function. Sleeps for `self.sleep_time` seconds, then
        returns a result. In the resulting mask, pixels to keep (detected food)
        are white (255) and pixels to remove (non-food) are black (0).

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
        with self.latest_depth_msg_lock:
            latest_depth_msg = self.latest_depth_msg
        with self.camera_info_lock:
            camera_info = self.camera_info
        result.header = latest_img_msg.header
        result.camera_info = camera_info
        img = self.bridge.compressed_imgmsg_to_cv2(latest_img_msg, "bgr8")
        width, height, _ = img.shape
        depth_img = cv2.imdecode(
            np.frombuffer(
                latest_depth_msg.data[len(_COMPRESSED_DEPTH_16UC1_HEADER) :], np.uint8
            ),
            cv2.IMREAD_UNCHANGED,
        )

        # Sleep (dummy segmentation)
        time.sleep(self.sleep_time)

        # Return the result. Detects 3 random polygons around seed_point
        cx, cy = seed_point
        for i in range(3):
            num_points = np.random.randint(3, 11)  # 3-10 points
            points = MultiPoint(
                [
                    [x, y]
                    for x, y in zip(
                        np.random.uniform(
                            max(cx - width / 8, 0),
                            min(cx + width / 8, width),
                            num_points,
                        ),
                        np.random.uniform(
                            max(cy - height / 8, 0),
                            min(cy + height / 8, height),
                            num_points,
                        ),
                    )
                ]
            )
            polygon = points.convex_hull
            x_min, y_min, x_max, y_max = polygon.bounds
            x_min = int(math.floor(x_min))
            x_max = int(math.ceil(x_max))
            y_min = int(math.floor(y_min))
            y_max = int(math.ceil(y_max))
            mask_img = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            cv2.fillPoly(
                mask_img,
                [np.array(polygon.exterior.coords, dtype=np.int32) - (x_min, y_min)],
                (255,),
            )
            # Create the message
            mask_msg = Mask()
            mask_msg.roi = RegionOfInterest(
                x_offset=x_min,
                y_offset=y_min,
                height=y_max - y_min,
                width=x_max - x_min,
                do_rectify=False,
            )
            mask_msg.mask = CompressedImage(
                format="jpeg",
                data=cv2.imencode(".jpg", mask_img)[1].tostring(),
            )
            mask_msg.rgb_image = CompressedImage(
                format="jpeg",
                data=cv2.imencode(".jpg", img[y_min:y_max, x_min:x_max])[1].tostring(),
            )
            mask_msg.depth_image = self.bridge.cv2_to_imgmsg(
                depth_img[y_min:y_max, x_min:x_max], "passthrough"
            )
            depth_points_in_mask = depth_img[y_min:y_max, x_min:x_max][mask_img > 0]
            mask_msg.average_depth = (
                np.median(depth_points_in_mask[depth_points_in_mask > 0]) / 1000.0
            )  # Convert to meters
            mask_msg.item_id = "dummy_food_id_%d" % (i)
            mask_msg.confidence = np.random.random()
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
