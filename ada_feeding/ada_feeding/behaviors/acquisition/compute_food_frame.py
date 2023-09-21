#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeFoodFrame behavior, which computes the
food frame from 
"""
# Standard imports
from typing import Union, Optional

# Third-party imports
import cv2 as cv
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3Stamped
import numpy as np
import py_trees
import pyrealsense2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_feeding_msgs.msg import Mask
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.helpers import BlackboardKey, quat_between_vectors
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding_perception.helpers import ros_msg_to_cv2_image


class ComputeFoodFrame(BlackboardBehavior):
    """
    Computes the food reference frame.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    # pylint: disable=too-many-arguments
    # These are effectively config definitions
    # They require a lot of arguments.

    def blackboard_inputs(
        self,
        ros2_node: Union[BlackboardKey, Node],
        camera_info: Union[BlackboardKey, CameraInfo],
        mask: Union[BlackboardKey, Mask],
        camera_frame: Union[BlackboardKey, str] = "camera_color_optical_frame",
        world_frame: Union[BlackboardKey, str] = "world",
        debug_food_frame: Union[BlackboardKey, str] = "food",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        ros2_node (Node): ROS2 Node for reading/writing TFs
        camera_info (geometry_msgs/CameraInfo): camera intrinsics matrix
        mask (ada_feeding_msgs/Mask): food context, see Mask.msg
        camera_frame (string): ID of the TF frame that the Mask is in
        world_frame (string): ID of the TF frame to represent the food frame in
        debug_food_frame (string): If len>0, TF frame to publish static transform
                                   (relative to world_frame) for debugging purposes
        """
        # pylint: disable=unused-argument
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        action_select_request: Optional[BlackboardKey],  # AcquisitionSelect.Request
        food_frame: Optional[BlackboardKey],  # TransformStamped
        debug_tf_publisher: Optional[
            BlackboardKey
        ] = None,  # StaticTransformBroadcaster
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        action_select_request (AcquisitionSelect.Request): request to send to AcquisitionSelect
                                                           (copies mask input)
        food_frame (geometry_msgs/TransformStamped): transform from world_frame to food frame
        debug_tf_publisher (StaticTransformBroadcaster): If set, store
                                                        static broadcaster here to keep it alive
                                                        for debugging purposes.
        """
        # pylint: disable=unused-argument
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    # pylint: disable=attribute-defined-outside-init
    # It is okay for attributes in behaviors to be
    # defined in the setup / initialise functions.

    def setup(self):
        """
        Middleware (i.e. TF) setup
        """

        # Create TF Tree
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self.blackboard_get("ros2_node")
        )

    def initialise(self):
        """
        Behavior initialization
        """

        # Construct camera intrinsics
        # See: https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-
        #      to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        camera_info = self.blackboard_get("camera_info")
        self.intrinsics = pyrealsense2.intrinsics()
        self.intrinsics.width = camera_info.width
        self.intrinsics.height = camera_info.height
        self.intrinsics.ppx = camera_info.K[2]
        self.intrinsics.ppy = camera_info.K[5]
        self.intrinsics.fx = camera_info.K[0]
        self.intrinsics.fy = camera_info.K[4]
        self.intrinsics.model = camera_info.distortion_model
        self.intrinsics.coeffs = list(camera_info.D)

    def update(self) -> py_trees.common.Status:
        """
        Behavior tick (DO NOT BLOCK)
        """
        # pylint: disable=too-many-locals
        # I think this is reasonable to understand
        # the logic of this function.

        # Check if we have the camera transform
        if not self.tf_buffer.can_transform(
            self.blackboard_get("world_frame"),
            self.blackboard_get("camera_frame"),
            rclpy.time.Time(),
        ):
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING
        transform = self.tf_buffer.lookup_transform(
            self.blackboard_get("world_frame"),
            self.blackboard_get("camera_frame"),
            rclpy.time.Time(),
        )

        # Set up return objects
        world_to_food_transform = TransformStamped()
        world_to_food_transform.header.stamp = rclpy.time.Time()
        world_to_food_transform.header.frame_id = self.blackboard_get("world_frame")
        world_to_food_transform.child_frame_id = self.blackboard_get("debug_food_frame")

        # De-project center of ROI
        mask = self.blackboard_get("mask")
        center_list = pyrealsense2.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [mask.x_offset + mask.width // 2, mask.y_offset + mask.height // 2],
            mask.average_depth,
        )
        center = PointStamped()
        center.header.frame_id = self.blackboard_get("camera_frame")
        center.point.x = center_list[0]
        center.point.y = center_list[1]
        center.point.z = center_list[2]
        center = tf2_ros.TransformRegistration().get(PointStamped)(center, transform)

        # Get angle from mask bounded ellipse
        # See: https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-
        #      about-the-angle-returned
        # Get Mask
        mask_cv = ros_msg_to_cv2_image(mask.mask)
        # Threshold and get contours
        _, mask_thresh = cv.threshold(mask_cv, 127, 255, 0)
        contours, _ = cv.findContours(mask_thresh, 1, 2)
        full_contours = np.vstack(contours)
        rect = cv.minAreaRect(full_contours)
        points = cv.boxPoints(rect)
        # Get direction of +X axix in pixel-space
        # Take longest side
        if np.linalg.norm(points[0] - points[1]) > np.linalg.norm(
            points[1] - points[2]
        ):
            point1 = points[0]
            point2 = points[1]
        else:
            point1 = points[1]
            point2 = points[2]
        # Get vector in camera frame
        point1 = pyrealsense2.rs2_deproject_pixel_to_point(
            self.intrinsics, [point1[0], point1[1]], mask.average_depth
        )
        point2 = pyrealsense2.rs2_deproject_pixel_to_point(
            self.intrinsics, [point2[0], point2[1]], mask.average_depth
        )
        x_pos = Vector3Stamped()
        x_pos.header.frame_id = self.blackboard_get("camera_frame")
        x_pos.vector.x = point1[0] - point2[0]
        x_pos.vector.y = point1[1] - point2[1]
        x_pos.vector.z = point1[2] - point2[2]
        # Transform to world frame
        x_pos = tf2_ros.TransformRegistration().get(Vector3Stamped)(x_pos, transform)
        # Project to world x-y plane
        x_pos.vector.z = 0.0

        # Convert to TransformStamped
        world_to_food_transform.transform.translation.x = center.point.x
        world_to_food_transform.transform.translation.y = center.point.y
        world_to_food_transform.transform.translation.z = center.point.z

        x_unit = Vector3Stamped()
        x_unit.vector.x = 1.0
        world_to_food_transform.transform.orientation = quat_between_vectors(
            x_unit.vector, x_pos.vector
        )

        # Write to blackboard outputs
        if len(self.blackboard_get("debug_food_frame")) > 0:
            stb = StaticTransformBroadcaster(self.blackboard_get("ros2_node"))
            stb.sendTransform(world_to_food_transform)
            self.blackboard_write("debug_tf_publisher", stb)
        self.blackboard_write("food_frame", world_to_food_transform)
        request = AcquisitionSelect.Request()
        request.food_context = mask
        self.blackboard_write("action_select_request", request)

        return py_trees.common.Status.SUCCESS
