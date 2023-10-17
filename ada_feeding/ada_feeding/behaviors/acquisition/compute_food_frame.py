#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeFoodFrame behavior, which computes the
food frame from the Mask provided from a perception algorithm.
"""
# Standard imports
from typing import Union, Optional

# Third-party imports
import cv2 as cv
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3Stamped
import numpy as np
from overrides import override
import py_trees
import pyrealsense2
import rclpy
from sensor_msgs.msg import CameraInfo
import tf2_ros

# Local imports
from ada_feeding_msgs.msg import Mask
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.helpers import (
    BlackboardKey,
    quat_between_vectors,
    get_tf_object,
    set_static_tf,
)
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding_perception.helpers import ros_msg_to_cv2_image


class ComputeFoodFrame(BlackboardBehavior):
    """
    Computes the food reference frame.
    See definition in AcquisitionSchema.msg
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
        camera_info: Union[BlackboardKey, CameraInfo],
        mask: Union[BlackboardKey, Mask],
        food_frame_id: Union[BlackboardKey, str] = "food",
        world_frame: Union[BlackboardKey, str] = "world",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        camera_info (geometry_msgs/CameraInfo): camera intrinsics matrix
        mask (ada_feeding_msgs/Mask): food context, see Mask.msg
        food_frame_id (string): If len>0, TF frame to publish static transform
                                   (relative to world_frame)
        world_frame (string): ID of the TF frame to represent the food frame in
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        action_select_request: Optional[BlackboardKey],  # AcquisitionSelect.Request
        food_frame: Optional[BlackboardKey],  # TransformStamped
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        action_select_request (AcquisitionSelect.Request): request to send to AcquisitionSelect
                                                           (copies mask input)
        food_frame (geometry_msgs/TransformStamped): transform from world_frame to food_frame
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get TF Listener from blackboard
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    @override
    def initialise(self):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Construct camera intrinsics
        # See: https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-
        #      to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        camera_info = self.blackboard_get("camera_info")
        self.intrinsics = pyrealsense2.intrinsics()
        self.intrinsics.width = camera_info.width
        self.intrinsics.height = camera_info.height
        self.intrinsics.ppx = camera_info.k[2]
        self.intrinsics.ppy = camera_info.k[5]
        self.intrinsics.fx = camera_info.k[0]
        self.intrinsics.fy = camera_info.k[4]
        if camera_info.distortion_model == "plumb_bob":
            self.intrinsics.model = pyrealsense2.distortion.brown_conrady
            self.intrinsics.coeffs = list(camera_info.d)
        elif camera_info.distortion_model == "equidistant":
            self.intrinsics.model = pyrealsense2.distortion.kannala_brandt4
            self.intrinsics.coeffs = list(camera_info.d)
        else:
            self.logger.warning(
                f"Unsupported camera distortion model: {camera_info.distortion_model}"
            )
            self.intrinsics.model = pyrealsense2.distortion.none

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=too-many-locals
        # I think this is reasonable to understand
        # the logic of this function.

        # pylint: disable=too-many-statements
        # We can't get around all the conversions
        # to ROS2 msg types, which take 3-4 statements each.

        camera_frame = self.blackboard_get("camera_info").header.frame_id
        world_frame = self.blackboard_get("world_frame")

        # Lock TF Buffer
        if self.tf_lock.locked():
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING
        transform = None
        with self.tf_lock:
            # Check if we have the camera transform
            if not self.tf_buffer.can_transform(
                world_frame,
                camera_frame,
                rclpy.time.Time(),
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                return py_trees.common.Status.RUNNING
            transform = self.tf_buffer.lookup_transform(
                world_frame,
                camera_frame,
                rclpy.time.Time(),
            )

        # Set up return objects
        world_to_food_transform = TransformStamped()
        world_to_food_transform.header.stamp = self.node.get_clock().now().to_msg()
        world_to_food_transform.header.frame_id = world_frame
        world_to_food_transform.child_frame_id = self.blackboard_get("food_frame_id")

        # De-project center of ROI
        mask = self.blackboard_get("mask")
        center_list = pyrealsense2.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [
                mask.roi.x_offset + mask.roi.width // 2,
                mask.roi.y_offset + mask.roi.height // 2,
            ],
            mask.average_depth,
        )
        center = PointStamped()
        center.header.frame_id = camera_frame
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
        x_pos.header.frame_id = camera_frame
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
        world_to_food_transform.transform.rotation = quat_between_vectors(
            x_unit.vector, x_pos.vector
        )

        # Write to blackboard outputs
        if len(self.blackboard_get("food_frame_id")) > 0:
            set_static_tf(world_to_food_transform, self.blackboard, self.node)
        self.blackboard_set("food_frame", world_to_food_transform)
        request = AcquisitionSelect.Request()
        request.food_context = mask
        self.blackboard_set("action_select_request", request)

        return py_trees.common.Status.SUCCESS
