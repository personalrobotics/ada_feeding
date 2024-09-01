#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeMouthFrame behavior, which takes in the PointStamped
containing the (x, y, z) position of the mouth in the camera frame, and returns the
pose of the mouth in the base frame.
"""

# Standard imports
from typing import Optional, Union

# Third-party imports
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3
from overrides import override
import py_trees
import rclpy
import tf2_ros

# Local imports
from ada_feeding.helpers import (
    BlackboardKey,
    quat_between_vectors,
    get_tf_object,
)
from ada_feeding.behaviors import BlackboardBehavior


class ComputeMouthFrame(BlackboardBehavior):
    """
    Compute the mouth PoseStamped in the base frame, where the position is the output
    of face detection and the orientation is facing the forkTip.
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
        detected_mouth_center: Union[BlackboardKey, PointStamped],
        timestamp: Union[BlackboardKey, rclpy.time.Time] = rclpy.time.Time(),
        world_frame: Union[BlackboardKey, str] = "root",  # +z will match this frame
        frame_to_orient_towards: Union[
            BlackboardKey, str
        ] = "forkTip",  # +x will point towards this frame
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        detected_mouth_center : Union[BlackboardKey, PointStamped]
            The detected mouth center in the camera frame.
        timestamp : Union[BlackboardKey, rclpy.time.Time]
            The timestamp of the detected mouth center (default 0 for latest).
        world_frame : Union[BlackboardKey, str]
            The target frame to transform the mouth center to.
        frame_to_orient_towards : Union[BlackboardKey, str]
            The frame that the mouth should orient towards.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self, mouth_pose: Optional[BlackboardKey]  # PoseStamped
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        mouth_pose : Optional[BlackboardKey]
            The PoseStamped of the mouth in the base frame.
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
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Validate inputs
        if not self.blackboard_exists(
            [
                "detected_mouth_center",
                "timestamp",
                "world_frame",
                "frame_to_orient_towards",
            ]
        ):
            self.logger.error(
                "Missing detected_mouth_center, timestamp, world_frame, or frame_to_orient_towards."
            )
            return py_trees.common.Status.FAILURE
        detected_mouth_center = self.blackboard_get("detected_mouth_center")
        timestamp = self.blackboard_get("timestamp")
        world_frame = self.blackboard_get("world_frame")
        frame_to_orient_towards = self.blackboard_get("frame_to_orient_towards")

        # Lock TF Buffer
        if self.tf_lock.locked():
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING
        with self.tf_lock:
            # Transform detected_mouth_center to world_frame
            if not self.tf_buffer.can_transform(
                world_frame,
                detected_mouth_center.header.frame_id,
                timestamp,
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                self.logger.warning("ComputeMouthFrame waiting on world/camera TF...")
                return py_trees.common.Status.RUNNING
            camera_transform = self.tf_buffer.lookup_transform(
                world_frame,
                detected_mouth_center.header.frame_id,
                timestamp,
            )
            mouth_point = tf2_ros.TransformRegistration().get(PointStamped)(
                detected_mouth_center, camera_transform
            )
            self.logger.info(f"Computed mouth point: {mouth_point.point}")

            # Transform frame_to_orient_towards to world_frame
            if not self.tf_buffer.can_transform(
                world_frame,
                frame_to_orient_towards,
                timestamp,
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                self.logger.warning("ComputeMouthFrame waiting on world/forkTip TF...")
                return py_trees.common.Status.RUNNING
            orientation_target_transform = self.tf_buffer.lookup_transform(
                world_frame,
                frame_to_orient_towards,
                timestamp,
            )
            self.logger.info(
                f"Computed orientation target transform: {orientation_target_transform.transform}"
            )

        # Get the yaw of the face frame
        x_unit = Vector3(x=1.0, y=0.0, z=0.0)
        x_pos = Vector3(
            x=orientation_target_transform.transform.translation.x
            - mouth_point.point.x,
            y=orientation_target_transform.transform.translation.y
            - mouth_point.point.y,
            z=0.0,
        )
        self.logger.info(f"Computed x_pos: {x_pos}")
        quat = quat_between_vectors(x_unit, x_pos)
        self.logger.info(f"Computed orientation: {quat}")

        # Create return object
        mouth_pose = PoseStamped()
        mouth_pose.header.frame_id = mouth_point.header.frame_id
        mouth_pose.header.stamp = mouth_point.header.stamp
        mouth_pose.pose.position = mouth_point.point
        mouth_pose.pose.orientation = quat

        # Write to blackboard outputs
        self.blackboard_set("mouth_pose", mouth_pose)

        return py_trees.common.Status.SUCCESS
