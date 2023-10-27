#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeMoveToMouthPosition behavior, which computes the
target position to move the robot's end effector to based on the detected mouth
center. Specifically, it transforms the detected mouth center from the camera
frame to the requested frame and then adds an offset.
"""
# Standard imports
import copy
from typing import Optional, Tuple, Union

# Third-party imports
from overrides import override
import py_trees
from rclpy.time import Time
from tf2_geometry_msgs import PointStamped  # pylint: disable=unused-import
import tf2_py as tf2

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    get_tf_object,
)


class ComputeMoveToMouthPosition(BlackboardBehavior):
    """
    A behavior that computes the target position to move the robot's end effector
    to based on the detected mouth center. Specifically, it transforms the
    detected mouth center from the camera frame to the requested frame and then
    adds an offset.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        face_detection_msg: Union[BlackboardKey, FaceDetection],
        frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
        position_offset: Union[BlackboardKey, Tuple[float, float, float]] = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        face_detection_msg: The face detection message.
        frame_id: The frame ID for the target position.
        position_offset: The offset to add to the target position, in `frame_id`.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        target_position: Optional[BlackboardKey],  # PointStamped
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        target_position: The target position to move the robot's end effector to
            will be written to this key, as a PointStamped in `frame_id`.
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
        # For transform approach -> end_effector_frame
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    def update(self) -> py_trees.common.Status:
        """
        Computes the target position to move the robot's end effector to based on
        the detected mouth center. Specifically, it transforms the detected mouth
        center from the camera frame to the requested frame and then adds an offset.
        It immediately returns either SUCCESS or FAILURE, and never returns RUNNING.
        """
        self.logger.info(f"{self.name} [ComputeMoveToMouthPosition::update()]")

        # Get the inputs from the blackboard
        face_detection_msg = self.blackboard_get("face_detection_msg")
        frame_id = self.blackboard_get("frame_id")
        position_offset = self.blackboard_get("position_offset")

        # Transform the face detection result to the base frame.
        if self.tf_lock.locked():
            return py_trees.common.Status.RUNNING
        with self.tf_lock:
            try:
                target_position = self.tf_buffer.transform(
                    face_detection_msg.detected_mouth_center, frame_id
                )
                self.logger.debug(
                    f"{self.name} [ComputeMoveToMouthPosition::update()] "
                    "face_detection.detected_mouth_center "
                    f"{face_detection_msg.detected_mouth_center} "
                    f"target_position {target_position}"
                )
            except tf2.ExtrapolationException as exc:
                # If the transform failed at the timestamp in the message, retry
                # with the latest transform
                self.logger.warning(
                    f"{self.name} [ComputeMoveToMouthPosition::update()] "
                    f"Transform failed at timestamp in message: {type(exc)}: {exc}. "
                    "Retrying with latest transform."
                )
                detected_mouth_center = copy.deepcopy(
                    face_detection_msg.detected_mouth_center
                )
                detected_mouth_center.header.stamp = Time().to_msg()
                try:
                    target_position = self.tf_buffer.transform(
                        detected_mouth_center,
                        frame_id,
                    )
                except tf2.ExtrapolationException as exc: # pylint: disable=redefined-outer-name
                    # If the transform doesn't exist, then return failure.
                    self.logger.error(
                        f"%{self.name} [ComputeMoveToMouthPosition::update()] "
                        f"Failed to transform face detection result: {type(exc)}: {exc}"
                    )
                    return py_trees.common.Status.FAILURE

        # Write the target position to the blackboard
        target_position.point.x += position_offset[0]
        target_position.point.y += position_offset[1]
        target_position.point.z += position_offset[2]
        self.blackboard_set("target_position", target_position)
        return py_trees.common.Status.SUCCESS
