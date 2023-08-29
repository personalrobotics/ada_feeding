#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeMoveToMouthPosition behavior, which computes the
target position to move the robot's end effector to based on the detected mouth
center. Specifically, it transforms the detected mouth center from the camera
frame to the requested frame and then adds an offset.
"""
# Standard imports
from typing import Tuple

# Third-party imports
import py_trees
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import PointStamped  # pylint: disable=unused-import
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Local imports


class ComputeMoveToMouthPosition(py_trees.behaviour.Behaviour):
    """
    A behavior that computes the target position to move the robot's end effector
    to based on the detected mouth center. Specifically, it transforms the
    detected mouth center from the camera frame to the requested frame and then
    adds an offset.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # A few over is fine. All are necessary.

    def __init__(
        self,
        name: str,
        node: Node,
        face_detection_input_key: str,
        target_position_output_key: str,
        target_position_frame_id: str,
        target_position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initializes the behavior.

        Parameters
        ----------
        name: The name of the behavior.
        node: The ROS node.
        face_detection_input_key: The key for the face detection input on the blackboard.
        target_position_output_key: The key for the target position output on the blackboard.
        target_position_frame_id: The frame ID for the target position.
        target_position_offset: The offset to add to the target position, in
            `target_position_frame_id`
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.face_detection_input_key = face_detection_input_key
        self.target_position_output_key = target_position_output_key
        self.target_position_frame_id = target_position_frame_id
        self.target_position_offset = target_position_offset

        # Initialization the blackboard for this behavior
        self.blackboard = self.attach_blackboard_client(
            name=name + "ComputeMoveToMouthPosition", namespace=name
        )
        # Read the results of face detection
        self.blackboard.register_key(
            key=self.face_detection_input_key, access=py_trees.common.Access.READ
        )
        # Write the target position
        self.blackboard.register_key(
            key=self.target_position_output_key, access=py_trees.common.Access.WRITE
        )

    # pylint: disable=attribute-defined-outside-init
    # It is reasonable to define attributes in setup, since that will be run once
    # to initialize the behavior.
    def setup(self, **kwargs) -> None:
        """
        Subscribe to tf2 transforms.
        """
        self.logger.info(f"{self.name} [ComputeMoveToMouthPosition::setup()]")
        # Initialize the tf2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

    def update(self) -> py_trees.common.Status:
        """
        Computes the target position to move the robot's end effector to based on
        the detected mouth center. Specifically, it transforms the detected mouth
        center from the camera frame to the requested frame and then adds an offset.
        It immediately returns either SUCCESS or FAILURE, and never returns RUNNING.
        """
        self.logger.info(f"{self.name} [ComputeMoveToMouthPosition::update()]")
        # Get the face detection message from the blackboard. If it doesn't
        # exist, then return failure.
        try:
            face_detection = self.blackboard.get(self.face_detection_input_key)
        except KeyError:
            self.logger.error(
                f"{self.name} [ComputeMoveToMouthPosition::update()] "
                "Face detection message not found in blackboard"
            )
            return py_trees.common.Status.FAILURE

        # Transform the face detection result to the base frame. If the
        # transform doesn't exist, then return failure.
        try:
            try:
                target_position = self.tf_buffer.transform(
                    face_detection.detected_mouth_center, self.target_position_frame_id
                )
                self.logger.info(
                    f"{self.name} [ComputeMoveToMouthPosition::update()] "
                    f"face_detection.detected_mouth_center {face_detection.detected_mouth_center} "
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
                face_detection.detected_mouth_center.header.stamp = Time().to_msg()
                target_position = self.tf_buffer.transform(
                    face_detection.detected_mouth_center,
                    self.target_position_frame_id,
                )
        except tf2.ExtrapolationException as exc:
            self.logger.error(
                f"%{self.name} [ComputeMoveToMouthPosition::update()] "
                f"Failed to transform face detection result to base frame: {type(exc)}: {exc}"
            )
            return py_trees.common.Status.FAILURE

        # Write the target position to the blackboard
        final_position = (
            target_position.point.x + self.target_position_offset[0],
            target_position.point.y + self.target_position_offset[1],
            target_position.point.z + self.target_position_offset[2],
        )
        self.blackboard.set(
            self.target_position_output_key,
            final_position,
        )
        return py_trees.common.Status.SUCCESS
