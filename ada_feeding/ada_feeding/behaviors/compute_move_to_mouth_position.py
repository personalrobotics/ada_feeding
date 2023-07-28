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
from tf2_geometry_msgs import PointStamped
import py_trees
from rclpy.node import Node
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
        target_position_offset: The offset to add to the target position, in `target_position_frame_id`
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.face_detection_input_key = face_detection_input_key
        self.target_position_output_key = target_position_output_key
        self.target_position_frame_id = target_position_frame_id

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
    
        
    def setup(self, **kwargs) -> None:
        """
        Subscribe to tf2 transforms.
        """
        self.logger.info("%s [ComputeMoveToMouthPosition::setup()]" % self.name)
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
        self.logger.info("%s [ComputeMoveToMouthPosition::update()]" % self.name)
        # Get the face detection message from the blackboard. If it doesn't
        # exist, then return failure.
        try:
            face_detection = self.blackboard.get(self.face_detection_input_key)
        except KeyError:
            self.logger.error(
                "%s [ComputeMoveToMouthPosition::update()] "
                "Face detection message not found in blackboard" % self.name
            )
            return py_trees.common.Status.FAILURE

        # Transform the face detection result to the base frame. If the
        # transform doesn't exist, then return failure.
        # raise Exception("%s | %s" % (type(face_detection.detected_mouth_center), type(PointStamped())))
        try:
            target_position = self.tf_buffer.transform(face_detection.detected_mouth_center, self.target_position_frame_id)
            self.logger.info("%s [ComputeMoveToMouthPosition::update()] face_detection.detected_mouth_center %s target_position %s " % (self.name, face_detection.detected_mouth_center, target_position))
        except Exception as e:
            self.logger.error(
                "%s [ComputeMoveToMouthPosition::update()] "
                "Failed to transform face detection result to base frame: %s" % (self.name, e)
            )
            return py_trees.common.Status.FAILURE

        # Write the target position to the blackboard
        self.blackboard.set(self.target_position_output_key, (target_position.point.x, target_position.point.y, target_position.point.z))
        return py_trees.common.Status.SUCCESS