#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeFoodFrame behavior, which computes the
food frame from 
"""
# Standard imports
from typing import Union

# Third-party imports
from geometry_msgs.msg import TransformStamped
import py_trees
import pyrealsense2
import rclpy
from rclpy.node import Node
import tf2_geometry_msgs
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_feeding_msgs.msg import Mask
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ComputeFoodFrame(BlackboardBehavior):
    """
    Computes the food reference frame 
    """

    def blackboard_inputs(
        self,
        ros2_node: Union[BlackboardKey, Node],
        camera_info: Union[BlackboardKey, CameraInfo],
        mask: Union[BlackboardKey, Mask],
        camera_frame: Union[BlackboardKey, str] = "camera_color_optical_frame",
        world_frame: Union[BlackboardKey, str] = "world",
    ) -> None:
        super().blackboard_inputs(**{key: value for key, value in locals().items() if key != 'self'})

    def blackboard_outputs(
        self,
        action_select_request: BlackboardKey,   # AcquisitionSelect.Request
        food_frame: BlackboardKey               # TransformStamped
    ) -> None:
        super().blackboard_outputs(**{key: value for key, value in locals().items() if key != 'self'})


    def setup(self):
        """
        Middleware (i.e. TF) setup
        """

        # Create TF Tree
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tfbuffer, self.blackboard_get("ros2_node"))

    def initialise(self):
        """
        Behavior initialization
        """

        # Construct camera intrinsics
        # See: https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        camera_info = self.blackboard_get("camera_info")
        self.intrinsics = pyrealsense2.intrinsics()
        self.intrinsics.width = camera_info.width
        self.intrinsics.height = camera_info.height
        self.intrinsics.ppx = camera_info.K[2]
        self.intrinsics.ppy = camera_info.K[5]
        self.intrinsics.fx = camera_info.K[0]
        self.intrinsics.fy = camera_info.K[4]
        self.intrinsics.model = camera_info.distortion_model
        self.intrinsics.coeffs = [i for i in camera_info.D]


    def update(self) -> py_trees.common.Status:
        """
        Behavior tick (DO NOT BLOCK)
        """
        
        # Check if we have the camera transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.blackboard_get("world_frame"),
                self.blackboard_get("camera_frame_frame"),
                rclpy.time.Time())
        except TransformException as ex:
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING

        # Set up return objects
        world_to_food_transform = TransformStamped()
        world_to_food_transform.header.stamp = rclpy.time.Time()
        world_to_food_transform.header.frame_id = self.blackboard_get("world_frame")
        world_to_food_transform.child_frame_id = "food"

        # De-project center of ROI
        mask = self.blackboard_get("mask")
        center_x = mask.x_offset + mask.width // 2
        center_y = mask.y_offset + mask.height // 2
        center_3d = 

        # Write to blackboard outputs

        return py_trees.common.Status.SUCCESS

class FlipFoodFrame(py_trees.behavior.Behavior):
    """
    TODO
    Take the food frame from the blackboard and 
    rotate it PI about the world frame +Z axis.
    """

    def __init__(
        self,
        name: str,
        node: Node,
        frame_blackboard_key: str,
        world_frame_id: str,
    ) -> None:
        # Initiatilize the behavior
        super().__init__(name=name)
        pass

    def setup(self, **kwargs) -> None:
        pass

    def update(self) -> py_trees.common.Status:
        pass
