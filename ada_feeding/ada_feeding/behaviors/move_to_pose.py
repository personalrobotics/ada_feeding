#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPose behavior tree, which moves the Jaco
arm to a specified end effector pose.
"""
# Standard imports
from asyncio import Future

# Third-party imports
import py_trees

# Local imports
from ada_feeding.behaviors import MoveTo


class MoveToPose(MoveTo):
    """
    A generic behavior for moving the Jaco arm a specified end effector pose.
    """

    def __init__(self, *args, **kwargs):
        """
        A generic behavior for moving the Jaco arm a specified joint configuration.
        """
        # Initiatilize the behavior
        super().__init__(*args, **kwargs)

        # Get inputs specific to MoveToPose from the Blackboard
        self.blackboard.register_key(key="position", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="quat_xyzw", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="frame_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="tolerance_position", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance_orientation", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="weight_position", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="weight_orientation", access=py_trees.common.Access.READ
        )

    def plan_async(self) -> Future:
        """
        Get the MoveToPose parameters from the blackboard and send an
        asynchronous planning request to MoveIt.
        """
        # Get all parameters for planning, resorting to default values if unset.
        position = self.blackboard.position  # required
        quat_xyzw = self.blackboard.quat_xyzw  # required
        try:
            target_link = self.blackboard.target_link
        except KeyError:
            target_link = None  # default value
        try:
            frame_id = self.blackboard.frame_id
        except KeyError:
            frame_id = None  # default value
        try:
            tolerance_position = self.blackboard.tolerance_position
        except KeyError:
            tolerance_position = 0.001  # default value
        try:
            tolerance_orientation = self.blackboard.tolerance_orientation
        except KeyError:
            tolerance_orientation = 0.001  # default value
        try:
            weight_position = self.blackboard.weight_position
        except KeyError:
            weight_position = 1.0  # default value
        try:
            weight_orientation = self.blackboard.weight_orientation
        except KeyError:
            weight_orientation = 1.0  # default value

        # Send a new goal to MoveIt
        return self.moveit2.plan_async(
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            tolerance_position=tolerance_position,
            tolerance_orientation=tolerance_orientation,
            weight_position=weight_position,
            weight_orientation=weight_orientation,
            cartesian=self.cartesian,
        )
