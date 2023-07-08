#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetPoseGoalConstraint decorator, which adds a
pose goal to any behavior that moves the robot using MoveIt2.
"""
# Third-party imports
import py_trees

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default


class SetPoseGoalConstraint(MoveToConstraint):
    """
    SetPoseGoalConstraint adds pose goal constraints to any
    behavior that moves the robot using MoveIt2.
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
    ):
        """
        Initialize the MoveToConstraint decorator.

        Parameters
        ----------
        name: The name of the behavior.
        child: The child behavior.
        """
        # Initiatilize the decorator
        super().__init__(name=name, child=child)

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " SetPoseGoalConstraint", namespace=name
        )
        self.blackboard.register_key(key="position", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="quat_xyzw", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="frame_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.READ
        )
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

    def set_constraint(self) -> None:
        """
        Sets the constraint. For example, this function can call:
          - self.moveit2.set_joint_goal(...)
          - self.moveit2.set_path_position_constraint(...)
          - and so on.
        """
        self.logger.info("%s [SetPoseGoalConstraint::set_constraint()]" % self.name)

        # Get all parameters for planning, resorting to default values if unset.
        position = self.blackboard.position  # required
        quat_xyzw = self.blackboard.quat_xyzw  # required
        frame_id = get_from_blackboard_with_default(self.blackboard, "frame_id", None)
        target_link = get_from_blackboard_with_default(
            self.blackboard, "target_link", None
        )
        tolerance_position = get_from_blackboard_with_default(
            self.blackboard, "tolerance_position", 0.001
        )
        tolerance_orientation = get_from_blackboard_with_default(
            self.blackboard, "tolerance_orientation", 0.001
        )
        weight_position = get_from_blackboard_with_default(
            self.blackboard, "weight_position", 1.0
        )
        weight_orientation = get_from_blackboard_with_default(
            self.blackboard, "weight_orientation", 1.0
        )

        # Set the constraint
        self.moveit2.set_pose_goal(
            position=position,
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            target_link=target_link,
            tolerance_position=tolerance_position,
            tolerance_orientation=tolerance_orientation,
            weight_position=weight_position,
            weight_orientation=weight_orientation,
        )
