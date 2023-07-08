#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import Optional, Tuple

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveTo
from ada_feeding.decorators import SetPoseGoalConstraint
from ada_feeding.trees import MoveToTree


class MoveToPoseTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToPose.
    """

    def __init__(
        self,
        action_type_class: str,
        position: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: float = 0.001,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        position: the target end effector position relative to the base link.
        quat_xyzw: the target end effector orientation relative to the base link.
        frame_id: the frame id of the target pose. If None, the base link is used.
        target_link: the link to move to the target pose. If None, the end effector
            link is used.
        tolerance_position: the tolerance for the end effector position.
        tolerance_orientation: the tolerance for the end effector orientation.
        weight_position: the weight for the end effector position.
        weight_orientation: the weight for the end effector orientation.
        """
        # Initialize MoveTo
        super().__init__(action_type_class)

        # Store the parameters for the move to pose behavior
        self.position = position
        self.quat_xyzw = quat_xyzw
        self.frame_id = frame_id
        self.target_link = target_link
        self.tolerance_position = tolerance_position
        self.tolerance_orientation = tolerance_orientation
        self.weight_position = weight_position
        self.weight_orientation = weight_orientation

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToPose behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        # Inputs for MoveToPose
        self.blackboard.register_key(
            key="position", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="quat_xyzw", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="frame_id", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="tolerance_position", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="tolerance_orientation", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="weight_position", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="weight_orientation", access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToPose to blackboard
        self.blackboard.position = self.position
        self.blackboard.quat_xyzw = self.quat_xyzw
        self.blackboard.frame_id = self.frame_id
        self.blackboard.target_link = self.target_link
        self.blackboard.tolerance_position = self.tolerance_position
        self.blackboard.tolerance_orientation = self.tolerance_orientation
        self.blackboard.weight_position = self.weight_position
        self.blackboard.weight_orientation = self.weight_orientation

        # Create the tree
        move_to = MoveTo(name, node)
        root = SetPoseGoalConstraint(name, move_to)
        root.logger = logger
        tree = py_trees.trees.BehaviourTree(root)

        return tree
