#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import Tuple

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveToPose
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
        """
        # Initialize MoveTo
        super().__init__(action_type_class)

        # Store the parameters for the move to pose behavior
        self.position = position
        self.quat_xyzw = quat_xyzw

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
            key="target_link", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="frame_id", access=py_trees.common.Access.WRITE
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

        # Create the tree
        root = MoveToPose(name, node)
        root.logger = logger
        tree = py_trees.trees.BehaviourTree(root)

        return tree
