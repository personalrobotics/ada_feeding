#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfiguration behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveTo
from ada_feeding.decorators import SetJointGoalConstraint
from ada_feeding.trees import MoveToTree


class MoveToConfigurationTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration.
    """

    def __init__(
        self,
        action_type_class: str,
        joint_positions: List[float],
        tolerance: float = 0.001,
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        joint_positions: The joint positions to move the robot arm to.
        tolerance: The tolerance for the joint positions.
        """
        # Initialize MoveToTree
        super().__init__(action_type_class)

        # Store the parameters
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance = tolerance

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToConfiguration behavior tree.

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
        # Inputs for MoveToConfiguration
        self.blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToConfiguration to blackboard
        self.blackboard.joint_positions = self.joint_positions
        self.blackboard.tolerance = self.tolerance

        # Create the tree
        move_to = MoveTo(name, node)
        root = SetJointGoalConstraint(name, move_to)
        root.logger = logger
        tree = py_trees.trees.BehaviourTree(root)

        return tree
