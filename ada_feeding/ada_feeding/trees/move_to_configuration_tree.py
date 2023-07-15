#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
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
        action_type_class_str: str,
        joint_positions: List[float],
        tolerance: float = 0.001,
        weight: float = 1.0,
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class_str: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        joint_positions: The joint positions to move the robot arm to.
        tolerance: The tolerance for the joint positions.
        weight: The weight for the joint goal constraint.
        """
        # Initialize MoveToTree
        super().__init__(action_type_class_str)

        # Store the parameters
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance = tolerance
        self.weight = weight

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
        # Separate blackboard namespaces for children
        joint_constraint_namespace_prefix = "joint_goal_constraint"
        move_to_namespace_prefix = "move_to"

        # Inputs for MoveToConfiguration
        joint_positions_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "joint_positions"]
        )
        self.blackboard.register_key(
            key=joint_positions_key, access=py_trees.common.Access.WRITE
        )
        tolerance_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "tolerance"]
        )
        self.blackboard.register_key(
            key=tolerance_key, access=py_trees.common.Access.WRITE
        )
        weight_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "weight"]
        )
        self.blackboard.register_key(
            key=weight_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToConfiguration to blackboard
        self.blackboard.set(joint_positions_key, self.joint_positions)
        self.blackboard.set(tolerance_key, self.tolerance)
        self.blackboard.set(weight_key, self.weight)

        # Create the MoveTo behavior
        move_to_name = Blackboard.separator.join([name, move_to_namespace_prefix])
        move_to = MoveTo(move_to_name, name, node)
        move_to.logger = logger

        # Add the joint goal constraint to the MoveTo behavior
        joint_goal_constaint_name = Blackboard.separator.join(
            [name, joint_constraint_namespace_prefix]
        )
        root = SetJointGoalConstraint(joint_goal_constaint_name, move_to)
        root.logger = logger

        tree = py_trees.trees.BehaviourTree(root)
        return tree
