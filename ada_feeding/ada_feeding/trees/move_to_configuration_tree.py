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

# pylint: disable=duplicate-code
# move_to_configuration_tree.py has similar code to move_to_pose.py when defining
# blackboard variables that are necessary for all movements (e.g., planner_id).
# This is not a problem.


class MoveToConfigurationTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    def __init__(
        self,
        joint_positions: List[float],
        tolerance: float = 0.001,
        weight: float = 1.0,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        joint_positions: The joint positions to move the robot arm to.
        tolerance: The tolerance for the joint positions.
        weight: The weight for the joint goal constraint.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        """
        # Initialize MoveToTree
        super().__init__()

        # Store the parameters
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance = tolerance
        self.weight = weight
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time

    # pylint: disable=too-many-locals
    # Unfortunately, many local variables are required here to isolate the keys
    # of similar constraints in the blackboard.
    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToConfiguration behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables.
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

        # Inputs for MoveTo
        planner_id_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "planner_id"]
        )
        self.blackboard.register_key(
            key=planner_id_key, access=py_trees.common.Access.WRITE
        )
        allowed_planning_time_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "allowed_planning_time"]
        )
        self.blackboard.register_key(
            key=allowed_planning_time_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToConfiguration to blackboard
        self.blackboard.set(joint_positions_key, self.joint_positions)
        self.blackboard.set(tolerance_key, self.tolerance)
        self.blackboard.set(weight_key, self.weight)
        self.blackboard.set(planner_id_key, self.planner_id)
        self.blackboard.set(allowed_planning_time_key, self.allowed_planning_time)

        # Create the MoveTo behavior
        move_to_name = Blackboard.separator.join([name, move_to_namespace_prefix])
        move_to = MoveTo(move_to_name, tree_root_name, node)
        move_to.logger = logger

        # Add the joint goal constraint to the MoveTo behavior
        joint_goal_constaint_name = Blackboard.separator.join(
            [name, joint_constraint_namespace_prefix]
        )
        root = SetJointGoalConstraint(joint_goal_constaint_name, move_to)
        root.logger = logger

        tree = py_trees.trees.BehaviourTree(root)
        return tree
