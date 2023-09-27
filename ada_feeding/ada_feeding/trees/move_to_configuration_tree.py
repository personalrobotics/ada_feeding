#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import List, Set

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveTo
from ada_feeding.decorators import ClearConstraints, SetJointGoalConstraint
from ada_feeding.helpers import (
    CLEAR_CONSTRAINTS_NAMESPACE_PREFIX,
    JOINT_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
    MOVE_TO_NAMESPACE_PREFIX,
    set_to_blackboard,
)
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
    # pylint: disable=dangerous-default-value
    # A mutable default value is okay since we don't change it in this function.
    def __init__(
        self,
        joint_positions: List[float],
        tolerance: float = 0.001,
        weight: float = 1.0,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        max_velocity_scaling_factor: float = 0.1,
        max_acceleration_scaling_factor: float = 0.1,
        keys_to_not_write_to_blackboard: Set[str] = set(),
        clear_constraints: bool = True,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        joint_positions: The joint positions to move the robot arm to.
        tolerance: The tolerance for the joint positions.
        weight: The weight for the joint goal constraint.
        pipeline_id: The pipeline ID to use for the MoveIt2 motion planner.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        max_velocity_scaling_factor: The maximum velocity scaling factor for the
            MoveIt2 motion planner.
        max_acceleration_scaling_factor: The maximum acceleration scaling factor
            for the MoveIt2 motion planner.
        keys_to_not_write_to_blackboard: A set of keys that should not be written
            Note that the keys need to be exact e.g., "move_to.cartesian,"
            "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
            etc.
        clear_constraints: Whether or not to put a ClearConstraints decorator at the top
            of this branch. If you will be adding additional Constraints on top of this
            tree, this should be False. Else (e.g., if this is a standalone tree), True.
        """
        # Initialize MoveToTree
        super().__init__()

        # Store the parameters
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance = tolerance
        self.weight = weight
        self.pipeline_id = pipeline_id
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.max_acceleration_scaling_factor = max_acceleration_scaling_factor
        self.keys_to_not_write_to_blackboard = keys_to_not_write_to_blackboard
        self.clear_constraints = clear_constraints

    # pylint: disable=too-many-locals
    # Unfortunately, many local variables are required here to isolate the keys
    # of similar constraints in the blackboard.
    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
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
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        # Separate blackboard namespaces for children
        joint_constraint_namespace_prefix = JOINT_GOAL_CONSTRAINT_NAMESPACE_PREFIX
        clear_constraints_namespace_prefix = CLEAR_CONSTRAINTS_NAMESPACE_PREFIX
        move_to_namespace_prefix = MOVE_TO_NAMESPACE_PREFIX

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
        pipeline_id_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "pipeline_id"]
        )
        self.blackboard.register_key(
            key=pipeline_id_key, access=py_trees.common.Access.WRITE
        )
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
        max_velocity_scaling_factor_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "max_velocity_scaling_factor"]
        )
        self.blackboard.register_key(
            key=max_velocity_scaling_factor_key, access=py_trees.common.Access.WRITE
        )
        max_acceleration_scaling_factor_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "max_acceleration_scaling_factor"]
        )
        self.blackboard.register_key(
            key=max_acceleration_scaling_factor_key,
            access=py_trees.common.Access.WRITE,
        )

        # Write the inputs to MoveToConfiguration to blackboard
        set_to_blackboard(
            self.blackboard,
            joint_positions_key,
            self.joint_positions,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            tolerance_key,
            self.tolerance,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            weight_key,
            self.weight,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            pipeline_id_key,
            self.pipeline_id,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            planner_id_key,
            self.planner_id,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            allowed_planning_time_key,
            self.allowed_planning_time,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            max_velocity_scaling_factor_key,
            self.max_velocity_scaling_factor,
            self.keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            self.blackboard,
            max_acceleration_scaling_factor_key,
            self.max_acceleration_scaling_factor,
            self.keys_to_not_write_to_blackboard,
        )

        # Create the MoveTo behavior
        move_to_name = Blackboard.separator.join([name, move_to_namespace_prefix])
        move_to = MoveTo(move_to_name, tree_root_name, node)

        # Add the joint goal constraint to the MoveTo behavior
        joint_goal_constaint_name = Blackboard.separator.join(
            [name, joint_constraint_namespace_prefix]
        )
        joint_constraints = SetJointGoalConstraint(
            joint_goal_constaint_name, move_to, node
        )

        # Clear the constraints
        if self.clear_constraints:
            clear_constraints_name = Blackboard.separator.join(
                [name, clear_constraints_namespace_prefix]
            )
            root = ClearConstraints(clear_constraints_name, joint_constraints, node)
        else:
            root = joint_constraints

        tree = py_trees.trees.BehaviourTree(root)
        return tree
