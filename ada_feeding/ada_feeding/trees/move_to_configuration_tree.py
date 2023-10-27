#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import List, Set

# Third-party imports
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2JointConstraint,
)
from ada_feeding.helpers import BlackboardKey
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
        node: Node,
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
        super().__init__(node)

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
    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                MoveIt2JointConstraint(
                    name="JointConstraint",
                    ns=name,
                    inputs={
                        "joint_positions": self.joint_positions,
                        "tolerance": self.tolerance,
                        "weight": self.weight,
                    },
                    outputs={
                        "constraints": BlackboardKey("goal_constraints"),
                    },
                ),
                MoveIt2Plan(
                    name="MoveToConfigurationPlan",
                    ns=name,
                    inputs={
                        "goal_constraints": BlackboardKey("goal_constraints"),
                        "pipeline_id": self.pipeline_id,
                        "planner_id": self.planner_id,
                        "allowed_planning_time": self.allowed_planning_time,
                        "max_velocity_scale": self.max_velocity_scaling_factor,
                        "max_acceleration_scale": self.max_acceleration_scaling_factor,
                    },
                    outputs={"trajectory": BlackboardKey("trajectory")},
                ),
                MoveIt2Execute(
                    name="MoveToConfigurationExecute",
                    ns=name,
                    inputs={"trajectory": BlackboardKey("trajectory")},
                    outputs={},
                ),
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
