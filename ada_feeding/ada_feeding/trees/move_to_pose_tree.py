#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import Optional, Set, Tuple, Union

# Third-party imports
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2PositionConstraint,
    MoveIt2OrientationConstraint,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.trees import MoveToTree

# pylint: disable=duplicate-code
# move_to_pose.py has similar code to move_to_configuration_tree.py when defining
# blackboard variables that are necessary for all movements (e.g., planner_id).
# This is not a problem.


class MoveToPoseTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToPose.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.
    # pylint: disable=dangerous-default-value
    # A mutable default value is okay since we don't change it in this function.
    def __init__(
        self,
        node: Node,
        position: Optional[Tuple[float, float, float]] = None,
        quat_xyzw: Optional[Tuple[float, float, float, float]] = None,
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: Union[float, Tuple[float, float, float]] = 0.001,
        parameterization: int = 0,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
        cartesian: bool = False,
        cartesian_jump_threshold: float = 0.0,
        cartesian_max_step: float = 0.0025,
        cartesian_fraction_threshold: float = 0.0,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTConnectkConfigDefault",
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
        position: the target end effector position relative to the base link.
        quat_xyzw: the target end effector orientation relative to the base link.
        frame_id: the frame id of the target pose. If None, the base link is used.
        target_link: the link to move to the target pose. If None, the end effector
            link is used.
        tolerance_position: the tolerance for the end effector position.
        tolerance_orientation: the tolerance for the end effector orientation.
        parameterization: the parameterization of the orientation goal constraint.
            0 is Euler angles, 1 is rotation vector
        weight_position: the weight for the end effector position.
        weight_orientation: the weight for the end effector orientation.
        cartesian: whether to use cartesian path planning.
        cartesian_jump_threshold: the jump threshold for cartesian path planning.
        cartesian_max_step: the maximum step for cartesian path planning.
        cartesian_fraction_threshold: Reject cartesian plans that don't reach
            at least this fraction of the path to the goal.
        pipeline_id: the pipeline to use for path planning.
        planner_id: the planner to use for path planning.
        allowed_planning_time: the allowed planning time for path planning.
        max_velocity_scaling_factor: the maximum velocity scaling factor for path
            planning.
        max_acceleration_scaling_factor: the maximum acceleration scaling factor for
            path planning.
        keys_to_not_write_to_blackboard: the keys to not write to the blackboard.
            Note that the keys need to be exact e.g., "move_to.cartesian,"
            "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
            etc.
        clear_constraints: Whether or not to put a ClearConstraints decorator at the top
            of this branch. If you will be adding additional Constraints on top of this
            tree, this should be False. Else (e.g., if this is a standalone tree), True.
        """
        # Initialize MoveTo
        super().__init__(node)

        # Store the parameters for the move to pose behavior
        self.position = position
        self.quat_xyzw = quat_xyzw
        self.frame_id = frame_id
        self.target_link = target_link
        self.tolerance_position = tolerance_position
        self.tolerance_orientation = tolerance_orientation
        self.parameterization = parameterization
        self.weight_position = weight_position
        self.weight_orientation = weight_orientation
        self.cartesian = cartesian
        self.cartesian_jump_threshold = cartesian_jump_threshold
        self.cartesian_max_step = cartesian_max_step
        self.cartesian_fraction_threshold = cartesian_fraction_threshold
        self.pipeline_id = pipeline_id
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.max_acceleration_scaling_factor = max_acceleration_scaling_factor
        self.keys_to_not_write_to_blackboard = keys_to_not_write_to_blackboard
        self.clear_constraints = clear_constraints

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
            children=[],
        )

        # Add the pose goal constraint
        goal_constraints_key = BlackboardKey("goal_constraints")
        if self.position is not None:
            root_seq.add_child(
                MoveIt2PositionConstraint(
                    name="PositionGoalConstraint",
                    ns=name,
                    inputs={
                        "position": self.position,
                        "frame_id": self.frame_id,
                        "target_link": self.target_link,
                        "tolerance": self.tolerance_position,
                        "weight": self.weight_position,
                    },
                    outputs={
                        "constraints": goal_constraints_key,
                    },
                )
            )
        if self.quat_xyzw is not None:
            inputs = {
                "quat_xyzw": self.quat_xyzw,
                "frame_id": self.frame_id,
                "target_link": self.target_link,
                "tolerance": self.tolerance_orientation,
                "weight": self.weight_orientation,
                "parameterization": self.parameterization,
            }
            if self.position is not None:
                inputs["constraints"] = goal_constraints_key
            root_seq.add_child(
                MoveIt2OrientationConstraint(
                    name="OrientationGoalConstraint",
                    ns=name,
                    inputs=inputs,
                    outputs={
                        "constraints": goal_constraints_key,
                    },
                )
            )

        # Add the planning behavior
        root_seq.add_child(
            py_trees.decorators.Timeout(
                name="MoveToPosePlanTimeout",
                # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                duration=10.0 * self.allowed_planning_time,
                child=MoveIt2Plan(
                    name="MoveToPosePlan",
                    ns=name,
                    inputs={
                        "goal_constraints": goal_constraints_key,
                        "pipeline_id": self.pipeline_id,
                        "planner_id": self.planner_id,
                        "allowed_planning_time": self.allowed_planning_time,
                        "max_velocity_scale": self.max_velocity_scaling_factor,
                        "max_acceleration_scale": self.max_acceleration_scaling_factor,
                        "cartesian": self.cartesian,
                        "cartesian_jump_threshold": self.cartesian_jump_threshold,
                        "cartesian_max_step": self.cartesian_max_step,
                        "cartesian_fraction_threshold": self.cartesian_fraction_threshold,
                    },
                    outputs={"trajectory": BlackboardKey("trajectory")},
                ),
            ),
        )

        # Add the execution behavior
        root_seq.add_child(
            MoveIt2Execute(
                name="MoveToPoseExecute",
                ns=name,
                inputs={"trajectory": BlackboardKey("trajectory")},
                outputs={},
            ),
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
