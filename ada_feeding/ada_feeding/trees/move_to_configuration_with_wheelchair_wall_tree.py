#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithWheelchairWallTree behaviour tree.
This tree was designed for the MoveToStagingConfiguration action, but can be
reused by other actions that want to move to a configuration within the scope
of adding a wall in front of the wheelchair.
"""
# pylint: disable=duplicate-code
# MoveFromMouth and MoveToMouth are inverses of each other, so it makes sense
# that they have similar code.

# Standard imports
from typing import List, Optional

# Third-party imports
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config, scoped_behavior
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_wheelchair_wall_behavior,
    get_remove_in_front_of_wheelchair_wall_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
)


class MoveToConfigurationWithWheelchairWallTree(MoveToTree):
    """
    A behaviour tree adds a wall in front of the wheelchair to the collision
    scene, moves to a specified configuration with optional orientation
    constraints, and then removes the wall from the collision scene. This
    class was designed for the MoveToStagingConfiguration action, but can be
    reused by other actions.

    TODO: Add functionality to not add the collision wall if the robot is in
    collision with it!
    https://github.com/ros-planning/moveit_msgs/blob/humble/srv/GetStateValidity.srv
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # This is intended to be a flexible tree.

    def __init__(
        self,
        node: Node,
        goal_configuration: List[float],
        goal_configuration_tolerance: float = 0.001,
        orientation_constraint_quaternion: Optional[List[float]] = None,
        orientation_constraint_tolerances: Optional[List[float]] = None,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        max_velocity_scaling_factor: float = 0.1,
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        goal_configuration: The joint positions to move the robot arm to.
        goal_configuration_tolerance: The tolerance for the joint positions.
        orientation_constraint_quaternion: The quaternion for the orientation
            constraint. If None, the orientation constraint is not used.
        orientation_constraint_tolerances: The tolerances for the orientation
            constraint, as a 3D rotation vector. If None, the orientation
            constraint is not used.
        allowed_planning_time_to: The allowed planning time for the MoveIt2
            motion planner.
        max_velocity_scaling_factor: The maximum velocity scaling factor for the
            MoveIt2 motion planner.
        force_threshold: The force threshold (N) for the ForceGateController.
        torque_threshold: The torque threshold (N*m) for the ForceGateController.
        """

        # pylint: disable=too-many-locals
        # These are all necessary due to all the behaviors MoveToMouth contains

        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters
        self.goal_configuration = goal_configuration
        assert len(self.goal_configuration) == 6, "Must provide 6 joint positions"
        self.goal_configuration_tolerance = goal_configuration_tolerance
        self.orientation_constraint_quaternion = orientation_constraint_quaternion
        self.orientation_constraint_tolerances = orientation_constraint_tolerances
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        ### Define Tree Logic

        in_front_of_wheelchair_wall_id = "in_front_of_wheelchair_wall"

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                # Retare the F/T sensor and set the F/T Thresholds
                pre_moveto_config(
                    name=name + "PreMoveToConfig",
                    toggle_watchdog_listener=False,
                    f_mag=self.force_threshold,
                    t_mag=self.torque_threshold,
                ),
                # Add a wall in front of the wheelchair to prevent the robot
                # from moving unnecessarily close to the user.
                scoped_behavior(
                    name=name + " InFrontOfWheelchairWallScope",
                    pre_behavior=get_add_in_front_of_wheelchair_wall_behavior(
                        name + "AddWheelchairWall",
                        in_front_of_wheelchair_wall_id,
                    ),
                    # Remove the wall in front of the wheelchair
                    post_behavior=get_remove_in_front_of_wheelchair_wall_behavior(
                        name + "RemoveWheelchairWall",
                        in_front_of_wheelchair_wall_id,
                    ),
                    # Move to the staging configuration
                    workers=[
                        # Goal configuration: staging configuration
                        MoveIt2JointConstraint(
                            name="StagingConfigurationGoalConstraint",
                            ns=name,
                            inputs={
                                "joint_positions": self.goal_configuration,
                                "tolerance": self.goal_configuration_tolerance,
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        # Orientation path constraint to keep the fork straight
                        MoveIt2OrientationConstraint(
                            name="KeepForkStraightPathConstraint",
                            ns=name,
                            inputs={
                                "quat_xyzw": self.orientation_constraint_quaternion,
                                "tolerance": self.orientation_constraint_tolerances,
                                "parameterization": 1,  # Rotation vector
                            },
                            outputs={
                                "constraints": BlackboardKey("path_constraints"),
                            },
                        ),
                        # Plan
                        MoveIt2Plan(
                            name="MoveToStagingConfigurationPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "path_constraints": BlackboardKey("path_constraints"),
                                "planner_id": self.planner_id,
                                "allowed_planning_time": self.allowed_planning_time,
                                "max_velocity_scale": self.max_velocity_scaling_factor,
                                "ignore_violated_path_constraints": True,
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        # Execute
                        MoveIt2Execute(
                            name="MoveToStagingConfigurationExecute",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
                            outputs={},
                        ),
                    ],
                ),
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
