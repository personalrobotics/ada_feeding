# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

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
from py_trees.behaviours import Success

# Local imports
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2ComputeFK,
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
)
from ada_feeding.behaviors.state import (
    GetJointStates,
    CheckArticutoolPathDynamicFeasibility,
    LoadPinocchioModel,
    ExtractPoseFromPosesByLink,
    ComputeSlerpMidpointOrientation,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config, scoped_behavior
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_face_wall_behavior,
    get_remove_in_front_of_face_wall_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
)

import numpy as np


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
        allowed_planning_time: float = 2.0,
        max_velocity_scaling_factor: float = 0.1,
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
        lock_joints: bool = False,
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
        self.allowed_planning_time = allowed_planning_time if not lock_joints else 15.0
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.lock_joints = lock_joints

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        ### Define Tree Logic

        constraints = [
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
        ]
        if self.lock_joints:
            dynamic_path_constraint_sequence = [
                # 1. Get the current joint state for the Jaco arm
                GetJointStates(
                    name="GetJacoStartState",
                    ns=name,
                    node=self._node,
                    inputs={
                        "joint_names": [
                            "j2n6s200_joint_1",
                            "j2n6s200_joint_2",
                            "j2n6s200_joint_3",
                            "j2n6s200_joint_4",
                            "j2n6s200_joint_5",
                            "j2n6s200_joint_6",
                        ]
                    },
                    outputs={
                        "joint_state": BlackboardKey("current_jaco_joint_state"),
                        "joint_positions": None,
                        "joint_names": None,
                    },
                ),
                # 2. Use FK to calculate the current EE pose
                MoveIt2ComputeFK(
                    name="GetStartEEPose",
                    ns=name,
                    inputs={
                        "group_name": "jaco_arm",
                        "joint_state": BlackboardKey("current_jaco_joint_state"),
                        "fk_link_names": ["tool_tip"],
                    },
                    outputs={
                        "fk_poses": BlackboardKey("start_ee_fk_poses"),
                        "success": None,
                    },
                ),
                ExtractPoseFromPosesByLink(
                    name="ExtractStartEEPose",
                    ns=name,
                    inputs={
                        "fk_poses": BlackboardKey("start_ee_fk_poses"),
                        "target_link_name": "tool_tip",
                        "requested_link_names": ["tool_tip"],
                    },
                    outputs={
                        "extracted_pose": BlackboardKey("start_ee_pose"),
                        "success": None,
                    },
                ),
                # 3. Use FK to calculate the goal EE pose
                MoveIt2ComputeFK(
                    name="GetGoalEEPose",
                    ns=name,
                    inputs={
                        "group_name": "jaco_arm",
                        "joint_state": self.goal_configuration,
                        "fk_link_names": ["tool_tip"],
                    },
                    outputs={
                        "fk_poses": BlackboardKey("goal_ee_fk_poses"),
                        "success": None,
                    },
                ),
                ExtractPoseFromPosesByLink(
                    name="ExtractGoalEEPose",
                    ns=name,
                    inputs={
                        "fk_poses": BlackboardKey("goal_ee_fk_poses"),
                        "target_link_name": "tool_tip",
                        "requested_link_names": ["tool_tip"],
                    },
                    outputs={
                        "extracted_pose": BlackboardKey("goal_ee_pose"),
                        "success": None,
                    },
                ),
                # 4. Compute the SLERP midpoint using the calculated poses
                ComputeSlerpMidpointOrientation(
                    name="CalculateStagingPathConstraint",
                    ns=name,
                    inputs={
                        "start_ee_pose": BlackboardKey("start_ee_pose"),
                        "goal_ee_pose": BlackboardKey("goal_ee_pose"),
                        "tolerance": (
                            np.deg2rad(20.0),
                            np.pi,
                            np.deg2rad(20.0),
                        ),
                    },
                    outputs={
                        "midpoint_orientation_quaternion": BlackboardKey(
                            "path_constraint_quat"
                        ),
                        "path_constraint_tolerance": BlackboardKey(
                            "path_constraint_tol"
                        ),
                    },
                ),
                # 5. Create the MoveIt2 orientation constraint message
                MoveIt2OrientationConstraint(
                    name="CreateStagingPathConstraintMsg",
                    ns=name,
                    inputs={
                        "quat_xyzw": BlackboardKey("path_constraint_quat"),
                        "tolerance": BlackboardKey("path_constraint_tol"),
                    },
                    outputs={"constraints": BlackboardKey("path_constraints")},
                ),
            ]
        else:
            dynamic_path_constraint_sequence = [
                # 1. Get the current joint state for the Jaco arm
                GetJointStates(
                    name="GetJacoStartState",
                    ns=name,
                    node=self._node,
                    inputs={
                        "joint_names": [
                            "j2n6s200_joint_1",
                            "j2n6s200_joint_2",
                            "j2n6s200_joint_3",
                            "j2n6s200_joint_4",
                            "j2n6s200_joint_5",
                            "j2n6s200_joint_6",
                        ]
                    },
                    outputs={
                        "joint_state": BlackboardKey("current_jaco_joint_state"),
                        "joint_positions": None,
                        "joint_names": None,
                    },
                ),
                # 2. Use FK to calculate the current EE pose
                MoveIt2ComputeFK(
                    name="GetStartEEPose",
                    ns=name,
                    inputs={
                        "group_name": "jaco_arm",
                        "joint_state": BlackboardKey("current_jaco_joint_state"),
                        "fk_link_names": ["j2n6s200_end_effector"],
                    },
                    outputs={
                        "fk_poses": BlackboardKey("start_ee_fk_poses"),
                        "success": None,
                    },
                ),
                ExtractPoseFromPosesByLink(
                    name="ExtractStartEEPose",
                    ns=name,
                    inputs={
                        "fk_poses": BlackboardKey("start_ee_fk_poses"),
                        "target_link_name": "j2n6s200_end_effector",
                        "requested_link_names": ["j2n6s200_end_effector"],
                    },
                    outputs={
                        "extracted_pose": BlackboardKey("start_ee_pose"),
                        "success": None,
                    },
                ),
                # 3. Use FK to calculate the goal EE pose
                MoveIt2ComputeFK(
                    name="GetGoalEEPose",
                    ns=name,
                    inputs={
                        "group_name": "jaco_arm",
                        "joint_state": self.goal_configuration,
                        "fk_link_names": ["j2n6s200_end_effector"],
                    },
                    outputs={
                        "fk_poses": BlackboardKey("goal_ee_fk_poses"),
                        "success": None,
                    },
                ),
                ExtractPoseFromPosesByLink(
                    name="ExtractGoalEEPose",
                    ns=name,
                    inputs={
                        "fk_poses": BlackboardKey("goal_ee_fk_poses"),
                        "target_link_name": "j2n6s200_end_effector",
                        "requested_link_names": ["j2n6s200_end_effector"],
                    },
                    outputs={
                        "extracted_pose": BlackboardKey("goal_ee_pose"),
                        "success": None,
                    },
                ),
                # 4. Compute the SLERP midpoint using the calculated poses
                ComputeSlerpMidpointOrientation(
                    name="CalculateStagingPathConstraint",
                    ns=name,
                    inputs={
                        "start_ee_pose": BlackboardKey("start_ee_pose"),
                        "goal_ee_pose": BlackboardKey("goal_ee_pose"),
                        "tolerance": (np.pi / 2, 2 * np.pi, np.pi / 4),
                    },
                    outputs={
                        "midpoint_orientation_quaternion": BlackboardKey(
                            "path_constraint_quat"
                        ),
                        "path_constraint_tolerance": BlackboardKey(
                            "path_constraint_tol"
                        ),
                    },
                ),
                # 5. Create the MoveIt2 orientation constraint message
                MoveIt2OrientationConstraint(
                    name="CreateStagingPathConstraintMsg",
                    ns=name,
                    inputs={
                        "quat_xyzw": BlackboardKey("path_constraint_quat"),
                        "tolerance": BlackboardKey("path_constraint_tol"),
                    },
                    outputs={"constraints": BlackboardKey("path_constraints")},
                ),
            ]

        if self.lock_joints:
            plan_and_execute_sequence = [
                # Plan
                py_trees.decorators.Timeout(
                    name="MoveToStagingConfigurationPlanTimeout",
                    # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                    duration=10.0 * self.allowed_planning_time,
                    child=MoveIt2Plan(
                        name="MoveToStagingConfigurationPlan",
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "path_constraints": BlackboardKey("path_constraints"),
                            "planner_id": self.planner_id,
                            "allowed_planning_time": self.allowed_planning_time,
                            "max_velocity_scale": self.max_velocity_scaling_factor,
                            "ignore_violated_path_constraints": False,
                            "group_name": "jaco_arm",
                            "target_link": "tool_tip",
                        },
                        outputs={"trajectory": BlackboardKey("trajectory")},
                    ),
                ),
                # Execute
                MoveIt2Execute(
                    name="MoveToStagingConfigurationExecute",
                    ns=name,
                    inputs={
                        "trajectory": BlackboardKey("trajectory"),
                        "group_name": "jaco_arm",
                    },
                    outputs={},
                ),
            ]
        else:
            plan_and_execute_sequence = [
                # Plan
                py_trees.decorators.Timeout(
                    name="MoveToStagingConfigurationPlanTimeout",
                    # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                    duration=10.0 * self.allowed_planning_time,
                    child=MoveIt2Plan(
                        name="MoveToStagingConfigurationPlan",
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "path_constraints": BlackboardKey("path_constraints"),
                            "planner_id": self.planner_id,
                            "allowed_planning_time": self.allowed_planning_time,
                            "max_velocity_scale": self.max_velocity_scaling_factor,
                            "ignore_violated_path_constraints": False,
                            "group_name": "jaco_arm",
                        },
                        outputs={"trajectory": BlackboardKey("trajectory")},
                    ),
                ),
                CheckArticutoolPathDynamicFeasibility(
                    name="CheckArticutoolDynamicFeasibilityForStaging",
                    ns=name,
                    inputs={
                        "pinocchio_model": BlackboardKey("pinocchio_model"),
                        "pinocchio_data": BlackboardKey("pinocchio_data"),
                        "jaco_joint_names_pin": [
                            "j2n6s200_joint_1",
                            "j2n6s200_joint_2",
                            "j2n6s200_joint_3",
                            "j2n6s200_joint_4",
                            "j2n6s200_joint_5",
                            "j2n6s200_joint_6",
                        ],
                        "jaco_ee_frame_id_pin": BlackboardKey("jaco_ee_frame_id_pin"),
                        "jaco_trajectory": BlackboardKey("trajectory"),
                        "articutool_pitch_limits_rad": (-np.pi / 2, np.pi / 2),
                        "articutool_roll_limits_rad": (-np.pi, np.pi),
                        "jaco_vel_indices_pin": BlackboardKey("jaco_vel_indices_pin"),
                        "articutool_max_joint_velocity": 4.0,
                    },
                    outputs={
                        "articutool_is_dynamic_feasible": BlackboardKey(
                            "articutool_can_maintain_leveling"
                        )
                    },
                ),
                # Execute
                MoveIt2Execute(
                    name="MoveToStagingConfigurationExecute",
                    ns=name,
                    inputs={
                        "trajectory": BlackboardKey("trajectory"),
                        "group_name": "jaco_arm",
                    },
                    outputs={},
                ),
            ]

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                LoadPinocchioModel(
                    name="LoadPinocchioModel",
                    ns=name,
                    inputs={
                        "urdf_file_path": "package://ada_moveit/config/ada.urdf.xacro",
                        "jaco_joint_names": [
                            "j2n6s200_joint_1",
                            "j2n6s200_joint_2",
                            "j2n6s200_joint_3",
                            "j2n6s200_joint_4",
                            "j2n6s200_joint_5",
                            "j2n6s200_joint_6",
                        ],
                        "articutool_joint_names": ["atool_joint1", "atool_joint2"],
                        "jaco_end_effector_link_name": "j2n6s200_end_effector",
                        "tool_tip_link_name": "tool_tip",
                    },
                    outputs={
                        "pinocchio_model": BlackboardKey("pinocchio_model"),
                        "pinocchio_data": BlackboardKey("pinocchio_data"),
                        "jaco_vel_indices_pin": BlackboardKey("jaco_vel_indices_pin"),
                        "articutool_vel_indices_pin": BlackboardKey(
                            "articutool_vel_indices_pin"
                        ),
                        "jaco_ee_frame_id_pin": BlackboardKey("jaco_ee_frame_id_pin"),
                        "tool_tip_frame_id_pin": BlackboardKey("tool_tip_frame_id_pin"),
                    },
                ),
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
                    # TODO: Revert this when not running benchmarks
                    # pre_behavior=(
                    #     Success()  # pylint: disable=abstract-class-instantiated
                    # ),
                    # post_behavior=(
                    #     Success()  # pylint: disable=abstract-class-instantiated
                    # ),
                    pre_behavior=get_add_in_front_of_face_wall_behavior(
                        name + "AddWheelchairWall",
                    ),
                    # Remove the wall in front of the wheelchair
                    post_behavior=get_remove_in_front_of_face_wall_behavior(
                        name + "RemoveWheelchairWall",
                    ),
                    # Move to the staging configuration
                    workers=constraints
                    + dynamic_path_constraint_sequence
                    + plan_and_execute_sequence,
                ),
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
