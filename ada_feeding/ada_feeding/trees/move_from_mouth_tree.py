#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveFromMouthTree behaviour tree and provides functions to
wrap that behaviour tree in a ROS2 action server.
"""
# pylint: disable=duplicate-code
# MoveFromMouth and MoveToMouth are inverses of each other, so it makes sense
# that they have similar code.

# Standard imports
from typing import List, Optional, Tuple

# Third-party imports
from geometry_msgs.msg import PoseStamped
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2PositionConstraint,
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
)
from ada_feeding.behaviors.ros import CreatePoseStamped
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config, scoped_behavior, servo_until_pose
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_wheelchair_wall_behavior,
    get_remove_in_front_of_wheelchair_wall_behavior,
)
from ada_feeding.trees import MoveToTree
from .start_servo_tree import StartServoTree
from .stop_servo_tree import StopServoTree


class MoveFromMouthTree(MoveToTree):
    """
    A behaviour tree that moves the robot back to the staging location in a
    cartesian motion, and then moves the robot to a specified end configuration.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    # Bite transfer is a big part of the robot's behavior, so it makes
    # sense that it has lots of attributes/arguments.

    def __init__(
        self,
        node: Node,
        staging_configuration_position: Tuple[float, float, float],
        staging_configuration_quat_xyzw: Tuple[float, float, float, float],
        end_configuration: Optional[List[float]] = None,
        staging_configuration_tolerance_position: float = 0.02,
        end_configuration_tolerance: float = 0.001,
        orientation_constraint_to_staging_configuration_quaternion: Optional[
            List[float]
        ] = None,
        orientation_constraint_to_staging_configuration_tolerances: Optional[
            List[float]
        ] = None,
        orientation_constraint_to_end_configuration_quaternion: Optional[
            List[float]
        ] = None,
        orientation_constraint_to_end_configuration_tolerances: Optional[
            List[float]
        ] = None,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time_to_staging_configuration: float = 0.5,
        allowed_planning_time_to_end_configuration: float = 0.5,
        max_linear_speed_to_staging_configuration: float = 0.1,
        max_angular_speed_to_staging_configuration: float = 0.3,
        max_velocity_scaling_factor_to_staging_configuration: float = 0.1,
        max_velocity_scaling_factor_to_end_configuration: float = 0.1,
        cartesian_jump_threshold_to_staging_configuration: float = 0.0,
        cartesian_max_step_to_staging_configuration: float = 0.0025,
        wheelchair_collision_object_id: str = "wheelchair_collision",
        force_threshold_to_staging_configuration: float = 1.0,
        torque_threshold_to_staging_configuration: float = 1.0,
        force_threshold_to_end_configuration: float = 4.0,
        torque_threshold_to_end_configuration: float = 4.0,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        staging_configuration_position: The position of the staging
            configuration.
        staging_configuration_quat_xyzw: The quaternion of the staging
            configuration.
        orientation_constraint_to_staging_configuration_quaternion: The
            quaternion for the orientation constraint to the staging
            configuration.
        orientation_constraint_to_staging_configuration_tolerances: The
            tolerances for the orientation constraint to the staging
            configuration.
        end_configuration: The joint positions to move the robot arm to after
            going to the staging configuration.
        staging_configuration_tolerance_position: The tolerance for the joint positions.
        end_configuration_tolerance: The tolerance for the joint positions.
        orientation_constraint_to_end_configuration_quaternion: The
            quaternion for the orientation constraint to the end configuration.
        orientation_constraint_to_end_configuration_tolerances: The
            tolerances for the orientation constraint to the end configuration.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time_to_staging_configuration: The allowed planning
            time for the MoveIt2 motion planner to move to the staging config.
        allowed_planning_time_to_end_configuration: The allowed planning
            time for the MoveIt2 motion planner to move to the end config.
        max_linear_speed_to_staging_configuration: The maximum linear speed
            (m/s) for the motion to the staging config.
        max_angular_speed_to_staging_configuration: The maximum angular speed
            (rad/s) for the motion to the staging config.
        max_velocity_scaling_factor_to_staging_configuration: The maximum
            velocity scaling factor for the MoveIt2 motion planner to move to
            the staging config.
        max_velocity_scaling_factor_to_end_configuration: The maximum
            velocity scaling factor for the MoveIt2 motion planner to move to
            the end config.
        cartesian_jump_threshold_to_staging_configuration: The cartesian
            jump threshold for the MoveIt2 motion planner to move to the
            staging config.
        cartesian_max_step_to_staging_configuration: The cartesian
            max step for the MoveIt2 motion planner to move to the
            staging config.
        wheelchair_collision_object_id: The ID of the wheelchair collision object
            in the MoveIt2 planning scene.
        force_threshold_to_staging_configuration: The force threshold for the
            motion to the staging config.
        torque_threshold_to_staging_configuration: The torque threshold for the
            motion to the staging config.
        force_threshold_to_end_configuration: The force threshold for the
            motion to the end config.
        torque_threshold_to_end_configuration: The torque threshold for the
            motion to the end config.
        """
        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters
        self.staging_configuration_position = staging_configuration_position
        self.staging_configuration_quat_xyzw = staging_configuration_quat_xyzw
        self.end_configuration = end_configuration
        if self.end_configuration is not None:
            assert len(self.end_configuration) == 6, "Must provide 6 joint positions"
        self.staging_configuration_tolerance_position = (
            staging_configuration_tolerance_position
        )
        self.end_configuration_tolerance = end_configuration_tolerance
        self.orientation_constraint_to_staging_configuration_quaternion = (
            orientation_constraint_to_staging_configuration_quaternion
        )
        self.orientation_constraint_to_staging_configuration_tolerances = (
            orientation_constraint_to_staging_configuration_tolerances
        )
        self.orientation_constraint_to_end_configuration_quaternion = (
            orientation_constraint_to_end_configuration_quaternion
        )
        self.orientation_constraint_to_end_configuration_tolerances = (
            orientation_constraint_to_end_configuration_tolerances
        )
        self.planner_id = planner_id
        self.allowed_planning_time_to_staging_configuration = (
            allowed_planning_time_to_staging_configuration
        )
        self.allowed_planning_time_to_end_configuration = (
            allowed_planning_time_to_end_configuration
        )
        self.max_linear_speed_to_staging_configuration = (
            max_linear_speed_to_staging_configuration
        )
        self.max_angular_speed_to_staging_configuration = (
            max_angular_speed_to_staging_configuration
        )
        self.max_velocity_scaling_factor_to_staging_configuration = (
            max_velocity_scaling_factor_to_staging_configuration
        )
        self.max_velocity_scaling_factor_to_end_configuration = (
            max_velocity_scaling_factor_to_end_configuration
        )
        self.cartesian_jump_threshold_to_staging_configuration = (
            cartesian_jump_threshold_to_staging_configuration
        )
        self.cartesian_max_step_to_staging_configuration = (
            cartesian_max_step_to_staging_configuration
        )
        self.wheelchair_collision_object_id = wheelchair_collision_object_id
        self.force_threshold_to_staging_configuration = (
            force_threshold_to_staging_configuration
        )
        self.torque_threshold_to_staging_configuration = (
            torque_threshold_to_staging_configuration
        )
        self.force_threshold_to_end_configuration = force_threshold_to_end_configuration
        self.torque_threshold_to_end_configuration = (
            torque_threshold_to_end_configuration
        )

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        ### Define tree logic

        in_front_of_wheelchair_wall_id = "in_front_of_wheelchair_wall"
        base_link = "j2n6s200_link_base"

        # The tree may or may not have orientation path constraints active
        def get_staging_path_constraints() -> py_trees.behaviour.Behaviour:
            if self.orientation_constraint_to_staging_configuration_quaternion is None:
                # pylint: disable=abstract-class-instantiated
                # This is fine, not sure why pylint doesn't like it
                return py_trees.behaviours.Success()
            else:
                # Orientation path constraint to keep the fork straight
                return MoveIt2OrientationConstraint(
                    name="KeepForkStraightPathConstraint",
                    ns=name,
                    inputs={
                        "quat_xyzw": (
                            self.orientation_constraint_to_staging_configuration_quaternion
                        ),
                        "tolerance": (
                            self.orientation_constraint_to_staging_configuration_tolerances
                        ),
                        "parameterization": 1,  # Rotation vector
                    },
                    outputs={
                        "constraints": BlackboardKey("path_constraints"),
                    },
                )
        if self.orientation_constraint_to_end_configuration_quaternion is None:
            # pylint: disable=abstract-class-instantiated
            # This is fine, not sure why pylint doesn't like it
            end_path_constraints = py_trees.behaviours.Success()
        else:
            # Orientation path constraint to keep the fork straight
            end_path_constraints = MoveIt2OrientationConstraint(
                name="KeepForkStraightPathConstraint",
                ns=name,
                inputs={
                    "quat_xyzw": (
                        self.orientation_constraint_to_end_configuration_quaternion
                    ),
                    "tolerance": (
                        self.orientation_constraint_to_end_configuration_tolerances
                    ),
                    "parameterization": 1,  # Rotation vector
                },
                outputs={
                    "constraints": BlackboardKey("path_constraints"),
                },
            )

        # Generate the fallback MoveIt2 behaviors to get to the staging configuration
        def moveit2_move_to_staging_pose(
            cartesian: bool,
        ) -> py_trees.behaviour.Behaviour:
            suffix = "Cartesian" if cartesian else "Kinematic"
            return py_trees.composites.Sequence(
                name=name + "MoveToStagingConfigurationViaMoveIt2"+suffix,
                memory=True,
                children=[
                    # Goal configuration: target position
                    MoveIt2PositionConstraint(
                        name="MoveToStagingPosePositionGoalConstraint"+suffix,
                        ns=name,
                        inputs={
                            "position": self.staging_configuration_position,
                            "tolerance": self.staging_configuration_tolerance_position,
                        },
                        outputs={
                            "constraints": BlackboardKey("goal_constraints"),
                        },
                    ),
                    # Goal configuration: target orientation
                    MoveIt2OrientationConstraint(
                        name="MoveToStagingPoseOrientationGoalConstraint"+suffix,
                        ns=name,
                        inputs={
                            "quat_xyzw": self.staging_configuration_quat_xyzw,
                            "tolerance": (0.6, 0.5, 0.5),
                            "parameterization": 1,  # Rotation vector
                            "constraints": BlackboardKey("goal_constraints"),
                        },
                        outputs={
                            "constraints": BlackboardKey("goal_constraints"),
                        },
                    ),
                    get_staging_path_constraints(),
                    # Plan
                    MoveIt2Plan(
                        name="MoveToStagingPosePlan"+suffix,
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "path_constraints": BlackboardKey("path_constraints"),
                            "planner_id": self.planner_id,
                            "allowed_planning_time": (
                                self.allowed_planning_time_to_staging_configuration
                            ),
                            "max_velocity_scale": (
                                self.max_velocity_scaling_factor_to_staging_configuration
                            ),
                            "cartesian": cartesian,
                            "cartesian_jump_threshold": (
                                self.cartesian_jump_threshold_to_staging_configuration
                            ),
                            "cartesian_fraction_threshold": 0.20,  # Fine if its low since the user can retry
                            "cartesian_max_step": (
                                self.cartesian_max_step_to_staging_configuration
                            ),
                        },
                        outputs={"trajectory": BlackboardKey("trajectory")},
                    ),
                    # Execute
                    MoveIt2Execute(
                        name="MoveToStagingPoseExecute"+suffix,
                        ns=name,
                        inputs={"trajectory": BlackboardKey("trajectory")},
                        outputs={},
                    ),
                ],  # End MoveToStagingConfigurationViaMoveIt2.children
            )  # End MoveToStagingConfigurationViaMoveIt2

        # Use a custom speed profile to do angular motions at the end.
        max_pose_distance = 0.0

        def speed(post_stamped: PoseStamped) -> Tuple[float, float]:
            """
            Always return the max linear velocity, but gradually increase
            the angular velocity as you get closer to the target position.
            In other words, tilt closer to the end of the motion.
            """
            # TODO: Consider making the speed slower closer to the mouth.
            nonlocal max_pose_distance
            pose_distance = (
                post_stamped.pose.position.x**2.0
                + post_stamped.pose.position.y**2.0
                + post_stamped.pose.position.z**2.0
            ) ** 0.5
            if pose_distance > max_pose_distance:
                max_pose_distance = pose_distance
            prop = ((max_pose_distance - pose_distance) / max_pose_distance) ** 1.0
            return (
                self.max_linear_speed_to_staging_configuration,
                self.max_angular_speed_to_staging_configuration * prop,
            )

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                # Allow collisions between the robot and the expanded wheelchair
                # collision object
                scoped_behavior(
                    name=name + " AllowWheelchairCollisionScope",
                    pre_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            StartServoTree(self._node)
                            .create_tree(name=name + "StartServoScopePre")
                            .root,
                        ],
                    ),
                    post_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            StopServoTree(self._node)
                            .create_tree(name=name + "StopServoScopePost")
                            .root,
                            pre_moveto_config(
                                name=name + "PreMoveToConfigScopePost",
                                re_tare=False,
                                f_mag=1.0,
                                param_service_name="~/set_servo_controller_parameters",
                            ),
                        ],
                    ),
                    workers=[
                        # Retare the F/T sensor and set the F/T Thresholds
                        pre_moveto_config(
                            name=name + "PreMoveToConfigToStaging",
                            toggle_watchdog_listener=False,
                            f_mag=self.force_threshold_to_staging_configuration,
                            t_mag=self.torque_threshold_to_staging_configuration,
                        ),
                        py_trees.composites.Selector(
                            name=name + "MoveToStagingConfigurationSelector",
                            memory=True,
                            children=[
                                # First, try to move to the staging configuration with Servo
                                py_trees.composites.Sequence(
                                    name=name + "MoveToStagingConfigurationViaServo",
                                    memory=True,
                                    children=[
                                        # Create the goal pose
                                        CreatePoseStamped(
                                            name=name + "CreateStagingPose",
                                            ns=name,
                                            inputs={
                                                "position": self.staging_configuration_position,
                                                "quaternion": self.staging_configuration_quat_xyzw,
                                                "frame_id": base_link,
                                            },
                                            outputs={
                                                "pose_stamped": BlackboardKey(
                                                    "goal_pose"
                                                ),
                                            },
                                        ),
                                        # Servo until the goal pose
                                        servo_until_pose(
                                            name=name + " MoveToStagingPoseServo",
                                            ns=name,
                                            target_pose_stamped_key=BlackboardKey(
                                                "goal_pose"
                                            ),
                                            tolerance_position=self.staging_configuration_tolerance_position,
                                            tolerance_orientation=(0.6, 0.5, 0.5),
                                            duration=10.0,
                                            round_decimals=3,
                                            speed=speed,
                                        ),
                                    ],  # End MoveToStagingConfigurationViaServo.children
                                ),  # End MoveToStagingConfigurationViaServo
                                # If that fails, try to move to the staging configuration with a
                                # MoveIt2 cartesian path
                                moveit2_move_to_staging_pose(cartesian=True),
                                # NOTE: Although we considered a kinematic plan,
                                # kinematic plans around the face can be dangerous
                                # and unexpected, so we decided to not use them.
                            ],  # End MoveToStagingConfigurationSelector.children
                        ),  # End MoveToStagingConfigurationSelector
                    ],  # End AllowWheelchairCollisionScope.workers
                ),  # End AllowWheelchairCollisionScope
            ],  # End root_seq.children
        )  # End root_seq

        # Move to the end configuration if it is provided
        if self.end_configuration is not None:
            root_seq.children.append(
                # Add the wall in front of the wheelchair to prevent the arm from
                # Moving closer to the user than it currently is.
                scoped_behavior(
                    name=name + " AddInFrontOfWheelchairWallScope",
                    pre_behavior=get_add_in_front_of_wheelchair_wall_behavior(
                        name + "AddInFrontOfWheelchairWallScopePre",
                        in_front_of_wheelchair_wall_id,
                    ),
                    post_behavior=get_remove_in_front_of_wheelchair_wall_behavior(
                        name + "RemoveInFrontOfWheelchairWallScopePost",
                        in_front_of_wheelchair_wall_id,
                    ),
                    workers=[
                        # Retare the F/T sensor and set the F/T Thresholds
                        pre_moveto_config(
                            name=name + "PreMoveToConfigToEnd",
                            toggle_watchdog_listener=False,
                            f_mag=self.force_threshold_to_end_configuration,
                            t_mag=self.torque_threshold_to_end_configuration,
                        ),
                        # Goal configuration: staging configuration
                        MoveIt2JointConstraint(
                            name="EndingConfigurationGoalConstraint",
                            ns=name,
                            inputs={
                                "joint_positions": self.end_configuration,
                                "tolerance": self.end_configuration_tolerance,
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        end_path_constraints,
                        # Plan
                        MoveIt2Plan(
                            name="MoveToEndingConfigurationPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "path_constraints": BlackboardKey("path_constraints"),
                                "planner_id": self.planner_id,
                                "allowed_planning_time": (
                                    self.allowed_planning_time_to_end_configuration
                                ),
                                "max_velocity_scale": (
                                    self.max_velocity_scaling_factor_to_end_configuration
                                ),
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        # Execute
                        MoveIt2Execute(
                            name="MoveToEndingConfigurationExecute",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
                            outputs={},
                        ),
                    ],  # End AddInFrontOfWheelchairWallScope.workers
                ),  # End AddInFrontOfWheelchairWallScope
            )  # End root_seq.children.append

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
