# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import pickle
import operator
from typing import List, Optional

# Third-party imports
from geometry_msgs.msg import Twist, TwistStamped, Vector3
import numpy as np
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
from py_trees.common import Status, ComparisonExpression
from py_trees.behaviours import Success
import py_trees_ros
from rcl_interfaces.srv import SetParameters
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from std_srvs.srv import Empty

# Local imports
from ada_feeding_msgs.action import AcquireFood
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.behaviors.acquisition import (
    ComputeFoodFrame,
    ComputeActionConstraints,
    ComputeActionTwist,
    RotateLocalApproachPoses,
    ConditionallyRotateFoodFrame,
    GenerateSkewerTiltCandidates,
    CalculateSkewerPoseForTilt,
)
from ada_feeding.behaviors.moveit2 import (
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
    MoveIt2PoseConstraint,
    MoveIt2PositionOffsetConstraint,
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2ComputeIK,
    MoveIt2ComputeFK,
    ServoMove,
    ToggleCollisionObject,
)
from ada_feeding.behaviors.state import (
    GetJointStates,
    ExtractJointsFromState,
    CombineJointStates,
    ExtractPoseFromPosesByLink,
    ExtractPoseComponents,
    CheckJacoDirectionalManipulability,
    CheckArticutoolPathOrientationFeasibility,
    CheckArticutoolPathLevelingFeasibility,
    CheckArticutoolPathDynamicFeasibility,
    LoadPinocchioModel,
    PublishPoseAsTf,
    ComputeSlerpMidpointOrientation,
)
from ada_feeding.behaviors.ros.msgs import StampPoseFromPose
from ada_feeding.behaviors.ros.tf import ApplyTransform
from ada_feeding.behaviors.articutool import (
    ExecuteArticutoolTrajectory,
    CallSetOrientationControl,
    SwitchArticutoolControllers,
    ComputeArticutoolLevelingJoints,
    TriggerArticutoolCalibration,
    ExecuteNamedPrimitive,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import (
    pre_moveto_config,
    scoped_behavior,
    retry_call_ros_service,
)
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_face_wall_behavior,
    get_remove_in_front_of_face_wall_behavior,
)
from ada_feeding.idioms.ft_thresh_utils import ft_thresh_satisfied
from ada_feeding.idioms.pre_moveto_config import set_parameter_response_all_success
from ada_feeding.trees import MoveToTree, StartServoTree, StopServoTree


# pylint: disable=too-many-lines
# This tree is the cruz of bite acquisition, hence is long.


class AcquireFoodTree(MoveToTree):
    """
    A behaviour tree to select and execute an acquisition
    action (see ada_feeding_msgs.action.AcquisitionSchema)
    for a given food mask in ada_feeding_msgs.action.AcquireFood.

    Tree Blackboard Inputs:
    - camera_info: See ComputeFoodFrame
    - mask: See ComputeFoodFrame
    - timestamp: See ComputeFoodFrame

    Tree Blackboard Outputs:

    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes
    # The many parameters makes this tree extremely configurable.

    def __init__(
        self,
        node: Node,
        resting_joint_positions: Optional[List[float]] = None,
        max_velocity_scaling_move_above: Optional[float] = 0.8,
        max_acceleration_scaling_move_above: Optional[float] = 0.8,
        max_velocity_scaling_move_into: Optional[float] = 1.0,
        max_acceleration_scaling_move_into: Optional[float] = 0.8,
        max_velocity_scaling_to_resting_configuration: Optional[float] = 0.8,
        max_acceleration_scaling_to_resting_configuration: Optional[float] = 0.8,
        pickle_goal_path: Optional[str] = None,
        allowed_planning_time_for_move_above: float = 1.0,
        allowed_planning_time_for_move_into: float = 0.5,
        allowed_planning_time_to_resting_configuration: float = 0.5,
        allowed_planning_time_for_recovery: float = 0.5,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        resting_joint_positions: Final joint position after acquisition
        max_velocity_scaling_move_above: Max velocity scaling for move above
        max_acceleration_scaling_move_above: Max acceleration scaling for move above
        max_velocity_scaling_move_into: Max velocity scaling for move into
        max_acceleration_scaling_move_into: Max acceleration scaling for move into
        max_velocity_scaling_to_resting_configuration: Max velocity scaling for move to resting configuration
        max_acceleration_scaling_to_resting_configuration: Max acceleration scaling for move to resting configuration
        pickle_goal_path: Path to pickle goal for debugging
        allowed_planning_time_for_move_above: Allowed planning time for move above
        allowed_planning_time_for_move_into: Allowed planning time for move into
        allowed_planning_time_to_resting_configuration: Allowed planning time for move to resting configuration
        allowed_planning_time_for_recovery: Allowed planning time for recovery
        """
        # Initialize ActionServerBT
        super().__init__(node)

        self.resting_joint_positions = resting_joint_positions
        self.max_velocity_scaling_move_above = max_velocity_scaling_move_above
        self.max_acceleration_scaling_move_above = max_acceleration_scaling_move_above
        self.max_velocity_scaling_move_into = max_velocity_scaling_move_into
        self.max_acceleration_scaling_move_into = max_acceleration_scaling_move_into
        self.max_velocity_scaling_to_resting_configuration = (
            max_velocity_scaling_to_resting_configuration
        )
        self.max_acceleration_scaling_to_resting_configuration = (
            max_acceleration_scaling_to_resting_configuration
        )
        self.pickle_goal_path = pickle_goal_path
        self.allowed_planning_time_for_move_above = allowed_planning_time_for_move_above
        self.allowed_planning_time_for_move_into = allowed_planning_time_for_move_into
        self.allowed_planning_time_to_resting_configuration = (
            allowed_planning_time_to_resting_configuration
        )
        self.allowed_planning_time_for_recovery = allowed_planning_time_for_recovery

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied by @override

        # pylint: disable=line-too-long
        # This is the entire tree rolled out.

        ### Blackboard Constants
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(key="zero_twist", access=py_trees.common.Access.WRITE)
        blackboard.zero_twist = TwistStamped(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="world",
            ),
            twist=Twist(),
        )

        # The max amount that each joint can move for any computed plan. Intended
        # to reduce swivels.
        max_path_len_joint = {
            "j2n6s200_joint_1": np.pi * 5.0 / 6.0,
            "j2n6s200_joint_2": np.pi / 4.0,
        }

        # Get the base lin to publish servo commands in
        base_link = "j2n6s200_link_base"

        ### Add Resting Position
        resting_position_behaviors = []
        if self.resting_joint_positions is not None:
            # Move back to resting position
            resting_position_behaviors.append(
                scoped_behavior(
                    name=name + " InFrontOfWheelchairWallScope",
                    # TODO: Revert this when not running benchmarks
                    pre_behavior=(
                        Success()  # pylint: disable=abstract-class-instantiated
                    ),
                    post_behavior=(
                        Success()  # pylint: disable=abstract-class-instantiated
                    ),
                    # pre_behavior=get_add_in_front_of_face_wall_behavior(
                    #     name + "AddWheelchairWall",
                    # ),
                    # # Remove the wall in front of the wheelchair
                    # post_behavior=get_remove_in_front_of_face_wall_behavior(
                    #     name + "RemoveWheelchairWall",
                    # ),
                    workers=[
                        py_trees_ros.service_clients.FromConstant(
                            name="ClearOctomap",
                            service_name="/clear_octomap",
                            service_type=Empty,
                            service_request=Empty.Request(),
                            # Default fail if service is down
                            wait_for_server_timeout_sec=0.0,
                        ),
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
                                "joint_state": BlackboardKey("current_jaco_joint_state")
                            },
                        ),
                        # 2. Use FK to calculate the current EE pose
                        MoveIt2ComputeFK(
                            name="GetStartEEPose",
                            ns=name,
                            inputs={
                                "group_name": "jaco_arm",
                                "joint_state": BlackboardKey(
                                    "current_jaco_joint_state"
                                ),
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
                        MoveIt2ComputeFK(
                            name="GetGoalEEPose",
                            ns=name,
                            inputs={
                                "group_name": "jaco_arm",
                                "joint_state": self.resting_joint_positions,
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
                            name="CalculateRestingPathConstraint",
                            ns=name,
                            inputs={
                                "start_ee_pose": BlackboardKey("start_ee_pose"),
                                "goal_ee_pose": BlackboardKey("goal_ee_pose"),
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
                        # Create the MoveIt2 orientation constraint message
                        MoveIt2OrientationConstraint(
                            name="CreateRestingPathConstraintMsg",
                            ns=name,
                            inputs={
                                "quat_xyzw": BlackboardKey("path_constraint_quat"),
                                "tolerance": BlackboardKey("path_constraint_tol"),
                            },
                            outputs={
                                "constraints": BlackboardKey(
                                    "resting_path_constraints"
                                ),
                            },
                        ),
                        MoveIt2JointConstraint(
                            name="RestingConstraint",
                            ns=name,
                            inputs={
                                "joint_positions": self.resting_joint_positions,
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        py_trees.decorators.Timeout(
                            name="RestingPlanTimeout",
                            # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                            duration=10.0
                            * self.allowed_planning_time_to_resting_configuration,
                            child=MoveIt2Plan(
                                name="RestingPlan",
                                ns=name,
                                inputs={
                                    "goal_constraints": BlackboardKey(
                                        "goal_constraints"
                                    ),
                                    "path_constraints": BlackboardKey(
                                        "resting_path_constraints"
                                    ),
                                    "max_velocity_scale": self.max_velocity_scaling_to_resting_configuration,
                                    "max_acceleration_scale": self.max_acceleration_scaling_to_resting_configuration,
                                    "allowed_planning_time": self.allowed_planning_time_to_resting_configuration,
                                    "group_name": "jaco_arm",
                                },
                                outputs={
                                    "trajectory": BlackboardKey("resting_trajectory")
                                },
                            ),
                        ),
                        CheckArticutoolPathDynamicFeasibility(
                            name="CheckArticutoolDynamicFeasibilityForResting",
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
                                "jaco_ee_frame_id_pin": BlackboardKey(
                                    "jaco_ee_frame_id_pin"
                                ),
                                "jaco_trajectory": BlackboardKey("resting_trajectory"),
                                "articutool_pitch_limits_rad": (-np.pi / 2, np.pi / 2),
                                "articutool_roll_limits_rad": (-np.pi, np.pi),
                                "jaco_vel_indices_pin": BlackboardKey(
                                    "jaco_vel_indices_pin"
                                ),
                                "articutool_max_joint_velocity": 4.0,
                            },
                            outputs={
                                "articutool_is_dynamic_feasible": BlackboardKey(
                                    "articutool_can_maintain_leveling"
                                )
                            },
                        ),
                        MoveIt2Execute(
                            name="Resting",
                            ns=name,
                            inputs={
                                "trajectory": BlackboardKey("resting_trajectory"),
                                "group_name": "jaco_arm",
                            },
                            outputs={},
                        ),
                    ],
                ),
            )

        ### Define Recovery Tree (if failure in Grasp/Extract)
        recovery_tree = py_trees.composites.Sequence(
            name="RecoverySequence",
            memory=True,
            children=[
                pre_moveto_config(
                    name="MaxFTRecoveryRetare",
                    re_tare=False,
                    f_mag=75.0,
                    param_service_name="~/set_cartesian_controller_parameters",
                ),
                # Clear Octomap
                py_trees_ros.service_clients.FromConstant(
                    name="ClearOctomap",
                    service_name="/clear_octomap",
                    service_type=Empty,
                    service_request=Empty.Request(),
                    # Default fail if service is down
                    wait_for_server_timeout_sec=0.0,
                ),
                # Recovery Attempts
                py_trees.composites.Selector(
                    name="RecoverySelector",
                    memory=True,
                    children=[
                        py_trees.decorators.Retry(
                            name="RecoveryServoRetry",
                            num_failures=3,
                            child=py_trees.composites.Sequence(
                                name="RecoveryServoSequence",
                                memory=True,
                                children=[
                                    ServoMove(
                                        name="RecoveryServo",
                                        ns=name,
                                        inputs={
                                            "default_frame_id": base_link,
                                            "twist": Twist(
                                                linear=Vector3(x=0.0, y=0.0, z=0.05),
                                                angular=Vector3(),
                                            ),  # Default 1s duration
                                            "pub_topic": "~/cartesian_twist_cmds",
                                            "servo_status_sub_topic": None,
                                        },
                                    ),  # Auto Zero-Twist on terminate()
                                    ft_thresh_satisfied(name="FTThreshSatisfied"),
                                ],
                            ),
                        ),  # End Attempt 1: RecoveryServoRetry
                        py_trees.composites.Sequence(
                            name="RecoveryCartesianSequence",
                            memory=True,
                            children=[
                                pre_moveto_config(
                                    name="MaxFTRecoveryCartesian",
                                    re_tare=False,
                                    f_mag=75.0,
                                ),  # Protected by scoped FTThresh in AcquireTree
                                MoveIt2PositionOffsetConstraint(
                                    name="RecoveryOffsetPose",
                                    ns=name,
                                    inputs={
                                        "offset": Vector3(x=0.0, y=0.0, z=0.03),
                                        # Default end effector link
                                        # Default base link frame
                                    },
                                    outputs={
                                        "constraints": BlackboardKey(
                                            "goal_constraints"
                                        ),
                                    },
                                ),
                                # A cartesian plan requires a position and orientation.
                                # Put no orientation constraint by setting high tolerances.
                                MoveIt2OrientationConstraint(
                                    name="",
                                    ns=name,
                                    inputs={
                                        "constraints": BlackboardKey(
                                            "goal_constraints"
                                        ),
                                        "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
                                        "tolerance": (
                                            2.0 * np.pi,
                                            2.0 * np.pi,
                                            2.0 * np.pi,
                                        ),
                                    },
                                    outputs={
                                        "constraints": BlackboardKey(
                                            "goal_constraints"
                                        ),
                                    },
                                ),
                                py_trees.decorators.Timeout(
                                    name="RecoveryOffsetPlanTimeout",
                                    # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                                    duration=10.0
                                    * self.allowed_planning_time_for_recovery,
                                    child=MoveIt2Plan(
                                        name="RecoveryOffsetPlan",
                                        ns=name,
                                        inputs={
                                            "goal_constraints": BlackboardKey(
                                                "goal_constraints"
                                            ),
                                            "max_velocity_scale": self.max_velocity_scaling_move_into,
                                            "max_acceleration_scale": self.max_acceleration_scaling_move_into,
                                            "cartesian": True,
                                            "cartesian_max_step": 0.001,
                                            "cartesian_fraction_threshold": 0.92,
                                            "allowed_planning_time": self.allowed_planning_time_for_recovery,
                                            "group_name": "jaco_arm",
                                        },
                                        outputs={
                                            "trajectory": BlackboardKey(
                                                "recovery_trajectory"
                                            )
                                        },
                                    ),
                                ),
                                MoveIt2Execute(
                                    name="RecoveryOffset",
                                    ns=name,
                                    inputs={
                                        "trajectory": BlackboardKey(
                                            "recovery_trajectory"
                                        ),
                                        "group_name": "jaco_arm",
                                    },
                                    outputs={},
                                ),
                            ],
                        ),  # End Attempt 2: MoveIt2 Cartesian Planner
                    ],  # End RecoverySelector.children
                ),  # End RecoverySelector
            ],  # End RecoverySequence.children
        )  # End RecoverySequence

        def post_acquisition_sequence() -> py_trees.behaviour.Behaviour:
            return py_trees.composites.Sequence(
                name="PreAcquisitionSequence",
                memory=True,
                children=[
                    CallSetOrientationControl(
                        name="DisableArticutoolOrientation",
                        ns=name,
                        inputs={
                            "control_mode": 0,
                        },
                        outputs={},
                    ),
                    GetJointStates(
                        name="GetJacoArmStateForLeveling",
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
                            ],
                        },
                        outputs={
                            "joint_state": BlackboardKey(
                                "current_jaco_arm_state_for_leveling_fk"
                            ),
                            "joint_positions": None,
                            "joint_names": None,
                        },
                    ),
                    MoveIt2ComputeFK(
                        name="GetJacoEEPoseForLeveling",
                        ns=name,
                        inputs={
                            "group_name": "jaco_arm",
                            "joint_state": BlackboardKey(
                                "current_jaco_arm_state_for_leveling_fk"
                            ),
                            "fk_link_names": ["j2n6s200_end_effector"],
                        },
                        outputs={
                            "fk_poses": BlackboardKey("current_jaco_arm_fk_poses"),
                            "success": None,
                        },
                    ),
                    ExtractPoseFromPosesByLink(
                        name="ExtractJacoEEPoseForLeveling",
                        ns=name,
                        inputs={
                            "fk_poses": BlackboardKey("current_jaco_arm_fk_poses"),
                            "target_link_name": "j2n6s200_end_effector",
                            "requested_link_names": ["j2n6s200_end_effector"],
                        },
                        outputs={
                            "extracted_pose": BlackboardKey(
                                "current_jaco_ee_world_pose_stamped"
                            ),
                            "success": None,
                        },
                    ),
                    ComputeArticutoolLevelingJoints(
                        name="ComputeLevelingAngles",
                        ns=name,
                        inputs={
                            "jaco_ee_world_pose": BlackboardKey(
                                "current_jaco_ee_world_pose_stamped"
                            ),
                        },
                        outputs={
                            "articutool_joint_positions": BlackboardKey(
                                "articutool_joint_positions"
                            ),
                            "articutool_leveling_ik_found": BlackboardKey(
                                "leveling_ik_success"
                            ),
                        },
                    ),
                    MoveIt2JointConstraint(
                        name="SetLevelingJointGoal",
                        ns=name,
                        inputs={
                            "joint_positions": BlackboardKey(
                                "articutool_joint_positions"
                            ),
                        },
                        outputs={
                            "constraints": BlackboardKey(
                                "articutool_leveling_constraints"
                            )
                        },
                    ),
                    py_trees.decorators.Timeout(
                        name="PlanToLevelArticutoolTimeout",
                        # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                        duration=10.0
                        * self.allowed_planning_time_to_resting_configuration,
                        child=MoveIt2Plan(
                            name="PlanToLevelArticutool",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey(
                                    "articutool_leveling_constraints"
                                ),
                                "group_name": "articutool",
                            },
                            outputs={
                                "trajectory": BlackboardKey(
                                    "level_articutool_trajectory"
                                )
                            },
                        ),
                    ),
                    SwitchArticutoolControllers(
                        name="SwitchArticutoolToJointTrajectory",
                        ns=name,
                        inputs={
                            "controllers_to_activate": ["joint_trajectory_controller"],
                            "controllers_to_deactivate": ["velocity_controller"],
                        },
                        outputs={
                            "switch_call_succeeded": None,
                            "switch_response_ok": None,
                        },
                    ),
                    ExecuteArticutoolTrajectory(
                        name="LevelArticutool",
                        ns=name,
                        inputs={
                            "trajectory": BlackboardKey("level_articutool_trajectory"),
                        },
                        outputs={
                            "action_goal_accepted": BlackboardKey("tool_goal_accepted"),
                            "action_result_code": BlackboardKey(
                                "tool_exec_result_code"
                            ),
                            "action_status": BlackboardKey("tool_action_status"),
                        },
                    ),
                    SwitchArticutoolControllers(
                        name="SwitchArticutoolToVelocity",
                        ns=name,
                        inputs={
                            "controllers_to_activate": ["velocity_controller"],
                            "controllers_to_deactivate": [
                                "joint_trajectory_controller"
                            ],
                        },
                        outputs={
                            "switch_call_succeeded": None,
                            "switch_response_ok": None,
                        },
                    ),
                    ExecuteNamedPrimitive(
                        name="RunPostAcquisitionPrimitive",
                        ns=name,
                        inputs={
                            "primitive_name": BlackboardKey(
                                "post_acquisition_action_name"
                            ),
                            "primitive_params": BlackboardKey(
                                "post_acquisition_action_params"
                            ),
                        },
                        outputs={
                            "primitive_status": None,
                        },
                    ),
                    CallSetOrientationControl(
                        name="SetArticutoolOrientation",
                        ns=name,
                        inputs={
                            "control_mode": 1,
                        },
                        outputs={},
                    ),
                ],
            )

        def pre_acquisition_sequence(
            flip_food_frame: bool = False,
            action: Optional[BlackboardKey] = None,
        ) -> py_trees.behaviour.Behaviour:
            return py_trees.composites.Sequence(
                name="PreAcquisitionSequence",
                memory=True,
                children=[
                    # Compute Food Frame
                    py_trees.decorators.Timeout(
                        name="ComputeFoodFrameTimeout",
                        duration=1.0,
                        child=ComputeFoodFrame(
                            name="ComputeFoodFrame",
                            ns=name,
                            inputs={
                                "camera_info": BlackboardKey("camera_info"),
                                "mask": BlackboardKey("mask"),
                                # NOTE: We override the goal message timestamp
                                # since sometimes there isn't a recent enough TF
                                "timestamp": rclpy.time.Time(),
                                # "timestamp": BlackboardKey("timestamp"),
                                # Default food_frame_id = "food"
                                # Default world_frame = "world"
                                "flip_food_frame": flip_food_frame,
                                "align_to_robot_base": False,
                            },
                            outputs={
                                "action_select_request": BlackboardKey(
                                    "action_request"
                                ),
                                "food_frame": BlackboardKey("initial_food_frame"),
                            },
                        ),
                    ),
                    # Get Action to Use
                    py_trees_ros.service_clients.FromBlackboard(
                        name="AcquisitionSelect",
                        service_name="~/action_select",
                        service_type=AcquisitionSelect,
                        # Need absolute Blackboard name
                        key_request=Blackboard.separator.join(
                            [name, BlackboardKey("action_request")]
                        ),
                        key_response=Blackboard.separator.join(
                            [name, BlackboardKey("action_response")]
                        ),
                        # Default fail if service is down
                        wait_for_server_timeout_sec=0.0,
                    ),
                    # Get MoveIt2 Constraints
                    py_trees.decorators.Timeout(
                        name="ComputeActionConstraintsTimeout",
                        duration=1.0,
                        child=ComputeActionConstraints(
                            name="ComputeActionConstraints",
                            ns=name,
                            inputs={
                                "action_select_response": BlackboardKey(
                                    "action_response"
                                ),
                                "action": action,
                                # Default move_above_dist_m = 0.05
                                # Default food_frame_id = "food"
                                # Default approach_frame_id = "approach"
                            },
                            outputs={
                                "move_above_pose": BlackboardKey(
                                    "move_above_pose_food_frame"
                                ),
                                "move_into_pose": BlackboardKey(
                                    "move_into_pose_food_frame"
                                ),
                                "approach_thresh": BlackboardKey("approach_thresh"),
                                "grasp_thresh": BlackboardKey("grasp_thresh"),
                                "ext_thresh": BlackboardKey("ext_thresh"),
                                "action": BlackboardKey("action"),
                                "action_index": BlackboardKey("action_index"),
                                "pre_move_into_primitive_name": BlackboardKey(
                                    "pre_move_into_action_name"
                                ),
                                "pre_move_into_primitive_params": BlackboardKey(
                                    "pre_move_into_action_params"
                                ),
                                "post_move_into_primitive_name": BlackboardKey(
                                    "post_move_into_action_name"
                                ),
                                "post_move_into_primitive_params": BlackboardKey(
                                    "post_move_into_action_params"
                                ),
                                "post_acquisition_primitive_name": BlackboardKey(
                                    "post_acquisition_action_name"
                                ),
                                "post_acquisition_primitive_params": BlackboardKey(
                                    "post_acquisition_action_params"
                                ),
                                "should_align_to_base": BlackboardKey(
                                    "should_align_to_base"
                                ),
                            },
                        ),
                    ),
                    ConditionallyRotateFoodFrame(
                        name="ConditionallyRotateFoodFrame",
                        ns=name,
                        inputs={
                            "initial_food_frame": BlackboardKey("initial_food_frame"),
                            "should_align_to_base": BlackboardKey(
                                "should_align_to_base"
                            ),
                        },
                        outputs={
                            "food_frame_updated": BlackboardKey("final_food_frame"),
                        },
                    ),
                    # Re-Tare FT Sensor and default to 4N threshold
                    pre_moveto_config(name="PreAcquireFTTare"),
                    # --- Prepare MoveAbove Pose for IK ---
                    StampPoseFromPose(
                        name="StampMoveAbovePoseFood",
                        ns=name,
                        inputs={
                            "input_pose": BlackboardKey("move_above_pose_food_frame"),
                            "frame_id": "food",
                        },
                        outputs={
                            "output_pose_stamped": BlackboardKey(
                                "move_above_pose_stamped_food"
                            )
                        },
                    ),
                    ApplyTransform(
                        name="TransformMoveAbovePoseToWorld",
                        ns=name,
                        inputs={
                            "stamped_msg": BlackboardKey(
                                "move_above_pose_stamped_food"
                            ),
                            "target_frame": "j2n6s200_link_base",
                        },
                        outputs={
                            "transformed_msg": BlackboardKey(
                                "tool_tip_move_above_pose_world"
                            )
                        },
                    ),
                    StampPoseFromPose(
                        name="StampMoveIntoPoseFood",
                        ns=name,
                        inputs={
                            "input_pose": BlackboardKey("move_into_pose_food_frame"),
                            "frame_id": "food",
                        },
                        outputs={
                            "output_pose_stamped": BlackboardKey(
                                "move_into_pose_stamped_food"
                            )
                        },
                    ),
                    ApplyTransform(
                        name="TransformMoveIntoPoseToWorld",
                        ns=name,
                        inputs={
                            "stamped_msg": BlackboardKey("move_into_pose_stamped_food"),
                            "target_frame": "j2n6s200_link_base",
                        },
                        outputs={
                            "transformed_msg": BlackboardKey(
                                "tool_tip_move_into_pose_world"
                            )
                        },
                    ),
                    PublishPoseAsTf(
                        name="PublishAboveFoodDebugFrame",
                        ns=name,
                        inputs={
                            "pose_to_publish": BlackboardKey(
                                "tool_tip_move_above_pose_world"
                            ),
                            "child_frame_id": "debug_above_pose",
                        },
                    ),
                    PublishPoseAsTf(
                        name="PublishIntoFoodDebugFrame",
                        ns=name,
                        inputs={
                            "pose_to_publish": BlackboardKey(
                                "tool_tip_move_into_pose_world"
                            ),
                            "child_frame_id": "debug_into_pose",
                        },
                    ),
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
                            "jaco_vel_indices_pin": BlackboardKey(
                                "jaco_vel_indices_pin"
                            ),
                            "articutool_vel_indices_pin": BlackboardKey(
                                "articutool_vel_indices_pin"
                            ),
                            "jaco_ee_frame_id_pin": BlackboardKey(
                                "jaco_ee_frame_id_pin"
                            ),
                            "tool_tip_frame_id_pin": BlackboardKey(
                                "tool_tip_frame_id_pin"
                            ),
                        },
                    ),
                ],
            )

        ### Define Tree Logic
        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                scoped_behavior(
                    name="Success",
                    # Set Approach F/T Thresh
                    pre_behavior=(
                        Success()  # pylint: disable=abstract-class-instantiated
                        # ToggleCollisionObject(
                        #     name="AllowTable",
                        #     ns=name,
                        #     inputs={
                        #         "collision_object_ids": ["table"],
                        #         "allow": True,
                        #     },
                        # )
                    ),
                    post_behavior=(
                        Success()  # pylint: disable=abstract-class-instantiated
                        # ToggleCollisionObject(
                        #     name="DisallowTable",
                        #     ns=name,
                        #     inputs={
                        #         "collision_object_ids": ["table"],
                        #         "allow": False,
                        #     },
                        # )
                    ),
                    on_preempt_timeout=5.0,
                    # Starts a new Sequence w/ Memory internally
                    workers=[
                        scoped_behavior(
                            name="OctomapAndTableCollision",
                            # Set Approach F/T Thresh
                            pre_behavior=py_trees.composites.Sequence(
                                name="AllowOctomapAndTable",
                                memory=True,
                                children=[
                                    ToggleCollisionObject(
                                        name="AllowOctomap",
                                        ns=name,
                                        inputs={
                                            "collision_object_ids": ["<octomap>"],
                                            "allow": True,
                                        },
                                    ),
                                    ToggleCollisionObject(
                                        name="AllowTable",
                                        ns=name,
                                        inputs={
                                            "collision_object_ids": ["table"],
                                            "allow": True,
                                        },
                                    ),
                                ],
                            ),
                            post_behavior=py_trees.composites.Sequence(
                                name="DisallowOctomapAndTable",
                                memory=True,
                                children=[
                                    ToggleCollisionObject(
                                        name="DisallowOctomap",
                                        ns=name,
                                        inputs={
                                            "collision_object_ids": ["<octomap>"],
                                            "allow": False,
                                        },
                                    ),
                                    ToggleCollisionObject(
                                        name="DisallowTable",
                                        ns=name,
                                        inputs={
                                            "collision_object_ids": ["table"],
                                            "allow": False,
                                        },
                                    ),
                                ],
                            ),
                            on_preempt_timeout=5.0,
                            workers=[
                                # Clear Octomap
                                py_trees_ros.service_clients.FromConstant(
                                    name="ClearOctomap",
                                    service_name="/clear_octomap",
                                    service_type=Empty,
                                    service_request=Empty.Request(),
                                    # Default fail if service is down
                                    wait_for_server_timeout_sec=0.0,
                                ),
                                py_trees.composites.Selector(
                                    name="BackupFlipFoodFrameSel",
                                    memory=True,
                                    children=[
                                        pre_acquisition_sequence(True),
                                        pre_acquisition_sequence(
                                            False, BlackboardKey("action")
                                        ),
                                    ],
                                ),
                                # This sequence implements the benchmark's robust planning logic.
                                # It finds a reachable skewer configuration, plans all motions,
                                # and then executes the first part of the motion.
                                CallSetOrientationControl(
                                    name="DisableArticutoolOrientation",
                                    ns=name,
                                    inputs={
                                        "control_mode": 0,
                                    },
                                    outputs={},
                                ),
                                MoveIt2JointConstraint(
                                    name="SetAtoolHomeGoal",
                                    ns=name,
                                    inputs={
                                        "joint_positions": [0.0, 0.0],
                                    },
                                    outputs={
                                        "constraints": BlackboardKey(
                                            "goal_constraints_home"
                                        )
                                    },
                                ),
                                MoveIt2Plan(
                                    name="PlanAtoolToPitch",
                                    ns=name,
                                    inputs={
                                        "goal_constraints": BlackboardKey(
                                            "goal_constraints_home"
                                        ),
                                        "group_name": "articutool",
                                    },
                                    outputs={
                                        "trajectory": BlackboardKey(
                                            "articutool_home_traj"
                                        )
                                    },
                                ),
                                SwitchArticutoolControllers(
                                    name="SwitchArticutoolToJointTrajectory",
                                    ns=name,
                                    inputs={
                                        "controllers_to_activate": [
                                            "joint_trajectory_controller"
                                        ],
                                        "controllers_to_deactivate": [
                                            "velocity_controller"
                                        ],
                                    },
                                    outputs={
                                        "switch_call_succeeded": None,
                                        "switch_response_ok": None,
                                    },
                                ),
                                ExecuteArticutoolTrajectory(
                                    name="ExecuteAtoolToHome",
                                    ns=name,
                                    inputs={
                                        "trajectory": BlackboardKey(
                                            "articutool_home_traj"
                                        )
                                    },
                                    outputs={
                                        "action_goal_accepted": BlackboardKey(
                                            "tool_goal_accepted"
                                        ),
                                        "action_result_code": BlackboardKey(
                                            "tool_exec_result_code"
                                        ),
                                        "action_status": BlackboardKey(
                                            "tool_action_status"
                                        ),
                                    },
                                ),
                                py_trees.composites.Sequence(
                                    name="DecoupledAcquisitionPlanAndMove",
                                    memory=True,
                                    children=[
                                        # A. Find a reachable skewer configuration by looping through tilt angles
                                        py_trees.composites.Sequence(
                                            name="FindReachableSkewerConfig",
                                            memory=True,
                                            children=[
                                                GenerateSkewerTiltCandidates(
                                                    name="GenerateTiltCandidates",
                                                    ns=name,
                                                    outputs={
                                                        "tilt_candidates_rad": BlackboardKey(
                                                            "tilt_candidates_rad"
                                                        ),
                                                        "tilt_index": BlackboardKey(
                                                            "tilt_index"
                                                        ),
                                                    },
                                                ),
                                                py_trees.decorators.Retry(
                                                    name="RetryWithNextTiltAngle",
                                                    num_failures=20,
                                                    child=py_trees.composites.Sequence(
                                                        name="AttemptSingleTiltAngle",
                                                        memory=True,
                                                        children=[
                                                            CalculateSkewerPoseForTilt(
                                                                name="CalculateCandidatePoses",
                                                                ns=name,
                                                                inputs={
                                                                    "tilt_candidates_rad": BlackboardKey(
                                                                        "tilt_candidates_rad"
                                                                    ),
                                                                    "tilt_index": BlackboardKey(
                                                                        "tilt_index"
                                                                    ),
                                                                    "tool_tip_move_above_pose_world": BlackboardKey(
                                                                        "tool_tip_move_above_pose_world"
                                                                    ),
                                                                    "tool_tip_move_into_pose_world": BlackboardKey(
                                                                        "tool_tip_move_into_pose_world"
                                                                    ),
                                                                    "initial_food_frame": BlackboardKey(
                                                                        "final_food_frame"
                                                                    ),
                                                                    "pinocchio_model": BlackboardKey(
                                                                        "pinocchio_model"
                                                                    ),
                                                                    "pinocchio_data": BlackboardKey(
                                                                        "pinocchio_data"
                                                                    ),
                                                                    "articutool_joint_names": [
                                                                        "atool_joint1",
                                                                        "atool_joint2",
                                                                    ],
                                                                },
                                                                outputs={
                                                                    "candidate_jaco_ee_above_pose": BlackboardKey(
                                                                        "candidate_jaco_ee_above_pose"
                                                                    ),
                                                                    "candidate_jaco_ee_into_pose": BlackboardKey(
                                                                        "candidate_jaco_ee_into_pose"
                                                                    ),
                                                                    "articutool_joint_positions": BlackboardKey(
                                                                        "candidate_atool_joint_positions"
                                                                    ),
                                                                    "tilt_index": BlackboardKey(
                                                                        "tilt_index"
                                                                    ),
                                                                },
                                                            ),
                                                            StampPoseFromPose(
                                                                name="StampCandidatePoseForIK",
                                                                ns=name,
                                                                inputs={
                                                                    "input_pose": BlackboardKey(
                                                                        "candidate_jaco_ee_above_pose"
                                                                    ),
                                                                    "frame_id": "world",
                                                                },
                                                                outputs={
                                                                    "output_pose_stamped": BlackboardKey(
                                                                        "stamped_candidate_pose"
                                                                    )
                                                                },
                                                            ),
                                                            MoveIt2ComputeIK(
                                                                name="CheckCandidatePoseReachable",
                                                                ns=name,
                                                                inputs={
                                                                    "target_pose": BlackboardKey(
                                                                        "stamped_candidate_pose"
                                                                    ),
                                                                    "group_name": "jaco_arm",
                                                                },
                                                                outputs={
                                                                    "ik_solution_joint_state": None,
                                                                    "success": BlackboardKey(
                                                                        "candidate_ik_success"
                                                                    ),
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                            ],
                                        ),
                                        # B. Plan all three trajectory segments now that we have a valid goal
                                        MoveIt2PoseConstraint(
                                            name="SetJacoAbovePoseGoal",
                                            ns=name,
                                            inputs={
                                                "pose": BlackboardKey(
                                                    "candidate_jaco_ee_above_pose"
                                                )
                                            },
                                            outputs={
                                                "constraints": BlackboardKey(
                                                    "goal_constraints_s1"
                                                )
                                            },
                                        ),
                                        MoveIt2Plan(
                                            name="PlanJacoToAbove",
                                            ns=name,
                                            inputs={
                                                "goal_constraints": BlackboardKey(
                                                    "goal_constraints_s1"
                                                ),
                                                "group_name": "jaco_arm",
                                                "max_velocity_scale": self.max_velocity_scaling_move_above,
                                                "max_acceleration_scale": self.max_acceleration_scaling_move_above,
                                                "allowed_planning_time": 10.0,
                                            },
                                            outputs={
                                                "trajectory": BlackboardKey(
                                                    "jaco_move_above_traj"
                                                ),
                                                "end_joint_state": BlackboardKey(
                                                    "jaco_move_above_end_joint_state"
                                                ),
                                            },
                                        ),
                                        MoveIt2JointConstraint(
                                            name="SetAtoolPitchGoal",
                                            ns=name,
                                            inputs={
                                                "joint_positions": BlackboardKey(
                                                    "candidate_atool_joint_positions"
                                                ),
                                            },
                                            outputs={
                                                "constraints": BlackboardKey(
                                                    "goal_constraints_s2"
                                                )
                                            },
                                        ),
                                        MoveIt2Plan(
                                            name="PlanAtoolToPitch",
                                            ns=name,
                                            inputs={
                                                "goal_constraints": BlackboardKey(
                                                    "goal_constraints_s2"
                                                ),
                                                "group_name": "articutool",
                                            },
                                            outputs={
                                                "trajectory": BlackboardKey(
                                                    "articutool_set_pitch_traj"
                                                ),
                                            },
                                        ),
                                        MoveIt2PoseConstraint(
                                            name="SetJacoIntoPoseGoal",
                                            ns=name,
                                            inputs={
                                                "pose": BlackboardKey(
                                                    "candidate_jaco_ee_into_pose"
                                                ),
                                                "tolerance_position": 0.01,
                                            },
                                            outputs={
                                                "constraints": BlackboardKey(
                                                    "goal_constraints_s3"
                                                )
                                            },
                                        ),
                                        MoveIt2Plan(
                                            name="PlanJacoToInto",
                                            ns=name,
                                            inputs={
                                                "goal_constraints": BlackboardKey(
                                                    "goal_constraints_s3"
                                                ),
                                                "group_name": "jaco_arm",
                                                "cartesian": True,
                                                "max_velocity_scale": self.max_velocity_scaling_move_into,
                                                "max_acceleration_scale": self.max_acceleration_scaling_move_into,
                                                "cartesian_max_step": 0.001,
                                                "cartesian_fraction_threshold": 0.92,
                                                "cartesian_jump_threshold": 5.0,
                                                "start_joint_state": BlackboardKey(
                                                    "jaco_move_above_end_joint_state"
                                                ),
                                                "max_path_len_joint": max_path_len_joint,
                                                "allowed_planning_time": 10.0,
                                            },
                                            outputs={
                                                "trajectory": BlackboardKey(
                                                    "jaco_move_into_traj"
                                                )
                                            },
                                        ),
                                        # C. Execute the first motion to move above the food
                                        MoveIt2Execute(
                                            name="ExecuteJacoToAbove",
                                            ns=name,
                                            inputs={
                                                "trajectory": BlackboardKey(
                                                    "jaco_move_above_traj"
                                                ),
                                                "group_name": "jaco_arm",
                                            },
                                            outputs={
                                                "error_code": None,
                                            },
                                        ),
                                        # D. Execute the Articutool pitch motion
                                        SwitchArticutoolControllers(
                                            name="SwitchArticutoolToJointTrajectory",
                                            ns=name,
                                            inputs={
                                                "controllers_to_activate": [
                                                    "joint_trajectory_controller"
                                                ],
                                                "controllers_to_deactivate": [
                                                    "velocity_controller"
                                                ],
                                            },
                                            outputs={
                                                "switch_call_succeeded": None,
                                                "switch_response_ok": None,
                                            },
                                        ),
                                        ExecuteArticutoolTrajectory(
                                            name="ExecuteAtoolToPitch",
                                            ns=name,
                                            inputs={
                                                "trajectory": BlackboardKey(
                                                    "articutool_set_pitch_traj"
                                                )
                                            },
                                            outputs={
                                                "action_goal_accepted": BlackboardKey(
                                                    "tool_goal_accepted"
                                                ),
                                                "action_result_code": BlackboardKey(
                                                    "tool_exec_result_code"
                                                ),
                                                "action_status": BlackboardKey(
                                                    "tool_action_status"
                                                ),
                                            },
                                        ),
                                    ],
                                ),
                                # If Anything goes wrong, reset FT to safe levels
                                scoped_behavior(
                                    name="SafeFTPreempt",
                                    # Set Approach F/T Thresh
                                    pre_behavior=py_trees.composites.Sequence(
                                        name=name,
                                        memory=True,
                                        children=[
                                            retry_call_ros_service(
                                                name="ApproachFTThresh",
                                                service_type=SetParameters,
                                                service_name="~/set_force_gate_controller_parameters",
                                                # Blackboard, not Constant
                                                request=None,
                                                # Need absolute Blackboard name
                                                key_request=Blackboard.separator.join(
                                                    [
                                                        name,
                                                        BlackboardKey(
                                                            "approach_thresh"
                                                        ),
                                                    ]
                                                ),
                                                key_response=Blackboard.separator.join(
                                                    [name, BlackboardKey("ft_response")]
                                                ),
                                                response_checks=[
                                                    py_trees.common.ComparisonExpression(
                                                        variable=Blackboard.separator.join(
                                                            [
                                                                name,
                                                                BlackboardKey(
                                                                    "ft_response"
                                                                ),
                                                            ]
                                                        ),
                                                        value=SetParameters.Response(),  # Unused
                                                        operator=set_parameter_response_all_success,
                                                    )
                                                ],
                                            ),
                                        ],
                                    ),
                                    post_behavior=py_trees.composites.Sequence(
                                        name=name,
                                        memory=True,
                                        children=[
                                            pre_moveto_config(
                                                name="PostAcquireFTSet", re_tare=False
                                            ),
                                        ],
                                    ),
                                    on_preempt_timeout=5.0,
                                    # Starts a new Sequence w/ Memory internally
                                    workers=[
                                        CallSetOrientationControl(
                                            name="SetArticutoolOrientation",
                                            ns=name,
                                            inputs={
                                                "control_mode": 0,
                                            },
                                            outputs={},
                                        ),
                                        SwitchArticutoolControllers(
                                            name="SwitchArticutoolToVelocity",
                                            ns=name,
                                            inputs={
                                                "controllers_to_activate": [
                                                    "velocity_controller"
                                                ],
                                                "controllers_to_deactivate": [
                                                    "joint_trajectory_controller"
                                                ],
                                            },
                                            outputs={
                                                "switch_call_succeeded": None,
                                                "switch_response_ok": None,
                                            },
                                        ),
                                        ExecuteNamedPrimitive(
                                            name="RunPreMoveIntoPrimitive",
                                            ns=name,
                                            inputs={
                                                "primitive_name": BlackboardKey(
                                                    "pre_move_into_action_name"
                                                ),
                                                "primitive_params": BlackboardKey(
                                                    "pre_move_into_action_params"
                                                ),
                                            },
                                            outputs={
                                                "primitive_status": None,
                                            },
                                        ),
                                        ### Move Into Food
                                        # E. Execute the final Jaco Cartesian insertion.
                                        # This is wrapped in FailureIsSuccess because we expect
                                        # it to be interrupted by the F/T sensor.
                                        py_trees.decorators.FailureIsSuccess(
                                            name="ExecuteJacoToInto_SucceedOnFT",
                                            child=MoveIt2Execute(
                                                name="ExecuteJacoToInto",
                                                ns=name,
                                                inputs={
                                                    "trajectory": BlackboardKey(
                                                        "jaco_move_into_traj"
                                                    ),
                                                    "group_name": "jaco_arm",
                                                },
                                                outputs={
                                                    "error_code": None,
                                                },
                                            ),
                                        ),
                                        CallSetOrientationControl(
                                            name="SetArticutoolOrientation",
                                            ns=name,
                                            inputs={
                                                "control_mode": 0,
                                            },
                                            outputs={},
                                        ),
                                        ExecuteNamedPrimitive(
                                            name="RunPostMoveIntoPrimitive",
                                            ns=name,
                                            inputs={
                                                "primitive_name": BlackboardKey(
                                                    "post_move_into_action_name"
                                                ),
                                                "primitive_params": BlackboardKey(
                                                    "post_move_into_action_params"
                                                ),
                                            },
                                            outputs={
                                                "primitive_status": None,
                                            },
                                        ),
                                        ### Scoped Behavior for Moveit2_Servo
                                        scoped_behavior(
                                            name="MoveIt2Servo",
                                            # Set Approach F/T Thresh
                                            pre_behavior=py_trees.composites.Sequence(
                                                name=name,
                                                memory=True,
                                                children=[
                                                    StartServoTree(
                                                        self._node,
                                                        servo_controller_name="jaco_arm_cartesian_controller",
                                                        start_moveit_servo=False,
                                                    )
                                                    .create_tree(
                                                        name="StartServoScoped"
                                                    )
                                                    .root,
                                                ],
                                            ),
                                            # Reset FT and Stop Servo
                                            post_behavior=py_trees.composites.Sequence(
                                                name=name,
                                                memory=True,
                                                children=[
                                                    pre_moveto_config(
                                                        name="PostServoFTSet",
                                                        re_tare=False,
                                                        f_mag=1.0,
                                                        param_service_name="~/set_cartesian_controller_parameters",
                                                    ),
                                                    StopServoTree(
                                                        self._node,
                                                        servo_controller_name="jaco_arm_cartesian_controller",
                                                        stop_moveit_servo=False,
                                                    )
                                                    .create_tree(name="StopServoScoped")
                                                    .root,
                                                ],
                                            ),
                                            on_preempt_timeout=5.0,
                                            # Starts a new Sequence w/ Memory internally
                                            workers=[
                                                py_trees.composites.Selector(
                                                    name="InFoodErrorSelector",
                                                    memory=True,
                                                    children=[
                                                        py_trees.composites.Sequence(
                                                            name="InFoodGraspExtract",
                                                            memory=True,
                                                            children=[
                                                                ### Grasp
                                                                retry_call_ros_service(
                                                                    name="GraspFTThresh",
                                                                    service_type=SetParameters,
                                                                    service_name="~/set_cartesian_controller_parameters",
                                                                    # Blackboard, not Constant
                                                                    request=None,
                                                                    # Need absolute Blackboard name
                                                                    key_request=Blackboard.separator.join(
                                                                        [
                                                                            name,
                                                                            BlackboardKey(
                                                                                "grasp_thresh"
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    key_response=Blackboard.separator.join(
                                                                        [
                                                                            name,
                                                                            BlackboardKey(
                                                                                "ft_response"
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    response_checks=[
                                                                        py_trees.common.ComparisonExpression(
                                                                            variable=Blackboard.separator.join(
                                                                                [
                                                                                    name,
                                                                                    BlackboardKey(
                                                                                        "ft_response"
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                            value=SetParameters.Response(),  # Unused
                                                                            operator=set_parameter_response_all_success,
                                                                        )
                                                                    ],
                                                                ),
                                                                ComputeActionTwist(
                                                                    name="ComputeGrasp",
                                                                    ns=name,
                                                                    inputs={
                                                                        "action": BlackboardKey(
                                                                            "action"
                                                                        ),
                                                                        "is_grasp": True,
                                                                    },
                                                                    outputs={
                                                                        "twist": BlackboardKey(
                                                                            "twist"
                                                                        ),
                                                                        "duration": BlackboardKey(
                                                                            "duration"
                                                                        ),
                                                                    },
                                                                ),
                                                                ServoMove(
                                                                    name="GraspServo",
                                                                    ns=name,
                                                                    inputs={
                                                                        "twist": BlackboardKey(
                                                                            "twist"
                                                                        ),
                                                                        "duration": BlackboardKey(
                                                                            "duration"
                                                                        ),
                                                                        "pub_topic": "~/cartesian_twist_cmds",
                                                                        "servo_status_sub_topic": None,
                                                                    },
                                                                ),  # Auto Zero-Twist on terminate()
                                                                ### Extraction
                                                                ComputeActionTwist(
                                                                    name="ComputeExtract",
                                                                    ns=name,
                                                                    inputs={
                                                                        "action": BlackboardKey(
                                                                            "action"
                                                                        ),
                                                                        "is_grasp": False,
                                                                    },
                                                                    outputs={
                                                                        "twist": BlackboardKey(
                                                                            "twist"
                                                                        ),
                                                                        "duration": BlackboardKey(
                                                                            "duration"
                                                                        ),
                                                                    },
                                                                ),
                                                                retry_call_ros_service(
                                                                    name="ExtractionFTThresh",
                                                                    service_type=SetParameters,
                                                                    service_name="~/set_cartesian_controller_parameters",
                                                                    # Blackboard, not Constant
                                                                    request=None,
                                                                    # Need absolute Blackboard name
                                                                    key_request=Blackboard.separator.join(
                                                                        [
                                                                            name,
                                                                            BlackboardKey(
                                                                                "ext_thresh"
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    key_response=Blackboard.separator.join(
                                                                        [
                                                                            name,
                                                                            BlackboardKey(
                                                                                "ft_response"
                                                                            ),
                                                                        ]
                                                                    ),
                                                                    response_checks=[
                                                                        py_trees.common.ComparisonExpression(
                                                                            variable=Blackboard.separator.join(
                                                                                [
                                                                                    name,
                                                                                    BlackboardKey(
                                                                                        "ft_response"
                                                                                    ),
                                                                                ]
                                                                            ),
                                                                            value=SetParameters.Response(),  # Unused
                                                                            operator=set_parameter_response_all_success,
                                                                        )
                                                                    ],
                                                                ),
                                                                ServoMove(
                                                                    name="ExtractServo",
                                                                    ns=name,
                                                                    inputs={
                                                                        "twist": BlackboardKey(
                                                                            "twist"
                                                                        ),
                                                                        "duration": BlackboardKey(
                                                                            "duration"
                                                                        ),
                                                                        "pub_topic": "~/cartesian_twist_cmds",
                                                                        "servo_status_sub_topic": None,
                                                                    },
                                                                ),  # Auto Zero-Twist on terminate()
                                                                ft_thresh_satisfied(
                                                                    name="CheckFTForkOffPlate"
                                                                ),
                                                            ],  # End InFoodGraspExtract.children
                                                        ),  # End InFoodGraspExtract
                                                        recovery_tree,
                                                    ],  # End InFoodErrorSelector.children
                                                ),  # End InFoodErrorSelector
                                            ],  # End MoveIt2Servo.workers
                                        ),  # End MoveIt2Servo
                                    ],  # End SafeFTPreempt.workers
                                ),  # End SafeFTPreempt
                                post_acquisition_sequence(),
                            ],  # End OctomapAndTableCollision.workers
                        ),  # OctomapAndTableCollision
                    ]
                    + resting_position_behaviors,  # End Success.workers
                ),  # End Success # TableCollision
            ],  # End root_seq.children
        )  # End root_seq

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)

    # Override goal to read arguments into local blackboard
    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree

        # Check goal type
        if not isinstance(goal, AcquireFood.Goal):
            return False

        # Pickle goal for debugging
        if self.pickle_goal_path is not None:
            with open(self.pickle_goal_path, "wb") as file:
                pickle.dump(goal, file)
            self._node.get_logger().info(f"Pickled goal to {self.pickle_goal_path}")

        # Write tree inputs to blackboard
        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(key="mask", access=py_trees.common.Access.WRITE)
        blackboard.mask = goal.detected_food
        blackboard.register_key(key="camera_info", access=py_trees.common.Access.WRITE)
        blackboard.camera_info = goal.camera_info
        blackboard.register_key(key="timestamp", access=py_trees.common.Access.WRITE)
        blackboard.timestamp = Time.from_msg(goal.header.stamp)

        # Adds MoveToVisitor for Feedback
        return super().send_goal(tree, goal)

    # Override result to handle timing outside MoveTo Behaviors
    @override
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree
        if action_type is not AcquireFood:
            return None

        # Get MoveTo Params
        feedback_msg = super().get_feedback(tree, action_type)

        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(
            key="action_response", access=py_trees.common.Access.READ
        )
        blackboard.register_key(key="action_index", access=py_trees.common.Access.READ)

        feedback_msg.action_info_populated = True
        try:
            # TODO: add posthoc
            feedback_msg.selection_id = blackboard.action_response.id
            feedback_msg.action_index = int(blackboard.action_index)
        except KeyError as e:
            self._node.get_logger().debug(
                f"Failed to populate action_info in AcquireFoodTree: {e}"
            )
            feedback_msg.action_info_populated = False
        return feedback_msg

    # Override result to add other elements to result msg
    @override
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree
        if action_type is not AcquireFood:
            return None
        # Get MoveTo Params
        response = super().get_result(tree, action_type)
        return response
