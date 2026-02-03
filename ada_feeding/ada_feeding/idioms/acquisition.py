# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines idioms (reusable behavior tree sequences) specifically
for the Articutool acquisition pipeline. This includes the "Plan-then-Verify"
logic, localized primitives, and active leveling sequences.
"""

from typing import List, Optional, Union

import numpy as np
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.node import Node
from std_srvs.srv import Empty
import rclpy.time

# Local imports
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.behaviors.acquisition import (
    ComputeFoodFrame,
    ComputeActionConstraints,
    ConditionallyRotateFoodFrame,
    GenerateSkewerTiltCandidates,
    CalculateSkewerPoseForTilt,
)
from ada_feeding.behaviors.moveit2 import (
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
    MoveIt2PoseConstraint,
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2ComputeIK,
    MoveIt2ComputeFK,
)
from ada_feeding.behaviors.state import (
    GetJointStates,
    ExtractPoseFromPosesByLink,
    CheckArticutoolPathDynamicFeasibility,
    PublishPoseAsTf,
    ComputeSlerpMidpointOrientation,
    CheckElbowUpConfiguration,
)
from ada_feeding.behaviors.ros.msgs import StampPoseFromPose
from ada_feeding.behaviors.ros.tf import ApplyTransform
from ada_feeding.behaviors.articutool import (
    ExecuteArticutoolTrajectory,
    CallSetOrientationControl,
    SwitchArticutoolControllers,
    ComputeArticutoolLevelingJoints,
    ExecuteNamedPrimitive,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config


def get_pre_acquisition_setup(
    name: str,
    ns: str,
    flip_food_frame: bool = False,
    action_key: Optional[BlackboardKey] = None,
) -> py_trees.behaviour.Behaviour:
    """
    Orchestrates the setup phase: Detecting food, selecting an action,
    computing constraints, and publishing debug frames.
    """
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
                    ns=ns,
                    inputs={
                        "camera_info": BlackboardKey("camera_info"),
                        "mask": BlackboardKey("mask"),
                        # Override timestamp to use latest
                        "timestamp": rclpy.time.Time(),
                        "flip_food_frame": flip_food_frame,
                    },
                    outputs={
                        "action_select_request": BlackboardKey("action_request"),
                        "food_frame": BlackboardKey("initial_food_frame"),
                    },
                ),
            ),
            # Get Action to Use
            py_trees_ros.service_clients.FromBlackboard(
                name="AcquisitionSelect",
                service_name="~/action_select",
                service_type=AcquisitionSelect,
                # Need absolute Blackboard name for FromBlackboard
                key_request=Blackboard.separator.join(
                    [ns, BlackboardKey("action_request")]
                ),
                key_response=Blackboard.separator.join(
                    [ns, BlackboardKey("action_response")]
                ),
                wait_for_server_timeout_sec=0.0,
            ),
            # Get MoveIt2 Constraints
            py_trees.decorators.Timeout(
                name="ComputeActionConstraintsTimeout",
                duration=1.0,
                child=ComputeActionConstraints(
                    name="ComputeActionConstraints",
                    ns=ns,
                    inputs={
                        "action_select_response": BlackboardKey("action_response"),
                        "action": action_key,
                    },
                    outputs={
                        "move_above_pose": BlackboardKey("move_above_pose_food_frame"),
                        "move_into_pose": BlackboardKey("move_into_pose_food_frame"),
                        "approach_thresh": BlackboardKey("approach_thresh"),
                        "grasp_thresh": BlackboardKey("grasp_thresh"),
                        "ext_thresh": BlackboardKey("ext_thresh"),
                        "retract_thresh": BlackboardKey("retract_thresh"),
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
                        "jaco_ee_tilt_angle_min": BlackboardKey(
                            "jaco_ee_tilt_angle_min"
                        ),
                        "jaco_ee_tilt_angle_max": BlackboardKey(
                            "jaco_ee_tilt_angle_max"
                        ),
                        "should_align_to_base": BlackboardKey("should_align_to_base"),
                    },
                ),
            ),
            ConditionallyRotateFoodFrame(
                name="ConditionallyRotateFoodFrame",
                ns=ns,
                inputs={
                    "initial_food_frame": BlackboardKey("initial_food_frame"),
                    "should_align_to_base": BlackboardKey("should_align_to_base"),
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
                ns=ns,
                inputs={
                    "input_pose": BlackboardKey("move_above_pose_food_frame"),
                    "frame_id": "food",
                },
                outputs={
                    "output_pose_stamped": BlackboardKey("move_above_pose_stamped_food")
                },
            ),
            ApplyTransform(
                name="TransformMoveAbovePoseToWorld",
                ns=ns,
                inputs={
                    "stamped_msg": BlackboardKey("move_above_pose_stamped_food"),
                    "target_frame": "j2n6s200_link_base",
                },
                outputs={
                    "transformed_msg": BlackboardKey("tool_tip_move_above_pose_world")
                },
            ),
            StampPoseFromPose(
                name="StampMoveIntoPoseFood",
                ns=ns,
                inputs={
                    "input_pose": BlackboardKey("move_into_pose_food_frame"),
                    "frame_id": "food",
                },
                outputs={
                    "output_pose_stamped": BlackboardKey("move_into_pose_stamped_food")
                },
            ),
            ApplyTransform(
                name="TransformMoveIntoPoseToWorld",
                ns=ns,
                inputs={
                    "stamped_msg": BlackboardKey("move_into_pose_stamped_food"),
                    "target_frame": "j2n6s200_link_base",
                },
                outputs={
                    "transformed_msg": BlackboardKey("tool_tip_move_into_pose_world")
                },
            ),
            PublishPoseAsTf(
                name="PublishAboveFoodDebugFrame",
                ns=ns,
                inputs={
                    "pose_to_publish": BlackboardKey("tool_tip_move_above_pose_world"),
                    "child_frame_id": "debug_above_pose",
                },
            ),
            PublishPoseAsTf(
                name="PublishIntoFoodDebugFrame",
                ns=ns,
                inputs={
                    "pose_to_publish": BlackboardKey("tool_tip_move_into_pose_world"),
                    "child_frame_id": "debug_into_pose",
                },
            ),
        ],
    )


def get_robust_move_above_sequence(
    name: str,
    ns: str,
    max_velocity_scaling_move_above: float,
    max_acceleration_scaling_move_above: float,
    max_velocity_scaling_move_into: float,
    max_acceleration_scaling_move_into: float,
    allowed_planning_time_for_move_above: float,
    allowed_planning_time_for_move_into: float,
) -> py_trees.behaviour.Behaviour:
    """
    Implements the "Plan-Then-Verify" idiom.
    It attempts to find a valid Articutool/Jaco configuration that satisfies:
    1. Kinematic reachability (IK)
    2. Elbow-Up configuration check
    3. Successful motion plan for the 'MoveAbove' segment
    """
    # The max amount that each joint can move for any computed plan.
    # Intended to reduce swivels.
    max_path_len_joint = {
        "j2n6s200_joint_1": np.pi / 2.0,
        "j2n6s200_joint_2": np.pi / 4.0,
    }

    return py_trees.composites.Sequence(
        name="DecoupledAcquisitionPlanAndMove",
        memory=True,
        children=[
            # A. Find a reachable skewer configuration AND a valid motion plan to it
            py_trees.composites.Sequence(
                name="FindReachableSkewerConfigAndPlan",
                memory=True,
                children=[
                    GenerateSkewerTiltCandidates(
                        name="GenerateTiltCandidates",
                        ns=ns,
                        inputs={
                            "jaco_ee_tilt_angle_min": BlackboardKey(
                                "jaco_ee_tilt_angle_min"
                            ),
                            "jaco_ee_tilt_angle_max": BlackboardKey(
                                "jaco_ee_tilt_angle_max"
                            ),
                        },
                        outputs={
                            "tilt_candidates_rad": BlackboardKey("tilt_candidates_rad"),
                            "tilt_index": BlackboardKey("tilt_index"),
                        },
                    ),
                    py_trees.decorators.Retry(
                        name="RetryWithNextTiltAngle",
                        num_failures=50,
                        child=py_trees.composites.Sequence(
                            name="AttemptSingleTiltAngle",
                            memory=True,
                            children=[
                                # 1. Calculate the geometric poses for the current tilt angle
                                CalculateSkewerPoseForTilt(
                                    name="CalculateCandidatePoses",
                                    ns=ns,
                                    inputs={
                                        "tilt_candidates_rad": BlackboardKey(
                                            "tilt_candidates_rad"
                                        ),
                                        "tilt_index": BlackboardKey("tilt_index"),
                                        "tool_tip_move_above_pose_world": BlackboardKey(
                                            "tool_tip_move_above_pose_world"
                                        ),
                                        "tool_tip_move_into_pose_world": BlackboardKey(
                                            "tool_tip_move_into_pose_world"
                                        ),
                                        "initial_food_frame": BlackboardKey(
                                            "final_food_frame"
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
                                        "tilt_index": BlackboardKey("tilt_index"),
                                    },
                                ),
                                # 2. Stamp the "above" pose for the IK check
                                StampPoseFromPose(
                                    name="StampCandidatePoseForIK",
                                    ns=ns,
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
                                # 3. Check if the pose is reachable via IK
                                MoveIt2ComputeIK(
                                    name="CheckCandidatePoseReachable",
                                    ns=ns,
                                    inputs={
                                        "target_pose": BlackboardKey(
                                            "stamped_candidate_pose"
                                        ),
                                        "group_name": "jaco_arm",
                                    },
                                    outputs={
                                        "ik_solution_joint_state": BlackboardKey(
                                            "candidate_ik_solution"
                                        ),
                                        "success": BlackboardKey(
                                            "candidate_ik_success"
                                        ),
                                    },
                                ),
                                # 4. Check if the IK solution is in an "elbow-up" configuration
                                CheckElbowUpConfiguration(
                                    name="ValidateElbowUp",
                                    ns=ns,
                                    inputs={
                                        "ik_solution_joint_state": BlackboardKey(
                                            "candidate_ik_solution"
                                        ),
                                        "jaco_joint_names": [
                                            "j2n6s200_joint_1",
                                            "j2n6s200_joint_2",
                                            "j2n6s200_joint_3",
                                            "j2n6s200_joint_4",
                                            "j2n6s200_joint_5",
                                            "j2n6s200_joint_6",
                                        ],
                                    },
                                ),
                                # 5. Create the goal constraint for this valid pose
                                MoveIt2PoseConstraint(
                                    name="SetJacoAbovePoseGoal",
                                    ns=ns,
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
                                # 6. Attempt to plan a motion to this valid pose
                                # If this fails, the whole sequence fails, and RetryWithNextTiltAngle
                                # will try the next tilt angle.
                                MoveIt2Plan(
                                    name="PlanJacoToAbove",
                                    ns=ns,
                                    inputs={
                                        "goal_constraints": BlackboardKey(
                                            "goal_constraints_s1"
                                        ),
                                        "group_name": "jaco_arm",
                                        "max_velocity_scale": max_velocity_scaling_move_above,
                                        "max_acceleration_scale": max_acceleration_scaling_move_above,
                                        "allowed_planning_time": allowed_planning_time_for_move_above,
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
                            ],
                        ),
                    ),
                ],
            ),
            # B. Plan the remaining trajectory segments
            #    (We only get here if the PlanJacoToAbove was successful)
            MoveIt2JointConstraint(
                name="SetAtoolPitchGoal",
                ns=ns,
                inputs={
                    "joint_positions": BlackboardKey("candidate_atool_joint_positions"),
                },
                outputs={"constraints": BlackboardKey("goal_constraints_s2")},
            ),
            MoveIt2Plan(
                name="PlanAtoolToPitch",
                ns=ns,
                inputs={
                    "goal_constraints": BlackboardKey("goal_constraints_s2"),
                    "group_name": "articutool",
                    "max_velocity_scale": 1.0,
                },
                outputs={
                    "trajectory": BlackboardKey("articutool_set_pitch_traj"),
                },
            ),
            MoveIt2PoseConstraint(
                name="SetJacoIntoPoseGoal",
                ns=ns,
                inputs={
                    "pose": BlackboardKey("candidate_jaco_ee_into_pose"),
                    "tolerance_position": 0.01,
                },
                outputs={"constraints": BlackboardKey("goal_constraints_s3")},
            ),
            MoveIt2Plan(
                name="PlanJacoToInto",
                ns=ns,
                inputs={
                    "goal_constraints": BlackboardKey("goal_constraints_s3"),
                    "group_name": "jaco_arm",
                    "cartesian": True,
                    "max_velocity_scale": max_velocity_scaling_move_into,
                    "max_acceleration_scale": max_acceleration_scaling_move_into,
                    "cartesian_max_step": 0.001,
                    "cartesian_fraction_threshold": 0.92,
                    "cartesian_jump_threshold": 0.0,
                    "start_joint_state": BlackboardKey(
                        "jaco_move_above_end_joint_state"
                    ),
                    "max_path_len_joint": max_path_len_joint,
                    "allowed_planning_time": allowed_planning_time_for_move_into,
                },
                outputs={"trajectory": BlackboardKey("jaco_move_into_traj")},
            ),
            # C. Execute the first motion to move above the food
            MoveIt2Execute(
                name="ExecuteJacoToAbove",
                ns=ns,
                inputs={
                    "trajectory": BlackboardKey("jaco_move_above_traj"),
                    "group_name": "jaco_arm",
                },
                outputs={
                    "error_code": None,
                },
            ),
            # D. Execute the Articutool pitch motion
            SwitchArticutoolControllers(
                name="SwitchArticutoolToJointTrajectory",
                ns=ns,
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
                name="ExecuteAtoolToPitch",
                ns=ns,
                inputs={"trajectory": BlackboardKey("articutool_set_pitch_traj")},
                outputs={
                    "action_goal_accepted": BlackboardKey("tool_goal_accepted"),
                    "action_result_code": BlackboardKey("tool_exec_result_code"),
                    "action_status": BlackboardKey("tool_action_status"),
                },
            ),
        ],
    )


def get_articutool_move_into_sequence(
    name: str, ns: str
) -> py_trees.behaviour.Behaviour:
    """
    Disables orientation control, switches to velocity for primitives,
    runs pre-move primitive, executes the Cartesian push, and runs post-move primitive.
    """
    return py_trees.composites.Sequence(
        name="ArticutoolMoveIntoFood",
        memory=True,
        children=[
            CallSetOrientationControl(
                name="SetArticutoolOrientation",
                ns=ns,
                inputs={
                    "control_mode": 0,
                },
                outputs={},
            ),
            SwitchArticutoolControllers(
                name="SwitchArticutoolToVelocity",
                ns=ns,
                inputs={
                    "controllers_to_activate": ["velocity_controller"],
                    "controllers_to_deactivate": ["joint_trajectory_controller"],
                },
                outputs={
                    "switch_call_succeeded": None,
                    "switch_response_ok": None,
                },
            ),
            ExecuteNamedPrimitive(
                name="RunPreMoveIntoPrimitive",
                ns=ns,
                inputs={
                    "primitive_name": BlackboardKey("pre_move_into_action_name"),
                    "primitive_params": BlackboardKey("pre_move_into_action_params"),
                },
                outputs={
                    "primitive_status": None,
                },
            ),
            ### Move Into Food
            # Execute the final Jaco Cartesian insertion.
            # This is wrapped in FailureIsSuccess because we expect
            # it to be interrupted by the F/T sensor.
            py_trees.decorators.FailureIsSuccess(
                name="ExecuteJacoToInto_SucceedOnFT",
                child=MoveIt2Execute(
                    name="ExecuteJacoToInto",
                    ns=ns,
                    inputs={
                        "trajectory": BlackboardKey("jaco_move_into_traj"),
                        "group_name": "jaco_arm",
                    },
                    outputs={},
                ),
            ),
        ],
    )


def get_post_acquisition_leveling_sequence(
    name: str,
    ns: str,
    node: Node,
    allowed_planning_time: float,
) -> py_trees.behaviour.Behaviour:
    """
    Performs the leveling procedure after acquisition.
    1. Compute FK of Jaco.
    2. Compute inverse kinematics for Articutool to align with gravity.
    3. Plan and execute Articutool motion.
    """
    return py_trees.composites.Sequence(
        name="PostAcquisitionSequence",
        memory=True,
        children=[
            CallSetOrientationControl(
                name="DisableArticutoolOrientation",
                ns=ns,
                inputs={
                    "control_mode": 0,
                },
                outputs={},
            ),
            GetJointStates(
                name="GetJacoArmStateForLeveling",
                ns=ns,
                node=node,
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
                ns=ns,
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
                ns=ns,
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
                ns=ns,
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
                ns=ns,
                inputs={
                    "joint_positions": BlackboardKey("articutool_joint_positions"),
                },
                outputs={
                    "constraints": BlackboardKey("articutool_leveling_constraints")
                },
            ),
            py_trees.decorators.Timeout(
                name="PlanToLevelArticutoolTimeout",
                duration=10.0 * allowed_planning_time,
                child=MoveIt2Plan(
                    name="PlanToLevelArticutool",
                    ns=ns,
                    inputs={
                        "goal_constraints": BlackboardKey(
                            "articutool_leveling_constraints"
                        ),
                        "group_name": "articutool",
                        "max_velocity_scale": 0.5,
                        "allowed_planning_time": 0.1,
                    },
                    outputs={
                        "trajectory": BlackboardKey("level_articutool_trajectory")
                    },
                ),
            ),
            SwitchArticutoolControllers(
                name="SwitchArticutoolToJointTrajectory",
                ns=ns,
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
                ns=ns,
                inputs={
                    "trajectory": BlackboardKey("level_articutool_trajectory"),
                },
                outputs={
                    "action_goal_accepted": BlackboardKey("tool_goal_accepted"),
                    "action_result_code": BlackboardKey("tool_exec_result_code"),
                    "action_status": BlackboardKey("tool_action_status"),
                },
            ),
        ],
    )


def get_post_retract_primitive_sequence(
    name: str, ns: str
) -> py_trees.behaviour.Behaviour:
    """
    Executes primitives after retraction (e.g., cleaning or settling food),
    then re-enables orientation control.
    """
    return py_trees.composites.Sequence(
        name="PostRetractSequence",
        memory=True,
        children=[
            SwitchArticutoolControllers(
                name="SwitchArticutoolToVelocity",
                ns=ns,
                inputs={
                    "controllers_to_activate": ["velocity_controller"],
                    "controllers_to_deactivate": ["joint_trajectory_controller"],
                },
                outputs={
                    "switch_call_succeeded": None,
                    "switch_response_ok": None,
                },
            ),
            ExecuteNamedPrimitive(
                name="RunPostAcquisitionPrimitive",
                ns=ns,
                inputs={
                    "primitive_name": BlackboardKey("post_acquisition_action_name"),
                    "primitive_params": BlackboardKey("post_acquisition_action_params"),
                },
                outputs={
                    "primitive_status": None,
                },
            ),
            CallSetOrientationControl(
                name="SetArticutoolOrientation",
                ns=ns,
                inputs={
                    "control_mode": 1,  # MODE_LEVELING
                },
                outputs={},
            ),
        ],
    )


def get_resting_sequence(
    name: str,
    ns: str,
    node: Node,
    resting_joint_positions: List[float],
    max_velocity_scale: float,
    max_acceleration_scale: float,
    allowed_planning_time: float,
) -> py_trees.behaviour.Behaviour:
    """
    Generates the sequence to move the arm to its resting position.
    Calculates a dynamic orientation constraint (SLERP) between the current
    pose and the resting pose to prevent spills during this transport.
    """
    return py_trees.composites.Sequence(
        name="PostRetractSequence",
        memory=True,
        children=[
            # 1. Get the current joint state for the Jaco arm
            GetJointStates(
                name="GetJacoStartState",
                ns=ns,
                node=node,
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
                outputs={"joint_state": BlackboardKey("current_jaco_joint_state")},
            ),
            # 2. Use FK to calculate the current EE pose
            MoveIt2ComputeFK(
                name="GetStartEEPose",
                ns=ns,
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
                ns=ns,
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
                ns=ns,
                inputs={
                    "group_name": "jaco_arm",
                    "joint_state": resting_joint_positions,
                    "fk_link_names": ["j2n6s200_end_effector"],
                },
                outputs={
                    "fk_poses": BlackboardKey("goal_ee_fk_poses"),
                    "success": None,
                },
            ),
            ExtractPoseFromPosesByLink(
                name="ExtractGoalEEPose",
                ns=ns,
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
                ns=ns,
                inputs={
                    "start_ee_pose": BlackboardKey("start_ee_pose"),
                    "goal_ee_pose": BlackboardKey("goal_ee_pose"),
                    "tolerance": (np.pi / 2, 2 * np.pi, np.pi / 4),
                },
                outputs={
                    "midpoint_orientation_quaternion": BlackboardKey(
                        "path_constraint_quat"
                    ),
                    "path_constraint_tolerance": BlackboardKey("path_constraint_tol"),
                },
            ),
            # Create the MoveIt2 orientation constraint message
            MoveIt2OrientationConstraint(
                name="CreateRestingPathConstraintMsg",
                ns=ns,
                inputs={
                    "quat_xyzw": BlackboardKey("path_constraint_quat"),
                    "tolerance": BlackboardKey("path_constraint_tol"),
                },
                outputs={
                    "constraints": BlackboardKey("resting_path_constraints"),
                },
            ),
            MoveIt2JointConstraint(
                name="RestingConstraint",
                ns=ns,
                inputs={
                    "joint_positions": resting_joint_positions,
                },
                outputs={
                    "constraints": BlackboardKey("goal_constraints"),
                },
            ),
            py_trees.decorators.Timeout(
                name="RestingPlanTimeout",
                # Increase allowed_planning_time to account for ROS2 overhead
                duration=10.0 * allowed_planning_time,
                child=MoveIt2Plan(
                    name="RestingPlan",
                    ns=ns,
                    inputs={
                        "goal_constraints": BlackboardKey("goal_constraints"),
                        "path_constraints": BlackboardKey("resting_path_constraints"),
                        "max_velocity_scale": max_velocity_scale,
                        "max_acceleration_scale": max_acceleration_scale,
                        "allowed_planning_time": allowed_planning_time,
                        "group_name": "jaco_arm",
                    },
                    outputs={"trajectory": BlackboardKey("resting_trajectory")},
                ),
            ),
            CheckArticutoolPathDynamicFeasibility(
                name="CheckArticutoolDynamicFeasibilityForResting",
                ns=ns,
                inputs={
                    "jaco_trajectory": BlackboardKey("resting_trajectory"),
                    "articutool_pitch_limits_rad": (-np.pi / 2, np.pi / 2),
                    "articutool_roll_limits_rad": (-np.pi, np.pi),
                    "articutool_max_joint_velocity": 4.0,
                    "num_trajectory_points_to_check": 5,
                },
                outputs={
                    "articutool_is_dynamic_feasible": BlackboardKey(
                        "articutool_can_maintain_leveling"
                    )
                },
            ),
            MoveIt2Execute(
                name="Resting",
                ns=ns,
                inputs={
                    "trajectory": BlackboardKey("resting_trajectory"),
                    "group_name": "jaco_arm",
                },
                outputs={},
            ),
        ],
    )
