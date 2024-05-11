#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import pickle
from typing import List, Optional

# Third-party imports
from geometry_msgs.msg import Twist, TwistStamped, Vector3
import numpy as np
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
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
)
from ada_feeding.behaviors.moveit2 import (
    MoveIt2JointConstraint,
    MoveIt2OrientationConstraint,
    MoveIt2PoseConstraint,
    MoveIt2PositionOffsetConstraint,
    MoveIt2Plan,
    MoveIt2Execute,
    ServoMove,
    ToggleCollisionObject,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import (
    pre_moveto_config,
    scoped_behavior,
    retry_call_ros_service,
)
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_wheelchair_wall_behavior,
    get_remove_in_front_of_wheelchair_wall_behavior,
)
from ada_feeding.idioms.ft_thresh_utils import ft_thresh_satisfied
from ada_feeding.idioms.pre_moveto_config import set_parameter_response_all_success
from ada_feeding.trees import MoveToTree, StartServoTree, StopServoTree


class AcquireFoodTree(MoveToTree):
    """
    A behvaior tree to select and execute an acquisition
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
            "j2n6s200_joint_2": np.pi / 2.0,
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
                    pre_behavior=get_add_in_front_of_wheelchair_wall_behavior(
                        name + "AddWheelchairWall",
                        "in_front_of_wheelchair_wall",
                    ),
                    # Remove the wall in front of the wheelchair
                    post_behavior=get_remove_in_front_of_wheelchair_wall_behavior(
                        name + "RemoveWheelchairWall",
                        "in_front_of_wheelchair_wall",
                    ),
                    workers=[
                        py_trees_ros.service_clients.FromConstant(
                            name="ClearOctomap",
                            service_name="/clear_octomap",
                            service_type=Empty,
                            service_request=Empty.Request(),
                            # Default fail if service is down
                            wait_for_server_timeout_sec=0.0,
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
                        MoveIt2Plan(
                            name="RestingPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "max_velocity_scale": self.max_velocity_scaling_to_resting_configuration,
                                "max_acceleration_scale": self.max_acceleration_scaling_to_resting_configuration,
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        MoveIt2Execute(
                            name="Resting",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
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
                                MoveIt2Plan(
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
                                    },
                                    outputs={"trajectory": BlackboardKey("trajectory")},
                                ),
                                MoveIt2Execute(
                                    name="RecoveryOffset",
                                    ns=name,
                                    inputs={"trajectory": BlackboardKey("trajectory")},
                                    outputs={},
                                ),
                            ],
                        ),  # End Attempt 2: MoveIt2 Cartesian Planner
                    ],  # End RecoverySelector.children
                ),  # End RecoverySelector
            ],  # End RecoverySequence.children
        )  # End RecoverySequence

        def move_above_plan(
            flip_food_frame: bool = False,
        ) -> py_trees.behaviour.Behaviour:
            return py_trees.composites.Sequence(
                name="MoveAbovePlanningSeq",
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
                            },
                            outputs={
                                "action_select_request": BlackboardKey(
                                    "action_request"
                                ),
                                "food_frame": None,
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
                                # Default move_above_dist_m = 0.05
                                # Default food_frame_id = "food"
                                # Default approach_frame_id = "approach"
                            },
                            outputs={
                                "move_above_pose": BlackboardKey("move_above_pose"),
                                "move_into_pose": BlackboardKey("move_into_pose"),
                                "approach_thresh": BlackboardKey("approach_thresh"),
                                "grasp_thresh": BlackboardKey("grasp_thresh"),
                                "ext_thresh": BlackboardKey("ext_thresh"),
                                "action": BlackboardKey("action"),
                            },
                        ),
                    ),
                    # Re-Tare FT Sensor and default to 4N threshold
                    pre_moveto_config(name="PreAcquireFTTare"),
                    ### Move Above Food
                    MoveIt2PoseConstraint(
                        name="MoveAbovePose",
                        ns=name,
                        inputs={
                            "pose": BlackboardKey("move_above_pose"),
                            "frame_id": "food",
                            "tolerance_orientation": [
                                0.01,
                                0.01,
                                0.01,
                            ],  # x, y, z rotvec
                            "parameterization": 1,
                        },
                        outputs={
                            "constraints": BlackboardKey("goal_constraints"),
                        },
                    ),
                    MoveIt2Plan(
                        name="MoveAbovePlan",
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "max_velocity_scale": self.max_velocity_scaling_move_above,
                            "max_acceleration_scale": self.max_acceleration_scaling_move_above,
                            "allowed_planning_time": 2.0,
                            "max_path_len_joint": max_path_len_joint,
                        },
                        outputs={
                            "trajectory": BlackboardKey("trajectory"),
                            "end_joint_state": BlackboardKey("test_into_joints"),
                        },
                    ),
                    ### Test MoveIntoFood
                    MoveIt2PoseConstraint(
                        name="MoveIntoPose",
                        ns=name,
                        inputs={
                            "pose": BlackboardKey("move_into_pose"),
                            "frame_id": "food",
                        },
                        outputs={
                            "constraints": BlackboardKey("goal_constraints"),
                        },
                    ),
                    MoveIt2Plan(
                        name="MoveIntoPlan",
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "max_velocity_scale": self.max_velocity_scaling_move_into,
                            "max_acceleration_scale": self.max_acceleration_scaling_move_into,
                            "cartesian": True,
                            "cartesian_max_step": 0.001,
                            "cartesian_fraction_threshold": 0.92,
                            "start_joint_state": BlackboardKey("test_into_joints"),
                            "max_path_len_joint": max_path_len_joint,
                        },
                        outputs={"trajectory": BlackboardKey("move_into_trajectory")},
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
                    name="TableCollision",
                    # Set Approach F/T Thresh
                    pre_behavior=ToggleCollisionObject(
                        name="AllowTable",
                        ns=name,
                        inputs={
                            "collision_object_ids": ["table"],
                            "allow": True,
                        },
                    ),
                    post_behavior=ToggleCollisionObject(
                        name="DisallowTable",
                        ns=name,
                        inputs={
                            "collision_object_ids": ["table"],
                            "allow": False,
                        },
                    ),
                    on_preempt_timeout=5.0,
                    # Starts a new Sequence w/ Memory internally
                    workers=[
                        scoped_behavior(
                            name="OctomapCollision",
                            # Set Approach F/T Thresh
                            pre_behavior=ToggleCollisionObject(
                                name="AllowOctomap",
                                ns=name,
                                inputs={
                                    "collision_object_ids": ["<octomap>"],
                                    "allow": True,
                                },
                            ),
                            post_behavior=ToggleCollisionObject(
                                name="DisallowOctomap",
                                ns=name,
                                inputs={
                                    "collision_object_ids": ["<octomap>"],
                                    "allow": False,
                                },
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
                                        move_above_plan(True),
                                        move_above_plan(False),
                                    ],
                                ),
                                MoveIt2Execute(
                                    name="MoveAbove",
                                    ns=name,
                                    inputs={"trajectory": BlackboardKey("trajectory")},
                                    outputs={},
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
                                        ### Move Into Food
                                        MoveIt2PoseConstraint(
                                            name="MoveIntoPose",
                                            ns=name,
                                            inputs={
                                                "pose": BlackboardKey("move_into_pose"),
                                                "frame_id": "food",
                                            },
                                            outputs={
                                                "constraints": BlackboardKey(
                                                    "goal_constraints"
                                                ),
                                            },
                                        ),
                                        # If this fails
                                        # Auto-fallback to precomputed MoveInto
                                        # From move_above_plan()
                                        py_trees.decorators.FailureIsSuccess(
                                            name="MoveIntoPlanFallbackPrecomputed",
                                            child=MoveIt2Plan(
                                                name="MoveIntoPlan",
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
                                                    "max_path_len_joint": max_path_len_joint,
                                                },
                                                outputs={
                                                    "trajectory": BlackboardKey(
                                                        "move_into_trajectory"
                                                    )
                                                },
                                            ),
                                        ),
                                        # MoveInto expect F/T failure
                                        py_trees.decorators.FailureIsSuccess(
                                            name="MoveIntoExecuteSucceed",
                                            child=MoveIt2Execute(
                                                name="MoveInto",
                                                ns=name,
                                                inputs={
                                                    "trajectory": BlackboardKey(
                                                        "move_into_trajectory"
                                                    )
                                                },
                                                outputs={},
                                            ),
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
                            ],  # End OctomapCollision.workers
                        ),  # OctomapCollision
                    ]
                    + resting_position_behaviors,  # End TableCollision.workers
                ),  # End TableCollision
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

        return super().get_feedback(tree, action_type)

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

        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(
            key="action_response", access=py_trees.common.Access.READ
        )
        blackboard.register_key(key="action", access=py_trees.common.Access.READ)

        # TODO: add posthoc
        response.selection_id = blackboard.action_response.id
        try:
            response.action_index = blackboard.action_response.actions.index(
                blackboard.action
            )
        except ValueError:
            response.status = response.STATUS_ACTION_NOT_TAKEN

        return response
