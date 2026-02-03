# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

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
from py_trees.behaviours import Success
import py_trees_ros
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from std_srvs.srv import Empty

# Local imports
from ada_feeding_msgs.action import AcquireFood

# Note: Most behavior imports have been moved to ada_feeding/idioms/acquisition.py
from ada_feeding.behaviors.acquisition import ComputeActionTwist
from ada_feeding.behaviors.moveit2 import (
    MoveIt2OrientationConstraint,
    MoveIt2PositionOffsetConstraint,
    MoveIt2Plan,
    MoveIt2Execute,
    ServoMove,
    ToggleCollisionObject,
)
from ada_feeding.behaviors.articutool import (
    CallSetOrientationControl,
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

# --- Modular Acquisition Idioms ---
from ada_feeding.idioms.acquisition import (
    get_pre_acquisition_setup,
    get_robust_move_above_sequence,
    get_articutool_move_into_sequence,
    get_post_acquisition_leveling_sequence,
    get_post_retract_primitive_sequence,
    get_resting_sequence,
)


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
        allowed_planning_time_for_move_into: float = 1.0,
        allowed_planning_time_to_resting_configuration: float = 6.0,
        allowed_planning_time_for_recovery: float = 1.0,
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

        # Get the base link to publish servo commands in
        base_link = "j2n6s200_link_base"

        ### Add Resting Position Logic
        resting_position_behaviors = []
        if self.resting_joint_positions is not None:
            # Move back to resting position using the new idiom
            resting_position_behaviors.append(
                scoped_behavior(
                    name=name + " InFrontOfWheelchairWallScope",
                    pre_behavior=get_add_in_front_of_face_wall_behavior(
                        name + "AddWheelchairWall",
                    ),
                    # Remove the wall in front of the wheelchair
                    post_behavior=get_remove_in_front_of_face_wall_behavior(
                        name + "RemoveWheelchairWall",
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
                        # Call the modular resting sequence idiom
                        get_resting_sequence(
                            name=name,
                            ns=name,
                            node=self._node,
                            resting_joint_positions=self.resting_joint_positions,
                            max_velocity_scale=self.max_velocity_scaling_to_resting_configuration,
                            max_acceleration_scale=self.max_acceleration_scaling_to_resting_configuration,
                            allowed_planning_time=self.allowed_planning_time_to_resting_configuration,
                        ),
                    ],
                ),
            )

        ### Define Recovery Tree (Inline)
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

        ### Main Tree Orchestration
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
                                # --- 1. Pre-Acquisition Setup ---
                                py_trees.composites.Selector(
                                    name="BackupFlipFoodFrameSel",
                                    memory=True,
                                    children=[
                                        get_pre_acquisition_setup(
                                            name, name, flip_food_frame=True
                                        ),
                                        get_pre_acquisition_setup(
                                            name,
                                            name,
                                            flip_food_frame=False,
                                            action_key=BlackboardKey("action"),
                                        ),
                                    ],
                                ),
                                # --- 2. Move Above (Plan-Then-Verify) ---
                                get_robust_move_above_sequence(
                                    name=name,
                                    ns=name,
                                    max_velocity_scaling_move_above=self.max_velocity_scaling_move_above,
                                    max_acceleration_scaling_move_above=self.max_acceleration_scaling_move_above,
                                    max_velocity_scaling_move_into=self.max_velocity_scaling_move_into,
                                    max_acceleration_scaling_move_into=self.max_acceleration_scaling_move_into,
                                    allowed_planning_time_for_move_above=self.allowed_planning_time_for_move_above,
                                    allowed_planning_time_for_move_into=self.allowed_planning_time_for_move_into,
                                ),
                                # --- 3. Execution & Safe Interaction ---
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
                                        # 3a. Move Into Food (Primitives + Cartesian)
                                        get_articutool_move_into_sequence(name, name),
                                        # 3b. MoveIt2 Servo (Interaction)
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
                                            on_preempt_timeout=60.0,
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
                                                                        "is_retract": False,
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
                                                                ),
                                                                # Run Post-Grasp Primitives (e.g. Post-Move-Into actions)
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
                                                                ### Extraction
                                                                ComputeActionTwist(
                                                                    name="ComputeExtract",
                                                                    ns=name,
                                                                    inputs={
                                                                        "action": BlackboardKey(
                                                                            "action"
                                                                        ),
                                                                        "is_grasp": False,
                                                                        "is_retract": False,
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
                                                                # 3c. Post-Acquisition Leveling
                                                                get_post_acquisition_leveling_sequence(
                                                                    name=name,
                                                                    ns=name,
                                                                    node=self._node,
                                                                    allowed_planning_time=self.allowed_planning_time_to_resting_configuration,
                                                                ),
                                                                ### Retract
                                                                retry_call_ros_service(
                                                                    name="RetractFTThresh",
                                                                    service_type=SetParameters,
                                                                    service_name="~/set_cartesian_controller_parameters",
                                                                    # Blackboard, not Constant
                                                                    request=None,
                                                                    # Need absolute Blackboard name
                                                                    key_request=Blackboard.separator.join(
                                                                        [
                                                                            name,
                                                                            BlackboardKey(
                                                                                "retract_thresh"
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
                                                                    name="ComputeRetract",
                                                                    ns=name,
                                                                    inputs={
                                                                        "action": BlackboardKey(
                                                                            "action"
                                                                        ),
                                                                        "is_grasp": False,
                                                                        "is_retract": True,
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
                                                                    name="RetractServo",
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
                                                                ),
                                                                # 3d. Post-Retract (Cleaning/Settling Primitives)
                                                                get_post_retract_primitive_sequence(
                                                                    name, name
                                                                ),
                                                            ],
                                                        ),
                                                        # 4. Recovery Strategy (Inline)
                                                        recovery_tree,
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]
                    + resting_position_behaviors,
                ),
            ],
        )

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
