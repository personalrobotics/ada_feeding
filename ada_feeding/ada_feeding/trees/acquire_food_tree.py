#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import List, Optional

# Third-party imports
from geometry_msgs.msg import Twist, TwistStamped
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
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
    MoveIt2PoseConstraint,
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

    Tree Blackboard Outputs:

    """

    def __init__(
        self,
        node: Node,
        resting_joint_positions: Optional[List[float]] = None,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        resting_joint_positions: Final joint position after acquisition
        """
        # Initialize ActionServerBT
        super().__init__(node)

        self.resting_joint_positions = resting_joint_positions

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
                                "max_velocity_scale": 0.8,
                                "max_acceleration_scale": 0.8,
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

        ### Define Tree Logic
        # NOTE: If octomap clearing ends up being an issue, we should
        # consider adding a call to the /clear_octomap service to this tree.
        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                scoped_behavior(
                    name="TableCollision",
                    # Set Approach F/T Thresh
                    pre_behavior=ToggleCollisionObject(
                        name="AllowTableAndOctomap",
                        ns=name,
                        inputs={
                            "collision_object_ids": ["table", "<octomap>"],
                            "allow": True,
                        },
                    ),
                    post_behavior=ToggleCollisionObject(
                        name="DisallowTableAndOctomap",
                        ns=name,
                        inputs={
                            "collision_object_ids": ["table", "<octomap>"],
                            "allow": False,
                        },
                    ),
                    on_preempt_timeout=5.0,
                    # Starts a new Sequence w/ Memory internally
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
                        # Compute Food Frame
                        py_trees.decorators.Timeout(
                            name="ComputeFoodFrameTimeout",
                            duration=1.0,
                            child=ComputeFoodFrame(
                                name="ComputeFoodFrame",
                                ns=name,
                                inputs={
                                    "camera_info": BlackboardKey("camera_info"),
                                    "mask": BlackboardKey("mask")
                                    # Default food_frame_id = "food"
                                    # Default world_frame = "world"
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
                                    0.001,
                                    0.001,
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
                                "max_velocity_scale": 0.8,
                                "max_acceleration_scale": 0.8,
                                "allowed_planning_time": 1.5,
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
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
                                            [name, BlackboardKey("approach_thresh")]
                                        ),
                                        key_response=Blackboard.separator.join(
                                            [name, BlackboardKey("ft_response")]
                                        ),
                                        response_checks=[
                                            py_trees.common.ComparisonExpression(
                                                variable=Blackboard.separator.join(
                                                    [name, BlackboardKey("ft_response")]
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
                                MoveIt2Plan(
                                    name="MoveIntoPlan",
                                    ns=name,
                                    inputs={
                                        "goal_constraints": BlackboardKey(
                                            "goal_constraints"
                                        ),
                                        "max_velocity_scale": 1.0,
                                        "max_acceleration_scale": 0.8,
                                        "cartesian": True,
                                        "cartesian_max_step": 0.001,
                                        "cartesian_fraction_threshold": 0.95,
                                    },
                                    outputs={"trajectory": BlackboardKey("trajectory")},
                                ),
                                # MoveInto expect F/T failure
                                py_trees.decorators.FailureIsSuccess(
                                    name="MoveIntoExecuteSucceed",
                                    child=MoveIt2Execute(
                                        name="MoveInto",
                                        ns=name,
                                        inputs={
                                            "trajectory": BlackboardKey("trajectory")
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
                                            StartServoTree(self._node)
                                            .create_tree(name="StartServoScoped")
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
                                                param_service_name="~/set_servo_controller_parameters",
                                            ),
                                            StopServoTree(self._node)
                                            .create_tree(name="StopServoScoped")
                                            .root,
                                        ],
                                    ),
                                    on_preempt_timeout=5.0,
                                    # Starts a new Sequence w/ Memory internally
                                    workers=[
                                        ### Grasp
                                        retry_call_ros_service(
                                            name="GraspFTThresh",
                                            service_type=SetParameters,
                                            service_name="~/set_servo_controller_parameters",
                                            # Blackboard, not Constant
                                            request=None,
                                            # Need absolute Blackboard name
                                            key_request=Blackboard.separator.join(
                                                [name, BlackboardKey("grasp_thresh")]
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
                                        ComputeActionTwist(
                                            name="ComputeGrasp",
                                            ns=name,
                                            inputs={
                                                "action": BlackboardKey("action"),
                                                "is_grasp": True,
                                            },
                                            outputs={
                                                "twist": BlackboardKey("twist"),
                                                "duration": BlackboardKey("duration"),
                                            },
                                        ),
                                        ServoMove(
                                            name="GraspServo",
                                            ns=name,
                                            inputs={
                                                "twist": BlackboardKey("twist"),
                                                "duration": BlackboardKey("duration"),
                                            },
                                        ),  # Auto Zero-Twist on terminate()
                                        ### Extraction
                                        ComputeActionTwist(
                                            name="ComputeExtract",
                                            ns=name,
                                            inputs={
                                                "action": BlackboardKey("action"),
                                                "is_grasp": False,
                                            },
                                            outputs={
                                                "twist": BlackboardKey("twist"),
                                                "duration": BlackboardKey("duration"),
                                            },
                                        ),
                                        retry_call_ros_service(
                                            name="ExtractionFTThresh",
                                            service_type=SetParameters,
                                            service_name="~/set_servo_controller_parameters",
                                            # Blackboard, not Constant
                                            request=None,
                                            # Need absolute Blackboard name
                                            key_request=Blackboard.separator.join(
                                                [name, BlackboardKey("ext_thresh")]
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
                                        ServoMove(
                                            name="ExtractServo",
                                            ns=name,
                                            inputs={
                                                "twist": BlackboardKey("twist"),
                                                "duration": BlackboardKey("duration"),
                                            },
                                        ),  # Auto Zero-Twist on terminate()
                                    ],  # End MoveIt2Servo.workers
                                ),  # End MoveIt2Servo
                            ],  # End SafeFTPreempt.workers
                        ),  # End SafeFTPreempt
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

        # Write tree inputs to blackboard
        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(key="mask", access=py_trees.common.Access.WRITE)
        blackboard.mask = goal.detected_food
        blackboard.register_key(key="camera_info", access=py_trees.common.Access.WRITE)
        blackboard.camera_info = goal.camera_info

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

        # TODO: add action_index, posthoc, action_select_hash
        return super().get_result(tree, action_type)
