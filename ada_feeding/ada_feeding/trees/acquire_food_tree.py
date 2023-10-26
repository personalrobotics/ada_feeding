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
from rclpy.qos import QoSProfile
from std_msgs.msg import Header

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.behaviors import ToggleCollisionObject
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
)
from ada_feeding.decorators import TimeoutFromBlackboard
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import (
    pre_moveto_config,
    scoped_behavior,
    retry_call_ros_service,
)
from ada_feeding.idioms.pre_moveto_config import set_parameter_response_all_success
from ada_feeding.trees import StartServoTree, StopServoTree
from ada_feeding.visitors import MoveToVisitor
from ada_feeding_msgs.action import AcquireFood
from ada_feeding_msgs.srv import AcquisitionSelect


class AcquireFoodTree(ActionServerBT):
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
        tree_root_name: str,  # DEPRECATED
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied by @override

        # TODO: remove tree_root_name
        # Sub-trees in general should not need knowledge of their parent.

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

        ### Define Tree Leaf Nodes
        start_servo_tree = StartServoTree(self._node)
        stop_servo_tree = StopServoTree(self._node)

        # FT Threshold Setting Nodes
        approach_ft_behavior = retry_call_ros_service(
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
        )
        grasp_ft_behavior = retry_call_ros_service(
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
                        [name, BlackboardKey("ft_response")]
                    ),
                    value=SetParameters.Response(),  # Unused
                    operator=set_parameter_response_all_success,
                )
            ],
        )
        ext_ft_behavior = retry_call_ros_service(
            name="ExtractionFTThresh",
            service_type=SetParameters,
            service_name="~/set_servo_controller_parameters",
            # Blackboard, not Constant
            request=None,
            # Need absolute Blackboard name
            key_request=Blackboard.separator.join([name, BlackboardKey("ext_thresh")]),
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
        )

        ### Define Tree Logic

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
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
                            "mask": BlackboardKey("mask")
                            # Default food_frame_id = "food"
                            # Default world_frame = "world"
                        },
                        outputs={
                            "action_select_request": BlackboardKey("action_request"),
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
                ComputeActionConstraints(
                    name="ComputeActionConstraints",
                    ns=name,
                    inputs={
                        "action_select_response": BlackboardKey("action_response"),
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
                # Re-Tare FT Sensor and default to 4N threshold
                pre_moveto_config(name="PreAquireFTTare"),
                ### Move Above Food
                MoveIt2PoseConstraint(
                    name="MoveAbovePose",
                    ns=name,
                    inputs={
                        "pose": BlackboardKey("move_above_pose"),
                        "frame_id": "food",
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
                # Also disable and re-enable table collision
                scoped_behavior(
                    name="SafeFTPreempt",
                    # Set Approach F/T Thresh
                    pre_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            approach_ft_behavior,
                            ToggleCollisionObject(
                                name="AllowTable",
                                node=self._node,
                                collision_object_ids=["table"],
                                allow=True,
                            ),
                        ],
                    ),
                    post_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            pre_moveto_config(name="PostAcquireFTSet", re_tare=False),
                            ToggleCollisionObject(
                                name="DisallowTable",
                                node=self._node,
                                collision_object_ids=["table"],
                                allow=False,
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
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        MoveIt2Plan(
                            name="MoveIntoPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "max_velocity_scale": 0.8,
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
                                inputs={"trajectory": BlackboardKey("trajectory")},
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
                                    start_servo_tree.create_tree(
                                        name="StartServoScoped", tree_root_name=name
                                    ).root,
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
                                    stop_servo_tree.create_tree(
                                        name="StopServoScoped", tree_root_name=name
                                    ).root,
                                ],
                            ),
                            on_preempt_timeout=5.0,
                            # Starts a new Sequence w/ Memory internally
                            workers=[
                                ### Grasp
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
                                grasp_ft_behavior,
                                py_trees.decorators.FailureIsSuccess(
                                    name="GraspSuceed",
                                    child=TimeoutFromBlackboard(
                                        name="MoveIt2ServoTimeout",
                                        ns=name,
                                        duration_key=BlackboardKey("duration"),
                                        child=py_trees.decorators.SuccessIsRunning(
                                            name="GraspRetryInfinite",
                                            child=py_trees_ros.publishers.FromBlackboard(
                                                name="GraspTwist",
                                                topic_name="~/servo_twist_cmds",
                                                topic_type=TwistStamped,
                                                qos_profile=QoSProfile(depth=1),
                                                blackboard_variable=Blackboard.separator.join(
                                                    [name, BlackboardKey("twist")]
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                py_trees_ros.publishers.FromBlackboard(
                                    name="StopTwist",
                                    topic_name="~/servo_twist_cmds",
                                    topic_type=TwistStamped,
                                    qos_profile=QoSProfile(depth=1),
                                    blackboard_variable=Blackboard.separator.join(
                                        [name, BlackboardKey("zero_twist")]
                                    ),
                                ),
                                ### Extraction
                                ComputeActionTwist(
                                    name="ComputeGrasp",
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
                                ext_ft_behavior,
                                py_trees.decorators.FailureIsSuccess(
                                    name="ExtractSucceed",
                                    child=TimeoutFromBlackboard(
                                        name="MoveIt2ServoTimeout",
                                        ns=name,
                                        duration_key=BlackboardKey("duration"),
                                        child=py_trees.decorators.SuccessIsRunning(
                                            name="ExtractRetryInfinite",
                                            child=py_trees_ros.publishers.FromBlackboard(
                                                name="ExtractTwist",
                                                topic_name="~/servo_twist_cmds",
                                                topic_type=TwistStamped,
                                                qos_profile=QoSProfile(depth=1),
                                                blackboard_variable=Blackboard.separator.join(
                                                    [name, BlackboardKey("twist")]
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                py_trees_ros.publishers.FromBlackboard(
                                    name="StopTwist",
                                    topic_name="~/servo_twist_cmds",
                                    topic_type=TwistStamped,
                                    qos_profile=QoSProfile(depth=1),
                                    blackboard_variable=Blackboard.separator.join(
                                        [name, BlackboardKey("zero_twist")]
                                    ),
                                ),
                            ],  # End MoveIt2Servo.workers
                        ),  # End MoveIt2Servo
                    ],  # End SafeFTPreempt.workers
                ),  # End SafeFTPreempt
            ],  # End root_seq.children
        )  # End root_seq

        ### Add Resting Position
        if self.resting_joint_positions is not None:
            root_seq.add_children(
                [
                    # Move back to resting position
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
                ]
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

        # Write tree inputs to blackboard
        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(key="mask", access=py_trees.common.Access.WRITE)
        blackboard.mask = goal.detected_food
        blackboard.register_key(key="camera_info", access=py_trees.common.Access.WRITE)
        blackboard.camera_info = goal.camera_info

        # Add MoveToVisitor for Feedback
        feedback_visitor = None
        for visitor in tree.visitors:
            if isinstance(visitor, MoveToVisitor):
                visitor.reinit()
                feedback_visitor = visitor
        if feedback_visitor is None:
            tree.add_visitor(MoveToVisitor(self._node))

        return True

    # Override result to handle timing outside MoveTo Behaviors
    @override
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree
        # TODO: This Feedback/Result logic w/ MoveToVisitor can exist in MoveToTree right now
        if action_type is not AcquireFood:
            return None

        feedback_msg = action_type.Feedback()

        # Get Feedback Visitor
        feedback_visitor = None
        for visitor in tree.visitors:
            if isinstance(visitor, MoveToVisitor):
                feedback_visitor = visitor

        # Copy everything from the visitor
        if feedback_visitor is not None:
            feedback = feedback_visitor.get_feedback()
            feedback_msg.is_planning = feedback.is_planning
            feedback_msg.planning_time = feedback.planning_time
            feedback_msg.motion_time = feedback.motion_time
            feedback_msg.motion_initial_distance = feedback.motion_initial_distance
            feedback_msg.motion_curr_distance = feedback.motion_curr_distance

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

        result_msg = action_type.Result()

        # If the tree succeeded, return success
        if tree.root.status == py_trees.common.Status.SUCCESS:
            result_msg.status = result_msg.STATUS_SUCCESS
        # If the tree failed, detemine whether it was a planning or motion failure
        elif tree.root.status == py_trees.common.Status.FAILURE:
            # Get Feedback Visitor to investigate failure cause
            feedback_visitor = None
            for visitor in tree.visitors:
                if isinstance(visitor, MoveToVisitor):
                    feedback_visitor = visitor
            if feedback_visitor is None:
                result_msg.status = result_msg.STATUS_UNKNOWN
            else:
                feedback = feedback_visitor.get_feedback()
                if feedback.is_planning:
                    result_msg.status = result_msg.STATUS_PLANNING_FAILED
                else:
                    result_msg.status = result_msg.STATUS_MOTION_FAILED
        # If the tree has an invalid status, return unknown
        elif tree.root.status == py_trees.common.Status.INVALID:
            result_msg.status = result_msg.STATUS_UNKNOWN
        # If the tree is running, the fact that `get_result` was called is
        # indicative of an error. Return unknown error.
        else:
            tree.root.logger.error(
                f"Called get_result with status RUNNING: {tree.root.status}"
            )
            result_msg.status = result_msg.STATUS_UNKNOWN

        # TODO: add action_index, posthoc, action_select_hash
        return result_msg
