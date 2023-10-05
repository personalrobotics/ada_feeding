#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports

# Third-party imports
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rcl_interfaces.srv import SetParameters

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.behaviors.acquisition import (
    ComputeFoodFrame,
    ComputeApproachConstraints,
    ComputeExtractConstraints,
)
from ada_feeding.behaviors.moveit2 import (
    MoveIt2OrientationConstraint,
    MoveIt2PoseConstraint,
    MoveIt2PositionConstraint,
    MoveIt2Plan,
    MoveIt2Execute,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import (
    pre_moveto_config,
    scoped_behavior,
    retry_call_ros_service,
)
from ada_feeding.idioms.pre_moveto_config import set_parameter_response_all_success
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

    @override
    def create_tree(
        self,
        name: str,
        tree_root_name: str,  # DEPRECATED
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied by @override

        # TODO: remove tree_root_name
        # Sub-trees in general should not need knowledge of their parent.

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
                ComputeApproachConstraints(
                    name="ComputeApproachConstraints",
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
                        "ft_thresh": BlackboardKey("ft_thresh"),
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
                scoped_behavior(
                    name="SafeFTPreempt",
                    # pylint: disable=abstract-class-instantiated
                    # It has trouble with py_trees meta classes
                    pre_behavior=py_trees.behaviours.Success(),
                    post_behavior=pre_moveto_config(
                        name="PostAcquireFTSet", re_tare=False
                    ),
                    on_preempt_timeout=5.0,
                    # Starts a new Sequence w/ Memory internally
                    workers=[
                        # Set Approach F/T Thresh
                        retry_call_ros_service(
                            name="ApproachFTThresh",
                            service_type=SetParameters,
                            service_name="~/set_force_gate_controller_parameters",
                            # Blackboard, not Constant
                            request=None,
                            # Need absolute Blackboard name
                            key_request=Blackboard.separator.join(
                                [name, BlackboardKey("ft_thresh")]
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
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        MoveIt2Execute(
                            name="MoveInto",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
                            outputs={},
                        ),
                        ### Extraction
                        ComputeExtractConstraints(
                            name="ComputeExtractConstraints",
                            ns=name,
                            inputs={
                                "action": BlackboardKey("action"),
                                # Default approach_frame_id = "approach"
                            },
                            outputs={
                                "extract_position": BlackboardKey("extract_position"),
                                "extract_orientation": BlackboardKey(
                                    "extract_orientation"
                                ),
                                "ft_thresh": BlackboardKey("ft_thresh"),
                                "ee_frame_id": BlackboardKey("ee_frame_id"),
                            },
                        ),
                        MoveIt2PositionConstraint(
                            name="ExtractPosition",
                            ns=name,
                            inputs={
                                "position": BlackboardKey("extract_position"),
                                "frame_id": "approach",
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        MoveIt2OrientationConstraint(
                            name="ExtractOrientation",
                            ns=name,
                            inputs={
                                "quat_xyzw": BlackboardKey("extract_orientation"),
                                "frame_id": BlackboardKey("ee_frame_id"),
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        MoveIt2Plan(
                            name="ExtractPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "max_velocity_scale": 0.8,
                                "max_acceleration_scale": 0.8,
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        MoveIt2Execute(
                            name="Extraction",
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
