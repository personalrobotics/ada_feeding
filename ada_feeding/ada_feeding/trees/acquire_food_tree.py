#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from overrides import override

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.behaviors.acquisition import ComputeFoodFrame
from ada_feeding.helpers import BlackboardKey
from ada_feeding.visitors import MoveIt2Visitor
from ada_feeding_msgs.action import AcquireFood


class AcquireFoodTree(ActionServerBT):
    """
    A behvaior tree to select and execute an acquisition
    action (see ada_feeding_msgs.action.AcquisitionSchema)
    for a given food mask in ada_feeding_msgs.action.AcquireFood.
    """

    def __init__(self):
        """
        Initializes tree-specific parameters.
        """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    # pylint: disable=too-many-locals
    # Unfortunately, many local variables are required here to isolate the keys
    # of similar constraints in the blackboard.
    # pylint: disable=attribute-defined-outside-init
    # It is reasonable for attributes that are tree-specific, or only relevant
    # after the tree is created, to be defined in `create_tree`.
    @override
    def create_tree(
        self,
        name: str,
        action_type: type,
        tree_root_name: str,  # DEPRECATED
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied by @override

        # TODO: remove tree_root_name
        # Sub-trees in general should not need knowledge of their parent.

        ## Create Local Variables
        self.blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        self.action_type = action_type
        self.node = node

        ### Inputs we expect on the blackboard on initialization
        # Note: WRITE access since send_goal could write to these
        # Mask for ComputeFoodFrame
        self.blackboard.register_key(key="mask", access=py_trees.common.Access.WRITE)
        # CameraInfo for ComputeFoodFrame
        self.blackboard.register_key(
            key="camera_info", access=py_trees.common.Access.WRITE
        )

        ### Define Tree Leaves and Subtrees

        # Add ComputeFoodFrame
        compute_food_frame = ComputeFoodFrame(
            "ComputeFoodFrame",
            self.blackboard.namespace,
            inputs={
                "ros2_node": node,
                "camera_info": BlackboardKey("camera_info"),
                "mask": BlackboardKey("mask")
                # Default food_frame_id = "food"
                # Default world_frame = "world"
            },
            outputs={"action_select_request": None, "food_frame": None},
        )

        ### Define Tree Logic

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name="RootSequence",
            memory=True,
            children=[
                compute_food_frame,
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
        self.blackboard.mask = goal.detected_food
        self.blackboard.camera_info = goal.camera_info

        # Top level tree
        # Add MoveIt2Visitor for Feedback:
        self.feedback_visitor = MoveIt2Visitor(self.node)
        tree.visitors.append(self.feedback_visitor)

        return True

    # Override result to handle timing outside MoveTo Behaviors
    @override
    def get_feedback(self, tree: py_trees.trees.BehaviourTree) -> object:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree
        feedback_msg = self.action_type.Feedback()

        # Copy everything from the visitor
        # TODO: This Feedback/Result logic w/ MoveIt2Visitor can exist in MoveToTree right now
        feedback = self.feedback_visitor.get_feedback()
        feedback_msg.is_planning = feedback.is_planning
        feedback_msg.planning_time = feedback.planning_time
        feedback_msg.motion_time = feedback.motion_time
        feedback_msg.motion_initial_distance = feedback.motion_initial_distance
        feedback_msg.motion_curr_distance = feedback.motion_curr_distance

        return feedback_msg

    # Override result to add other elements to result msg
    @override
    def get_result(self, tree: py_trees.trees.BehaviourTree) -> object:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree
        result_msg = self.action_type.Result()

        # If the tree succeeded, return success
        if tree.root.status == py_trees.common.Status.SUCCESS:
            result_msg.status = result_msg.STATUS_SUCCESS
        # If the tree failed, detemine whether it was a planning or motion failure
        elif tree.root.status == py_trees.common.Status.FAILURE:
            feedback = self.feedback_visitor.get_feedback()
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
