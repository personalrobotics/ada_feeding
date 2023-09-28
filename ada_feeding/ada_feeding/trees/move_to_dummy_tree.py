#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToDummy behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports

# Third-party imports
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveToDummy
from ada_feeding import ActionServerBT

# pylint: disable=duplicate-code
# Since this is a dummy node, it will by design have lots of overlap with the
# real node.


class MoveToDummyTree(ActionServerBT):
    """
    A dummy behavior tree that mimics the interface of the MoveAbovePlate
    behavior tree.
    """

    def __init__(
        self,
        node: Node,
        dummy_plan_time: float = 2.5,
        dummy_motion_time: float = 7.5,
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        dummy_plan_time: How many seconds this dummy node should spend in planning.
        dummy_motion_time: How many seconds this dummy node should spend in motion.
        """
        super().__init__(node)

        # Set the dummy motion parameters
        self.dummy_plan_time = dummy_plan_time
        self.dummy_motion_time = dummy_motion_time

        # Cache the tree so that it can be reused
        self.tree = None
        self.blackboard = None

    @override
    def create_tree(
        self,
        name: str,
        tree_root_name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the behaviors in the tree
        if self.tree is None:
            root = MoveToDummy(name, self.dummy_plan_time, self.dummy_motion_time)
            # Create the tree
            self.tree = py_trees.trees.BehaviourTree(root)
            # Create the blackboard
            self.blackboard = py_trees.blackboard.Client(
                name=name + " Tree", namespace=tree_root_name
            )
            self.blackboard.register_key(
                key="goal", access=py_trees.common.Access.WRITE
            )
            self.blackboard.register_key(
                key="is_planning", access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key="planning_time", access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key="motion_time", access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key="motion_initial_distance", access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key="motion_curr_distance", access=py_trees.common.Access.READ
            )

        return self.tree

    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied from @override

        # Write the goal to blackboard
        self.blackboard.goal = goal
        return True

    @override
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied from @override

        feedback_msg = action_type.Feedback()
        if self.blackboard.exists("is_planning"):
            feedback_msg.is_planning = self.blackboard.is_planning
            planning_time = self.blackboard.planning_time
            feedback_msg.planning_time.sec = int(planning_time)
            feedback_msg.planning_time.nanosec = int(
                (planning_time - int(planning_time)) * 1e9
            )
            motion_time = self.blackboard.motion_time
            feedback_msg.motion_time.sec = int(motion_time)
            feedback_msg.motion_time.nanosec = int(
                (motion_time - int(motion_time)) * 1e9
            )
            if not feedback_msg.is_planning:
                feedback_msg.motion_initial_distance = (
                    self.blackboard.motion_initial_distance
                )
                feedback_msg.motion_curr_distance = self.blackboard.motion_curr_distance
        return feedback_msg

    @override
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied from @override

        result = action_type.Result()
        # If the tree succeeded, return success
        if tree.root.status == py_trees.common.Status.SUCCESS:
            result.status = result.STATUS_SUCCESS
        # If the tree failed, detemine whether it was a planning or motion failure
        elif tree.root.status == py_trees.common.Status.FAILURE:
            if self.blackboard.exists("is_planning"):
                if self.blackboard.is_planning:
                    result.status = result.STATUS_PLANNING_FAILED
                else:
                    result.status = result.STATUS_MOTION_FAILED
            else:
                result.status = result.STATUS_UNKNOWN
        # If the tree has an invalid status, return unknown
        elif tree.root.status == py_trees.common.Status.INVALID:
            result.status = result.STATUS_UNKNOWN
        # If the tree is running, the fact that `get_result` was called is
        # indicative of an error. Return unknown error.
        else:
            tree.root.logger.warn(
                f"Called get_result with status RUNNING: {tree.root.status}"
            )
            result.status = result.STATUS_UNKNOWN

        return result
