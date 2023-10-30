#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the abstract MoveTo behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.

This module is intended to consolidate all the logic for generating feedback
and results, which is shared across all MoveTo actions, into one place.
"""

# Standard imports
from abc import ABC

# Third-party imports
from overrides import override
import py_trees

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.visitors import MoveToVisitor


class MoveToTree(ActionServerBT, ABC):
    """
    An abstract behavior tree for any behavior that moves the robot and provides
    results and feedback as specified in ada_feeding_msgs.action.MoveTo or
    ada_feeding_msgs.action.AcquireFood.
    """

    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        """
        Sends the goal of the action to the behavior tree.

        The default `send_goal` in MoveToTree actually ignores the goal and
        merely initializes the MoveToVisitor for feedback. This is because
        the default `MoveTo.action` has an empty goal. Any subclass that wants
        to use the goal should override this function, but should also call
        `super().send_goal(tree, goal)` to initialize the MoveToVisitor.

        Parameters
        ----------
        tree: The behavior tree that is being executed.
        goal: The ROS goal message sent to the action server.

        Returns
        -------
        success: Whether the goal was successfully sent to the behavior tree.
        """

        # Add MoveToVisitor for Feedback
        feedback_visitor = None
        for visitor in tree.visitors:
            if isinstance(visitor, MoveToVisitor):
                # Re-initialize existing MoveToVisitor
                visitor.reinit()
                feedback_visitor = visitor
        if feedback_visitor is None:
            tree.add_visitor(MoveToVisitor(self._node))
        return True

    @override
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
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

    @override
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
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

        return result_msg
