#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveAbovePlate behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging

# Third-party imports
import py_trees

# Local imports
from ada_feeding.behaviors import MoveToDummy
from ada_feeding import ActionServerBT
from ada_feeding_msgs.action import MoveTo


class MoveAbovePlate(ActionServerBT):
    """
    A dummy behavior tree that mimics the interface of the MoveAbovePlate
    behavior tree.
    """

    def __init__(
        self, dummy_plan_time: float = 2.5, dummy_motion_time: float = 7.5
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        dummy_plan_time: How many seconds this dummy node should spend in planning.
        dummy_motion_time: How many seconds this dummy node should spend in motion.
        """
        self.dummy_plan_time = dummy_plan_time
        self.dummy_motion_time = dummy_motion_time

        # Cache the tree so that it can be reused
        self.tree = None

    def create_tree(
        self, name: str, logger: logging.Logger
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveAbovePlate behavior tree.

        Currently, this only has one behavior in it, MoveToDummy. Eventually,
        this should be replaced with a behavior tree that actually moves the
        robot arm above the plate.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        # Create the behaviors in the tree
        if self.tree is None:
            root = MoveToDummy(name, self.dummy_plan_time, self.dummy_motion_time)
            root.logger = logger
            # Create the tree
            self.tree = py_trees.trees.BehaviourTree(root)

        return self.tree

    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: MoveTo.Goal) -> bool:
        """
        Sends the goal from the action client to the behavior tree.

        This function is called before the behavior tree is executed.

        Parameters
        ----------
        tree: The behavior tree that is being executed.
        goal: The ROS goal message to be sent to the behavior tree.

        Returns
        -------
        success: Whether the goal was sent successfully.
        """
        # For MoveAbovePlate, there is no goal to send
        return True

    def preempt_goal(self, tree: py_trees.trees.BehaviourTree) -> bool:
        """
        Preempts the currently running goal on the behavior tree.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        success: Whether the preempt request was sent successfully.
        """
        tree.root.blackboard.preempt_requested = True
        return True

    def was_preempted(self, tree: py_trees.trees.BehaviourTree) -> bool:
        """
        Checks whether the tree has completely processed a preempt request.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        success: Whether the preempt request was sent successfully.
        """
        return (
            tree.root.status == py_trees.common.Status.INVALID
            and tree.root.blackboard.exists("was_preempted")
            and tree.root.blackboard.was_preempted
        )

    def get_feedback(self, tree: py_trees.trees.BehaviourTree) -> MoveTo.Feedback:
        """
        Traverses the tree to generate a feedback message for the MoveTo action.

        This function is used as a post-tick handler for the behavior tree.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        feedback: The ROS feedback message to be sent to the action client.
        """
        feedback_msg = MoveTo.Feedback()
        if tree.root.blackboard.exists("is_planning"):
            feedback_msg.is_planning = tree.root.blackboard.is_planning
            planning_time = tree.root.blackboard.planning_time
            feedback_msg.planning_time.sec = int(planning_time)
            feedback_msg.planning_time.nanosec = int(
                (planning_time - int(planning_time)) * 1e9
            )
            if not feedback_msg.is_planning:
                feedback_msg.motion_initial_distance = (
                    tree.root.blackboard.motion_initial_distance
                )
                feedback_msg.motion_curr_distance = (
                    tree.root.blackboard.motion_curr_distance
                )
        return feedback_msg

    def get_result(self, tree: py_trees.trees.BehaviourTree) -> MoveTo.Result:
        """
        Traverses the tree to generate a result message for the MoveTo action.

        This function is called after the behavior tree terminates.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        result: The ROS result message to be sent to the action client.
        """
        result = MoveTo.Result()
        # If the tree succeeded, return success
        if tree.root.status == py_trees.common.Status.SUCCESS:
            result.status = result.STATUS_SUCCESS
        # If the tree failed, detemine whether it was a planning or motion failure
        elif tree.root.status == py_trees.common.Status.FAILURE:
            if tree.root.blackboard.exists("is_planning"):
                if tree.root.blackboard.is_planning:
                    result.status = result.STATUS_PLANNING_FAILED
                else:
                    result.status = result.STATUS_MOTION_FAILED
            else:
                result.status = result.STATUS_UNKNOWN
        # If the tree has an invalid status, return unknown
        elif tree.root.status == py_trees.common.Status.INVALID:
            if (
                tree.root.blackboard.exists("was_preempted")
                and tree.root.blackboard.was_preempted
            ):
                result.status = result.STATUS_CANCELED
            else:
                result.status = result.STATUS_UNKNOWN
        # If the tree is running, the fact that `get_result` was called is
        # indicative of an error. Return unknown error.
        else:
            tree.root.logger.warn(
                "Called get_result with status RUNNING: %s" % tree.root.status
            )
            result.status = result.STATUS_UNKNOWN

        return result
