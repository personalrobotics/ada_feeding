#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToDummy behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports

# Third-party imports
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

        # Set the dummy motion parameters
        self.dummy_plan_time = dummy_plan_time
        self.dummy_motion_time = dummy_motion_time

        # Cache the tree so that it can be reused
        self.tree = None
        self.blackboard = None

    # pylint: disable=attribute-defined-outside-init
    # It is reasonable for attributes that are tree-specific, or only relevant
    # after the tree is created, to be defined in `create_tree`.
    def create_tree(
        self,
        name: str,
        action_type: type,
        tree_root_name: str,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveAbovePlate behavior tree.

        Currently, this only has one behavior in it, MoveToDummy. Eventually,
        this should be replaced with a behavior tree that actually moves the
        robot arm above the plate.

        Parameters
        ----------
        name: The name of the behavior tree.
        action_type: the type for the action, as a class.
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """

        # pylint: disable=too-many-arguments
        # One over is fine.

        # Store the action type
        self.action_type = action_type

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

    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
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
        # Write the goal to blackboard
        self.blackboard.goal = goal
        return True

    def get_feedback(self, tree: py_trees.trees.BehaviourTree) -> object:
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
        feedback_msg = self.action_type.Feedback()
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

    def get_result(self, tree: py_trees.trees.BehaviourTree) -> object:
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
        result = self.action_type.Result()
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
            tree.root.logger.warning(
                f"Called get_result with status RUNNING: {tree.root.status}"
            )
            result.status = result.STATUS_UNKNOWN

        return result
