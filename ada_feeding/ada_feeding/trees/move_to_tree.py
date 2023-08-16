#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the abstract MoveTo behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.

This module is intended to consolidate all the logic for generating feedback
and results, which is shared across all MoveTo actions, into one place.
Subclasses should only need to define create_move_to_tree.
"""

# Standard imports
from abc import ABC, abstractmethod
import logging

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.helpers import import_from_string
from ada_feeding import ActionServerBT


class MoveToTree(ActionServerBT, ABC):
    """
    An abstract behvaior tree for any behavior that moves the robot and provides
    results and feedback as specified in ada_feeding_msgs.action.MoveTo or
    ada_feeding_msgs.action.AcquireFood.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes tree-specific parameters.
        """
        pass

    def create_tree(
        self,
        name: str,
        action_type: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Create the behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        action_type: full name of a Python class for the associated action,
            can be converted to a type object with `import_from_string` in
            helpers.py
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        self.create_blackboard(name)

        # Import the action type
        self.action_type_class = import_from_string(action_type_class_str)
        self.action_type_class_str = action_type_class_str

        return self.create_move_to_tree(name, action_type_class_str, logger, node)

    def create_blackboard(self, name: str) -> None:
        """
        Creates the blackboard for the behavior tree, and defines the blackboard
        keys necessary to set the goal and send feedback.

        Parameters
        ----------
        name: The name of the behavior tree.
        """
        # Create the blackboard
        self.blackboard = py_trees.blackboard.Client(
            name=name + " Tree", namespace=name
        )
        # Goal that is passed from the ROS2 Action Server
        self.blackboard.register_key(key="goal", access=py_trees.common.Access.WRITE)
        # Feedback from MoveToConfiguration for the ROS2 Action Server
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

    @abstractmethod
    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Create the behavior tree. By the time this function is called, the
        blackboard has already been created and been defined with the keys
        necessary to set the goal and send feedback. Therefore, this function
        is just responsible for defining the blackboard keys specific to the
        movement behavior, and creating the tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        raise NotImplementedError("create_move_to_tree not implemented")

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
        feedback_msg = self.action_type_class.Feedback()
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
        result = self.action_type_class.Result()
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
                "Called get_result with status RUNNING: %s" % tree.root.status
            )
            result.status = result.STATUS_UNKNOWN

        return result
