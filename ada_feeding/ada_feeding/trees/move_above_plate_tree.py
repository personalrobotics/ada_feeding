#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveAbovePlate behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveToConfiguration
from ada_feeding.helpers import import_from_string
from ada_feeding import ActionServerBT


class MoveAbovePlateTree(ActionServerBT):
    """
    Move the robot arm to a fixed above-plate position.
    """

    def __init__(
        self,
        action_type_class: str,
        joint_positions: List[float],
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        joint_positions: The joint positions to move the robot arm to.
        """
        # Import the action type
        self.action_type_class = import_from_string(action_type_class)

        # Store the joint positions
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"

    def create_tree(
        self,
        name: str,
        logger: logging.Logger,
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
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        # Create the blackboard
        self.blackboard = py_trees.blackboard.Client(name=name + " Tree", namespace=name)
        # Inputs for MoveToConfiguration
        self.blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="joint_names", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key="cartesian", access=py_trees.common.Access.WRITE
        )
        # Goal that is passed from the ROS2 Action Server
        self.blackboard.register_key(
            key="goal", access=py_trees.common.Access.WRITE
        )
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

        # Write the inputs to MoveToConfiguration to blackboard
        self.blackboard.joint_positions = self.joint_positions

        # Create the tree
        root = MoveToConfiguration(name, node)
        root.logger = logger
        tree = py_trees.trees.BehaviourTree(root)
            
        return tree

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
