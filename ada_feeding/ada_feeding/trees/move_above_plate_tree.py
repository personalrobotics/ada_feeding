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
import yaml

# Local imports
from ada_feeding.behaviors import MoveTo
from ada_feeding.helpers import import_from_string
from ada_feeding import ActionServerBT


class MoveAbovePlateTree(ActionServerBT):
    """
    MoveAbovePlateTree is a behavior tree subclassing ActionServerBT which is an interface for behavior trees to be wrapped in an action server.
    """

    def __init__(
        self,
        action_type_class: str
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        """
        # Import the action type
        self.action_type_class = import_from_string(action_type_class)

        # Cache the tree so that it can be reused
        self.tree = None
        self.blackboard = None

    def create_tree(
        self, name: str, logger: logging.Logger
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveAbovePlate behavior tree that moves the robot arm above the plate.

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
            # parallel root
            root = MoveTo(name)
            root.logger = logger
            # Create the tree
            self.tree = py_trees.trees.BehaviourTree(root)
            #root = py_trees.composites.Parallel(
             #       name="Move_Above_Plate_Tree_Root",
              #      policy=py_trees.common.ParallelPolicy.SuccessOnAll(
               #     synchronise=False
                #)
            #)

            # read goal from yaml file
            # with open('../config/feeding_goal_config.yaml', 'r') as file:
               # parameter_service = yaml.safe_load(file)
            # location_goal = parameter_service['above_plate']
            location_goal = [-2.11666, 3.34967, 2.04129, -2.30031, -2.34026, 2.9545]
            print(location_goal)
            # create blackboard and write goal in blackboard
            self.blackboard = py_trees.blackboard.Client(name=name + " Tree")
            self.blackboard.register_key(
                key="goal", access=py_trees.common.Access.WRITE
            )
            self.blackboard.register_key(
                key="is_planning", access=py_trees.common.Access.READ
            )
            # set blackboard key value
            self.blackboard.goal = location_goal
            print(self.blackboard.goal)
            # add child to root
            # root.add_child(blackboard)
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
        feedback_msg = self.action_type_class.Feedback()
        if self.blackboard.exists("is_planning"):
            feedback_msg.is_planning = self.blackboard.is_planning
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