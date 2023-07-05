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
from ada_feeding.behaviors import MoveToDummy
from ada_feeding.helpers import import_from_string
from ada_feeding import ActionServerBT


class MoveAbovePlateTree(ActionServerBT):
    """
    MoveAbovePlate behavior tree subclassing ActionServerBT which is an interface for behavior trees to be wrapped in an action server.
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
        plan_time: How many seconds this node should spend in planning.
        motion_time: How many seconds this node should spend in motion.
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
            root = py_trees.composites.Parallel(
                    name="Move_Above_Plate_Tree_Root",
                    policy=py_trees.common.ParallelPolicy.SuccessOnAll(
                    synchronise=False
                )
            )

            # read goal from yaml file
            with open('../../config/feeding_goal_config.yaml', 'r') as file:
                parameter_service = yaml.safe_load(file)
            location_goal = parameter_service['above_plate']
            print(location_goal)
            # create blackboard and write goal in blackboard
            self.blackboard = py_trees.blackboard.Client(name=name + " Tree")
            self.blackboard.register_key(
                key="goal", access=py_trees.common.Access.WRITE
            )
            # set blackboard key value
            self.blackboard.goal = location_goal
            print(self.blackboard.goal)
            # add child to root
            root.add_child(blackboard)
            # Create the tree
            self.tree = py_trees.trees.BehaviourTree(root)
        return self.tree