#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToOffsetTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import Optional, Tuple, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import MoveTo
from ada_feeding.decorators import (
    SetPositionGoalConstraint,
    SetOrientationGoalConstraint,
)
from ada_feeding.trees import MoveToTree

class MoveToOffsetTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToOffset.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    def __init__(
        self,
        distance: float,
        direction: str,
        allowed_planning_time: float = 0.5
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        distance: the offset in milimeter to move the end effector.
        direction: the direction to move the end effector.
        allowed_planning_time: the allowed planning time for path planning.
        """
        # Initialize MoveTo
        super().__init__()

        # Store the parameters for the move to offset behavior
        self.distance = distance
        self.direction = direction
        self.allowed_planning_time = allowed_planning_time

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToOffset behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot to an offset.
        """
        # Separate blackboard namespaces for children
        distance_constraint_namespace_prefix = "distance_goal_constraint"
        direction_constraint_namespace_prefix = "direction_goal_constraint"
        move_to_namespace_prefix = "move_to"

        # Distance constraints
        distance_key = Blackboard.separator.join(
            [distance_constraint_namespace_prefix, "distance"]
        )
        self.blackboard.register_key(
            key=distance_key, access=py_trees.common.Access.WRITE
        )

        # Direction constraints
        direction_key = Blackboard.separator.join(
            [direction_constraint_namespace_prefix, "direction"]
        )
        self.blackboard.register_key(
            key=direction_key, access=py_trees.common.Access.WRITE
        )

        # MoveTo inputs
        allowed_planning_time_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "allowed_planning_time"]
        )
        self.blackboard.register_key(
            key=allowed_planning_time_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToOffset to blackboard
        self.blackboard.set(distance_key, self.distance)
        self.blackboard.set(direction_key, self.direction)
        self.blackboard.set(allowed_planning_time_key, self.allowed_planning_time)

        # Create the MoveTo behavior
        move_to_name = Blackboard.separator.join([name, move_to_namespace_prefix])
        move_to = MoveTo(move_to_name, name, node)
        move_to.logger = logger

        # Add the distance goal constraint to the MoveTo behavior
        distance_goal_constaint_name = Blackboard.separator.join(
            [name, distance_constraint_namespace_prefix]
        )
        distance_constraint = SetPositionGoalConstraint(
            distance_goal_constaint_name, move_to
        )
        distance_constraint.logger = logger

        # Add the direction goal constraint to the MoveTo behavior
        direction_goal_constaint_name = Blackboard.separator.join(
            [name, direction_constraint_namespace_prefix]
        )
        root = SetOrientationGoalConstraint(
            direction_goal_constaint_name, distance_constraint
        )
        root.logger = logger

        tree = py_trees.trees.BehaviourTree(root)
        return tree