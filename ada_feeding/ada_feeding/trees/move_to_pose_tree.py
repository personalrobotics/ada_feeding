#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import Optional, Tuple

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


class MoveToPoseTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToPose.
    """

    def __init__(
        self,
        action_type_class: str,
        position: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_orientation: float = 0.001,
        weight_position: float = 1.0,
        weight_orientation: float = 1.0,
        cartesian: bool = False,
    ) -> None:
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        position: the target end effector position relative to the base link.
        quat_xyzw: the target end effector orientation relative to the base link.
        frame_id: the frame id of the target pose. If None, the base link is used.
        target_link: the link to move to the target pose. If None, the end effector
            link is used.
        tolerance_position: the tolerance for the end effector position.
        tolerance_orientation: the tolerance for the end effector orientation.
        weight_position: the weight for the end effector position.
        weight_orientation: the weight for the end effector orientation.
        cartesian: whether to use cartesian path planning.
        """
        # Initialize MoveTo
        super().__init__(action_type_class)

        # Store the parameters for the move to pose behavior
        self.position = position
        self.quat_xyzw = quat_xyzw
        self.frame_id = frame_id
        self.target_link = target_link
        self.tolerance_position = tolerance_position
        self.tolerance_orientation = tolerance_orientation
        self.weight_position = weight_position
        self.weight_orientation = weight_orientation
        self.cartesian = cartesian

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToPose behavior tree.

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
        # Separate blackboard namespaces for children
        position_constraint_namespace_prefix = "position_goal_constraint"
        orientation_constraint_namespace_prefix = "orientation_goal_constraint"
        move_to_namespace_prefix = "move_to"

        # Position constraints
        position_key = Blackboard.separator.join(
            [position_constraint_namespace_prefix, "position"]
        )
        self.blackboard.register_key(
            key=position_key, access=py_trees.common.Access.WRITE
        )
        position_frame_id_key = Blackboard.separator.join(
            [position_constraint_namespace_prefix, "frame_id"]
        )
        self.blackboard.register_key(
            key=position_frame_id_key, access=py_trees.common.Access.WRITE
        )
        position_target_link_key = Blackboard.separator.join(
            [position_constraint_namespace_prefix, "target_link"]
        )
        self.blackboard.register_key(
            key=position_target_link_key, access=py_trees.common.Access.WRITE
        )
        position_tolerance_key = Blackboard.separator.join(
            [position_constraint_namespace_prefix, "tolerance"]
        )
        self.blackboard.register_key(
            key=position_tolerance_key, access=py_trees.common.Access.WRITE
        )
        position_weight_key = Blackboard.separator.join(
            [position_constraint_namespace_prefix, "weight"]
        )
        self.blackboard.register_key(
            key=position_weight_key, access=py_trees.common.Access.WRITE
        )

        # Orientation constraints
        orientation_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "quat_xyzw"]
        )
        self.blackboard.register_key(
            key=orientation_key, access=py_trees.common.Access.WRITE
        )
        orientation_frame_id_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "frame_id"]
        )
        self.blackboard.register_key(
            key=orientation_frame_id_key, access=py_trees.common.Access.WRITE
        )
        orientation_target_link_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "target_link"]
        )
        self.blackboard.register_key(
            key=orientation_target_link_key, access=py_trees.common.Access.WRITE
        )
        orientation_tolerance_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "tolerance"]
        )
        self.blackboard.register_key(
            key=orientation_tolerance_key, access=py_trees.common.Access.WRITE
        )
        orientation_weight_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "weight"]
        )
        self.blackboard.register_key(
            key=orientation_weight_key, access=py_trees.common.Access.WRITE
        )

        # MoveTo inputs
        cartesian_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "cartesian"]
        )
        self.blackboard.register_key(
            key=cartesian_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToPose to blackboard
        self.blackboard.set(position_key, self.position)
        self.blackboard.set(position_frame_id_key, self.frame_id)
        self.blackboard.set(position_target_link_key, self.target_link)
        self.blackboard.set(position_tolerance_key, self.tolerance_position)
        self.blackboard.set(position_weight_key, self.weight_position)
        self.blackboard.set(orientation_key, self.quat_xyzw)
        self.blackboard.set(orientation_frame_id_key, self.frame_id)
        self.blackboard.set(orientation_target_link_key, self.target_link)
        self.blackboard.set(orientation_tolerance_key, self.tolerance_orientation)
        self.blackboard.set(orientation_weight_key, self.weight_orientation)
        self.blackboard.set(cartesian_key, self.cartesian)

        # Create the MoveTo behavior
        move_to_name = Blackboard.separator.join([name, move_to_namespace_prefix])
        move_to = MoveTo(move_to_name, name, node)
        move_to.logger = logger

        # Add the position goal constraint to the MoveTo behavior
        position_goal_constaint_name = Blackboard.separator.join(
            [name, position_constraint_namespace_prefix]
        )
        position_constraint = SetPositionGoalConstraint(
            position_goal_constaint_name, move_to
        )
        position_constraint.logger = logger

        # Add the orientation goal constraint to the MoveTo behavior
        orientation_goal_constaint_name = Blackboard.separator.join(
            [name, orientation_constraint_namespace_prefix]
        )
        root = SetOrientationGoalConstraint(
            orientation_goal_constaint_name, position_constraint
        )
        root.logger = logger

        tree = py_trees.trees.BehaviourTree(root)
        return tree
