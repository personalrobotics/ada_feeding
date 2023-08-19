#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToTeleopTree behavior tree and provides functions to
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
    SetJointGoalConstraint,
)
from ada_feeding.trees import MoveToTree


class MoveToTeleopTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToTeleop.
    """

    def __init__(
        self,
        action_type_class: str,
        position: Tuple[float, float, float],
        joint_positions: List[float],
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position: float = 0.001,
        tolerance_joint_positions: float = 0.001,
        weight_position: float = 1.0,
        weight_joint_positions: float = 1.0,
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
        self.joint_positions = joint_positions
        self.frame_id = frame_id
        self.target_link = target_link
        self.tolerance_position = tolerance_position
        self.tolerance_joint_positions = tolerance_joint_positions
        self.weight_position = weight_position
        self.weight_joint_positions = weight_joint_positions
        self.cartesian = cartesian

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToTeleop behavior tree.

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
        joint_constraint_namespace_prefix = "joint_goal_constraint"
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

        # joint_positions constraints
        joint_positions_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "joint_positions"]
        )
        self.blackboard.register_key(
            key=joint_positions_key, access=py_trees.common.Access.WRITE
        )
        joint_positions_frame_id_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "frame_id"]
        )
        self.blackboard.register_key(
            key=joint_frame_id_key, access=py_trees.common.Access.WRITE
        )
        joint_positions_target_link_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "target_link"]
        )
        self.blackboard.register_key(
            key=joint_positions_target_link_key, access=py_trees.common.Access.WRITE
        )
        joint_positions_tolerance_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "tolerance"]
        )
        self.blackboard.register_key(
            key=joint_positions_tolerance_key, access=py_trees.common.Access.WRITE
        )
        joint_positions_weight_key = Blackboard.separator.join(
            [joint_constraint_namespace_prefix, "weight"]
        )
        self.blackboard.register_key(
            key=joint_positions_weight_key, access=py_trees.common.Access.WRITE
        )

        # MoveTo inputs
        cartesian_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "cartesian"]
        )
        self.blackboard.register_key(
            key=cartesian_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToTeleop to blackboard
        self.blackboard.set(position_key, self.position)
        self.blackboard.set(position_frame_id_key, self.frame_id)
        self.blackboard.set(position_target_link_key, self.target_link)
        self.blackboard.set(position_tolerance_key, self.tolerance_position)
        self.blackboard.set(position_weight_key, self.weight_position)
        self.blackboard.set(joint_positions_key, self.joint_positions)
        self.blackboard.set(joint_positions_frame_id_key, self.frame_id)
        self.blackboard.set(joint_positions_target_link_key, self.target_link)
        self.blackboard.set(joint_positions_tolerance_key, self.tolerance_orientation)
        self.blackboard.set(joint_positions_weight_key, self.weight_orientation)
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
        joint_goal_constaint_name = Blackboard.separator.join(
            [name, joint_constraint_namespace_prefix]
        )
        root = SetJointGoalConstraint(
            joint_goal_constaint_name, position_constraint
        )
        root.logger = logger

        tree = py_trees.trees.BehaviourTree(root)
        return tree
