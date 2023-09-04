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

# pylint: disable=duplicate-code
# move_to_pose.py has similar code to move_to_configuration_tree.py when defining
# blackboard variables that are necessary for all movements (e.g., planner_id).
# This is not a problem.


class MoveToOffsetTree(MoveToTree):
    """
    A behavior tree that consists of a single behavior, MoveToOffset.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    def __init__(
        self,
        dustance: float,
        direction: str,
        allowed_planning_time: float = 0.5
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        position: the target end effector position relative to the base link.
        quat_xyzw: the target end effector orientation relative to the base link.
        frame_id: the frame id of the target offset. If None, the base link is used.
        target_link: the link to move to the target offset. If None, the end effector
            link is used.
        tolerance_position: the tolerance for the end effector position.
        tolerance_orientation: the tolerance for the end effector orientation.
        parameterization: the parameterization of the orientation goal constraint.
            0 is Euler angles, 1 is rotation vector
        weight_position: the weight for the end effector position.
        weight_orientation: the weight for the end effector orientation.
        cartesian: whether to use cartesian path planning.
        planner_id: the planner to use for path planning.
        allowed_planning_time: the allowed planning time for path planning.
        """
        # Initialize MoveTo
        super().__init__()

        # Store the parameters for the move to offset behavior
        self.distance = distance
        self.direction = direction
        self.allowed_planning_time = allowed_planning_time

    # pylint: disable=too-many-locals, too-many-statements
    # Unfortunately, many locals/statements are required here to isolate the keys
    # of similar constraints in the blackboard.
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
        orientation_parameterization_key = Blackboard.separator.join(
            [orientation_constraint_namespace_prefix, "parameterization"]
        )
        self.blackboard.register_key(
            key=orientation_parameterization_key, access=py_trees.common.Access.WRITE
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
        planner_id_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "planner_id"]
        )
        self.blackboard.register_key(
            key=planner_id_key, access=py_trees.common.Access.WRITE
        )
        allowed_planning_time_key = Blackboard.separator.join(
            [move_to_namespace_prefix, "allowed_planning_time"]
        )
        self.blackboard.register_key(
            key=allowed_planning_time_key, access=py_trees.common.Access.WRITE
        )

        # Write the inputs to MoveToOffset to blackboard
        self.blackboard.set(position_key, self.position)
        self.blackboard.set(position_frame_id_key, self.frame_id)
        self.blackboard.set(position_target_link_key, self.target_link)
        self.blackboard.set(position_tolerance_key, self.tolerance_position)
        self.blackboard.set(position_weight_key, self.weight_position)
        self.blackboard.set(orientation_key, self.quat_xyzw)
        self.blackboard.set(orientation_frame_id_key, self.frame_id)
        self.blackboard.set(orientation_target_link_key, self.target_link)
        self.blackboard.set(orientation_tolerance_key, self.tolerance_orientation)
        self.blackboard.set(orientation_parameterization_key, self.parameterization)
        self.blackboard.set(orientation_weight_key, self.weight_orientation)
        self.blackboard.set(cartesian_key, self.cartesian)
        self.blackboard.set(planner_id_key, self.planner_id)
        self.blackboard.set(allowed_planning_time_key, self.allowed_planning_time)

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