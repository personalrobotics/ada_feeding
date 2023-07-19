#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithPosePathConstraintsTree behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List, Tuple, Optional, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.decorators import SetPositionPathConstraint, SetOrientationPathConstraint
from ada_feeding.trees import MoveToTree, MoveToConfigurationTree


class MoveToConfigurationWithPosePathConstraintsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration.
    """

    def __init__(
        self,
        action_type_class_str: str,
        # Required parameters for moving to a configuration
        joint_positions: List[float],
        # Required parameters for the pose path constraint
        position: Tuple[float, float, float] = None,
        quat_xyzw: Tuple[float, float, float, float] = None,
        # Optional parameters for moving to a configuration
        tolerance_joint_goal: float = 0.001,
        weight_joint_goal: float = 1.0,
        planner_id: str = "RRTstarkConfigDefault",
        # Optional parameters for the pose path constraint
        frame_id: Optional[str] = None,
        target_link: Optional[str] = None,
        tolerance_position_path: float = 0.001,
        tolerance_orientation_path: Union[float, Tuple[float, float, float]] = 0.001,
        parameterization_orientation_path: int = 0,
        weight_position_path: float = 1.0,
        weight_orientation_path: float = 1.0,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class_str: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
        joint_positions: The joint positions for the goal constraint.
        position: the position for the path constraint.
        quat_xyzw: the orientation for the path constraint.
        tolerance_joint_goal: The tolerance for the joint goal constraint.
        weight_joint_goal: The weight for the joint goal constraint.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        frame_id: the frame id of the target pose, for the pose path constraint.
            If None, the base link is used.
        target_link: the link to move to the target pose, for the pose path
            constraint. If None, the end effector link is used.
        tolerance_position_path: the tolerance for the end effector position,
            for the pose path constraint.
        tolerance_orientation_path: the tolerance for the end effector orientation,
            for the pose path constraint.
        parameterization_orientation_path: the parameterization for the end effector
            orientation, for the pose path constraint. 0 is Euler angles, 1 is
            rotation vector.
        weight_position_path: the weight for the end effector position path constraint.
        weight_orientation_path: the weight for the end effector orientation path constraint.
        """
        # Initialize MoveToTree
        self.action_type_class_str = action_type_class_str
        super().__init__(action_type_class_str)

        # Store the parameters for the joint goal constraint
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance_joint_goal = tolerance_joint_goal
        self.weight_joint_goal = weight_joint_goal
        self.planner_id = planner_id

        # Store the parameters for the pose path constraint
        self.position = position
        self.quat_xyzw = quat_xyzw
        self.frame_id = frame_id
        self.target_link = target_link
        self.tolerance_position_path = tolerance_position_path
        self.tolerance_orientation_path = tolerance_orientation_path
        self.parameterization_orientation_path = parameterization_orientation_path
        self.weight_position_path = weight_position_path
        self.weight_orientation_path = weight_orientation_path

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToConfiguration behavior tree.

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
        # First, create the MoveToConfiguration behavior tree, in the same
        # namespace as this tree
        move_to_configuration_root = MoveToConfigurationTree(
            action_type_class_str=self.action_type_class_str,
            joint_positions=self.joint_positions,
            tolerance=self.tolerance_joint_goal,
            weight=self.weight_joint_goal,
            planner_id=self.planner_id,
        ).create_tree(name, logger, node).root

        # Separate blackboard namespaces for decorators
        if self.position is not None:
            position_constraint_namespace_prefix = "position_path_constraint"
        if self.quat_xyzw is not None:
            orientation_constraint_namespace_prefix = "orientation_path_constraint"

        # Position constraints
        if self.position is not None:
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
        if self.quat_xyzw is not None:
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

        # Write the inputs to MoveToConfigurationWithPosePathConstraintsTree to blackboard
        if self.position is not None:
            self.blackboard.set(position_key, self.position)
            self.blackboard.set(position_frame_id_key, self.frame_id)
            self.blackboard.set(position_target_link_key, self.target_link)
            self.blackboard.set(position_tolerance_key, self.tolerance_position_path)
            self.blackboard.set(position_weight_key, self.weight_position_path)
        if self.quat_xyzw is not None:
            self.blackboard.set(orientation_key, self.quat_xyzw)
            self.blackboard.set(orientation_frame_id_key, self.frame_id)
            self.blackboard.set(orientation_target_link_key, self.target_link)
            self.blackboard.set(orientation_tolerance_key, self.tolerance_orientation_path)
            self.blackboard.set(orientation_parameterization_key, self.parameterization_orientation_path)
            self.blackboard.set(orientation_weight_key, self.weight_orientation_path)

        # Add the position goal constraint to the MoveToConfiguration root
        if self.position is not None:
            position_goal_constaint_name = Blackboard.separator.join(
                [name, position_constraint_namespace_prefix]
            )
            position_constraint = SetPositionPathConstraint(
                position_goal_constaint_name, move_to_configuration_root
            )
            position_constraint.logger = logger
        else:
            position_constraint = move_to_configuration_root

        # Add the orientation goal constraint to the MoveTo behavior
        if self.quat_xyzw is not None:
            orientation_goal_constaint_name = Blackboard.separator.join(
                [name, orientation_constraint_namespace_prefix]
            )
            root = SetOrientationPathConstraint(
                orientation_goal_constaint_name, position_constraint
            )
            root.logger = logger
        else:
            root = position_constraint

        tree = py_trees.trees.BehaviourTree(root)
        return tree
