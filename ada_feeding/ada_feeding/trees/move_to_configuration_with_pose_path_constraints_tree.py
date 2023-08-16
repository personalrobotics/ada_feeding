#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithPosePathConstraintsTree behavior
tree and provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List, Tuple, Optional, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import add_pose_path_constraints
from ada_feeding.trees import MoveToTree, MoveToConfigurationTree


class MoveToConfigurationWithPosePathConstraintsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration while
    honoring pose path constraints.
    """

    def __init__(
        self,
        # Required parameters for moving to a configuration
        joint_positions_goal: List[float],
        # Optional parameters for moving to a configuration
        tolerance_joint_goal: float = 0.001,
        weight_joint_goal: float = 1.0,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        # Optional parameters for the pose path constraint
        position_path: Tuple[float, float, float] = None,
        quat_xyzw_path: Tuple[float, float, float, float] = None,
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
        joint_positions_goal: The joint positions for the goal constraint.
        tolerance_joint_goal: The tolerance for the joint goal constraint.
        weight_joint_goal: The weight for the joint goal constraint.
        planner_id: The planner ID to use for MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        position_path: the position for the path constraint.
        quat_xyzw_path: the orientation for the path constraint.
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
        super().__init__()

        # Store the parameters for the joint goal constraint
        self.joint_positions_goal = joint_positions_goal
        assert len(self.joint_positions_goal) == 6, "Must provide 6 joint positions"
        self.tolerance_joint_goal = tolerance_joint_goal
        self.weight_joint_goal = weight_joint_goal
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time

        # Store the parameters for the pose path constraint
        self.position_path = position_path
        self.quat_xyzw_path = quat_xyzw_path
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
        Creates the MoveToConfigurationWithPosePathConstraintsTree behavior tree.

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
        move_to_configuration_root = (
            MoveToConfigurationTree(
                joint_positions=self.joint_positions_goal,
                tolerance=self.tolerance_joint_goal,
                weight=self.weight_joint_goal,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time,
            )
            .create_tree(name, self.action_type_class_str, logger, node)
            .root
        )

        # Add the pose path constraints
        root = add_pose_path_constraints(
            child=move_to_configuration_root,
            name=name,
            blackboard=self.blackboard,
            logger=logger,
            set_blackboard_variables=True,
            position_path=self.position_path,
            quat_xyzw_path=self.quat_xyzw_path,
            frame_id_path=self.frame_id,
            target_link_path=self.target_link,
            tolerance_position_path=self.tolerance_position_path,
            tolerance_orientation_path=self.tolerance_orientation_path,
            parameterization_orientation_path=self.parameterization_orientation_path,
            weight_position_path=self.weight_position_path,
            weight_orientation_path=self.weight_orientation_path,
        )

        tree = py_trees.trees.BehaviourTree(root)
        return tree
