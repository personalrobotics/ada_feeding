#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithPosePathConstraintsTree behavior
tree and provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List, Optional, Tuple, Set, Union

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import add_pose_path_constraints
from ada_feeding.trees import MoveToTree, MoveToConfigurationTree

# pylint: disable=duplicate-code
# move_to_configuration_with_pose_path_constraints_tree.py has similar code to
# move_to_pose_with_pose_path_constraints_tree.py when calling `add_pose_path_constraints`.
# This is not a problem.


class MoveToConfigurationWithPosePathConstraintsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration while
    honoring pose path constraints.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.
    # pylint: disable=dangerous-default-value
    # A mutable default value is okay since we don't change it in this function.
    def __init__(
        self,
        # Required parameters for moving to a configuration
        joint_positions_goal: List[float],
        # Optional parameters for moving to a configuration
        tolerance_joint_goal: float = 0.001,
        weight_joint_goal: float = 1.0,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        max_velocity_scaling_factor: float = 0.1,
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
        keys_to_not_write_to_blackboard: Set[str] = set(),
        clear_constraints: bool = True,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        joint_positions_goal: The joint positions for the goal constraint.
        tolerance_joint_goal: The tolerance for the joint goal constraint.
        weight_joint_goal: The weight for the joint goal constraint.
        pipeline_id: The pipeline ID to use for MoveIt2 motion planning.
        planner_id: The planner ID to use for MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        max_velocity_scaling_factor: The maximum velocity scaling factor for the
            MoveIt2 motion planner.
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
        keys_to_not_write_to_blackboard: the keys to not write to the blackboard.
            Note that the keys need to be exact e.g., "move_to.cartesian,"
            "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
            etc.
        clear_constraints: Whether or not to put a ClearConstraints decorator at the top
            of this branch. If you will be adding additional Constraints on top of this
            tree, this should be False. Else (e.g., if this is a standalone tree), True.
        """
        # Initialize MoveToTree
        super().__init__()

        # Store the parameters for the joint goal constraint
        self.joint_positions_goal = joint_positions_goal
        assert len(self.joint_positions_goal) == 6, "Must provide 6 joint positions"
        self.tolerance_joint_goal = tolerance_joint_goal
        self.weight_joint_goal = weight_joint_goal
        self.pipeline_id = pipeline_id
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor

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

        self.keys_to_not_write_to_blackboard = keys_to_not_write_to_blackboard
        self.clear_constraints = clear_constraints

    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToConfigurationWithPosePathConstraintsTree behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables.
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
                pipeline_id=self.pipeline_id,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time,
                max_velocity_scaling_factor=self.max_velocity_scaling_factor,
                keys_to_not_write_to_blackboard=self.keys_to_not_write_to_blackboard,
                clear_constraints=False,
            )
            .create_tree(name, self.action_type, tree_root_name, logger, node)
            .root
        )

        # Add the pose path constraints
        root = add_pose_path_constraints(
            child=move_to_configuration_root,
            name=name,
            blackboard=self.blackboard,
            logger=logger,
            keys_to_not_write_to_blackboard=self.keys_to_not_write_to_blackboard,
            position_path=self.position_path,
            quat_xyzw_path=self.quat_xyzw_path,
            frame_id_path=self.frame_id,
            target_link_path=self.target_link,
            tolerance_position_path=self.tolerance_position_path,
            tolerance_orientation_path=self.tolerance_orientation_path,
            parameterization_orientation_path=self.parameterization_orientation_path,
            weight_position_path=self.weight_position_path,
            weight_orientation_path=self.weight_orientation_path,
            node=node,
            clear_constraints=self.clear_constraints,
        )

        tree = py_trees.trees.BehaviourTree(root)
        return tree
