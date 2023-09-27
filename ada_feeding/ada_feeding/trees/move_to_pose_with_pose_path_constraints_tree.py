#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseWithPosePathConstraintsTree behavior tree and
provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import Optional, Set, Tuple, Union

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import add_pose_path_constraints
from ada_feeding.trees import MoveToTree, MoveToPoseTree

# pylint: disable=duplicate-code
# move_to_pose_with_pose_path_constraints_tree.py has similar code to
# move_to_configuration_with_pose_path_constraints_tree.py when calling `add_pose_path_constraints`.
# This is not a problem.


class MoveToPoseWithPosePathConstraintsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified end effector pose, while
    honoring pose path constraints.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.
    # pylint: disable=dangerous-default-value
    # A mutable default value is okay since we don't change it in this function.
    def __init__(
        self,
        # Required parameters for moving to a pose
        position_goal: Optional[Tuple[float, float, float]] = None,
        quat_xyzw_goal: Optional[Tuple[float, float, float, float]] = None,
        # Optional parameters for moving to a pose
        frame_id_goal: Optional[str] = None,
        target_link_goal: Optional[str] = None,
        tolerance_position_goal: float = 0.001,
        tolerance_orientation_goal: Union[float, Tuple[float, float, float]] = 0.001,
        parameterization_orientation_goal: int = 0,
        weight_position_goal: float = 1.0,
        weight_orientation_goal: float = 1.0,
        cartesian: bool = False,
        cartesian_jump_threshold: float = 0.0,
        cartesian_max_step: float = 0.0025,
        cartesian_fraction_threshold: float = 0.0,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        max_velocity_scaling_factor: float = 0.1,
        max_acceleration_scaling_factor: float = 0.1,
        # Optional parameters for the pose path constraint
        position_path: Tuple[float, float, float] = None,
        quat_xyzw_path: Tuple[float, float, float, float] = None,
        frame_id_path: Optional[str] = None,
        target_link_path: Optional[str] = None,
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
        position_goal: the target position relative to frame_id.
        quat_xyzw_goal: the target orientation relative to frame_id.
        frame_id_goal: the frame id of the target pose. If None, the base link is used.
        target_link_goal: the link to move to the target pose. If None, the end effector
            link is used.
        tolerance_position_goal: the tolerance for the goal position.
        tolerance_orientation_goal: the tolerance for the goal orientation.
        parameterization_orientation_goal: the parameterization for the goal
            orientation tolerance.
        weight_position_goal: the weight for the position goal.
        weight_orientation_goal: the weight for the orientation goal.
        cartesian: whether to use cartesian path planning.
        cartesian_jump_threshold: the jump threshold for cartesian path planning.
        cartesian_max_step: the maximum step for cartesian path planning.
        cartesian_fraction_threshold: if a cartesian plan does not reach at least
            this fraction of the way to the goal, the plan is rejected.
        pipeline_id: the pipeline ID to use for MoveIt2 motion planning.
        planner_id: the planner ID to use for MoveIt2 motion planning.
        allowed_planning_time: the allowed planning time for the MoveIt2 motion
            planner.
        max_velocity_scaling_factor: the maximum velocity scaling factor for
            MoveIt2 motion planning.
        max_acceleration_scaling_factor: the maximum acceleration scaling factor
            for MoveIt2 motion planning.
        position_path: the target position relative to frame_id for path constraints.
        quat_xyzw_path: the target orientation relative to frame_id for path constraints.
        frame_id_path: the frame id of the target pose for path constraints. If None,
            the base link is used.
        target_link_path: the link to move to the target pose for path constraints.
            If None, the end effector link is used.
        tolerance_position_path: the tolerance for the path position.
        tolerance_orientation_path: the tolerance for the path orientation.
        parameterization_orientation_path: the parameterization for the path
            orientation tolerance.
        weight_position_path: the weight for the position path.
        weight_orientation_path: the weight for the orientation path.
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

        # Store the parameters for the move to pose behavior
        self.position_goal = position_goal
        self.quat_xyzw_goal = quat_xyzw_goal
        self.frame_id_goal = frame_id_goal
        self.target_link_goal = target_link_goal
        self.tolerance_position_goal = tolerance_position_goal
        self.tolerance_orientation_goal = tolerance_orientation_goal
        self.parameterization_orientation_goal = parameterization_orientation_goal
        self.weight_position_goal = weight_position_goal
        self.weight_orientation_goal = weight_orientation_goal
        self.cartesian = cartesian
        self.cartesian_jump_threshold = cartesian_jump_threshold
        self.cartesian_max_step = cartesian_max_step
        self.cartesian_fraction_threshold = cartesian_fraction_threshold
        self.pipeline_id = pipeline_id
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.max_acceleration_scaling_factor = max_acceleration_scaling_factor

        # Store the parameters for the pose path constraint
        self.position_path = position_path
        self.quat_xyzw_path = quat_xyzw_path
        self.frame_id_path = frame_id_path
        self.target_link_path = target_link_path
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
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToPoseWithPosePathConstraintsTree behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """
        # First, create the MoveToPose behavior tree, in the same
        # namespace as this tree
        move_to_pose_root = (
            MoveToPoseTree(
                position=self.position_goal,
                quat_xyzw=self.quat_xyzw_goal,
                frame_id=self.frame_id_goal,
                target_link=self.target_link_goal,
                tolerance_position=self.tolerance_position_goal,
                tolerance_orientation=self.tolerance_orientation_goal,
                parameterization=self.parameterization_orientation_goal,
                weight_position=self.weight_position_goal,
                weight_orientation=self.weight_orientation_goal,
                cartesian=self.cartesian,
                cartesian_jump_threshold=self.cartesian_jump_threshold,
                cartesian_max_step=self.cartesian_max_step,
                cartesian_fraction_threshold=self.cartesian_fraction_threshold,
                pipeline_id=self.pipeline_id,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time,
                max_velocity_scaling_factor=self.max_velocity_scaling_factor,
                max_acceleration_scaling_factor=self.max_acceleration_scaling_factor,
                keys_to_not_write_to_blackboard=self.keys_to_not_write_to_blackboard,
                clear_constraints=False,
            )
            .create_tree(name, self.action_type, tree_root_name, node)
            .root
        )

        root = add_pose_path_constraints(
            child=move_to_pose_root,
            name=name,
            blackboard=self.blackboard,
            keys_to_not_write_to_blackboard=self.keys_to_not_write_to_blackboard,
            position_path=self.position_path,
            quat_xyzw_path=self.quat_xyzw_path,
            frame_id_path=self.frame_id_path,
            target_link_path=self.target_link_path,
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
