#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToPoseWithPosePathConstraintsTree behavior tree and
provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import Tuple, Optional, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import add_pose_path_constraints
from ada_feeding.trees import MoveToTree, MoveToPoseTree


class MoveToPoseWithPosePathConstraintsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified end effector pose, while
    honoring pose path constraints.
    """

    def __init__(
        self,
        action_type_class_str: str,
        # Required parameters for moving to a pose
        position_goal: Tuple[float, float, float],
        quat_xyzw_goal: Tuple[float, float, float, float],
        # Optional parameters for moving to a pose
        frame_id_goal: Optional[str] = None,
        target_link_goal: Optional[str] = None,
        tolerance_position_goal: float = 0.001,
        tolerance_orientation_goal: Union[float, Tuple[float, float, float]] = 0.001,
        parameterization_orientation_goal: int = 0,
        weight_position_goal: float = 1.0,
        weight_orientation_goal: float = 1.0,
        cartesian: bool = False,
        planner_id: str = "RRTstarkConfigDefault",
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
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class_str: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
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
        planner_id: the planner ID to use for the MoveIt2 motion planning.
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
        """
        # Initialize MoveToTree
        self.action_type_class_str = action_type_class_str
        super().__init__(action_type_class_str)

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
        self.planner_id = planner_id

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

    def create_move_to_tree(
        self,
        name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToPoseWithPosePathConstraintsTree behavior tree.

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
        # First, create the MoveToPose behavior tree, in the same
        # namespace as this tree
        move_to_pose_root = MoveToPoseTree(
            action_type_class_str = self.action_type_class_str,
            position = self.position_goal,
            quat_xyzw = self.quat_xyzw_goal,
            frame_id = self.frame_id_goal,
            target_link = self.target_link_goal,
            tolerance_position = self.tolerance_position_goal,
            tolerance_orientation = self.tolerance_orientation_goal,
            parameterization = self.parameterization_orientation_goal,
            weight_position = self.weight_position_goal,
            weight_orientation = self.weight_orientation_goal,
            cartesian = self.cartesian,
            planner_id = self.planner_id,
        ).create_tree(name, logger, node).root

        root = add_pose_path_constraints(
            child = move_to_pose_root,
            name = name,
            blackboard = self.blackboard,
            logger = logger,
            set_blackboard_variables = True,
            position_path = self.position_path,
            quat_xyzw_path = self.quat_xyzw_path,
            frame_id_path = self.frame_id_path,
            target_link_path = self.target_link_path,
            tolerance_position_path = self.tolerance_position_path,
            tolerance_orientation_path = self.tolerance_orientation_path,
            parameterization_orientation_path = self.parameterization_orientation_path,
            weight_position_path = self.weight_position_path,
            weight_orientation_path = self.weight_orientation_path,
        )

        # # Separate blackboard namespaces for decorators
        # if self.position is not None:
        #     position_constraint_namespace_prefix = "position_path_constraint"
        # if self.quat_xyzw is not None:
        #     orientation_constraint_namespace_prefix = "orientation_path_constraint"

        # # Position constraints
        # if self.position is not None:
        #     position_key = Blackboard.separator.join(
        #         [position_constraint_namespace_prefix, "position"]
        #     )
        #     self.blackboard.register_key(
        #         key=position_key, access=py_trees.common.Access.WRITE
        #     )
        #     position_frame_id_key = Blackboard.separator.join(
        #         [position_constraint_namespace_prefix, "frame_id"]
        #     )
        #     self.blackboard.register_key(
        #         key=position_frame_id_key, access=py_trees.common.Access.WRITE
        #     )
        #     position_target_link_key = Blackboard.separator.join(
        #         [position_constraint_namespace_prefix, "target_link"]
        #     )
        #     self.blackboard.register_key(
        #         key=position_target_link_key, access=py_trees.common.Access.WRITE
        #     )
        #     position_tolerance_key = Blackboard.separator.join(
        #         [position_constraint_namespace_prefix, "tolerance"]
        #     )
        #     self.blackboard.register_key(
        #         key=position_tolerance_key, access=py_trees.common.Access.WRITE
        #     )
        #     position_weight_key = Blackboard.separator.join(
        #         [position_constraint_namespace_prefix, "weight"]
        #     )
        #     self.blackboard.register_key(
        #         key=position_weight_key, access=py_trees.common.Access.WRITE
        #     )

        # # Orientation constraints
        # if self.quat_xyzw is not None:
        #     orientation_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "quat_xyzw"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_key, access=py_trees.common.Access.WRITE
        #     )
        #     orientation_frame_id_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "frame_id"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_frame_id_key, access=py_trees.common.Access.WRITE
        #     )
        #     orientation_target_link_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "target_link"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_target_link_key, access=py_trees.common.Access.WRITE
        #     )
        #     orientation_tolerance_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "tolerance"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_tolerance_key, access=py_trees.common.Access.WRITE
        #     )
        #     orientation_parameterization_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "parameterization"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_parameterization_key, access=py_trees.common.Access.WRITE
        #     )
        #     orientation_weight_key = Blackboard.separator.join(
        #         [orientation_constraint_namespace_prefix, "weight"]
        #     )
        #     self.blackboard.register_key(
        #         key=orientation_weight_key, access=py_trees.common.Access.WRITE
        #     )

        # # Write the inputs to MoveToPoseWithPosePathConstraintsTree to blackboard
        # if self.position is not None:
        #     self.blackboard.set(position_key, self.position)
        #     self.blackboard.set(position_frame_id_key, self.frame_id)
        #     self.blackboard.set(position_target_link_key, self.target_link)
        #     self.blackboard.set(position_tolerance_key, self.tolerance_position_path)
        #     self.blackboard.set(position_weight_key, self.weight_position_path)
        # if self.quat_xyzw is not None:
        #     self.blackboard.set(orientation_key, self.quat_xyzw)
        #     self.blackboard.set(orientation_frame_id_key, self.frame_id)
        #     self.blackboard.set(orientation_target_link_key, self.target_link)
        #     self.blackboard.set(orientation_tolerance_key, self.tolerance_orientation_path)
        #     self.blackboard.set(orientation_parameterization_key, self.parameterization_orientation_path)
        #     self.blackboard.set(orientation_weight_key, self.weight_orientation_path)

        # # Add the position goal constraint to the MoveToConfiguration root
        # if self.position is not None:
        #     position_goal_constaint_name = Blackboard.separator.join(
        #         [name, position_constraint_namespace_prefix]
        #     )
        #     position_constraint = SetPositionPathConstraint(
        #         position_goal_constaint_name, move_to_configuration_root
        #     )
        #     position_constraint.logger = logger
        # else:
        #     position_constraint = move_to_configuration_root

        # # Add the orientation goal constraint to the MoveTo behavior
        # if self.quat_xyzw is not None:
        #     orientation_goal_constaint_name = Blackboard.separator.join(
        #         [name, orientation_constraint_namespace_prefix]
        #     )
        #     root = SetOrientationPathConstraint(
        #         orientation_goal_constaint_name, position_constraint
        #     )
        #     root.logger = logger
        # else:
        #     root = position_constraint

        tree = py_trees.trees.BehaviourTree(root)
        return tree
