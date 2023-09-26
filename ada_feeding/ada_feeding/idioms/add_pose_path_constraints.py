#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the add_pose_path_constraints idiom, which takes in a
behavior that consists of MoveTo with any number of MoveToConstraint decorators
on it, and adds a SetPositionPathConstraint and/or SetOrientationPathConstraint.
"""

# Standard imports
import logging
from typing import Set, Tuple, Optional, Union

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.decorators import (
    ClearConstraints,
    SetPositionPathConstraint,
    SetOrientationPathConstraint,
)
from ada_feeding.helpers import (
    CLEAR_CONSTRAINTS_NAMESPACE_PREFIX,
    POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX,
    ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX,
    set_to_blackboard,
)


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
# Unfortunately, all these arguments/locals/statements are necessary to add
# pose path constraints to a behavior tree -- the point of putting them
# into one function is to reduce the number of arguments/locals/statements
# needed in other functions.
# pylint: disable=dangerous-default-value
# A mutable default value is okay since we don't change it in this function.
def add_pose_path_constraints(
    child: py_trees.behaviour.Behaviour,
    name: str,
    blackboard: py_trees.blackboard.Blackboard,
    node: Node,
    keys_to_not_write_to_blackboard: Set[str] = set(),
    # Optional parameters for the pose path constraint
    position_path: Optional[Tuple[float, float, float]] = None,
    quat_xyzw_path: Optional[Tuple[float, float, float, float]] = None,
    frame_id_path: Optional[str] = None,
    target_link_path: Optional[str] = None,
    tolerance_position_path: float = 0.001,
    tolerance_orientation_path: Union[float, Tuple[float, float, float]] = 0.001,
    parameterization_orientation_path: int = 0,
    weight_position_path: float = 1.0,
    weight_orientation_path: float = 1.0,
    clear_constraints: bool = True,
) -> py_trees.behaviour.Behaviour:
    """
    Adds a SetPositionPathConstraint and/or SetOrientationPathConstraint to a
    behavior that consists of MoveTo with any number of MoveToConstraint
    decorators on it. Optionally sets the blackboard variables for the path
    constraint.

    Parameters
    ----------
    child: The child behavior. Must be an instance of MoveTo or MoveToConstraint.
    name: The name for the tree that this behavior is in.
    blackboard: The blackboard for the tree that this behavior is in.
    node: The ROS2 node that the MoveIt2 object is associated with.
    keys_to_not_write_to_blackboard: the keys to not write to the blackboard.
        Note that the keys need to be exact e.g., "move_to.cartesian,"
        "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
        etc.
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
    clear_constraints: Whether or not to put a ClearConstraints decorator at the top
        of this branch. If you will be adding additional Constraints on top of this
        tree, this should be False. Else (e.g., if this is a standalone tree), True.
    """

    # Separate blackboard namespaces for decorators
    if position_path is not None:
        position_path_constraint_namespace_prefix = (
            POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX
        )
    if quat_xyzw_path is not None:
        orientation_path_constraint_namespace_prefix = (
            ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX
        )
    clear_constraints_namespace_prefix = CLEAR_CONSTRAINTS_NAMESPACE_PREFIX

    # Position constraints
    if position_path is not None:
        position_key = Blackboard.separator.join(
            [position_path_constraint_namespace_prefix, "position"]
        )
        blackboard.register_key(key=position_key, access=py_trees.common.Access.WRITE)
        position_frame_id_key = Blackboard.separator.join(
            [position_path_constraint_namespace_prefix, "frame_id"]
        )
        blackboard.register_key(
            key=position_frame_id_key, access=py_trees.common.Access.WRITE
        )
        position_target_link_key = Blackboard.separator.join(
            [position_path_constraint_namespace_prefix, "target_link"]
        )
        blackboard.register_key(
            key=position_target_link_key, access=py_trees.common.Access.WRITE
        )
        position_tolerance_key = Blackboard.separator.join(
            [position_path_constraint_namespace_prefix, "tolerance"]
        )
        blackboard.register_key(
            key=position_tolerance_key, access=py_trees.common.Access.WRITE
        )
        position_weight_key = Blackboard.separator.join(
            [position_path_constraint_namespace_prefix, "weight"]
        )
        blackboard.register_key(
            key=position_weight_key, access=py_trees.common.Access.WRITE
        )

    # Orientation constraints
    if quat_xyzw_path is not None:
        orientation_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "quat_xyzw"]
        )
        blackboard.register_key(
            key=orientation_key, access=py_trees.common.Access.WRITE
        )
        orientation_frame_id_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "frame_id"]
        )
        blackboard.register_key(
            key=orientation_frame_id_key, access=py_trees.common.Access.WRITE
        )
        orientation_target_link_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "target_link"]
        )
        blackboard.register_key(
            key=orientation_target_link_key, access=py_trees.common.Access.WRITE
        )
        orientation_tolerance_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "tolerance"]
        )
        blackboard.register_key(
            key=orientation_tolerance_key, access=py_trees.common.Access.WRITE
        )
        orientation_parameterization_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "parameterization"]
        )
        blackboard.register_key(
            key=orientation_parameterization_key, access=py_trees.common.Access.WRITE
        )
        orientation_weight_key = Blackboard.separator.join(
            [orientation_path_constraint_namespace_prefix, "weight"]
        )
        blackboard.register_key(
            key=orientation_weight_key, access=py_trees.common.Access.WRITE
        )

    # Write the inputs to the pose path constraints to blackboard
    if position_path is not None:
        set_to_blackboard(
            blackboard, position_key, position_path, keys_to_not_write_to_blackboard
        )
        set_to_blackboard(
            blackboard,
            position_frame_id_key,
            frame_id_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            position_target_link_key,
            target_link_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            position_tolerance_key,
            tolerance_position_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            position_weight_key,
            weight_position_path,
            keys_to_not_write_to_blackboard,
        )
    if quat_xyzw_path is not None:
        set_to_blackboard(
            blackboard, orientation_key, quat_xyzw_path, keys_to_not_write_to_blackboard
        )
        set_to_blackboard(
            blackboard,
            orientation_frame_id_key,
            frame_id_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            orientation_target_link_key,
            target_link_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            orientation_tolerance_key,
            tolerance_orientation_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            orientation_parameterization_key,
            parameterization_orientation_path,
            keys_to_not_write_to_blackboard,
        )
        set_to_blackboard(
            blackboard,
            orientation_weight_key,
            weight_orientation_path,
            keys_to_not_write_to_blackboard,
        )

    # Add the position path constraint to the child behavior
    if position_path is not None:
        position_path_constaint_name = Blackboard.separator.join(
            [name, position_path_constraint_namespace_prefix]
        )
        position_constraint = SetPositionPathConstraint(
            position_path_constaint_name, child, node
        )
    else:
        position_constraint = child

    # Add the orientation goal constraint to the position constriant behavior
    if quat_xyzw_path is not None:
        orientation_goal_constaint_name = Blackboard.separator.join(
            [name, orientation_path_constraint_namespace_prefix]
        )
        orientation_constraint = SetOrientationPathConstraint(
            orientation_goal_constaint_name, position_constraint, node
        )
    else:
        orientation_constraint = position_constraint

    # Clear any previous constraints that may be in the MoveIt2 object
    if clear_constraints:
        clear_constraints_name = Blackboard.separator.join(
            [name, clear_constraints_namespace_prefix]
        )
        clear_constraints_behavior = ClearConstraints(
            clear_constraints_name, orientation_constraint, node
        )
    else:
        clear_constraints_behavior = orientation_constraint

    return clear_constraints_behavior
