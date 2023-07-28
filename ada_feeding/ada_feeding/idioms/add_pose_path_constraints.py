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

# Local imports
from ada_feeding.decorators import (
    SetPositionPathConstraint,
    SetOrientationPathConstraint,
)
from ada_feeding.helpers import (
    POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX,
    ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX,
)


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
# Unfortunately, all these arguments/locals/statements are necessary to add
# pose path constraints to a behavior tree -- the point of putting them
# into one function is to reduce the number of arguments/locals/statements
# needed in other functions.
def add_pose_path_constraints(
    child: py_trees.behaviour.Behaviour,
    name: str,
    blackboard: py_trees.blackboard.Blackboard,
    logger: logging.Logger,
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
    logger: The logger for the tree that this behavior is in.
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
    """

    # Separate blackboard namespaces for decorators
    if position_path is not None:
        position_path_constraint_namespace_prefix = POSITION_PATH_CONSTRAINT_NAMESPACE_PREFIX
    if quat_xyzw_path is not None:
        orientation_path_constraint_namespace_prefix = ORIENTATION_PATH_CONSTRAINT_NAMESPACE_PREFIX

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
        if position_key not in keys_to_not_write_to_blackboard:
            blackboard.set(position_key, position_path)
        if position_frame_id_key not in keys_to_not_write_to_blackboard:
            blackboard.set(position_frame_id_key, frame_id_path)
        if position_target_link_key not in keys_to_not_write_to_blackboard:
            blackboard.set(position_target_link_key, target_link_path)
        if position_tolerance_key not in keys_to_not_write_to_blackboard:
            blackboard.set(position_tolerance_key, tolerance_position_path)
        if position_weight_key not in keys_to_not_write_to_blackboard:
            blackboard.set(position_weight_key, weight_position_path)
    if quat_xyzw_path is not None:
        if orientation_key not in keys_to_not_write_to_blackboard:
            blackboard.set(orientation_key, quat_xyzw_path)
        if orientation_frame_id_key not in keys_to_not_write_to_blackboard:
            blackboard.set(orientation_frame_id_key, frame_id_path)
        if orientation_target_link_key not in keys_to_not_write_to_blackboard:
            blackboard.set(orientation_target_link_key, target_link_path)
        if orientation_tolerance_key not in keys_to_not_write_to_blackboard:
            blackboard.set(orientation_tolerance_key, tolerance_orientation_path)
        if orientation_parameterization_key not in keys_to_not_write_to_blackboard:
            blackboard.set(
                orientation_parameterization_key, parameterization_orientation_path
            )
        if orientation_weight_key not in keys_to_not_write_to_blackboard:
            blackboard.set(orientation_weight_key, weight_orientation_path)

    # Add the position path constraint to the child behavior
    if position_path is not None:
        position_path_constaint_name = Blackboard.separator.join(
            [name, position_path_constraint_namespace_prefix]
        )
        position_constraint = SetPositionPathConstraint(
            position_path_constaint_name, child
        )
        position_constraint.logger = logger
    else:
        position_constraint = child

    # Add the orientation goal constraint to the position constriant behavior
    if quat_xyzw_path is not None:
        orientation_goal_constaint_name = Blackboard.separator.join(
            [name, orientation_path_constraint_namespace_prefix]
        )
        orientation_constraint = SetOrientationPathConstraint(
            orientation_goal_constaint_name, position_constraint
        )
        orientation_constraint.logger = logger
    else:
        orientation_constraint = position_constraint

    return orientation_constraint
