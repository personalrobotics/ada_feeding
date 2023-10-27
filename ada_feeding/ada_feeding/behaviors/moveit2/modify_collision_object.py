#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ModifyCollisionObject behavior, which adds, moves, or
removes a collision object in MoveIt's planning scene.
"""
# Standard imports
from enum import Enum
from typing import List, Optional, Tuple, Union

# Third-party imports
from geometry_msgs.msg import Point, PointStamped, Quaternion, QuaternionStamped
from overrides import override
import py_trees

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
)


class ModifyCollisionObjectOperation(Enum):
    """
    An enum for the operation to perform on the collision object.
    """

    ADD = 0
    REMOVE = 1
    MOVE = 2


class ModifyCollisionObject(BlackboardBehavior):
    """
    A behavior that adds, moves, or removes a collision object. Note that this
    behavior does not wait for the collision object to be added, moved, or
    removed before returning SUCCESS/FAILURE.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        operation: Union[BlackboardKey, ModifyCollisionObjectOperation],
        collision_object_id: Union[BlackboardKey, str],
        collision_object_position: Union[
            BlackboardKey,
            Optional[Union[PointStamped, Point, Tuple[float, float, float]]],
        ] = None,
        collision_object_orientation: Union[
            BlackboardKey,
            Optional[
                Union[QuaternionStamped, Quaternion, Tuple[float, float, float, float]]
            ],
        ] = None,
        prim_type: Union[BlackboardKey, Optional[int]] = None,
        dims: Union[BlackboardKey, Optional[List[float]]] = None,
        mesh_filepath: Union[BlackboardKey, Optional[str]] = None,
        frame_id: Union[BlackboardKey, Optional[str]] = None,
        position_offset: Union[BlackboardKey, Tuple[float, float, float]] = (
            0.0,
            0.0,
            0.0,
        ),
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        operation: The operation to perform on the collision object (ADD, REMOVE, MOVE).
        collision_object_id: The ID for the collision object in the MoveIt planning scene.
        collision_object_position: The position to move the collision object to. Required
            for ADD and MOVE. Its frame_id must match `collision_object_orientation`.
        collision_object_orientation: The orientation to move the collision object to.
            Required for ADD and MOVE. Its frame_id must match `collision_object_position`.
        prim_type: The type of primitive to add. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD. For `prim_type` options, see:
            https://github.com/ros2/common_interfaces/blob/humble/shape_msgs/msg/SolidPrimitive.msg
        dims: The dimensions of the collision object. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD. For `dims` details, see:
            https://github.com/ros2/common_interfaces/blob/humble/shape_msgs/msg/SolidPrimitive.msg
        mesh_filepath: The filepath to the mesh to add. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD.
        frame_id: The frame ID for the collision object pose. Only used if the operation is
            ADD or MOVE and the collision object position/orientation do not have a frame_id.
        position_offset: The offset to apply to the collision object position. Only used if
            the operation is ADD or MOVE.
        """
        # pylint: disable=unused-argument, duplicate-code, too-many-arguments
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        self.logger.info(f"{self.name} [ModifyCollisionObject::setup()]")

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    @override
    def initialise(self):
        # Docstring copied from @override

        self.logger.info(f"{self.name} [ModifyCollisionObject::initialise()]")

        # Check that the right parameters have been passed
        operation = self.blackboard_get("operation")
        if operation in set(
            (ModifyCollisionObjectOperation.ADD, ModifyCollisionObjectOperation.MOVE)
        ):
            collision_object_position = self.blackboard_get("collision_object_position")
            collision_object_orientation = self.blackboard_get(
                "collision_object_orientation"
            )
            if collision_object_position is None:
                raise ValueError(
                    "The collision object position input key must be specified for ADD "
                    "and MOVE operations."
                )
            if collision_object_orientation is None:
                raise ValueError(
                    "The collision object orientation input key must be specified for "
                    "ADD and MOVE operations."
                )
        if operation == ModifyCollisionObjectOperation.ADD:
            mesh_filepath = self.blackboard_get("mesh_filepath")
            prim_type = self.blackboard_get("prim_type")
            dims = self.blackboard_get("dims")
            if mesh_filepath is None and (prim_type is None or dims is None):
                raise ValueError(
                    "If `mesh_filepath` is None, then both `prim_type` and `dims` "
                    "must be specified."
                )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        self.logger.info(f"{self.name} [ModifyCollisionObject::update()]")

        # Get the blackboard inputs for all operations
        operation = self.blackboard_get("operation")
        collision_object_id = self.blackboard_get("collision_object_id")

        # Remove the collision object
        if operation == ModifyCollisionObjectOperation.REMOVE:
            with self.moveit2_lock:
                self.moveit2.remove_collision(collision_object_id)
            return py_trees.common.Status.SUCCESS

        # Get the blackboard inputs for ADD and MOVE operations
        collision_object_position = self.blackboard_get("collision_object_position")
        collision_object_orientation = self.blackboard_get(
            "collision_object_orientation"
        )
        frame_id = self.blackboard_get("frame_id")
        mesh_filepath = self.blackboard_get("mesh_filepath")
        prim_type = self.blackboard_get("prim_type")
        dims = self.blackboard_get("dims")
        position_offset = self.blackboard_get("position_offset")

        # Update types
        if isinstance(collision_object_position, PointStamped):
            frame_id = collision_object_position.header.frame_id
            collision_object_position = Point(
                x=collision_object_position.point.x + position_offset[0],
                y=collision_object_position.point.y + position_offset[1],
                z=collision_object_position.point.z + position_offset[2],
            )
        elif isinstance(collision_object_position, tuple):
            collision_object_position = Point(
                x=collision_object_position[0] + position_offset[0],
                y=collision_object_position[1] + position_offset[1],
                z=collision_object_position[2] + position_offset[2],
            )
        if isinstance(collision_object_orientation, QuaternionStamped):
            frame_id = collision_object_orientation.header.frame_id
            collision_object_orientation = collision_object_orientation.quaternion

        # Move the collision object
        with self.moveit2_lock:
            if operation == ModifyCollisionObjectOperation.ADD:
                # Add the collision object
                if mesh_filepath is not None:
                    self.moveit2.add_collision_mesh(
                        mesh_filepath,
                        collision_object_id,
                        collision_object_position,
                        collision_object_orientation,
                        frame_id=frame_id,
                    )
                else:
                    self.moveit2.add_collision_primitive(
                        collision_object_id,
                        dims,
                        collision_object_position,
                        collision_object_orientation,
                        prim_type,
                        frame_id=frame_id,
                    )
            elif operation == ModifyCollisionObjectOperation.MOVE:
                # Move the collision object
                self.moveit2.move_collision(
                    collision_object_id,
                    collision_object_position,
                    collision_object_orientation,
                    frame_id,
                )

        # Return success
        return py_trees.common.Status.SUCCESS
