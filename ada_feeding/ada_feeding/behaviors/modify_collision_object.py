#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ModifyCollisionObject behavior, which adds, moves, or
removes a collision object in MoveIt's planning scene.
"""
# Standard imports
from enum import Enum
from typing import List, Optional, Tuple

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.helpers import get_moveit2_object


class ModifyCollisionObjectOperation(Enum):
    """
    An enum for the operation to perform on the collision object.
    """

    ADD = 0
    REMOVE = 1
    MOVE = 2


class ModifyCollisionObject(py_trees.behaviour.Behaviour):
    """
    A behavior that adds, moves, or removes a collision object.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # A few over is fine. All are necessary.

    def __init__(
        self,
        name: str,
        node: Node,
        operation: ModifyCollisionObjectOperation,
        collision_object_id: str,
        collision_object_position_input_key: Optional[str] = None,
        collision_object_orientation_input_key: Optional[str] = None,
        prim_type: Optional[int] = None,
        dims: Optional[List[float]] = None,
        mesh_filepath: Optional[str] = None,
        position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initializes the modify collision object behavior.

        Parameters
        ----------
        name: The name of the behavior.
        node: The ROS node to associate the publishes with.
        operation: The operation to perform on the collision object (ADD, REMOVE, MOVE).
        collision_object_id: The ID for the collision object in the MoveIt planning scene.
        collision_object_position_input_key: The key for the collision object pose input
            on the blackboard. Required for ADD and MOVE. This key should contain a list
            of size 3.
        collision_object_orientation_input_key: The key for the collision object orientation
            input on the blackboard. Required for ADD and MOVE. This key should contain a
            list of size 4.
        prim_type: The type of primitive to add. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD. For `prim_type` options, see:
            https://github.com/ros2/common_interfaces/blob/humble/shape_msgs/msg/SolidPrimitive.msg
        dims: The dimensions of the collision object. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD. For `dims` details, see:
            https://github.com/ros2/common_interfaces/blob/humble/shape_msgs/msg/SolidPrimitive.msg
        mesh_filepath: The filepath to the mesh to add. Either `prim_type` *and* `dims`, *or*
            `mesh_filepath` are required for ADD.
        position_offset: The offset to *add to* to the collision object position.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.operation = operation
        self.collision_object_id = collision_object_id
        self.collision_object_position_input_key = collision_object_position_input_key
        self.collision_object_orientation_input_key = (
            collision_object_orientation_input_key
        )
        self.prim_type = prim_type
        self.dims = dims
        self.mesh_filepath = mesh_filepath
        self.position_offset = position_offset

        # Check that the right parameters have been passed
        if self.operation in set(
            (ModifyCollisionObjectOperation.ADD, ModifyCollisionObjectOperation.MOVE)
        ):
            if self.collision_object_position_input_key is None:
                raise ValueError(
                    "The collision object position input key must be specified for ADD "
                    "and MOVE operations."
                )
            if self.collision_object_orientation_input_key is None:
                raise ValueError(
                    "The collision object orientation input key must be specified for "
                    "ADD and MOVE operations."
                )
        if self.operation == ModifyCollisionObjectOperation.ADD:
            if self.mesh_filepath is None and (
                self.prim_type is None or self.dims is None
            ):
                raise ValueError(
                    "If `mesh_filepath` is None, then both `prim_type` and `dims` "
                    "must be specified."
                )

        # Initialize the blackboard for this behavior
        self.blackboard = self.attach_blackboard_client(
            name=name + " ModifyCollisionObject", namespace=name
        )
        # Read the position to move the collision object to
        if self.collision_object_position_input_key is not None:
            self.blackboard.register_key(
                key=self.collision_object_position_input_key,
                access=py_trees.common.Access.READ,
            )
        # Read the orientation to move the collision object to
        if self.collision_object_orientation_input_key is not None:
            self.blackboard.register_key(
                key=self.collision_object_orientation_input_key,
                access=py_trees.common.Access.READ,
            )
        # Read the frame ID for the pose to move the collision object to
        if self.collision_object_position_input_key is not None:
            self.blackboard.register_key(
                key="frame_id",
                access=py_trees.common.Access.READ,
            )

        # Get the MoveIt2 interface
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    def update(self) -> py_trees.common.Status:
        """
        Adds, moves, or removes the collision object, as specified by the
        operation. If the operation is to add or move, it gets the latest
        pose from the blackboard. This behavior does not wait for the message
        to be processed by MoveIt before returning success.
        """
        self.logger.info(f"{self.name} [ModifyCollisionObject::update()]")

        # Remove the collision object
        if self.operation == ModifyCollisionObjectOperation.REMOVE:
            with self.moveit2_lock:
                self.moveit2.remove_collision(self.collision_object_id)
            return py_trees.common.Status.SUCCESS

        # If we are adding/moving, first get the pose to move the collision object to
        try:
            collision_object_position = self.blackboard.get(
                self.collision_object_position_input_key
            )
            collision_object_orientation = self.blackboard.get(
                self.collision_object_orientation_input_key
            )
            frame_id = self.blackboard.get("frame_id")
        except KeyError:
            # If the collision object pose is not on the blackboard, return failure
            self.logger.error(
                "The collision object pose is not on the blackboard. "
                "Returning failure."
            )
            return py_trees.common.Status.FAILURE

        # Add the position offset
        collision_object_position = [
            collision_object_position[0] + self.position_offset[0],
            collision_object_position[1] + self.position_offset[1],
            collision_object_position[2] + self.position_offset[2],
        ]

        # Move the collision object
        with self.moveit2_lock:
            if self.operation == ModifyCollisionObjectOperation.ADD:
                # Add the collision object
                if self.mesh_filepath is not None:
                    self.moveit2.add_collision_mesh(
                        self.mesh_filepath,
                        self.collision_object_id,
                        collision_object_position,
                        collision_object_orientation,
                        frame_id=frame_id,
                    )
                else:
                    self.moveit2.add_collision_primitive(
                        self.collision_object_id,
                        self.dims,
                        collision_object_position,
                        collision_object_orientation,
                        self.prim_type,
                        frame_id=frame_id,
                    )
            elif self.operation == ModifyCollisionObjectOperation.MOVE:
                self.moveit2.move_collision(
                    self.collision_object_id,
                    collision_object_position,
                    collision_object_orientation,
                    frame_id,
                )

        # Return success
        return py_trees.common.Status.SUCCESS
