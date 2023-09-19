#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveCollisionObject behavior, which moves a collision
object in MoveIt's planning scene to a specified pose.
"""
# Standard imports
from typing import Tuple

# Third-party imports
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
import py_trees
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Local imports
from ada_feeding.helpers import get_moveit2_object


class MoveCollisionObject(py_trees.behaviour.Behaviour):
    """
    A behavior that moves a collision object to a specified pose. This behavior
    assumes that the collision object is already in MoveIt's planning scene and
    consists of a single mesh.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # A few over is fine. All are necessary.

    def __init__(
        self,
        name: str,
        node: Node,
        collision_object_id: str,
        collision_object_position_input_key: str,
        collision_object_orientation_input_key: str,
        position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initializes the behavior. Requires the blackboard to pass it a position
        (3-tuple) and orientation (4-tuple, quaternion) to move the collision
        object to, as well as a frame_id for the pose.

        Parameters
        ----------
        name: The name of the behavior.
        node: The ROS node to associate the publishes with.
        collision_object_id: The ID for the collision object in the MoveIt planning scene.
        collision_object_position_input_key: The key for the collision object pose input
            on the blackboard. This should be a list of size 3.
        collision_object_orientation_input_key: The key for the collision object orientation
            input on the blackboard. This should be a list of size 4.
        position_offset: The offset to *add to* to the collision object position.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.collision_object_id = collision_object_id
        self.collision_object_position_input_key = collision_object_position_input_key
        self.collision_object_orientation_input_key = (
            collision_object_orientation_input_key
        )
        self.position_offset = position_offset

        # Initialize the blackboard for this behavior
        self.blackboard = self.attach_blackboard_client(
            name=name + " MoveCollisionObject", namespace=name
        )
        # Read the position to move the collision object to
        self.blackboard.register_key(
            key=self.collision_object_position_input_key,
            access=py_trees.common.Access.READ,
        )
        # Read the orientation to move the collision object to
        self.blackboard.register_key(
            key=self.collision_object_orientation_input_key,
            access=py_trees.common.Access.READ,
        )
        # Read the frame ID for the pose to move the collision object to
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
        Gets the name of the collision object to move and the pose to move it to
        from the blackboard and publishes a message to MoveIt to move that object
        to that pose. This behavior does not wait for the collision object to be
        moved to the pose before returning success.

        Note that this behavior asusmes that the collision object to be moved
        consists of a single mesh.
        """
        self.logger.info(f"{self.name} [MoveCollisionObject::update()]")
        # Get the pose to move the collision object to
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

        # Add the offset
        collision_object_position = [
            collision_object_position[0] + self.position_offset[0],
            collision_object_position[1] + self.position_offset[1],
            collision_object_position[2] + self.position_offset[2],
        ]

        # Move the collision object
        with self.moveit2_lock:
            self.moveit2.move_collision(
                self.collision_object_id,
                collision_object_position,
                collision_object_orientation,
                frame_id,
            )

        # Return success
        return py_trees.common.Status.SUCCESS
