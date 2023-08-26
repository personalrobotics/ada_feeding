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
        reverse_position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
            on the blackboard.
        collision_object_orientation_input_key: The key for the collision object orientation
            input on the blackboard.
        reverse_position_offset: The offset to *subtract from* to the collision object position.
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
        self.reverse_position_offset = reverse_position_offset

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

        # Create the publisher
        self.publisher = self.node.create_publisher(
            CollisionObject,
            "/collision_object",
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=1),
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
        # Get the pose to move the collision object to
        try:
            collision_object_position = self.blackboard.get(
                self.collision_object_position_input_key
            )
            collision_object_orientation = self.blackboard.get(
                self.collision_object_orientation_input_key
            )
        except KeyError:
            # If the collision object pose is not on the blackboard, return failure
            self.logger.error(
                "The collision object pose is not on the blackboard. "
                "Returning failure."
            )
            return py_trees.common.Status.FAILURE

        # Create the collision object
        collision_object = CollisionObject()
        collision_object.id = self.collision_object_id
        collision_object.header.frame_id = self.blackboard.frame_id
        pose = Pose()
        pose.position.x = collision_object_position[0] - self.reverse_position_offset[0]
        pose.position.y = collision_object_position[1] - self.reverse_position_offset[1]
        pose.position.z = collision_object_position[2] - self.reverse_position_offset[2]
        pose.orientation.x = collision_object_orientation[0]
        pose.orientation.y = collision_object_orientation[1]
        pose.orientation.z = collision_object_orientation[2]
        pose.orientation.w = collision_object_orientation[3]
        collision_object.pose = pose
        collision_object.operation = CollisionObject.MOVE

        # Publish the collision object
        self.publisher.publish(collision_object)

        # Return success
        return py_trees.common.Status.SUCCESS
