#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ToggleCollisionObject behavior, which (dis)allows
collisions between the robot and a collision object already in
MoveIt's planning scene.
"""

# Standard imports

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.helpers import get_moveit2_object


class ToggleCollisionObject(py_trees.behaviour.Behaviour):
    """
    ToggleCollisionObject is a behavior that (dis)allows collisions
    between the robot and a collision object already in MoveIt's planning
    scene.
    """

    def __init__(
        self,
        name: str,
        node: Node,
        collision_object_id: str,
        allow: bool,
    ) -> None:
        """
        Initializes the behavior.

        Parameters
        ----------
        name: The name of the behavior.
        node: The ROS node to associate the publishers with.
        collision_object_id: The ID for the collision object in the MoveIt planning scene.
        allow: If True, then collisions between the robot and collision object ID are
            *allowed* e.g., a plan that collides with the object will be considered valid.
            If False, then collisions between the robot and collision object ID are
            *disallowed* e.g., a plan that collides with the object will be considered invalid.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.collision_object_id = collision_object_id
        self.allow = allow

        # Get the MoveIt2 object.
        self.blackboard = self.attach_blackboard_client(
            name=name + " ToggleCollisionObject", namespace=name
        )
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    def update(self) -> py_trees.common.Status:
        """
        (Dis)allow collisions between the robot and a collision
        object already in MoveIt's planning scene.

        Returns
        -------
        status: The status of the behavior.
        """
        self.logger.info(f"{self.name} [ToggleCollisionObject::update()]")
        # (Dis)allow collisions between the robot and the collision object
        with self.moveit2_lock:
            succ = self.moveit2.allow_collisions(self.collision_object_id, self.allow)

        # Return success
        if succ:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
