#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ToggleCollisionObject behavior, which (dis)allows
collisions between the robot and a collision object already in
MoveIt's planning scene.
"""

# Standard imports
from typing import List, Union

# Third-party imports
from overrides import override
import py_trees

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
)


class ToggleCollisionObject(BlackboardBehavior):
    """
    ToggleCollisionObject is a behavior that (dis)allows collisions
    between the robot and a collision object already in MoveIt's planning
    scene.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        collision_object_ids: Union[BlackboardKey, List[str]],
        allow: Union[BlackboardKey, bool],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        collision_object_ids: The IDs for the collision objects in the MoveIt planning scene.
        allow: If True, then collisions between the robot and collision object ID are
            *allowed* e.g., a plan that collides with the object will be considered valid.
            If False, then collisions between the robot and collision object ID are
            *disallowed* e.g., a plan that collides with the object will be considered invalid.
        """
        # pylint: disable=unused-argument, duplicate-code
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

        self.logger.info(f"{self.name} [ToggleCollisionObject::setup()]")

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    @override
    def initialise(self) -> None:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        self.logger.info(f"{self.name} [ToggleCollisionObject::initialise()]")

        self.service_future = None
        self.curr_collision_object_id_i = 0
        self.all_collision_object_succ = True

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # The attributes pylint thinks were "defined" here
        # are actually defined in initialise, which is okay.

        collision_object_ids = self.blackboard_get("collision_object_ids")

        self.logger.info(f"{self.name} [ToggleCollisionObject::update()]")
        # (Dis)allow collisions between the robot and the collision object
        if self.service_future is None:
            # Check if we have processed all collision objects
            if self.curr_collision_object_id_i >= len(collision_object_ids):
                # We have processed all collision objects
                if self.all_collision_object_succ:
                    # We have processed all collision objects and they were all successful
                    return py_trees.common.Status.SUCCESS
                return py_trees.common.Status.FAILURE

            # Get the next collision object ID
            collision_object_id = collision_object_ids[self.curr_collision_object_id_i]
            self.logger.info(
                f"{self.name} [ToggleCollisionObject::update()] "
                f"collision_object_id: {collision_object_id}"
            )
            with self.moveit2_lock:
                service_future = self.moveit2.allow_collisions(
                    collision_object_id, self.blackboard_get("allow")
                )
            if service_future is None:  # service not available
                self.all_collision_object_succ = False
                self.curr_collision_object_id_i += 1
                return py_trees.common.Status.RUNNING
            self.service_future = service_future
            return py_trees.common.Status.RUNNING

        # Check if the service future is done
        if self.service_future.done():
            with self.moveit2_lock:
                succ = self.moveit2.process_allow_collision_future(self.service_future)
            # Process success/failure
            self.logger.info(
                f"{self.name} [ToggleCollisionObject::update()] "
                f"service_future: {succ}"
            )
            if not succ:
                self.all_collision_object_succ = False
            self.curr_collision_object_id_i += 1
            self.service_future = None
            return py_trees.common.Status.RUNNING

        # Return running
        return py_trees.common.Status.RUNNING
