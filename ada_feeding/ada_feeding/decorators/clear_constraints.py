#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ClearConstraints decorator, which clears all constraints
on the MoveIt2 object.
"""
# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_moveit2_object

# pylint: disable=duplicate-code
# All the constraints have similar code when registering and setting blackboard
# keys, since the parameters for constraints are similar. This is not a problem.


class ClearConstraints(MoveToConstraint):
    """
    ClearConstraints clears all constraints on the MoveIt2 object. This
    Should be at the top of a branch of constraints and the MoveTo behavior,
    in case any previous behavior left lingering constraints (e.g., due to
    an error).
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
        node: Node,
    ):
        """
        Initialize the MoveToConstraint decorator.

        Parameters
        ----------
        name: The name of the behavior.
        child: The child behavior.
        """
        # Initiatilize the decorator
        super().__init__(name=name, child=child)

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " ClearConstraints", namespace=name
        )

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            node,
        )

    def set_constraint(self) -> None:
        """
        Sets the joint goal constraint.
        """
        self.logger.info(f"{self.name} [ClearConstraints::set_constraint()]")

        # Set the constraint
        with self.moveit2_lock:
            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()
