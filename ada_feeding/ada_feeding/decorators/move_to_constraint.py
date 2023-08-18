#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConstraint abstract decorator. This decorator
can be added to any behavior that is either of type MoveTo or MoveToConstraint.
This decorator handles the logic for adding constraints (e.g., goal constraints,
path constraints) before the child behavior starts executing, and clearing the
constraints after the child behavior finishes executing.
"""
# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import py_trees
from pymoveit2 import MoveIt2

# Local imports
from ada_feeding.behaviors import MoveTo


class MoveToConstraint(py_trees.decorators.Decorator, ABC):
    """
    An abstract decorator to add constraints to any behavior that moves
    the robot using MoveIt2.
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
    ):
        """
        Initialize the MoveToConstraint decorator.

        Parameters
        ----------
        name: The name of the behavior.
        child: The child behavior.
        """
        # Check the child behavior type
        if not isinstance(child, (MoveTo, MoveToConstraint)):
            raise TypeError(
                "%s [MoveToConstraint::__init__()] Child must be of type MoveTo or MoveToConstraint!"
                % name
            )

        # Initiatilize the decorator
        super().__init__(name=name, child=child)

    @property
    def moveit2(self) -> MoveIt2:
        """
        Get the MoveIt2 interface.
        """
        return self.decorated.moveit2

    def initialise(self) -> None:
        """
        Set the constraints before the child behavior starts executing.
        """
        self.logger.info("%s [MoveToConstraint::initialise()]" % self.name)

        # Set the constraint
        self.set_constraint()

    @abstractmethod
    def set_constraint(self) -> None:
        """
        Sets the constraint. For example, this function can call:
          - self.moveit2.set_joint_goal(...)
          - self.moveit2.set_path_position_constraint(...)
          - and so on.
        """
        raise NotImplementedError("set_constraint not implemented")

    def update(self) -> py_trees.common.Status:
        """
        Just pass through the child's status
        """
        return self.decorated.status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Clear the constraints.
        """
        self.logger.info(
            "%s [MoveToConstraint::terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

        # Clear the constraints
        self.moveit2.clear_goal_constraints()
        self.moveit2.clear_path_constraints()
