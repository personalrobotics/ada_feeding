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

# Local imports


class MoveToConstraint(py_trees.decorators.Decorator, ABC):
    """
    An abstract decorator to add constraints to any behavior that moves
    the robot using MoveIt2.
    """

    def initialise(self) -> None:
        """
        Set the constraints before the child behavior starts executing.
        """
        self.logger.info(f"{self.name} [MoveToConstraint::initialise()]")

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
