#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetJointGoalConstraint decorator, which adds a joint goal to any
behavior that moves the robot using MoveIt2.
"""
# Third-party imports
import py_trees

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default


class SetJointGoalConstraint(MoveToConstraint):
    """
    SetJointGoalConstraint adds joint goal constraints to any behavior that moves
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
        # Initiatilize the decorator
        super().__init__(name=name, child=child)

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " SetJointGoalConstraint", namespace=name
        )
        self.blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="joint_names", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)

    def set_constraint(self) -> None:
        """
        Sets the joint goal constraint.
        """
        self.logger.info("%s [SetJointGoalConstraint::set_constraint()]" % self.name)

        # Get all parameters for planning, resorting to default values if unset.
        joint_positions = self.blackboard.joint_positions  # required
        joint_names = get_from_blackboard_with_default(
            self.blackboard, "joint_names", None
        )
        tolerance = get_from_blackboard_with_default(
            self.blackboard, "tolerance", 0.001
        )
        weight = get_from_blackboard_with_default(self.blackboard, "weight", 1.0)

        # Set the constraint
        self.moveit2.set_joint_goal(
            joint_positions=joint_positions,
            joint_names=joint_names,
            tolerance=tolerance,
            weight=weight,
        )
