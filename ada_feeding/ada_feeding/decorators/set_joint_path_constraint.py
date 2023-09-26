#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetJointPathConstraint decorator, which adds a path constraint
to keep specified joints within a specified tolerance of a specified position.
"""
# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default, get_moveit2_object

# pylint: disable=duplicate-code
# All the constraints have similar code when registering and setting blackboard
# keys, since the parameters for constraints are similar. This is not a problem.


class SetJointPathConstraint(MoveToConstraint):
    """
    SetJointPathConstraint adds a path constraint to keep specified joints within a
    specified tolerance of a specified position.
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
        node: The ROS node to associate the service call with.
        """
        # Initiatilize the decorator
        super().__init__(name=name, child=child)

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " SetJointPathConstraint", namespace=name
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

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            node,
        )

    def set_constraint(self) -> None:
        """
        Sets the joint path constraint.
        """
        self.logger.info(f"{self.name} [SetJointPathConstraint::set_constraint()]")

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
        with self.moveit2_lock:
            self.moveit2.set_path_joint_constraint(
                joint_positions=joint_positions,
                joint_names=joint_names,
                tolerance=tolerance,
                weight=weight,
            )
