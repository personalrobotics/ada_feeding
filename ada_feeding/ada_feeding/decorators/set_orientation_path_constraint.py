#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetOrientationPathConstraint decorator, which adds a path
constraint that keeps a specified frame within a secified tolerance of a
specified orientation.
"""
# Third-party imports
import py_trees

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default


class SetOrientationPathConstraint(MoveToConstraint):
    """
    SetOrientationPathConstraint adds a path constraint that keeps a specified frame
    within a secified tolerance of a specified orientation.
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
            name=name + " SetOrientationPathConstraint", namespace=name
        )
        self.blackboard.register_key(
            key="quat_xyzw", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="frame_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="parameterization", access=py_trees.common.Access.READ
        )

    def set_constraint(self) -> None:
        """
        Sets the orientation goal constraint.
        """
        self.logger.info(
            "%s [SetOrientationPathConstraint::set_constraint()]" % self.name
        )

        # Get all parameters for planning, resorting to default values if unset.
        quat_xyzw = self.blackboard.quat_xyzw  # required
        frame_id = get_from_blackboard_with_default(self.blackboard, "frame_id", None)
        target_link = get_from_blackboard_with_default(
            self.blackboard, "target_link", None
        )
        tolerance = get_from_blackboard_with_default(
            self.blackboard, "tolerance", 0.001
        )
        weight = get_from_blackboard_with_default(self.blackboard, "weight", 1.0)
        parameterization = get_from_blackboard_with_default(
            self.blackboard, "parameterization", 0
        )

        # Set the constraint
        self.moveit2.set_path_orientation_constraint(
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            weight=weight,
            parameterization=parameterization,
        )
