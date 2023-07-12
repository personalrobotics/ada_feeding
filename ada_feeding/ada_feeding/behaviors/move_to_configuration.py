#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfiguration behavior tree, which moves the Jaco
arm to a specified joint configuration.
"""
# Standard imports
from asyncio import Future

# Third-party imports
import py_trees

# Local imports
from ada_feeding.behaviors import MoveTo


class MoveToConfiguration(MoveTo):
    """
    A generic behavior for moving the Jaco arm a specified joint configuration.
    """

    def __init__(self, *args, **kwargs):
        """
        A generic behavior for moving the Jaco arm a specified joint configuration.
        """
        # Initiatilize the behavior
        super().__init__(*args, **kwargs)

        # Get inputs specific to MoveToConfiguration from the Blackboard
        self.blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)

    def plan_async(self) -> Future:
        """
        Get the MoveToConfiguration parameters from the blackboard and send an
        asynchronous planning request to MoveIt.
        """
        # Get all parameters for planning, resorting to default values if unset.
        joint_positions = self.blackboard.joint_positions  # required
        try:
            tolerance = self.blackboard.tolerance
        except KeyError:
            tolerance = 0.001  # default value
        try:
            weight = self.blackboard.weight
        except KeyError:
            weight = 1.0  # default value

        # Send a new goal to MoveIt
        return self.moveit2.plan_async(
            joint_positions=joint_positions,
            joint_names=self.joint_names,
            tolerance_joint_position=tolerance,
            weight_joint_position=weight,
            cartesian=self.cartesian,
        )
