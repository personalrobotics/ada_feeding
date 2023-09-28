#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeMoveAbove behavior, which computes the
food frame from 
"""
# Standard imports
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import TransformStamped
import numpy as np
import py_trees
import rclpy
from rclpy.node import Node
import tf2_ros

# Local imports
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ComputeMoveAbove(BlackboardBehavior):
    """
    (1) Selects an action from AcquisitionSelect service response.
    (2) Computes the MoveAbove Pose in the World Frame
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    # pylint: disable=too-many-arguments
    # These are effectively config definitions
    # They require a lot of arguments.

    def blackboard_inputs(
        self,
        world_to_food: Union[BlackboardKey, TransformStamped],
        action_select_response: Union[BlackboardKey, AcquisitionSelect.Response],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        world_to_food (geometry_msgs/TransformStamped): transform from world_frame to food_frame
        action_select_request (AcquisitionSelect.Response): response received from AcquisitionSelect
        """
        # pylint: disable=unused-argument
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        action: Optional[BlackboardKey],  # AcquisitionSchema
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        TODO
        """
        # pylint: disable=unused-argument
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """
        Middleware (i.e. TF) setup
        """

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # TODO
        pass

    def initialise(self):
        """
        Behavior initialization
        """

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # TODO
        pass

    def update(self) -> py_trees.common.Status:
        """
        Behavior tick (DO NOT BLOCK)
        """
        # pylint: disable=too-many-locals
        # I think this is reasonable to understand
        # the logic of this function.

        # pylint: disable=too-many-statements
        # We can't get around all the conversions
        # to ROS2 msg types, which take 3-4 statements each.

        # TODO

        return py_trees.common.Status.SUCCESS
