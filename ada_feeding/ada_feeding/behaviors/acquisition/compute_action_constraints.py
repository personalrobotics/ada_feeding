#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines behaviors that take an AcquisitionSchema.msg object
(or return from AcquisitionSelect.srv) and computes the outputs
needed to send MoveIt2Plan calls.
"""

# Standard imports
from copy import deepcopy
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Point, Pose
import numpy as np
from overrides import override
import py_trees
from rcl_interfaces.srv import SetParameters
import ros2_numpy

# Local imports
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.helpers import (
    BlackboardKey,
    set_static_tf,
)
from ada_feeding.behaviors import BlackboardBehavior


class ComputeApproachConstraints(BlackboardBehavior):
    """
    Checks AcquisitionSelect response, implements stochastic
    policy choice, and then decomposes into individual
    BlackboardKey objects for MoveIt2 Behaviors.
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
        action_select_response: Union[BlackboardKey, AcquisitionSelect.Response],
        move_above_dist_m: Union[BlackboardKey, float] = 0.05,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        action_select_response: response from AcquisitionSelect.srv
        move_above_dist_m: how far from the food to start
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        move_above_pose: Optional[BlackboardKey],  # Pose, in Food Frame
        move_into_pose: Optional[BlackboardKey],  # Pose, in Food Frame
        ft_thresh: Optional[BlackboardKey],  # SetParameters.Request
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        move_above_pose: Pose constraint when moving above food
        move_into_pose: Pose constraint when moving into food
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

    @staticmethod
    def create_ft_thresh_request(f_mag: float, t_mag: float) -> SetParameters.Request:
        parameters = []
        for key, val in [
            ("fMag", f_mag),
            ("fx", 0.0),
            ("fy", 0.0),
            ("fz", 0.0),
            ("tMag", t_mag),
            ("tx", 0.0),
            ("ty", 0.0),
            ("tz", 0.0),
        ]:
            parameters.append(
                Parameter(
                    name=f"wrench_threshold.{key}",
                    value=ParameterValue(
                        type=ParameterType.PARAMETER_DOUBLE, double_value=val
                    ),
                )
            )
        return SetParameters.Request(parameters=parameters)

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists("action_select_response"):
            self.logger.error("Missing Action Select Response")
            return py_trees.common.Status.FAILURE
        response = self.blackboard_get("action_select_response")
        if response.status != "Success":
            self.logger.error(f"Bad Action Select Response: {response.status}")
            return py_trees.common.Status.FAILURE
        prob = np.array(response.probabilities)
        if (
            len(response.actions) == 0
            or len(response.actions) != len(response.probabilities)
            or not np.isclose(np.sum(prob), 1.0)
        ):
            self.logger.error(f"Malformed action select response: {response}")
            return py_trees.common.Status.FAILURE

        # Sample Action
        index = np.random.choice(np.arange(prob.size), p=prob)
        action = response.actions[index]

        ### Calculate Approach Frame
        # TODO: Calculate Approach Frame
        identity = TransformStamped()
        identity.transform = ros2_numpy.msgify(Transform, np.eye(4))
        identity.header.stamp = self.node.get_clock().now()
        identity.header.frame_id = "food"
        identity.child_frame_id = "approach"
        set_static_tf(identity, self.blackboard, self.node)

        ### Construct Constraints

        # Re-scale pre-transform to move_above_dist_m
        position = ros2_numpy.numpify(action.pre_transform.position)
        if np.isclose(np.linalg.norm(position), 0.0):
            self.logger.error(
                f"Malformed action pre_transform: {action.pre_transform.position}"
            )
            return py_trees.common.Status.FAILURE
        position = (
            position
            * self.blackboard_get("move_above_dist_m")
            / np.linalg.norm(position)
        )
        action.pre_transform.position = ros2_numpy.msgify(Point, position)
        self.blackboard_set("move_above_pose", action.pre_transform)

        # Calculate Approach Target (in food frame)
        move_into_pose = Pose()
        move_into_pose.orientation = deepcopy(action.pre_transform.orientation)
        offset = ros2_numpy.numpify(action.pre_offset)
        move_into_pose.position = ros2_numpy.msgify(Point, offset)
        self.blackboard_set("move_into_pose", move_into_pose)

        # Calculate Approach F/T Thresholds
        self.blackboard_set(
            "approach_ft_thresh",
            ComputeActionConstraints.create_ft_thresh_request(
                action.pre_force, aciton.pre_torque
            ),
        )

        return py_trees.common.Status.SUCCESS
