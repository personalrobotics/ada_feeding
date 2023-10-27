#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines ROS utility behaviors.
"""

# Standard imports
from typing import Union, Optional, Any

# Third-party imports
from overrides import override
import py_trees
import rclpy

# Local imports
from ada_feeding.helpers import BlackboardKey
from .blackboard_behavior import BlackboardBehavior


class UpdateTimestamp(BlackboardBehavior):
    """
    Adds a custom timestamp (or current timestamp)
    to any stamped message object
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
        stamped_msg: Union[BlackboardKey, Any],
        timestamp: Union[BlackboardKey, Optional[rclpy.time.Time]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        stamped_msg: Any ROS msg with a header
        timestamp: if None, use current time
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        stamped_msg: Optional[BlackboardKey],  # Same type as input
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        stamped_msg: Any ROS msg with a header
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

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists("stamped_msg") or not self.blackboard_exists(
            "timestamp"
        ):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        msg = self.blackboard_get("stamped_msg")
        time = self.blackboard_get("timestamp")
        if time is None:
            time = self.node.get_clock().now()

        try:
            msg.header.stamp = time.to_msg()
        except AttributeError as error:
            self.logger.error(f"Malformed Stamped Message. Error: {error}")
            return py_trees.common.Status.FAILURE

        self.blackboard_set("stamped_msg", msg)
        return py_trees.common.Status.SUCCESS
