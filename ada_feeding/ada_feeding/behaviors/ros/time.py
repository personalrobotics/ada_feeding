#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines utility behaviors related to ROS time.
"""

# Standard imports
from typing import Optional

# Third-party imports
from overrides import override
import py_trees

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class TrackHzInitialize(BlackboardBehavior):
    """
    This behavior initializes the inputs to a TrackHz behavior.
    The reason we need a separate behavior (as opposed to the initialise function)
    is that we may want to initialise TrackHz from elsewhere in the tree.
    (e.g., initialize before a composite without memory that contains TrackHz).
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_outputs(
        self,
        num_ticks: BlackboardKey = BlackboardKey("num_ticks"),  # int
        start_time: BlackboardKey = BlackboardKey("start_time"),  # float, secs
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        num_ticks: Num times this behavior has been ticked since initialization. Shoulg
            be the same as the input.
        start_time: Time when this behavior was initialized. Should be the same as the input.
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

        # Initialize the number of ticks and start time
        self.blackboard_set("num_ticks", 0)
        self.blackboard_set(
            "start_time", self.node.get_clock().now().nanoseconds / 10.0**9
        )

        # Return SUCCESS
        return py_trees.common.Status.SUCCESS


class TrackHz(BlackboardBehavior):
    """
    The TrackHz behavior keeps track of the frequency with which it is ticked.
    It always returns SUCCESS.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self, num_ticks: BlackboardKey, start_time: BlackboardKey  # int  # float, secs
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        num_ticks: Num times this behavior has been ticked since initialization.
        start_time: Time when this behavior was initialized.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        hz: Optional[BlackboardKey],  # float
        num_ticks: BlackboardKey,  # int
        start_time: BlackboardKey,  # float, secs
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        hz: Frequency with which this behavior has been ticked since initialization.
        num_ticks: Num times this behavior has been ticked since initialization. Shoulg
            be the same as the input.
        start_time: Time when this behavior was initialized. Should be the same as the input.
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
        if not self.blackboard_exists(["num_ticks", "start_time"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        # Validate inputs
        num_ticks = self.blackboard_get("num_ticks")
        start_time = self.blackboard_get("start_time")

        # Update the number of ticks
        self.blackboard_set("num_ticks", num_ticks + 1)

        # Compute the elapsed time
        elapsed_time = self.node.get_clock().now().nanoseconds / 10.0**9 - start_time

        # Compute the frequency
        hz = num_ticks / elapsed_time

        # Set the frequency
        self.blackboard_set("hz", hz)

        # Return SUCCESS
        return py_trees.common.Status.SUCCESS
