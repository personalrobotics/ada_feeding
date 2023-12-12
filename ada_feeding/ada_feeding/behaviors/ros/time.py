#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines utility behaviors related to ROS time.
"""

# Standard imports
from typing import Optional, Union

# Third-party imports
from overrides import override
import py_trees

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class TrackHz(BlackboardBehavior):
    """
    The TrackHz behavior keeps track of the frequency with which it is ticked.
    It is often used within sequences/selectors without memory to track the
    # tick rate. It always returns SUCCESS.

    Note: This node tracks the rate ever since it was initialized. To trigger
    re-initialization, use the UnsetBlackboardVariable behavior.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        num_ticks: BlackboardKey,  # int
        start_time: BlackboardKey,  # float, secs
        default_hz: Union[BlackboardKey, int] = 0,  # int
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        num_ticks: Num times this behavior has been ticked since initialization.
        start_time: Time when this behavior was initialized.
        hz: Returned when the behavior is initialized, until the behavior has
            computed an empirical rate.
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

        # Initialize the node on the first tick
        if not self.blackboard_exists(["num_ticks", "start_time"]):
            self.logger.debug("Initializing TrackHz")

            # Initialize num_ticks and start_time
            self.blackboard_set("num_ticks", 1)
            self.blackboard_set(
                "start_time", self.node.get_clock().now().nanoseconds / 10.0**9
            )

            # Return the default_hz
            default_hz = 0
            if self.blackboard_exists(["default_hz"]):
                default_hz = self.blackboard_get("default_hz")
            self.blackboard_set("hz", default_hz)
            return py_trees.common.Status.SUCCESS

        # Update num_ticks
        num_ticks = self.blackboard_get("num_ticks")
        self.blackboard_set("num_ticks", num_ticks + 1)

        # Get the elapsed time
        start_time = self.blackboard_get("start_time")
        elapsed_time = self.node.get_clock().now().nanoseconds / 10.0**9 - start_time

        # Set the frequency
        hz = num_ticks / elapsed_time
        self.logger.debug(f"TrackHz rate: {hz} hz")
        self.blackboard_set("hz", hz)

        # Return SUCCESS
        return py_trees.common.Status.SUCCESS
