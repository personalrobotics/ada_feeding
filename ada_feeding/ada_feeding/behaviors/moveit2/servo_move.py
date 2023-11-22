#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ServoMove behavior, which publishes
a Twist to the Moveit2 Servo node.
"""

# Standard imports
from typing import Optional, Union

# Third-party imports
from overrides import override
from geometry_msgs.msg import Twist, TwistStamped
import py_trees
from rclpy.duration import Duration
from rclpy.qos import QoSProfile

# Local imports
from ada_feeding.helpers import BlackboardKey, duration_to_float, float_to_duration
from ada_feeding.behaviors import BlackboardBehavior


class ServoMove(BlackboardBehavior):
    """
    Publishes a Twist (or TwistStamped) to the
    MoveIt2Servo object for specified duration.
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
        twist: Union[BlackboardKey, Twist, TwistStamped],
        duration: Union[BlackboardKey, Duration, float] = Duration(seconds=1.0),
        pub_topic: Union[BlackboardKey, str] = "~/servo_twist_cmds",
        pub_qos: Union[BlackboardKey, QoSProfile] = QoSProfile(depth=1),
        default_frame_id: Union[BlackboardKey, str] = "world",
        curr_distance: Optional[BlackboardKey] = None,  # float
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        twist: Twist or TwistStamped to publish, with updated time stamps.
        duration: How long to publish the twist. If negative, then run forever.
        pub_topic: Where to publish servo TwistStamped messages
        pub_qos: QoS for publisher
        default_frame_id: frame_id to use if Twist type is provided.
        curr_distance: If None, use time as distance. Otherwise, this
            BlackboardKey stores the remaining distance.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
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
    def initialise(self) -> None:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Create publisher
        self.pub = self.node.create_publisher(
            TwistStamped,
            self.blackboard_get("pub_topic"),
            self.blackboard_get("pub_qos"),
        )

        # Record start time
        self.first_tick = True
        self.start_time = self.node.get_clock().now()
        self.initial_distance = 0.0
        self.curr_distance = 0.0

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init, too-many-return-statements
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Validate inputs
        if not self.blackboard_exists(["twist", "duration"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        # Publish twist
        twist = self.blackboard_get("twist")
        if isinstance(twist, Twist):
            twist = TwistStamped()
            twist.header.frame_id = self.blackboard_get("default_frame_id")
            twist.twist = self.blackboard_get("twist")
        twist.header.stamp = self.node.get_clock().now().to_msg()
        self.pub.publish(twist)

        # Write the remaining distance
        duration = self.blackboard_get("duration")
        self.curr_distance = self.blackboard_get("curr_distance")
        if self.curr_distance is None:
            self.curr_distance = duration_to_float(duration) - duration_to_float(
                self.node.get_clock().now() - self.start_time
            )
        if self.first_tick:
            self.initial_distance = self.curr_distance
            self.first_tick = False

        # Return success if duration is exceeded. If duration is negative, then
        # run forever
        if isinstance(duration, float):
            duration = float_to_duration(duration)
        if duration.nanoseconds >= 0 and self.node.get_clock().now() > (
            self.start_time + duration
        ):
            return py_trees.common.Status.SUCCESS

        # Servo is still executing
        return py_trees.common.Status.RUNNING

    @override
    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Docstring copied from @override

        # Publish Zero Twist
        zero_twist = TwistStamped()
        zero_twist.header.frame_id = (
            self.blackboard_get("twist").header.frame_id
            if isinstance(self.blackboard_get("twist"), TwistStamped)
            else self.blackboard_get("default_frame_id")
        )
        zero_twist.header.stamp = self.node.get_clock().now().to_msg()

        self.pub.publish(zero_twist)
