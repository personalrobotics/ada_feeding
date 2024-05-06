#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ServoMove behavior, which publishes
a Twist to the Moveit2 Servo node.
"""

# Standard imports
from enum import Enum
from threading import Lock
from typing import Optional, Union

# Third-party imports
from overrides import override
from geometry_msgs.msg import Twist, TwistStamped
import py_trees
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from std_msgs.msg import Int8

# Local imports
from ada_feeding.helpers import BlackboardKey, float_to_duration
from ada_feeding.behaviors import BlackboardBehavior


class ServoMove(BlackboardBehavior):
    """
    Publishes a Twist (or TwistStamped) to the
    MoveIt2Servo object for specified duration or until it receives a terminating
    status code.
    """

    # TODO: Consider having this node fail if the F/T threshold is exceeded.
    # This would involve having the controller publish a status when the F/T
    # threshold is exceeded, and then having this node subscribe to that status
    # and fail if it is exceeded.

    # See here for servo status codes:
    # https://github.com/ros-planning/moveit2/blob/3144e6eb555d6265ecd1240d9932122a8f78290a/moveit_ros/moveit_servo/include/moveit_servo/status_codes.h#L46
    class ServoStatus(Enum):
        """
        Enum for servo status.
        """

        INVALID = -1
        NO_WARNING = 0
        DECELERATE_FOR_APPROACHING_SINGULARITY = 1
        HALT_FOR_SINGULARITY = 2
        DECELERATE_FOR_COLLISION = 3
        HALT_FOR_COLLISION = 4
        JOINT_BOUND = 5
        DECELERATE_FOR_LEAVING_SINGULARITY = 6

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
        default_frame_id: Union[BlackboardKey, str] = "root",
        status_on_timeout: Union[
            BlackboardKey, py_trees.common.Status
        ] = py_trees.common.Status.SUCCESS,
        fail_near_collision: Union[BlackboardKey, bool] = True,
        fail_on_collision: Union[BlackboardKey, bool] = True,
        fail_on_singularity: Union[BlackboardKey, bool] = True,
        servo_status_sub_topic: Union[BlackboardKey, Optional[str]] = "~/servo_status",
        servo_status_sub_qos: Union[BlackboardKey, QoSProfile] = QoSProfile(depth=1),
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
        status_on_timeout: What status to return if once duration is reached.
        fail_near_collision: Whether to fail if near a collision.
        fail_on_collision: Whether to fail if in collision.
        fail_on_singularity: Whether to fail if in a singularity.
        servo_status_sub_topic: Topic to subscribe to for servo status. If None,
            don't subscribe to the servo status
        servo_status_sub_qos: QoS for servo status subscriber.
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

        # Create publisher
        self.pub = self.node.create_publisher(
            TwistStamped,
            self.blackboard_get("pub_topic"),
            self.blackboard_get("pub_qos"),
        )

        # Subscribe to the servo status.
        if self.blackboard_get("servo_status_sub_topic") is not None:
            self.latest_servo_status_lock = Lock()
            self.latest_servo_status = 0
            self.fail_on_servo_status = {}
            if self.blackboard_exists(
                [
                    "fail_near_collision",
                    "fail_on_collision",
                    "fail_on_singularity",
                ]
            ) and (
                self.blackboard_get("fail_near_collision")
                or self.blackboard_get("fail_on_collision")
                or self.blackboard_get("fail_on_singularity")
            ):
                # Create the subscriber
                self.servo_status_sub = self.node.create_subscription(
                    Int8,
                    self.blackboard_get("servo_status_sub_topic"),
                    self.servo_status_callback,
                    self.blackboard_get("servo_status_sub_qos"),
                )

                # Set the fail_on_servo_status
                if self.blackboard_get("fail_on_singularity"):
                    code = ServoMove.ServoStatus.HALT_FOR_SINGULARITY.value
                    name = ServoMove.ServoStatus.HALT_FOR_SINGULARITY.name
                    self.fail_on_servo_status[code] = name
                if self.blackboard_get("fail_near_collision"):
                    code = ServoMove.ServoStatus.DECELERATE_FOR_COLLISION.value
                    name = ServoMove.ServoStatus.DECELERATE_FOR_COLLISION.name
                    self.fail_on_servo_status[code] = name
                if self.blackboard_get("fail_on_collision"):
                    code = ServoMove.ServoStatus.HALT_FOR_COLLISION.value
                    name = ServoMove.ServoStatus.HALT_FOR_COLLISION.name
                    self.fail_on_servo_status[code] = name

    @override
    def initialise(self) -> None:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Record start time
        self.start_time = self.node.get_clock().now()

    def servo_status_callback(self, msg: Int8) -> None:
        """
        Callback for servo status subscriber.

        Parameters
        ----------
        msg: The servo status message.
        """
        with self.latest_servo_status_lock:
            # pylint: disable=attribute-defined-outside-init
            self.latest_servo_status = msg.data

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init, too-many-return-statements
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Validate inputs
        if not self.blackboard_exists(["twist", "duration", "status_on_timeout"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        # Check if servo status is in a fail_on_servo_status
        if self.blackboard_get("servo_status_sub_topic") is not None:
            with self.latest_servo_status_lock:
                if self.latest_servo_status in self.fail_on_servo_status:
                    self.logger.error(
                        f"Received servo status {self.fail_on_servo_status[self.latest_servo_status]}. Failing."
                    )
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

        # Return success if duration is exceeded. If duration is negative, then
        # run forever
        if isinstance(duration, float):
            duration = float_to_duration(duration)
        if duration.nanoseconds >= 0 and self.node.get_clock().now() > (
            self.start_time + duration
        ):
            status = self.blackboard_get("status_on_timeout")
            if status == py_trees.common.Status.FAILURE:
                self.logger.error("ServoMove timed out.")
            return status

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
