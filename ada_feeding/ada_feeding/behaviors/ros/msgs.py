#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines utility behaviors for interacting with and updating ROS
messages.
"""

# Standard imports
from typing import Any, Callable, List, Optional, Tuple, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Quaternion,
    QuaternionStamped,
    TwistStamped,
    Vector3,
)
import numpy as np
from overrides import override
import py_trees
import rclpy
from rclpy.duration import Duration
import ros2_numpy
from scipy.spatial.transform import Rotation as R
from tf2_geometry_msgs import PointStamped, PoseStamped

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


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
        timestamp: Union[BlackboardKey, rclpy.time.Time, Duration] = Duration(
            seconds=0.0
        ),
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        stamped_msg: Any ROS msg with a header
        timestamp: If a duration, add it to current time.
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
        if not self.blackboard_exists(["stamped_msg", "timestamp"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        msg = self.blackboard_get("stamped_msg")
        time = self.blackboard_get("timestamp")
        if isinstance(time, Duration):
            time = self.node.get_clock().now() + time

        try:
            msg.header.stamp = time.to_msg()
        except AttributeError as error:
            self.logger.error(f"Malformed Stamped Message. Error: {error}")
            return py_trees.common.Status.FAILURE

        self.blackboard_set("stamped_msg", msg)
        return py_trees.common.Status.SUCCESS


class CreatePoseStamped(BlackboardBehavior):
    """
    Create a PoseStamped from a position (which can be a PointStamped, Point,
    or List[float]) and quaternion (which can be a QuaternionStamped,
    Quaternion, or List[float]).
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        position: Union[
            BlackboardKey,
            PointStamped,
            Point,
            List[float],
            Tuple[float],
        ],
        quaternion: Union[
            BlackboardKey,
            QuaternionStamped,
            Quaternion,
            List[float],
            Tuple[float],
        ],
        frame_id: Union[BlackboardKey, Optional[str]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        position: The position of the pose. Must be in the same frame as the
            quaternion.
        quaternion: The orientation of the pose. Must be in the same frame as
            the position.
        frame_id: The frame of the pose. Only used if neither the position nor
            quaternion are stamped.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        pose_stamped: Optional[BlackboardKey],  # PoseStamped
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        pose_stamped: The pose stamped.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists(["position", "quaternion", "frame_id"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        position = self.blackboard_get("position")
        quaternion = self.blackboard_get("quaternion")
        frame_id = self.blackboard_get("frame_id")

        pose_stamped = PoseStamped()
        got_frame = False
        if isinstance(position, PointStamped):
            pose_stamped.header = position.header
            got_frame = True
            pose_stamped.pose.position = position.point
        elif isinstance(position, Point):
            pose_stamped.pose.position = position
        elif isinstance(position, (list, tuple)):
            pose_stamped.pose.position = Point(
                x=position[0],
                y=position[1],
                z=position[2],
            )
        if isinstance(quaternion, QuaternionStamped):
            if not got_frame:
                pose_stamped.header = quaternion.header
                got_frame = True
            pose_stamped.pose.orientation = quaternion.quaternion
        elif isinstance(quaternion, Quaternion):
            pose_stamped.pose.orientation = quaternion
        elif isinstance(quaternion, (list, tuple)):
            pose_stamped.pose.orientation = Quaternion(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3],
            )

        if not got_frame:
            if frame_id is None:
                self.logger.error("Must specify `frame_id`")
                return py_trees.common.Status.FAILURE
            pose_stamped.header.frame_id = frame_id

        self.blackboard_set("pose_stamped", pose_stamped)
        return py_trees.common.Status.SUCCESS


class PoseStampedToTwistStamped(BlackboardBehavior):
    """
    Converts a PoseStamped message, which represents a displacement from the origin
    in a particular frame to a target pose in that frame, to a TwistStamped
    message, representing the linear and angular velocities to achieve that
    displacement.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        pose_stamped: Union[BlackboardKey, PoseStamped],
        speed: Union[
            BlackboardKey,
            Callable[[PoseStamped], Tuple[float, float]],
            Tuple[float, float],
        ] = (0.1, 0.3),
        hz: Union[BlackboardKey, float] = 10.0,
        round_decimals: Union[BlackboardKey, Optional[int]] = None,
        angular_override: Union[BlackboardKey, Optional[Vector3]] = None,
        linear_override: Union[BlackboardKey, Optional[Vector3]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        pose_stamped: The pose stamped representing the displacement from the
            origin to a target pose.
        speed: The speed to move at. If a tuple, then the first element is the
            linear speed (m/s) and the second element is the angular speed (rad/s).
            If a callable, then it is a function that takes in a PoseStamped
            representing the displacement from the origin to a target pose and
            returns a tuple of the linear speed (m/s) and angular speed (rad/s).
        hz: the frequency at which this behavior is ticked (i.e., the twist is recomputed).
        round_decimals: If not None, round the linear and angular velocities to
            this many decimal places.
        angular_override: If not None, use this angular velocity instead of computing
            it from the PoseStamped.
        linear_override: If not None, use this linear velocity instead of computing
            it from the PoseStamped.
        """
        # pylint: disable=unused-argument, duplicate-code, too-many-arguments
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        twist_stamped: Optional[BlackboardKey],  # TwistStamped
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        twist_stamped: The twist stamped representing the linear and angular
            velocities to move from the origin to the target pose.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=too-many-locals
        # One over is fine

        # Input Validation
        if not self.blackboard_exists(
            [
                "pose_stamped",
                "speed",
                "hz",
                "round_decimals",
                "angular_override",
                "linear_override",
            ]
        ):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        pose_stamped = self.blackboard_get("pose_stamped")
        speed = self.blackboard_get("speed")
        hz = self.blackboard_get("hz")
        if isinstance(speed, tuple):
            max_linear_speed, max_angular_speed = speed
        else:
            max_linear_speed, max_angular_speed = speed(pose_stamped)

        # Get how much to round decimals
        round_decimals = self.blackboard_get("round_decimals")

        # For the linear velocity, normalize the pose's position and multiply
        # it by the linear_speed
        linear_msg = self.blackboard_get("linear_override")
        if linear_msg is None:
            # Compute the linear velocity from the PoseStamped
            linear_displacement = ros2_numpy.numpify(pose_stamped.pose.position)
            linear_distance = np.linalg.norm(linear_displacement)
            linear_speed = min(linear_distance * hz, max_linear_speed)
            linear_velocity = linear_displacement / linear_distance * linear_speed

            # Round it
            if round_decimals is not None:
                linear_velocity = np.round(
                    linear_velocity,
                    round_decimals,
                )

            # Convert to a msg
            linear_msg = ros2_numpy.msgify(Vector3, linear_velocity)

        # For the angular velocity, convert the pose's orientation to a
        # rotation vector, normalize it, and multiply it by the angular_speed
        angular_msg = self.blackboard_get("angular_override")
        if angular_msg is None:
            # Compute the angular velocity from the PoseStamped
            angular_displacement = R.from_quat(
                ros2_numpy.numpify(pose_stamped.pose.orientation)
            ).as_rotvec()
            angular_distance = np.linalg.norm(angular_displacement)
            angular_speed = min(angular_distance * hz, max_angular_speed)
            angular_velocity = angular_displacement / angular_distance * angular_speed

            # Round it
            if round_decimals is not None:
                angular_velocity = np.round(
                    angular_velocity,
                    round_decimals,
                )

            # Convert to a msg
            angular_msg = ros2_numpy.msgify(Vector3, angular_velocity)

        # Create the twist stamped message
        twist_stamped = TwistStamped()
        twist_stamped.header = pose_stamped.header
        twist_stamped.twist.linear = linear_msg
        twist_stamped.twist.angular = angular_msg

        self.blackboard_set("twist_stamped", twist_stamped)
        return py_trees.common.Status.SUCCESS
