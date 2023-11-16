#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines utility behaviors for interacting with and updating ROS
messages.
"""

# Standard imports
from typing import Any, List, Optional, Tuple, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Quaternion,
    QuaternionStamped,
)
from overrides import override
import py_trees
import rclpy
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
        if not self.blackboard_exists(["stamped_msg", "timestamp"]):
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
