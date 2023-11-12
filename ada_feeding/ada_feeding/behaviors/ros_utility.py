#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines ROS utility behaviors.
"""

# Standard imports
from typing import Any, List, Optional, Tuple, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    QuaternionStamped,
    Transform,
    TransformStamped,
    Vector3,
)
import numpy as np
from overrides import override
import py_trees
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
import ros2_numpy
from std_msgs.msg import Header
from tf2_geometry_msgs import PointStamped, PoseStamped, Vector3Stamped
import tf2_py as tf2
from tf2_ros import TypeException

# Local imports
from ada_feeding.helpers import (
    BlackboardKey,
    get_tf_object,
    set_static_tf,
)
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


class GetTransform(BlackboardBehavior):
    """
    Look up a transform between two frames.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        target_frame: Union[BlackboardKey, str],
        source_frame: Union[BlackboardKey, str],
        time: Union[BlackboardKey, Time] = Time(),
        timeout: Union[BlackboardKey, Duration] = Duration(seconds=0.0),
        new_type: Union[BlackboardKey, type] = TransformStamped,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        target_frame: Name of the frame to transform into.
        source_frame: Name of the input frame.
        time: The time at which to get the transform. (default, 0, gets the latest)
        timeout: Time to wait for the target frame to become available.
            Note that the tree ticking will block for this duration, so it is
            recommended that this is kept at 0.0 (the default value).
        new_type: The type of the transform to return. Must be either TransformStamped
            or PoseStamped.
        """
        # pylint: disable=unused-argument, duplicate-code, too-many-arguments
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        transform: Optional[BlackboardKey],  # new_type
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        transform: The transform between the two frames
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

        # Get Node
        self.node = kwargs["node"]

        # Get TF Listener from blackboard
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists(
            ["target_frame", "source_frame", "time", "timeout", "new_type"]
        ):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        target_frame = self.blackboard_get("target_frame")
        source_frame = self.blackboard_get("source_frame")
        time = self.blackboard_get("time")
        timeout = self.blackboard_get("timeout")
        new_type = self.blackboard_get("new_type")
        if new_type not in [TransformStamped, PoseStamped]:
            self.logger.error(
                f"Invalid type {new_type}. Must be either TransformStamped or PoseStamped"
            )
            return py_trees.common.Status.FAILURE

        if self.tf_lock.locked():
            return py_trees.common.Status.RUNNING

        with self.tf_lock:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, time, timeout
                )
            except (
                tf2.ConnectivityException,
                tf2.ExtrapolationException,
                tf2.InvalidArgumentException,
                tf2.LookupException,
                tf2.TimeoutException,
                tf2.TransformException,
            ) as error:
                self.logger.error(f"Could not get transform. Error: {error}")
                return py_trees.common.Status.FAILURE

        if new_type == PoseStamped:
            transform = PoseStamped(
                header=transform.header,
                pose=Pose(
                    position=Point(
                        x=transform.transform.translation.x,
                        y=transform.transform.translation.y,
                        z=transform.transform.translation.z,
                    ),
                    orientation=transform.transform.rotation,
                ),
            )

        self.blackboard_set("transform", transform)
        return py_trees.common.Status.SUCCESS


class SetStaticTransform(BlackboardBehavior):
    """
    Add a static transform to the TF tree.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        transform: Union[BlackboardKey, TransformStamped, PoseStamped],
        child_frame_id: Union[BlackboardKey, Optional[str]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        transform: The transform to add to the TF tree.
        child_frame_id: The child frame of the transform. Only used if transform
            is a PoseStamped.
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

        # Get Node
        self.node = kwargs["node"]

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists(["transform", "child_frame_id"]):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        transform = self.blackboard_get("transform")
        self.node.get_logger().info(f"Setting static transform: {transform}")
        if isinstance(transform, PoseStamped):
            # Convert PoseStamped to TransformStamped
            transform = TransformStamped(
                header=transform.header,
                child_frame_id=self.blackboard_get("child_frame_id"),
                transform=Transform(
                    translation=Vector3(
                        x=transform.pose.position.x,
                        y=transform.pose.position.y,
                        z=transform.pose.position.z,
                    ),
                    rotation=transform.pose.orientation,
                ),
            )

        if set_static_tf(
            transform,
            self.blackboard,
            self.node,
        ):
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class ApplyTransform(BlackboardBehavior):
    """
    Apply a Transform, either passed as an argument or from the TF tree,
    to a PointStamped, PoseStamped, or Vector3Stamped.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        stamped_msg: Union[BlackboardKey, PointStamped, PoseStamped, Vector3Stamped],
        target_frame=Union[BlackboardKey, Optional[str]],
        transform: Union[BlackboardKey, Optional[TransformStamped]] = None,
        timeout: Union[BlackboardKey, Duration] = Duration(seconds=0.0),
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        stamped_msg: The object to transform. Note if the timestamp is 0, it
            gets the latest transform.
        target_frame: The frame to transform into. If set, look up the transform
            from `stamped_msg` to `target_frame` in the TF tree. Else, apply
            the fixed transform passed in as `transform`.
        transform: The transform to apply to `stamped_msg`. Must be set if
            `target_frame` is None. Ignored if `target_frame` is not None.
        timeout: Time to wait for the target frame to become available.
            Note that the tree ticking will block for this duration, so it is
            recommended that this is kept at 0.0 (the default value).
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        transformed_msg: Optional[BlackboardKey],  # same type as stamped_msg
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        stamped_msg: The transformed stamped message.
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

        # Get Node
        self.node = kwargs["node"]

        # Get TF Listener from blackboard
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists(
            ["stamped_msg", "target_frame", "transform", "timeout"]
        ):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        stamped_msg = self.blackboard_get("stamped_msg")
        target_frame = self.blackboard_get("target_frame")
        transform = self.blackboard_get("transform")
        timeout = self.blackboard_get("timeout")
        if target_frame is None and transform is None:
            self.logger.error("Must specify either `target_frame` or `transform`")
            return py_trees.common.Status.FAILURE

        if self.tf_lock.locked():
            return py_trees.common.Status.RUNNING

        transformed_msg = None
        with self.tf_lock:
            if target_frame is not None:
                # Compute the transform from the TF tree
                try:
                    transformed_msg = self.tf_buffer.transform(
                        stamped_msg,
                        target_frame,
                        timeout,
                    )
                except (
                    tf2.ConnectivityException,
                    tf2.ExtrapolationException,
                    tf2.InvalidArgumentException,
                    tf2.LookupException,
                    tf2.TimeoutException,
                    tf2.TransformException,
                    TypeException,
                ) as error:
                    self.logger.error(f"Could not get transform. Error: {error}")
                    return py_trees.common.Status.FAILURE

        if transformed_msg is None:
            # Apply the fixed transform
            transform_matrix = ros2_numpy.numpify(transform.transform)
            self.node.get_logger().info(f"Transform Matrix: {transform_matrix}")
            header = Header(
                stamp=stamped_msg.header.stamp,
                frame_id=transform.child_frame_id,
            )
            if isinstance(stamped_msg, PointStamped):
                stamped_vec = ros2_numpy.numpify(stamped_msg.point, hom=True).reshape(
                    (-1, 1)
                )
                transformed_msg = PointStamped(
                    header=header,
                    point=ros2_numpy.msgify(
                        Point, np.matmul(transform_matrix, stamped_vec)
                    ),
                )
            elif isinstance(stamped_msg, PoseStamped):
                stamped_matrix = ros2_numpy.numpify(stamped_msg.pose)
                self.node.get_logger().info(f"Stamped Matrix: {stamped_matrix}")
                transformed_msg = PoseStamped(
                    header=header,
                    pose=ros2_numpy.msgify(
                        Pose,
                        np.matmul(transform_matrix, stamped_matrix),
                    ),
                )
                self.node.get_logger().info(f"Transformed Pose: {transformed_msg}")
            elif isinstance(stamped_msg, Vector3Stamped):
                stamped_vec = ros2_numpy.numpify(stamped_msg.vector, hom=True).reshape(
                    (-1, 1)
                )
                transformed_msg = Vector3Stamped(
                    header=header,
                    vector=ros2_numpy.msgify(
                        Vector3,
                        np.matmul(transform_matrix, stamped_vec),
                    ),
                )
            else:
                self.logger.error(f"Unsupported message type: {type(stamped_msg)}")
                return py_trees.common.Status.FAILURE

        self.blackboard_set("transformed_msg", transformed_msg)
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
