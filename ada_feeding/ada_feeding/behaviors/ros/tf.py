#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines utility behaviors for interacting with the TF tree.
"""

# Standard imports
from typing import Optional, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Pose,
    Transform,
    TransformStamped,
    Twist,
    TwistStamped,
    Vector3,
)
import numpy as np
from overrides import override
import py_trees
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
from ada_feeding.behaviors import BlackboardBehavior


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
        self.node.get_logger().debug(f"Setting static transform: {transform}")
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
        stamped_msg: Union[BlackboardKey, PointStamped, PoseStamped, Vector3Stamped, TwistStamped],
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
        linear_distance: Optional[BlackboardKey] = None,  # float
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        stamped_msg: The transformed stamped message.
        linear_distance: The linear distance between the transformed message and
            the origin.
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

        # pylint: disable=too-many-return-statements, too-many-branches
        # This is a complex behavior, so we need to check a lot of things.

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
                    if isinstance(stamped_msg, TwistStamped):
                        linear = Vector3Stamped(
                            header = stamped_msg.header,
                            vector = stamped_msg.twist.linear,
                        )
                        linear_transformed = self.tf_buffer.transform(
                            linear,
                            target_frame,
                            timeout,
                        )
                        angular = Vector3Stamped(
                            header = stamped_msg.header,
                            vector = stamped_msg.twist.angular,
                        )
                        angular_transformed = self.tf_buffer.transform(
                            angular,
                            target_frame,
                            timeout,
                        )
                        transformed_msg = TwistStamped(
                            header = Header(
                                stamp = stamped_msg.header.stamp,
                                frame_id = target_frame,
                            ),
                            twist = Twist(
                                linear = linear_transformed.vector,
                                angular = angular_transformed.vector,
                            ),
                        )
                    else:
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
                transformed_msg = PoseStamped(
                    header=header,
                    pose=ros2_numpy.msgify(
                        Pose,
                        np.matmul(transform_matrix, stamped_matrix),
                    ),
                )
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
            elif isinstance(stamped_msg, TwistStamped):
                linear_transformed = ros2_numpy.msgify(
                    Vector3,
                    np.matmul(transform_matrix, ros2_numpy.numpify(stamped_msg.twist.linear, hom=True).reshape((-1, 1))),
                )
                angular_transformed = ros2_numpy.msgify(
                    Vector3,
                    np.matmul(transform_matrix, ros2_numpy.numpify(stamped_msg.twist.angular, hom=True).reshape((-1, 1))),
                )
                transformed_msg = TwistStamped(
                    header = header,
                    twist = Twist(
                        linear = linear_transformed,
                        angular = angular_transformed,
                    ),
                )
            else:
                self.logger.error(f"Unsupported message type: {type(stamped_msg)}")
                return py_trees.common.Status.FAILURE

        # Write the transformed_msg
        self.blackboard_set("transformed_msg", transformed_msg)

        # Write the linear distance
        if isinstance(transformed_msg, PoseStamped):
            linear_msg = transformed_msg.pose.position
        elif isinstance(transformed_msg, PointStamped):
            linear_msg = transformed_msg.point
        elif isinstance(transformed_msg, Vector3Stamped):
            linear_msg = transformed_msg.vector
        elif isinstance(transformed_msg, TwistStamped):
            linear_msg = transformed_msg.twist.linear
        else:
            self.logger.error(f"Unsupported message type: {type(transformed_msg)}")
            return py_trees.common.Status.FAILURE
        self.blackboard_set(
            "linear_distance",
            np.linalg.norm(ros2_numpy.numpify(linear_msg)),
        )

        return py_trees.common.Status.SUCCESS
