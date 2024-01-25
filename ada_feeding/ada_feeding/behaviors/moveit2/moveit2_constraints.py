#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines behaviors that package individual
blackboard keys / constants into a dictionary
of constraints that can be passed to MoveIt2Plan.
"""

# Standard imports
from typing import Any, Union, Optional, Dict, List, Tuple

# Third-party imports
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    PointStamped,
    Point,
    QuaternionStamped,
    Quaternion,
    Vector3Stamped,
    Vector3,
)
import numpy as np
from overrides import override
import py_trees
import rclpy
import ros2_numpy

# Local imports
from ada_feeding.behaviors.moveit2.moveit2_plan import MoveIt2ConstraintType
from ada_feeding.helpers import BlackboardKey, get_tf_object, get_moveit2_object
from ada_feeding.behaviors import BlackboardBehavior


class MoveIt2JointConstraint(BlackboardBehavior):
    """
    Adds Joint Constraint to Blackboard Dictionary
    See pymoveit2:set_joint_goal() for more info
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
        joint_positions: Union[BlackboardKey, List[float]],
        joint_names: Union[BlackboardKey, Optional[List[str]]] = None,
        tolerance: Union[BlackboardKey, float] = 0.001,
        weight: Union[BlackboardKey, float] = 1.0,
        constraints: Union[
            BlackboardKey, Optional[List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """
        Blackboard Inputs

        Docstring copied from set_joint_goal():
        With `joint_names` specified, `joint_positions` can be
        defined for specific joints in an arbitrary order. Otherwise, first **n** joints
        passed into the constructor is used, where **n** is the length of `joint_positions`.

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        constraints: Optional[
            BlackboardKey
        ],  # List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        if not self.blackboard_exists(
            ["joint_positions", "joint_names", "tolerance", "weight", "constraints"]
        ):
            self.logger.error("MoveIt2Constraint: Missing input arguments.")
            return py_trees.common.Status.FAILURE

        constraint = (
            MoveIt2ConstraintType.JOINT,
            {
                "joint_positions": self.blackboard_get("joint_positions"),
                "joint_names": self.blackboard_get("joint_names"),
                "tolerance": self.blackboard_get("tolerance"),
                "weight": self.blackboard_get("weight"),
            },
        )

        constraints = self.blackboard_get("constraints")
        if constraints is None:
            constraints = []
        else:
            constraints = constraints.copy()
        constraints.append(constraint)
        self.blackboard_set("constraints", constraints)
        return py_trees.common.Status.SUCCESS


class MoveIt2PositionOffsetConstraint(BlackboardBehavior):
    """
    Adds Position Offset Constraint to Blackboard Dictionary
    Similar to Position Constraint + current position of target_link
    See pymoveit2:set_position_goal() for more info
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
        offset: Union[
            BlackboardKey, Union[Vector3Stamped, Vector3, Tuple[float, float, float]]
        ],
        frame_id: Union[BlackboardKey, Optional[str]] = None,
        target_link: Union[BlackboardKey, Optional[str]] = None,
        tolerance: Union[BlackboardKey, float] = 0.001,
        weight: Union[BlackboardKey, float] = 1.0,
        constraints: Union[
            BlackboardKey, Optional[List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """
        Blackboard Inputs

        Docstring copied from set_position_goal():
        Set Cartesian position goal of `target_link` with respect to `frame_id`.
          - `frame_id` defaults to the base link
          - `target_link` defaults to end effector

        Position goal == offset + position of target_link in frame_id

        Note: if position is Vector3Stamped,
              use position.header.frame_id for frame_id (if not "")

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        constraints: Optional[
            BlackboardKey
        ],  # List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
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

        # Get the MoveIt2 object, don't need lock (const read only)
        self.moveit2, _ = get_moveit2_object(
            self.blackboard,
            self.node,
        )

        # Get TF Listener from blackboard
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        if not self.blackboard_exists(
            [
                "offset",
                "frame_id",
                "target_link",
                "tolerance",
                "weight",
                "constraints",
            ]
        ):
            self.logger.error(
                "MoveIt2PositionOffsetConstraint: Missing input arguments."
            )
            self.logger.error(str(self.blackboard))
            return py_trees.common.Status.FAILURE

        if self.tf_lock.locked():
            return py_trees.common.Status.RUNNING

        offset = self.blackboard_get("offset")
        frame_id = (
            offset.header.frame_id
            if (isinstance(offset, Vector3Stamped) and len(offset.header.frame_id) > 0)
            else self.moveit2.base_link_name
            if self.blackboard_get("frame_id") is None
            else self.blackboard_get("frame_id")
        )
        target_link = (
            self.moveit2.end_effector_name
            if self.blackboard_get("target_link") is None
            else self.blackboard_get("target_link")
        )

        # Get current position
        transform = None
        with self.tf_lock:
            # Check if we have the target relative to base
            if not self.tf_buffer.can_transform(
                frame_id,
                target_link,
                rclpy.time.Time(),
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                self.logger.warning("PositionOffsetConstraint waiting on ee/base TF...")
                return py_trees.common.Status.RUNNING
            transform = self.tf_buffer.lookup_transform(
                frame_id,
                target_link,
                rclpy.time.Time(),
            )

        offset_np = (
            ros2_numpy.numpify(offset.vector)
            if isinstance(offset, Vector3Stamped)
            else ros2_numpy.numpify(offset)
            if isinstance(offset, Vector3)
            else np.array(offset)
        )
        offset_pos = ros2_numpy.msgify(
            Point, ros2_numpy.numpify(transform.transform.translation) + offset_np
        )

        constraint = (
            MoveIt2ConstraintType.POSITION,
            {
                "position": offset_pos,
                "frame_id": frame_id,
                "target_link": target_link,
                "tolerance": self.blackboard_get("tolerance"),
                "weight": self.blackboard_get("weight"),
            },
        )

        constraints = self.blackboard_get("constraints")
        if constraints is None:
            constraints = []
        else:
            constraints = constraints.copy()
        constraints.append(constraint)
        self.blackboard_set("constraints", constraints)
        return py_trees.common.Status.SUCCESS


class MoveIt2PositionConstraint(BlackboardBehavior):
    """
    Adds Position Constraint to Blackboard Dictionary
    See pymoveit2:set_position_goal() for more info
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
        position: Union[
            BlackboardKey, Union[PointStamped, Point, Tuple[float, float, float]]
        ],
        frame_id: Union[BlackboardKey, Optional[str]] = None,
        target_link: Union[BlackboardKey, Optional[str]] = None,
        tolerance: Union[BlackboardKey, float] = 0.001,
        weight: Union[BlackboardKey, float] = 1.0,
        constraints: Union[
            BlackboardKey, Optional[List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """
        Blackboard Inputs

        Docstring copied from set_position_goal():
        Set Cartesian position goal of `target_link` with respect to `frame_id`.
          - `frame_id` defaults to the base link
          - `target_link` defaults to end effector

        Note: if position is PointStamped,
              use position.header.frame_id for frame_id (if not "")

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        constraints: Optional[
            BlackboardKey
        ],  # List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        if not self.blackboard_exists(
            [
                "position",
                "frame_id",
                "target_link",
                "tolerance",
                "weight",
                "constraints",
            ]
        ):
            self.logger.error("MoveIt2PositionConstraint: Missing input arguments.")
            self.logger.error(str(self.blackboard))
            return py_trees.common.Status.FAILURE

        position = self.blackboard_get("position")
        constraint = (
            MoveIt2ConstraintType.POSITION,
            {
                "position": position.point
                if isinstance(position, PointStamped)
                else position
                if isinstance(position, Point)
                else ros2_numpy.msgify(Point, np.array(position)),
                "frame_id": position.header.frame_id
                if (
                    isinstance(position, PointStamped)
                    and len(position.header.frame_id) > 0
                )
                else self.blackboard_get("frame_id"),
                "target_link": self.blackboard_get("target_link"),
                "tolerance": self.blackboard_get("tolerance"),
                "weight": self.blackboard_get("weight"),
            },
        )

        constraints = self.blackboard_get("constraints")
        if constraints is None:
            constraints = []
        else:
            constraints = constraints.copy()
        constraints.append(constraint)
        self.blackboard_set("constraints", constraints)
        return py_trees.common.Status.SUCCESS


class MoveIt2OrientationConstraint(BlackboardBehavior):
    """
    Adds Orientation Constraint to Blackboard Dictionary
    See pymoveit2:set_orientation_goal() for more info
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
        quat_xyzw: Union[
            BlackboardKey,
            Union[QuaternionStamped, Quaternion, Tuple[float, float, float, float]],
        ],
        frame_id: Union[BlackboardKey, Optional[str]] = None,
        target_link: Union[BlackboardKey, Optional[str]] = None,
        tolerance: Union[
            BlackboardKey, Union[float, Tuple[float, float, float]]
        ] = 0.001,
        weight: Union[BlackboardKey, float] = 1.0,
        parameterization: int = 0,  # 0: Euler, 1: Rotation Vector
        constraints: Union[
            BlackboardKey, Optional[List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """
        Blackboard Inputs

        Docstring copied from set_position_goal():
        Set Cartesian orientation goal of `target_link` with respect to `frame_id`.
          - `frame_id` defaults to the base link
          - `target_link` defaults to end effector

        Note: if quat_xyzw is QuaternionStamped,
              use quat_xyzw.header.frame_id for frame_id. (if not "")

        Details on parameterization:
        https://github.com/ros-planning/moveit_msgs/blob/master/msg/OrientationConstraint.msg

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        constraints: Optional[
            BlackboardKey
        ],  # List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        if not self.blackboard_exists(
            [
                "quat_xyzw",
                "frame_id",
                "target_link",
                "tolerance",
                "weight",
                "constraints",
                "parameterization",
            ]
        ):
            self.logger.error("MoveIt2OrientationConstraint: Missing input arguments.")
            return py_trees.common.Status.FAILURE

        quat_xyzw = self.blackboard_get("quat_xyzw")
        constraint = (
            MoveIt2ConstraintType.ORIENTATION,
            {
                "quat_xyzw": quat_xyzw.quaternion
                if isinstance(quat_xyzw, QuaternionStamped)
                else quat_xyzw
                if isinstance(quat_xyzw, Quaternion)
                else ros2_numpy.msgify(Quaternion, np.array(quat_xyzw)),
                "frame_id": quat_xyzw.header.frame_id
                if (
                    isinstance(quat_xyzw, QuaternionStamped)
                    and len(quat_xyzw.header.frame_id) > 0
                )
                else self.blackboard_get("frame_id"),
                "target_link": self.blackboard_get("target_link"),
                "tolerance": self.blackboard_get("tolerance"),
                "weight": self.blackboard_get("weight"),
                "parameterization": self.blackboard_get("parameterization"),
            },
        )

        constraints = self.blackboard_get("constraints")
        if constraints is None:
            constraints = []
        else:
            constraints = constraints.copy()
        constraints.append(constraint)
        self.blackboard_set("constraints", constraints)
        return py_trees.common.Status.SUCCESS


class MoveIt2PoseConstraint(BlackboardBehavior):
    """
    Adds Pose Constraint to Blackboard Dictionary
    See pymoveit2:set_pose_goal() for more info.
    This is a direct combo of position + orientation
    useful for Pose/PoseStamped ROS objects.
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
        pose: Union[BlackboardKey, Union[PoseStamped, Pose]],
        frame_id: Union[BlackboardKey, Optional[str]] = None,
        target_link: Union[BlackboardKey, Optional[str]] = None,
        tolerance_position: Union[BlackboardKey, float] = 0.001,
        tolerance_orientation: Union[
            BlackboardKey, Union[float, Tuple[float, float, float]]
        ] = 0.001,
        weight_position: Union[BlackboardKey, float] = 1.0,
        weight_orientation: Union[BlackboardKey, float] = 1.0,
        parameterization: int = 0,  # 0: Euler, 1: Rotation Vector
        constraints: Union[
            BlackboardKey, Optional[List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
    ) -> None:
        """
        Blackboard Inputs

        Docstring copied from set_position_goal():
        Set Cartesian orientation goal of `target_link` with respect to `frame_id`.
          - `frame_id` defaults to the base link
          - `target_link` defaults to end effector

        Note: if pose is PoseStamped `frame_id` is pose.header.frame_id (if not "")

        Details on parameterization:
        https://github.com/ros-planning/moveit_msgs/blob/master/msg/OrientationConstraint.msg

        Parameters
        ----------
        constraints: previous set of constraints to append to
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        constraints: Optional[
            BlackboardKey
        ],  # List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        constraints: list of constraints to send to MoveIt2Plan
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        if not self.blackboard_exists(
            [
                "pose",
                "frame_id",
                "target_link",
                "tolerance_position",
                "weight_position",
                "tolerance_orientation",
                "weight_orientation",
                "constraints",
                "parameterization",
            ]
        ):
            self.logger.error("MoveIt2Constraint: Missing input arguments.")
            return py_trees.common.Status.FAILURE

        pose = self.blackboard_get("pose")
        constraint_orient = (
            MoveIt2ConstraintType.ORIENTATION,
            {
                "quat_xyzw": pose.pose.orientation
                if isinstance(pose, PoseStamped)
                else pose.orientation,
                "frame_id": pose.header.frame_id
                if (isinstance(pose, PoseStamped) and len(pose.header.frame_id) > 0)
                else self.blackboard_get("frame_id"),
                "target_link": self.blackboard_get("target_link"),
                "tolerance": self.blackboard_get("tolerance_orientation"),
                "weight": self.blackboard_get("weight_orientation"),
                "parameterization": self.blackboard_get("parameterization"),
            },
        )

        constraint_pos = (
            MoveIt2ConstraintType.POSITION,
            {
                "position": pose.pose.position
                if isinstance(pose, PoseStamped)
                else pose.position,
                "frame_id": pose.header.frame_id
                if (isinstance(pose, PoseStamped) and len(pose.header.frame_id) > 0)
                else self.blackboard_get("frame_id"),
                "target_link": self.blackboard_get("target_link"),
                "tolerance": self.blackboard_get("tolerance_position"),
                "weight": self.blackboard_get("weight_position"),
            },
        )

        constraints = self.blackboard_get("constraints")
        if constraints is None:
            constraints = []
        else:
            constraints = constraints.copy()
        constraints.append(constraint_orient)
        constraints.append(constraint_pos)
        self.blackboard_set("constraints", constraints)
        return py_trees.common.Status.SUCCESS
