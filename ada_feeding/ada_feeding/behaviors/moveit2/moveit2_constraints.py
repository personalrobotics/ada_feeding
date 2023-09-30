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
)
from overrides import override
import py_trees

# Local imports
from ada_feeding.behaviors.moveit2.moveit2_plan import MoveIt2ConstraintType
from ada_feeding.helpers import BlackboardKey
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

        # pylint: disable=too-many-boolean-expressions
        # This is just checking all inputs, should be
        # easy to read.
        if (
            not self.blackboard_exists("joint_positions")
            or not self.blackboard_exists("joint_names")
            or not self.blackboard_exists("tolerance")
            or not self.blackboard_exists("weight")
            or not self.blackboard_exists("constraints")
        ):
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

        # pylint: disable=too-many-boolean-expressions
        # This is just checking all inputs, should be
        # easy to read.
        if (
            not self.blackboard_exists("position")
            or not self.blackboard_exists("frame_id")
            or not self.blackboard_exists("target_link")
            or not self.blackboard_exists("tolerance")
            or not self.blackboard_exists("weight")
            or not self.blackboard_exists("constraints")
        ):
            return py_trees.common.Status.FAILURE

        position = self.blackboard_get("position")
        constraint = (
            MoveIt2ConstraintType.POSITION,
            {
                "position": position.point
                if isinstance(position, PointStamped)
                else position,
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

        # pylint: disable=too-many-boolean-expressions
        # This is just checking all inputs, should be
        # easy to read.
        if (
            not self.blackboard_exists("quat_xyzw")
            or not self.blackboard_exists("frame_id")
            or not self.blackboard_exists("target_link")
            or not self.blackboard_exists("tolerance")
            or not self.blackboard_exists("weight")
            or not self.blackboard_exists("constraints")
            or not self.blackboard_exists("parameterization")
        ):
            return py_trees.common.Status.FAILURE

        quat_xyzw = self.blackboard_get("quat_xyzw")
        constraint = (
            MoveIt2ConstraintType.ORIENTATION,
            {
                "quat_xyzw": quat_xyzw.quaternion
                if isinstance(quat_xyzw, QuaternionStamped)
                else quat_xyzw,
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

        Note: if pose is PoseStamped:
          - `frame_id` is pose.header.frame_id (if not "")
          - `target_link` is pose.child_frame_id (if not "")

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

        # pylint: disable=too-many-boolean-expressions
        # This is just checking all inputs, should be
        # easy to read.
        if (
            not self.blackboard_exists("pose")
            or not self.blackboard_exists("frame_id")
            or not self.blackboard_exists("target_link")
            or not self.blackboard_exists("tolerance_position")
            or not self.blackboard_exists("weight_position")
            or not self.blackboard_exists("tolerance_orientation")
            or not self.blackboard_exists("weight_orientation")
            or not self.blackboard_exists("constraints")
            or not self.blackboard_exists("parameterization")
        ):
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
                "target_link": pose.child_frame_id
                if (isinstance(pose, PoseStamped) and len(pose.child_frame_id) > 0)
                else self.blackboard_get("target_link"),
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
                "target_link": pose.child_frame_id
                if (isinstance(pose, PoseStamped) and len(pose.child_frame_id) > 0)
                else self.blackboard_get("target_link"),
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
