#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetOrientationPathConstraint decorator, which adds a path
constraint that keeps a specified frame within a secified tolerance of a
specified orientation.
"""
# Standard imports
from typing import List, Optional

# Third-party imports
import numpy as np
import py_trees
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default, get_moveit2_object

# pylint: disable=duplicate-code
# All the constraints have similar code when registering and setting blackboard
# keys, since the parameters for constraints are similar. This is not a problem.


class SetOrientationPathConstraint(MoveToConstraint):
    """
    SetOrientationPathConstraint adds a path constraint that keeps a specified frame
    within a secified tolerance of a specified orientation.
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
        node: Node,
        fk_poses_key: Optional[str] = None,
        fk_links_key: Optional[str] = None,
    ):
        """
        Initialize the MoveToConstraint decorator.

        Parameters
        ----------
        name: The name of the behavior.
        child: The child behavior.
        node: The ROS node to associate the service call with.
        fk_poses_key: The key where the forward kinematics poses are stored on the
            blackboard. If None, do not check whether the orientation constraint
            is satisfied by the start configuration.
        fk_links_key: The key where the forward kinematics links are stored on the
            blackboard. If None, do not check whether the orientation constraint
            is satisfied by the start configuration.
        """
        # pylint: disable=too-many-arguments
        # All are necessary.

        # Initiatilize the decorator
        super().__init__(name=name, child=child)
        self.fk_poses_key = fk_poses_key
        self.fk_links_key = fk_links_key

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " SetOrientationPathConstraint", namespace=name
        )
        self.blackboard.register_key(
            key="quat_xyzw", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="frame_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="parameterization", access=py_trees.common.Access.READ
        )
        if self.fk_poses_key is not None and self.fk_links_key is not None:
            self.blackboard.register_key(
                key=self.fk_poses_key, access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key=self.fk_links_key, access=py_trees.common.Access.READ
            )

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            node,
        )

    @staticmethod
    def normalize_absolute_angle(angle: float) -> float:
        """
        Normalizes an angle to the interval [-pi, +pi] and then take the absolute value
        The returned values will be in the following range [0, +pi]
        """
        normalized_angle = np.fmod(np.fabs(angle), 2 * np.pi)
        return min(2 * np.pi - normalized_angle, normalized_angle)

    def is_constraint_satisfied_at_start(
        self,
        quat_xyzw: Tuple[float, float, float, float],
        frame_id: Optional[str],
        target_link: Optional[str],
        tolerance: float,
        parameterization: int,
    ) -> bool:
        """
        Checks whether the orientation constraint is satisfied by the start
        configuration.

        Parameters
        ----------
        quat_xyzw: The quaternion to check.
        frame_id: The frame in which the quaternion is defined.
        target_link: The link to which the quaternion is attached.
        tolerance: The tolerance of the orientation constraint.
        parameterization: The parameterization of the orientation constraint.

        Returns
        -------
        constraint_satisfied_at_start: True if the orientation constraint is
            satisfied by the start configuration. False if it is not, **or** if
            this function is unable to determine whether it is.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # All arguments are necessary.

        # Check whether the user asked to check the starting FK.
        if self.fk_poses_key is None or self.fk_links_key is None:
            self.logger.debug(
                f"{self.name} [SetOrientationPathConstraint::"
                "is_constraint_satisfied_at_start()] "
                "Will not check whether the constraint is satisfied at start, "
                "as specified by the parameters."
            )
            return False

        # Get the FK links and poses from the balckboard
        try:
            fk_links = self.blackboard.get(self.fk_links_key)
            fk_poses = self.blackboard.get(self.fk_poses_key)
        except KeyError:
            self.logger.warning(
                f"{self.name} [SetOrientationPathConstraint::is_constraint_satisfied_at_start()] "
                "The forward kinematics links and poses were not found on the "
                "blackboard. Setting the orientation path constraint regardless "
                "of the starting FK."
            )
            return False

        # Get the index of the target link in the forward kinematics links.
        try:
            target_link_index = fk_links.index(target_link)
        except ValueError:
            self.logger.warning(
                f"{self.name} [SetOrientationPathConstraint::set_constraint()] "
                "The target link was not found in the forward kinematics "
                "links. Setting the orientation path constraint regardless "
                "of the starting FK."
            )
            return False

        # Verify that we have a pose for the link
        if target_link_index >= len(fk_poses):
            self.logger.warning(
                f"{self.name} [SetOrientationPathConstraint::set_constraint()] "
                "The target link index is out of bounds. Setting the "
                "orientation path constraint regardless of the starting FK."
            )
            return False

        # Get the pose of the target link
        target_link_pose = fk_poses[target_link_index]

        # Verify that the pose is in the right frame_id
        if frame_id is None:
            with self.moveit2_lock:
                frame_id = self.moveit2.base_link_name
        if target_link_pose.header.frame_id != frame_id:
            self.logger.warning(
                f"{self.name} [SetOrientationPathConstraint::set_constraint()] "
                f"The target link pose is in frame {target_link_pose.header.frame_id}, "
                f"not the requested frame {frame_id}. Setting the orientation path "
                "constraint regardless of the starting FK."
            )
            return False

        # Check if the constraint is satisfied. This function is designed to mimic:
        # https://github.com/ros-planning/moveit2/blob/94b4bc2e7952f8fa84ab484e4bc2ad8977a6102e/moveit_core/kinematic_constraints/src/kinematic_constraint.cpp#L670
        start_quat_xyzw = [
            target_link_pose.pose.orientation.x,
            target_link_pose.pose.orientation.y,
            target_link_pose.pose.orientation.z,
            target_link_pose.pose.orientation.w,
        ]
        start_rot_matrix = R.from_quat(start_quat_xyzw)
        target_rot_matrix = R.from_quat(quat_xyzw)
        diff = start_rot_matrix.inv * target_rot_matrix
        # Check parameterization See below for more:
        # https://github.com/ros-planning/moveit_msgs/blob/humble/msg/OrientationConstraint.msg
        abs_tolerance = np.fabs(tolerance)
        if parameterization == 0:  # Euler Angles
            diff_xyz = diff.as_euler("XYZ")
            if (
                SetOrientationPathConstraint.normalize_absolute_angle(diff_xyz[0])
                > abs_tolerance[2]
            ):
                diff_xyz[2] = diff_xyz[0]
                diff_xyz[0] = 0.0
            # Account for angle wrapping
            diff_xyz = np.array(
                [
                    SetOrientationPathConstraint.normalize_absolute_angle(angle)
                    for angle in diff_xyz
                ],
                dtype=diff_xyz.dtype,
            )
        else:  # Rotation Vector
            diff_xyz = np.fabs(diff.as_rotvec())
        # Get the result
        constraint_satisfied = np.all(diff_xyz <= abs_tolerance)
        if constraint_satisfied:
            self.logger.info(
                f"{self.name} [SetOrientationPathConstraint::"
                "is_constraint_satisfied_at_start()] "
                "The orientation constraint is satisfied at start."
            )
        else:
            self.logger.info(
                f"{self.name} [SetOrientationPathConstraint::"
                "is_constraint_satisfied_at_start()] "
                "The orientation constraint is not satisfied at start. "
                "Will not set the orientation path constraint."
            )
        return constraint_satisfied

    def set_constraint(self) -> None:
        """
        Sets the orientation goal constraint.
        """
        self.logger.info(
            f"{self.name} [SetOrientationPathConstraint::set_constraint()]"
        )

        # Get all parameters for planning, resorting to default values if unset.
        quat_xyzw = self.blackboard.quat_xyzw  # required
        frame_id = get_from_blackboard_with_default(self.blackboard, "frame_id", None)
        target_link = get_from_blackboard_with_default(
            self.blackboard, "target_link", None
        )
        tolerance = get_from_blackboard_with_default(
            self.blackboard, "tolerance", 0.001
        )
        weight = get_from_blackboard_with_default(self.blackboard, "weight", 1.0)
        parameterization = get_from_blackboard_with_default(
            self.blackboard, "parameterization", 0
        )

        # Check whether the constraint is satisfied by the start configuration.
        constraint_satisfied_at_start = self.is_constraint_satisfied_at_start(
            quat_xyzw=quat_xyzw,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
            parameterization=parameterization,
        )

        # Set the constraint
        if constraint_satisfied_at_start:
            with self.moveit2_lock:
                self.moveit2.set_path_orientation_constraint(
                    quat_xyzw=quat_xyzw,
                    frame_id=frame_id,
                    target_link=target_link,
                    tolerance=tolerance,
                    weight=weight,
                    parameterization=parameterization,
                )
