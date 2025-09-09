# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the ComputeSlerpMidpointOrientation behavior.
It takes start and goal joint configurations, performs Forward Kinematics (FK)
to find the end-effector orientation at each, and uses SLERP to compute
the midpoint orientation. This is used to create a dynamic path constraint
for transport motions, mirroring the logic from the Articutool benchmark.
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple

# Third-party imports
from geometry_msgs.msg import Quaternion, Pose, PoseStamped
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation as R, Slerp
import py_trees
from py_trees.common import Status
import rclpy.node

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey

# Import the benchmark's PinocchioModel to perform FK
from ada_feeding.articutool_benchmark.feeding_benchmark.kinematics import PinocchioModel


class ComputeSlerpMidpointOrientation(BlackboardBehavior):
    """
    Computes a midpoint orientation between two end-effector poses using SLERP.
    """

    def blackboard_inputs(
        self,
        start_ee_pose: Union[BlackboardKey, Pose, PoseStamped],
        goal_ee_pose: Union[BlackboardKey, Pose, PoseStamped],
    ) -> None:
        """Define blackboard inputs."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        midpoint_orientation_quaternion: Optional[BlackboardKey],
        path_constraint_tolerance: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def _check_and_log_pose(
        self,
        pose_name: str,
        pose_to_check: Union[Pose, PoseStamped],
        R_target: R,
        tolerances_rad: Tuple[float, float, float],
    ) -> bool:
        """Helper to check a single pose and log a detailed report."""
        orientation = (
            pose_to_check.pose.orientation
            if isinstance(pose_to_check, PoseStamped)
            else pose_to_check.orientation
        )
        R_current = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        R_error = R_target.inv() * R_current
        euler_error_rad = R_error.as_euler("xyz")

        pitch_error_rad, yaw_error_rad, roll_error_rad = euler_error_rad
        pitch_tol_rad, yaw_tol_rad, roll_tol_rad = tolerances_rad

        pitch_ok = abs(pitch_error_rad) <= pitch_tol_rad
        yaw_ok = abs(yaw_error_rad) <= yaw_tol_rad
        roll_ok = abs(roll_error_rad) <= roll_tol_rad
        is_satisfied = pitch_ok and yaw_ok and roll_ok

        self.logger.info(
            f"[{self.name}] Analyzing {pose_name} against path constraint..."
        )
        self.logger.info(f"[{self.name}]   - Constraint Satisfied: {is_satisfied}")

        details = {
            "Pitch (X)": (pitch_error_rad, pitch_tol_rad, pitch_ok),
            "Yaw (Y)": (yaw_error_rad, yaw_tol_rad, yaw_ok),
            "Roll (Z)": (roll_error_rad, roll_tol_rad, roll_ok),
        }
        for axis, (err_rad, tol_rad, status_ok) in details.items():
            err_deg = math.degrees(err_rad)
            tol_deg = math.degrees(tol_rad)
            status_str = "OK" if status_ok else "FAILED"
            self.logger.info(
                f"[{self.name}]     - {axis:<10}: Error = {err_deg:6.1f}°, Tolerance = ±{tol_deg:.1f}° -> {status_str}"
            )
        return is_satisfied

    @override
    def update(self) -> Status:
        """Perform FK, calculate SLERP, validate poses, and write outputs."""
        try:
            start_pose = self.blackboard_get("start_ee_pose")
            goal_pose = self.blackboard_get("goal_ee_pose")

            # --- 1. Compute SLERP Midpoint (same as before) ---
            start_orientation = (
                start_pose.pose.orientation
                if isinstance(start_pose, PoseStamped)
                else start_pose.orientation
            )
            goal_orientation = (
                goal_pose.pose.orientation
                if isinstance(goal_pose, PoseStamped)
                else goal_pose.orientation
            )
            R_start = R.from_quat(
                [
                    start_orientation.x,
                    start_orientation.y,
                    start_orientation.z,
                    start_orientation.w,
                ]
            )
            R_goal = R.from_quat(
                [
                    goal_orientation.x,
                    goal_orientation.y,
                    goal_orientation.z,
                    goal_orientation.w,
                ]
            )

            if np.dot(R_start.as_quat(), R_goal.as_quat()) < 0:
                R_goal = R_goal * R.from_quat([0, 0, 0, -1])

            key_rots = R.from_matrix([R_start.as_matrix(), R_goal.as_matrix()])
            slerp = Slerp([0, 1], key_rots)
            R_midpoint = slerp(0.5)

            # --- 2. Validate Start and Goal Poses Against the New Constraint ---
            benchmark_tolerance = (math.pi / 2, 2 * math.pi, math.pi / 4)
            start_ok = self._check_and_log_pose(
                "Start Pose", start_pose, R_midpoint, benchmark_tolerance
            )
            goal_ok = self._check_and_log_pose(
                "Goal Pose", goal_pose, R_midpoint, benchmark_tolerance
            )

            # --- 3. Return SUCCESS only if both poses are valid ---
            if start_ok and goal_ok:
                q_midpoint_xyzw = R_midpoint.as_quat()
                midpoint_quat_msg = Quaternion(
                    x=q_midpoint_xyzw[0],
                    y=q_midpoint_xyzw[1],
                    z=q_midpoint_xyzw[2],
                    w=q_midpoint_xyzw[3],
                )
                self.blackboard_set(
                    "midpoint_orientation_quaternion", midpoint_quat_msg
                )
                self.blackboard_set("path_constraint_tolerance", benchmark_tolerance)
                self.feedback_message = (
                    "Constraint computed and validated successfully."
                )
                return Status.SUCCESS
            else:
                self.feedback_message = (
                    "Start or Goal pose violates the generated path constraint."
                )
                # No need to log here, as the helper function already did.
                return Status.FAILURE

        except Exception as e:
            self.feedback_message = (
                f"Failed to compute or validate SLERP constraint: {e}"
            )
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
