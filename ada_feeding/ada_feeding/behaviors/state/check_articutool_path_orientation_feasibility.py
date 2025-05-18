# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the CheckArticutoolPathOrientationFeasibility behavior.
This behavior checks if the Articutool can maintain a desired world orientation
for its tool tip throughout a given Cartesian trajectory of the Jaco end-effector,
using an analytical IK adapted from benchmark scripts.
"""

# Standard imports
import os
import math
from typing import Union, Optional, List, Tuple

# Third-party imports
from geometry_msgs.msg import PoseStamped, Quaternion as QuaternionMsg, Pose
from moveit_msgs.msg import RobotTrajectory
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation  # For R.from_quat and R.apply
import py_trees
import py_trees.blackboard
from py_trees.common import Status
import rclpy.node

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CheckArticutoolPathOrientationFeasibility(BlackboardBehavior):
    """
    Checks if the Articutool can maintain a desired world orientation for its
    tool tip throughout a given Cartesian trajectory of the Jaco end-effector.
    It uses an analytical IK adapted from the orientation_constrained_planner_benchmark.py script.
    """

    # Constants adapted from benchmark for IK logic
    EPSILON = 1e-6

    def blackboard_inputs(
        self,
        jaco_ee_cartesian_trajectory: Union[BlackboardKey, RobotTrajectory],
        desired_tool_tip_world_orientation: Union[BlackboardKey, QuaternionMsg],
        articutool_joint_names: Union[
            BlackboardKey, List[str]
        ],  # Expected: ["pitch_joint", "roll_joint"] order
        articutool_pitch_limits_rad: Union[
            BlackboardKey, Tuple[float, float]
        ],  # (min, max)
        articutool_roll_limits_rad: Union[
            BlackboardKey, Tuple[float, float]
        ],  # (min, max)
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 10,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        jaco_ee_cartesian_trajectory: RobotTrajectory for the Jaco end-effector's Cartesian path.
                                      Expected to contain poses in multi_dof_joint_trajectory.
        desired_tool_tip_world_orientation: QuaternionMsg for the target world orientation of the tool_tip.
        articutool_joint_names: List of Articutool joint names, e.g., ["atool_joint1", "atool_joint2"].
                                The order should match how limits are provided and how IK solutions are interpreted.
        articutool_pitch_limits_rad: Tuple (min_pitch, max_pitch) in radians for the first Articutool joint.
        articutool_roll_limits_rad: Tuple (min_roll, max_roll) in radians for the second Articutool joint.
        num_trajectory_points_to_check: Number of points to sample and check along the trajectory.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_is_orientation_feasible: Optional[
            BlackboardKey
        ],  # -> Optional[bool]
        # Optional outputs for debugging:
        # first_failing_articutool_config: Optional[BlackboardKey], # -> Optional[List[float]]
        # first_failing_jaco_ee_pose_idx: Optional[BlackboardKey], # -> Optional[int]
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        articutool_is_orientation_feasible: Boolean, True if Articutool can maintain orientation.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[rclpy.node.Node] = None
        # Store limits internally after setup for easier access
        self._pitch_limits: Optional[Tuple[float, float]] = None
        self._roll_limits: Optional[Tuple[float, float]] = None

    @override
    def setup(self, **kwargs):
        """Gets the ROS2 node and retrieves fixed parameters like joint limits."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            return  # Cannot proceed

        # Retrieve and store joint limits once during setup
        try:
            self._pitch_limits = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits = self.blackboard_get("articutool_roll_limits_rad")
            if not (
                isinstance(self._pitch_limits, tuple)
                and len(self._pitch_limits) == 2
                and isinstance(self._roll_limits, tuple)
                and len(self._roll_limits) == 2
            ):
                self.logger.error(
                    f"[{self.name}] Articutool joint limits are not valid tuples of size 2."
                )
                self._pitch_limits = None  # Invalidate
                self._roll_limits = None  # Invalidate
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Blackboard key error for joint limits: {e}"
            )

    def _normalize_angle(self, angle: float) -> float:
        """Normalizes an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_benchmark_style(
        self, target_y_axis_in_jaco_hand_frame: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Analytical IK solver for the Articutool, adapted from the benchmark script.
        It solves for (theta_p, theta_r) to align the Articutool's effective
        pointing Y-axis with the target_y_axis_in_jaco_hand_frame.

        The benchmark's IK implies the Articutool's pointing Y-axis in its base frame is:
        [-sin(theta_r), cos(theta_p)cos(theta_r), sin(theta_p)cos(theta_r)]

        Parameters
        ----------
        target_y_axis_in_jaco_hand_frame : np.ndarray
            A 3D vector representing the desired orientation of the Articutool's
            pointing Y-axis, expressed in the Jaco hand frame.

        Returns
        -------
        List[Tuple[float, float]]
            A list of (theta_p, theta_r) solutions in radians.
        """
        vx, vy, vz = target_y_axis_in_jaco_hand_frame
        solutions: List[Tuple[float, float]] = []

        # From benchmark: asin_arg = np.clip(-vx, -1.0, 1.0)
        # This means: target_x_in_jaco_hand = -sin(theta_r)
        if not (-1.0 - self.EPSILON <= -vx <= 1.0 + self.EPSILON):
            self.logger.debug(
                f"[{self.name}] IK: -vx ({vx:.4f}) out of range for asin."
            )
            return []

        asin_arg_for_theta_r = np.clip(-vx, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_theta_r)
        theta_r_sol2 = self._normalize_angle(
            math.pi - theta_r_sol1
        )  # Second solution for asin

        candidate_thetas_r = [theta_r_sol1]
        if not math.isclose(theta_r_sol1, theta_r_sol2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_sol2)

        # self.logger.debug(f"IK: Target Y in JacoHand: [{vx:.3f}, {vy:.3f}, {vz:.3f}]")
        # self.logger.debug(f"IK: Candidate theta_r values: {[round(r, 3) for r in candidate_thetas_r]}")

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)

            # From benchmark: theta_p_sol = math.atan2(vz / cos_theta_r, vy / cos_theta_r)
            # This means: target_y_in_jaco_hand = cos(theta_p)cos(theta_r)
            #             target_z_in_jaco_hand = sin(theta_p)cos(theta_r)
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                # If cos(theta_r) is 0, then sin(theta_r) is +/-1.
                # This means -vx should be +/-1 (target_x_in_jaco_hand is +/-1).
                # Also, target_y_in_jaco_hand and target_z_in_jaco_hand must be 0.
                if math.isclose(vy, 0.0, abs_tol=self.EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=self.EPSILON
                ):
                    # theta_p is indeterminate (or can be chosen freely, e.g., 0)
                    # This configuration means the tool is pointing purely along Jaco hand's X-axis.
                    theta_p_sol = 0.0  # Choose a default, like 0
                    solutions.append((theta_p_sol, theta_r))
                    # self.logger.debug(f"IK: cos(theta_r) is zero, vy, vz also zero. Added solution ({theta_p_sol:.3f}, {theta_r:.3f})")
                # else:
                # self.logger.debug(f"IK: cos(theta_r) is zero, but vy or vz is non-zero. Infeasible for this theta_r.")
                continue  # Move to next theta_r candidate if any

            # Normal case: cos(theta_r) is not zero
            # theta_p = atan2( (sin(theta_p)cos(theta_r)) , (cos(theta_p)cos(theta_r)) )
            # theta_p = atan2( vz , vy )
            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
            # self.logger.debug(f"IK: For theta_r={theta_r:.3f}, got theta_p={theta_p_sol:.3f}. Added solution.")

        return solutions

    @override
    def initialise(self) -> None:
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("articutool_is_orientation_feasible", None)
        # self.blackboard_set("first_failing_articutool_config", None)
        # self.blackboard_set("first_failing_jaco_ee_pose_idx", None)

    @override
    def update(self) -> Status:
        if not self.node:
            self.feedback_message = "Node not initialized"
            return Status.FAILURE
        if self._pitch_limits is None or self._roll_limits is None:
            self.feedback_message = "Articutool joint limits not set up correctly."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        try:
            trajectory: RobotTrajectory = self.blackboard_get(
                "jaco_ee_cartesian_trajectory"
            )
            desired_orientation_msg: QuaternionMsg = self.blackboard_get(
                "desired_tool_tip_world_orientation"
            )
            num_points_to_check: int = self.blackboard_get(
                "num_trajectory_points_to_check"
            )

            if not isinstance(
                trajectory, RobotTrajectory
            ):  # No points check needed if not RobotTrajectory
                self.feedback_message = (
                    "Input 'jaco_ee_cartesian_trajectory' is not a RobotTrajectory."
                )
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE
            if not isinstance(desired_orientation_msg, QuaternionMsg):
                self.feedback_message = (
                    "Input 'desired_tool_tip_world_orientation' is not a Quaternion."
                )
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            # Check if trajectory has points
            # We expect poses in multi_dof_joint_trajectory for Cartesian paths
            if not trajectory.multi_dof_joint_trajectory.points:
                if trajectory.joint_trajectory.points:
                    self.feedback_message = "Trajectory has joint points but no multi_dof (Cartesian) points. FK needed but not implemented here."
                    self.logger.error(f"[{self.name}] {self.feedback_message}")
                else:
                    self.feedback_message = "Input Jaco EE trajectory is empty."
                    self.logger.warn(
                        f"[{self.name}] {self.feedback_message} Considering feasible as no points to fail."
                    )
                # If no points, it's vacuously feasible, or could be failure depending on requirements.
                # Let's say it's feasible if no points to check.
                self.blackboard_set("articutool_is_orientation_feasible", True)
                return Status.SUCCESS

            R_World_TipTarget = Rotation.from_quat(
                [
                    desired_orientation_msg.x,
                    desired_orientation_msg.y,
                    desired_orientation_msg.z,
                    desired_orientation_msg.w,
                ]
            )
            # This is the Y-axis of the desired tool tip orientation, expressed in World frame.
            # This is the vector that the Articutool's "pointing Y-axis" should align with in the world.
            y_axis_TipTarget_InWorld = R_World_TipTarget.apply(
                np.array([0.0, 1.0, 0.0])
            )

            jaco_ee_poses_from_traj: List[Pose] = []
            for mjt_point in trajectory.multi_dof_joint_trajectory.points:
                if mjt_point.transforms:
                    transform = mjt_point.transforms[
                        0
                    ]  # Assuming first transform is the EE
                    pose = Pose()
                    pose.position.x = transform.translation.x
                    pose.position.y = transform.translation.y
                    pose.position.z = transform.translation.z
                    pose.orientation = transform.rotation
                    jaco_ee_poses_from_traj.append(pose)

            if not jaco_ee_poses_from_traj:
                self.feedback_message = (
                    "No poses extracted from multi_dof_joint_trajectory."
                )
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                self.blackboard_set(
                    "articutool_is_orientation_feasible", True
                )  # Vacuously true
                return Status.SUCCESS

            indices_to_check = []
            if len(jaco_ee_poses_from_traj) == 1:
                indices_to_check = [0]
            elif num_points_to_check >= len(
                jaco_ee_poses_from_traj
            ):  # Check all if fewer than num_points
                indices_to_check = list(range(len(jaco_ee_poses_from_traj)))
            elif num_points_to_check > 0:
                indices_to_check = np.linspace(
                    0, len(jaco_ee_poses_from_traj) - 1, num_points_to_check, dtype=int
                )

            if (
                not indices_to_check and num_points_to_check > 0
            ):  # Should not happen if jaco_ee_poses_from_traj is not empty
                self.logger.warn(
                    f"[{self.name}] No indices to check despite having trajectory points and num_points_to_check > 0."
                )
                self.blackboard_set("articutool_is_orientation_feasible", True)
                return Status.SUCCESS

            for point_idx_in_indices, actual_traj_idx in enumerate(indices_to_check):
                jaco_ee_pose_msg: Pose = jaco_ee_poses_from_traj[actual_traj_idx]

                R_World_JacoEE = Rotation.from_quat(
                    [
                        jaco_ee_pose_msg.orientation.x,
                        jaco_ee_pose_msg.orientation.y,
                        jaco_ee_pose_msg.orientation.z,
                        jaco_ee_pose_msg.orientation.w,
                    ]
                )

                # Transform the desired tool tip Y-axis (in World) into the Jaco EE (Articutool Base) frame
                target_y_axis_for_ik_in_jaco_hand = R_World_JacoEE.inv().apply(
                    y_axis_TipTarget_InWorld
                )

                # Normalize for robustness
                norm_target_y = np.linalg.norm(target_y_axis_for_ik_in_jaco_hand)
                if norm_target_y < self.EPSILON:
                    self.feedback_message = f"Target Y-axis for IK became zero vector at trajectory point {actual_traj_idx}."
                    self.logger.error(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_orientation_feasible", False)
                    return Status.FAILURE
                target_y_axis_for_ik_in_jaco_hand /= norm_target_y

                ik_solutions = self._solve_articutool_ik_benchmark_style(
                    target_y_axis_for_ik_in_jaco_hand
                )

                found_valid_solution_for_waypoint = False
                best_solution_this_waypoint = None

                for theta_p_sol, theta_r_sol in ik_solutions:
                    # Normalize angles before checking limits
                    theta_p_norm = self._normalize_angle(theta_p_sol)
                    theta_r_norm = self._normalize_angle(theta_r_sol)

                    if (
                        self._pitch_limits[0] - self.EPSILON
                        <= theta_p_norm
                        <= self._pitch_limits[1] + self.EPSILON
                        and self._roll_limits[0] - self.EPSILON
                        <= theta_r_norm
                        <= self._roll_limits[1] + self.EPSILON
                    ):
                        found_valid_solution_for_waypoint = True
                        best_solution_this_waypoint = (
                            theta_p_norm,
                            theta_r_norm,
                        )  # Take the first valid one
                        break

                if not found_valid_solution_for_waypoint:
                    self.feedback_message = (
                        f"Articutool IK failed or solution out of limits at trajectory point index {actual_traj_idx}. "
                        f"Target Y in JacoHand: {np.round(target_y_axis_for_ik_in_jaco_hand, 3)}. "
                        f"Raw IK solutions: {[(round(s[0], 3), round(s[1], 3)) for s in ik_solutions]}"
                    )
                    self.logger.warn(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_orientation_feasible", False)
                    # self.blackboard_set("first_failing_jaco_ee_pose_idx", actual_traj_idx)
                    return Status.FAILURE

                self.logger.debug(
                    f"[{self.name}] Pt {actual_traj_idx}: JacoEE Ori (quat xyzw): "
                    f"[{jaco_ee_pose_msg.orientation.x:.2f}, ..., {jaco_ee_pose_msg.orientation.w:.2f}], "
                    f"Target Y_JacoHand: {np.round(target_y_axis_for_ik_in_jaco_hand, 2)}, "
                    f"Articutool q_sol: [{best_solution_this_waypoint[0]:.3f}, {best_solution_this_waypoint[1]:.3f}]"
                )

            self.feedback_message = (
                "Articutool can maintain orientation throughout Jaco trajectory."
            )
            self.logger.info(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_orientation_feasible", True)
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
