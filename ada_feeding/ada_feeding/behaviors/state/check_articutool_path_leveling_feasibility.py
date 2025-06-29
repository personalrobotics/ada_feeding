# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the CheckPathLevelingFeasibility behavior.

This behavior checks if the Articutool can maintain a level orientation
for its tool tip throughout a given trajectory of the Jaco arm. It ensures
not only that an IK solution exists at each point, but that a continuous
path of solutions can be traced through the joint space of the Articutool,
preventing control instabilities near kinematic singularities.
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple

# Third-party imports
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation
import py_trees
import py_trees.blackboard
from py_trees.common import Status
import rclpy.node
import pinocchio as pin

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CheckArticutoolPathLevelingFeasibility(BlackboardBehavior):
    """
    Checks Articutool's ability to maintain a level orientation for its tool tip
    throughout a Jaco arm trajectory. It verifies that a continuous, within-limits
    path of IK solutions exists for the Articutool.
    """

    EPSILON = 1e-6
    WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])

    def blackboard_inputs(
        self,
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        jaco_joint_names_pin: Union[BlackboardKey, List[str]],
        jaco_ee_frame_id_pin: Union[BlackboardKey, int],
        jaco_trajectory: Union[BlackboardKey, RobotTrajectory, JointTrajectory],
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 20,
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        is_leveling_path_feasible: Optional[BlackboardKey],
    ) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[rclpy.node.Node] = None
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._jaco_joint_names_pin: Optional[List[str]] = None
        self._jaco_ee_frame_id_pin: Optional[int] = None
        self._pitch_limits_rad: Optional[Tuple[float, float]] = None
        self._roll_limits_rad: Optional[Tuple[float, float]] = None
        self._pinocchio_ready = False

    @override
    def setup(self, **kwargs):
        """Gets the ROS2 node."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )

    @override
    def initialise(self) -> None:
        """Reads Pinocchio model and other static info from blackboard."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("is_leveling_path_feasible", None)
        self._pinocchio_ready = self._get_pinocchio_essentials_from_blackboard()

    def _get_pinocchio_essentials_from_blackboard(self) -> bool:
        """Reads Pinocchio model, data, and relevant IDs/names from blackboard."""
        if self._pinocchio_ready:
            return True
        try:
            self._pin_model = self.blackboard_get("pinocchio_model")
            self._pin_data = self.blackboard_get("pinocchio_data")
            self._jaco_joint_names_pin = self.blackboard_get("jaco_joint_names_pin")
            self._jaco_ee_frame_id_pin = self.blackboard_get("jaco_ee_frame_id_pin")
            self._pitch_limits_rad = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits_rad = self.blackboard_get("articutool_roll_limits_rad")
            return True
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Failed to get Pinocchio info from blackboard: {e}."
            )
            return False

    def _get_jaco_ee_poses_from_trajectory(
        self, trajectory_input: Union[RobotTrajectory, JointTrajectory]
    ) -> Optional[List[Pose]]:
        """Extracts or computes Jaco end-effector poses from a trajectory message."""
        if not self._pinocchio_ready:
            return None

        jaco_ee_world_poses: List[Pose] = []
        # Simplified logic: Assumes trajectory is a JointTrajectory for the Jaco arm.
        # A more robust version would handle RobotTrajectory and other cases like in the original file.
        if not isinstance(trajectory_input, JointTrajectory):
            self.logger.error(
                f"[{self.name}] This simplified checker only supports JointTrajectory input."
            )
            return None

        jt_points = trajectory_input.points
        jt_joint_names = trajectory_input.joint_names
        q_robot = pin.neutral(self._pin_model)

        for point_data in jt_points:
            temp_jaco_map = {
                name: point_data.positions[i] for i, name in enumerate(jt_joint_names)
            }

            for j_name_pin in self._jaco_joint_names_pin:
                if j_name_pin in temp_jaco_map:
                    joint_id = self._pin_model.getJointId(j_name_pin)
                    joint_obj = self._pin_model.joints[joint_id]
                    theta = temp_jaco_map[j_name_pin]
                    if joint_obj.nq == 2:  # Revolute
                        q_robot[joint_obj.idx_q] = math.cos(theta)
                        q_robot[joint_obj.idx_q + 1] = math.sin(theta)

            pin.forwardKinematics(self._pin_model, self._pin_data, q_robot)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            jaco_ee_transform_pin: pin.SE3 = self._pin_data.oMf[
                self._jaco_ee_frame_id_pin
            ]

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                jaco_ee_transform_pin.translation
            )
            quat_xyzw = Rotation.from_matrix(jaco_ee_transform_pin.rotation).as_quat()
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = quat_xyzw
            jaco_ee_world_poses.append(pose)

        return jaco_ee_world_poses

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to be within [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik(
        self, target_y_in_atool_base: np.ndarray
    ) -> List[np.ndarray]:
        """
        Solves the analytical IK for the Articutool to achieve a level orientation.
        `target_y_in_atool_base` is the desired "up" vector of the tool tip, expressed in the Articutool's base frame.
        """
        vx, vy, vz = target_y_in_atool_base
        solutions = []

        # Solve for theta_r
        asin_arg = -vx
        if not (-1.0 - self.EPSILON <= asin_arg <= 1.0 + self.EPSILON):
            return []

        theta_r_sol1 = math.asin(np.clip(asin_arg, -1.0, 1.0))
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = {
            self._normalize_angle(theta_r_sol1),
            self._normalize_angle(theta_r_sol2),
        }

        # Solve for theta_p for each valid theta_r
        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            if abs(cos_theta_r) > self.EPSILON:
                theta_p = math.atan2(vz, vy)
                solutions.append(np.array([self._normalize_angle(theta_p), theta_r]))

        return solutions

    @override
    def update(self) -> Status:
        if not self.node or not self._pinocchio_ready:
            self.feedback_message = "Node or Pinocchio model not ready."
            return Status.FAILURE

        try:
            trajectory_input = self.blackboard_get("jaco_trajectory")
            num_points_to_check = self.blackboard_get("num_trajectory_points_to_check")

            jaco_ee_poses = self._get_jaco_ee_poses_from_trajectory(trajectory_input)

            if jaco_ee_poses is None:
                self.feedback_message = (
                    "Could not extract Jaco EE poses from trajectory."
                )
                return Status.FAILURE

            if not jaco_ee_poses:
                self.logger.warn(
                    f"[{self.name}] Trajectory has no waypoints. Assuming feasible."
                )
                self.blackboard_set("is_leveling_path_feasible", True)
                return Status.SUCCESS

            indices_to_check = np.linspace(
                0, len(jaco_ee_poses) - 1, num_points_to_check, dtype=int
            )

            last_valid_solution = None

            for traj_idx in indices_to_check:
                ee_pose = jaco_ee_poses[traj_idx]
                R_World_JacoEE = Rotation.from_quat(
                    [
                        ee_pose.orientation.x,
                        ee_pose.orientation.y,
                        ee_pose.orientation.z,
                        ee_pose.orientation.w,
                    ]
                )

                # Transform world "up" vector into the Jaco EE's local frame
                target_y_in_atool_base = R_World_JacoEE.inv().apply(
                    self.WORLD_UP_VECTOR
                )

                # Get all IK solutions for this orientation
                ik_solutions = self._solve_articutool_ik(target_y_in_atool_base)

                # Filter for solutions that are within joint limits
                valid_solutions = [
                    sol
                    for sol in ik_solutions
                    if (
                        self._pitch_limits_rad[0] <= sol[0] <= self._pitch_limits_rad[1]
                        and self._roll_limits_rad[0]
                        <= sol[1]
                        <= self._roll_limits_rad[1]
                    )
                ]

                if not valid_solutions:
                    self.feedback_message = f"Infeasible: No valid IK solution at trajectory point {traj_idx}."
                    self.logger.warn(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("is_leveling_path_feasible", False)
                    return Status.FAILURE

                # Select the best solution based on continuity
                if last_valid_solution is None:
                    chosen_solution = valid_solutions[0]
                else:
                    # Subsequent points: choose solution closest to the previous chosen solution
                    distances = [
                        np.linalg.norm(sol - last_valid_solution)
                        for sol in valid_solutions
                    ]
                    chosen_solution = valid_solutions[np.argmin(distances)]

                last_valid_solution = chosen_solution

            self.feedback_message = "Path is feasible for continuous leveling."
            self.logger.info(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("is_leveling_path_feasible", True)
            return Status.SUCCESS

        except Exception as e:
            self.feedback_message = f"Unexpected error during feasibility check: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
