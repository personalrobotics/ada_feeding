# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the CheckArticutoolPathDynamicFeasibility behavior.

This behavior extends the kinematic feasibility check by also verifying if the
Articutool has the dynamic capability (i.e., sufficient motor velocity) to
maintain a level orientation throughout a given trajectory of the Jaco arm.
It combines the kinematic check with a dynamic analysis at each waypoint.
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple
import traceback

from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation
import py_trees
from py_trees.common import Status
import rclpy.node
import pinocchio as pin

from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class CheckArticutoolPathDynamicFeasibility(BlackboardBehavior):
    """
    Checks if a Jaco trajectory is dynamically feasible for the Articutool.

    For each point along a trajectory, this behavior performs a two-part check:
    1.  Kinematic Check: Can the Articutool *be* level at this Jaco pose?
        (Same as CheckArticutoolPathLevelingFeasibility).
    2.  Dynamic Check: Is the Articutool *fast enough* to counteract the
        tilting motion induced by the Jaco arm's movement?

    If both checks pass for all waypoints, it returns SUCCESS. Otherwise, FAILURE.
    """

    EPSILON = 1e-6
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])

    def blackboard_inputs(
        self,
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        jaco_joint_names_pin: Union[BlackboardKey, List[str]],
        jaco_vel_indices_pin: Union[BlackboardKey, List[int]],
        jaco_ee_frame_id_pin: Union[BlackboardKey, int],
        jaco_trajectory: Union[BlackboardKey, RobotTrajectory, JointTrajectory],
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_max_joint_velocity: Union[BlackboardKey, float],
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 10,
    ) -> None:
        """Define blackboard inputs."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_is_dynamic_feasible: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        """Initialize the behavior."""
        super().__init__(name=name, **kwargs)
        self.node: Optional[rclpy.node.Node] = None
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._jaco_joint_names_pin: Optional[List[str]] = None
        self._jaco_vel_indices_pin: Optional[List[int]] = None
        self._jaco_ee_frame_id_pin: Optional[int] = None
        self._pitch_limits_rad: Optional[Tuple[float, float]] = None
        self._roll_limits_rad: Optional[Tuple[float, float]] = None
        self._max_atool_vel: Optional[float] = None
        self._pinocchio_ready = False

    @override
    def setup(self, **kwargs):
        """Get the ROS2 node from kwargs."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )

    def _get_pinocchio_essentials_from_blackboard(self) -> bool:
        """Read Pinocchio model, data, and relevant IDs/names from blackboard."""
        if self._pinocchio_ready:
            return True
        try:
            self._pin_model = self.blackboard_get("pinocchio_model")
            self._pin_data = self.blackboard_get("pinocchio_data")
            self._jaco_joint_names_pin = self.blackboard_get("jaco_joint_names_pin")
            self._jaco_vel_indices_pin = self.blackboard_get("jaco_vel_indices_pin")
            self._jaco_ee_frame_id_pin = self.blackboard_get("jaco_ee_frame_id_pin")
            self._pitch_limits_rad = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits_rad = self.blackboard_get("articutool_roll_limits_rad")
            self._max_atool_vel = self.blackboard_get("articutool_max_joint_velocity")
            self._pinocchio_ready = True
            return True
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Failed to get required info from blackboard: {e}."
            )
            return False

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_for_leveling(
        self, target_y_axis_in_atool_base: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Analytical IK solver based on the reconciled kinematic model.
        Solves for Articutool (pitch, roll) to align its tool_tip Y-axis
        with the given target vector expressed in the Articutool's base frame.
        """
        vx, vy, vz = target_y_axis_in_atool_base
        solutions: List[Tuple[float, float]] = []

        asin_arg_for_tr = -vx
        if not (-1.0 - self.EPSILON <= asin_arg_for_tr <= 1.0 + self.EPSILON):
            return []

        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = [theta_r_sol1]
        if not math.isclose(theta_r_sol1, theta_r_sol2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_sol2)

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                if math.isclose(vy, 0.0, abs_tol=self.EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=self.EPSILON
                ):
                    solutions.append((0.0, self._normalize_angle(theta_r)))
                continue

            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
        return solutions

    def _compute_articutool_jacobian(
        self, theta_p: float, theta_r: float
    ) -> np.ndarray:
        """
        Computes the 3x2 analytical Jacobian for the Articutool leveling task.
        This must be derived from the same FK model that the IK inverts.
        """
        cp, sp = math.cos(theta_p), math.sin(theta_p)
        cr, sr = math.cos(theta_r), math.sin(theta_r)

        # Partial derivatives of y_F0 = [-sr, cp*cr, sp*cr]^T
        # d/d(theta_p)
        j11 = 0
        j21 = -sp * cr
        j31 = cp * cr

        # d/d(theta_r)
        j12 = -cr
        j22 = -cp * sr
        j32 = -sp * sr

        return np.array([[j11, j12], [j21, j22], [j31, j32]])

    def _get_jaco_trajectory_points(
        self, trajectory_input: Union[RobotTrajectory, JointTrajectory]
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Extracts joint positions and velocities from a trajectory message."""
        if isinstance(trajectory_input, RobotTrajectory):
            jt = trajectory_input.joint_trajectory
        elif isinstance(trajectory_input, JointTrajectory):
            jt = trajectory_input
        else:
            return None

        points = []
        for point in jt.points:
            positions = np.array(point.positions)
            velocities = np.array(point.velocities)
            points.append((positions, velocities))
        return points

    @override
    def update(self) -> Status:
        """Execute the behavior's logic."""
        if not self.node or not self._get_pinocchio_essentials_from_blackboard():
            self.feedback_message = (
                "Behavior not properly initialized or blackboard inputs missing."
            )
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE

        try:
            trajectory_input = self.blackboard_get("jaco_trajectory")
            num_points = self.blackboard_get("num_trajectory_points_to_check")

            jaco_points = self._get_jaco_trajectory_points(trajectory_input)

            if not jaco_points:
                self.logger.warn(
                    f"[{self.name}] Trajectory is empty. Assuming feasible."
                )
                self.blackboard_set("articutool_is_dynamic_feasible", True)
                return Status.SUCCESS

            indices = np.linspace(0, len(jaco_points) - 1, num_points, dtype=int)

            last_valid_atool_q = None

            for idx in indices:
                q_jaco_partial, v_jaco_partial = jaco_points[idx]

                # Construct the full configuration vector for Pinocchio
                q_full_robot = pin.neutral(self._pin_model)
                for i, joint_name in enumerate(self._jaco_joint_names_pin):
                    joint_id = self._pin_model.getJointId(joint_name)
                    joint_obj = self._pin_model.joints[joint_id]
                    theta = q_jaco_partial[i]
                    if joint_obj.nq == 2 and joint_obj.nv == 1:
                        q_full_robot[joint_obj.idx_q] = math.cos(theta)
                        q_full_robot[joint_obj.idx_q + 1] = math.sin(theta)
                    elif joint_obj.nq == 1 and joint_obj.nv == 1:
                        q_full_robot[joint_obj.idx_q] = theta

                pin.forwardKinematics(
                    self._pin_model, self._pin_data, q_full_robot
                )  # Use full vector
                pin.updateFramePlacements(self._pin_model, self._pin_data)
                T_world_atool_base = self._pin_data.oMf[self._jaco_ee_frame_id_pin]
                R_world_atool_base = Rotation.from_matrix(T_world_atool_base.rotation)

                # 2. Transform world "up" vector to Articutool's base frame
                target_y_in_atool_base = R_world_atool_base.inv().apply(
                    self.WORLD_Z_UP_VECTOR
                )

                # 3. Solve Articutool IK for leveling
                ik_solutions = self._solve_articutool_ik_for_leveling(
                    target_y_in_atool_base
                )

                valid_solutions = [
                    sol
                    for sol in ik_solutions
                    if self._pitch_limits_rad[0] - self.EPSILON
                    <= sol[0]
                    <= self._pitch_limits_rad[1] + self.EPSILON
                    and self._roll_limits_rad[0] - self.EPSILON
                    <= sol[1]
                    <= self._roll_limits_rad[1] + self.EPSILON
                ]

                if not valid_solutions:
                    self.feedback_message = (
                        f"Path is kinematically infeasible at point {idx}."
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

                # Choose best solution (e.g., closest to previous)
                if last_valid_atool_q is None:
                    q_atool = valid_solutions[0]
                else:
                    distances = [
                        np.linalg.norm(np.array(sol) - last_valid_atool_q)
                        for sol in valid_solutions
                    ]
                    q_atool = valid_solutions[np.argmin(distances)]
                last_valid_atool_q = np.array(q_atool)

                # --- DYNAMIC FEASIBILITY CHECK ---

                # 1. Calculate disturbance velocity from Jaco arm
                J_jaco_full = pin.computeFrameJacobian(
                    self._pin_model,
                    self._pin_data,
                    q_full_robot,  # Use full vector
                    self._jaco_ee_frame_id_pin,
                    pin.ReferenceFrame.WORLD,
                )

                # Construct full velocity vector
                v_full_robot = np.zeros(self._pin_model.nv)
                v_full_robot[self._jaco_vel_indices_pin] = v_jaco_partial

                v_jaco_full = J_jaco_full @ v_full_robot
                omega_disturbance_world = v_jaco_full[3:6]
                omega_correction_local = -R_world_atool_base.inv().apply(
                    omega_disturbance_world
                )

                # 3. Compute Articutool's Jacobian at the required configuration
                J_atool = self._compute_articutool_jacobian(q_atool[0], q_atool[1])

                # 4. Solve for required Articutool joint velocities
                try:
                    J_atool_pinv = np.linalg.pinv(J_atool, rcond=1e-4)
                    q_dot_atool_required = J_atool_pinv @ omega_correction_local
                except np.linalg.LinAlgError:
                    self.feedback_message = (
                        f"Articutool Jacobian is singular at point {idx}."
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

                # 5. The final check
                if np.linalg.norm(q_dot_atool_required) > self._max_atool_vel:
                    self.feedback_message = (
                        f"Path is dynamically infeasible at point {idx}. "
                        f"Required vel: {np.linalg.norm(q_dot_atool_required):.2f} rad/s > "
                        f"Max vel: {self._max_atool_vel:.2f} rad/s"
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

            # If all checks pass
            self.feedback_message = "Trajectory is dynamically feasible."
            self.blackboard_set("articutool_is_dynamic_feasible", True)
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(
                f"[{self.name}] {self.feedback_message}\n{traceback.format_exc()}"
            )
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")

    @override
    def initialise(self) -> None:
        """Reset blackboard output on initialization."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("articutool_is_dynamic_feasible", None)
