# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the ComputeArticutoolLevelingJoints behavior.
Given the Jaco arm's end-effector world pose, this behavior calculates
the Articutool joint angles (pitch, roll) required to make the
Articutool's tool_tip Y-axis point upwards against gravity (world +Z).
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple

# Third-party imports
from geometry_msgs.msg import Pose, PoseStamped, Quaternion as QuaternionMsg
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation as R
import py_trees
from py_trees.common import Status, Access
import rclpy
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import BlackboardBehavior  # Assuming this is your base class
from ada_feeding.helpers import BlackboardKey


class ComputeArticutoolLevelingJoints(BlackboardBehavior):
    """
    Calculates Articutool joint angles for its tool_tip Y-axis to align with World +Z.
    """

    EPSILON = 1e-6
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])  # Target direction in World Frame

    def blackboard_inputs(
        self,
        jaco_ee_world_pose: Union[
            BlackboardKey, Pose, PoseStamped
        ],  # Current pose of Jaco EE in world
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]] = (
            -math.pi / 2,
            math.pi / 2,
        ),
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]] = (
            -math.pi,
            math.pi,
        ),
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_joint_positions: Union[BlackboardKey, List[float]],
        articutool_leveling_ik_found: Optional[BlackboardKey],
    ) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[Node] = None
        self._initialized_properly = False

    @override
    def setup(self, **kwargs):
        try:
            self.node = kwargs["node"]
            self._initialized_properly = True
            self.logger.debug(f"[{self.name}] Behavior initialized.")
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            self._initialized_properly = False
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during setup: {e}")
            self._initialized_properly = False

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_for_leveling(
        self, target_y_axis_in_jaco_hand_frame: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Analytical IK solver for the 2-DOF Articutool.
        Input: target_y_axis_in_jaco_hand_frame (3D numpy array) - This is the desired
               pointing direction of the Articutool's tool_tip Y-axis, expressed in the
               Jaco Hand Frame (Articutool's base frame, F_JH).
        Output: List of (theta_p, theta_r) solutions in radians.
        This IK is based on the one used in the benchmark and CheckArticutoolPathOrientationFeasibility.
        It assumes the relationship where the input vector (vx, vy, vz) from F_JH
        is related to the Articutool's FK $y_{tip}^{AB}(\theta_p, \theta_r) = (X_{AB}, Y_{AB}, Z_{AB})$
        after transformation $y_{tip}^{AB} = R_z(-\pi/2) \cdot (v_x, v_y, v_z)^T = (v_y, -v_x, v_z)^T$.
        So, the IK solves for:
        $\cos\theta_p \cos\theta_r = v_y$
        $\sin\theta_r = -v_x$
        $\sin\theta_p \cos\theta_r = v_z$
        """
        vx, vy, vz = (
            target_y_axis_in_jaco_hand_frame  # This is the target vector in Jaco Hand Frame
        )
        solutions: List[Tuple[float, float]] = []

        # From sin(theta_r) = -vx
        asin_arg_for_tr = -vx
        if not (-1.0 - self.EPSILON <= asin_arg_for_tr <= 1.0 + self.EPSILON):
            self.logger.debug(
                f"[{self.name}] IK: asin_arg_for_tr ({asin_arg_for_tr:.4f}) out of range for -vx={-vx:.4f}."
            )
            return []  # No real solution for theta_r

        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = [theta_r_sol1]
        if not math.isclose(theta_r_sol1, theta_r_sol2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_sol2)

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            # From cos(theta_p)cos(theta_r) = vy  AND  sin(theta_p)cos(theta_r) = vz
            # If cos(theta_r) is near zero (singularity):
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                # Then vy and vz must also be near zero.
                # If so, theta_p is indeterminate but can be chosen (e.g., 0).
                # This case also means sin(theta_r) = +/-1, so -vx = +/-1.
                if math.isclose(vy, 0.0, abs_tol=self.EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=self.EPSILON
                ):
                    # This implies target vector was (∓1, 0, 0) in Jaco Hand Frame.
                    # This would mean Artic Y is trying to point along Jaco Hand +/-X.
                    solutions.append(
                        (0.0, self._normalize_angle(theta_r))
                    )  # Choose theta_p = 0
                # else: (vy or vz is not zero, but cos_theta_r is zero) -> No solution
                continue  # Skip to next theta_r if singular and conditions not met

            # Regular case: cos_theta_r is not zero
            # theta_p = atan2(sin_theta_p, cos_theta_p)
            # theta_p = atan2(vz / cos_theta_r, vy / cos_theta_r) which simplifies to atan2(vz, vy)
            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
        return solutions

    @override
    def update(self) -> Status:
        if not self._initialized_properly:
            self.feedback_message = "Behavior not properly initialized."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_leveling_ik_found", False)
            return Status.FAILURE

        try:
            jaco_ee_pose_input: Union[Pose, PoseStamped] = self.blackboard_get(
                "jaco_ee_world_pose"
            )
            pitch_limits: Tuple[float, float] = self.blackboard_get(
                "articutool_pitch_limits_rad"
            )
            roll_limits: Tuple[float, float] = self.blackboard_get(
                "articutool_roll_limits_rad"
            )

            current_jaco_ee_pose: Pose
            if isinstance(jaco_ee_pose_input, PoseStamped):
                current_jaco_ee_pose = jaco_ee_pose_input.pose
            elif isinstance(jaco_ee_pose_input, Pose):
                current_jaco_ee_pose = jaco_ee_pose_input
            else:
                self.feedback_message = (
                    "jaco_ee_world_pose is not a Pose or PoseStamped."
                )
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                self.blackboard_set("articutool_leveling_ik_found", False)
                return Status.FAILURE

            # 1. Extract R_World_JacoEE from the Jaco EE's world pose
            R_world_jacoee = R.from_quat(
                [
                    current_jaco_ee_pose.orientation.x,
                    current_jaco_ee_pose.orientation.y,
                    current_jaco_ee_pose.orientation.z,
                    current_jaco_ee_pose.orientation.w,
                ]
            )

            # 2. Desired tool_tip Y-axis in World Frame is WORLD_Z_UP_VECTOR
            # Transform this into the Jaco EE frame (F_JH)
            # target_Ytip_in_JacoEE = R_JacoEE_World * WORLD_Z_UP_VECTOR
            target_Ytip_in_JacoEE = R_world_jacoee.inv().apply(self.WORLD_Z_UP_VECTOR)

            self.logger.debug(
                f"[{self.name}] Jaco EE R_W_EE: {R_world_jacoee.as_quat(canonical=False)}"
            )
            self.logger.debug(
                f"[{self.name}] Target Y_tip in JacoEE frame: {np.round(target_Ytip_in_JacoEE, 3)}"
            )

            # 3. Solve Articutool IK
            ik_solutions = self._solve_articutool_ik_for_leveling(target_Ytip_in_JacoEE)

            if not ik_solutions:
                self.feedback_message = "Articutool IK found no solutions for leveling."
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                self.blackboard_set("articutool_leveling_ik_found", False)
                return (
                    Status.FAILURE
                )  # Or SUCCESS if allowing no solution to mean "do nothing"

            # 4. Select a valid solution (e.g., first one within limits, or closest to current)
            # For simplicity, taking the first valid one.
            valid_solution_found = False
            best_pitch = None
            best_roll = None

            for theta_p_sol, theta_r_sol in ik_solutions:
                # Normalize again just to be safe, though IK solver should do it
                tp_norm = self._normalize_angle(theta_p_sol)
                tr_norm = self._normalize_angle(theta_r_sol)

                if (
                    pitch_limits[0] - self.EPSILON
                    <= tp_norm
                    <= pitch_limits[1] + self.EPSILON
                    and roll_limits[0] - self.EPSILON
                    <= tr_norm
                    <= roll_limits[1] + self.EPSILON
                ):
                    best_pitch = tp_norm
                    best_roll = tr_norm
                    valid_solution_found = True
                    self.logger.info(
                        f"[{self.name}] Found valid Articutool leveling solution: "
                        f"Pitch={math.degrees(best_pitch):.1f} deg, Roll={math.degrees(best_roll):.1f} deg"
                    )
                    self.blackboard_set(
                        "articutool_joint_positions", [best_pitch, best_roll]
                    )
                    self.blackboard_set("articutool_leveling_ik_found", True)
                    self.feedback_message = "Articutool leveling IK solution found."
                    return Status.SUCCESS

            self.feedback_message = (
                "Articutool IK solutions for leveling are out of joint limits."
            )
            self.logger.warn(
                f"[{self.name}] {self.feedback_message} Solutions: {[(math.degrees(s[0]), math.degrees(s[1])) for s in ik_solutions]}"
            )
            self.blackboard_set("articutool_leveling_ik_found", False)
            return Status.FAILURE

        except KeyError as e:  # For blackboard_get errors
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_leveling_ik_found", False)
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error in {self.name}: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_leveling_ik_found", False)
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
