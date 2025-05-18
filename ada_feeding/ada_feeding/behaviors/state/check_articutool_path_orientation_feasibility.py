# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the CheckArticutoolPathOrientationFeasibility behavior.
This behavior checks if the Articutool can maintain a desired world orientation
for its tool tip throughout a given trajectory of the Jaco arm.
It uses Pinocchio for FK if the Jaco trajectory is in joint space,
and an analytical IK for the Articutool.
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple

# Third-party imports
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion as QuaternionMsg,
    Pose,
    TransformStamped,
)
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
    JointTrajectory,
    MultiDOFJointTrajectoryPoint,
)
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


class CheckArticutoolPathOrientationFeasibility(BlackboardBehavior):
    """
    Checks Articutool's ability to maintain tool tip orientation during Jaco trajectory.
    Uses Pinocchio for FK if Jaco trajectory is joint-space.
    """

    EPSILON = 1e-6

    def blackboard_inputs(
        self,
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        # Jaco info for FK if needed
        jaco_joint_names_pin: Union[
            BlackboardKey, List[str]
        ],  # Names as in Pinocchio model for q mapping
        jaco_ee_frame_id_pin: Union[
            BlackboardKey, int
        ],  # Pinocchio frame ID for Jaco EE
        # Articutool info for its own q mapping if full robot q is constructed
        articutool_joint_names_pin: Union[
            BlackboardKey, List[str]
        ],  # Names as in Pinocchio model
        # Trajectory and target
        jaco_trajectory: Union[
            BlackboardKey, RobotTrajectory, JointTrajectory
        ],  # Can be full RobotTrajectory or just JointTrajectory
        desired_tool_tip_world_orientation: Union[BlackboardKey, QuaternionMsg],
        # Articutool specific limits for its IK
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 10,
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_is_orientation_feasible: Optional[BlackboardKey],
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
        self._articutool_joint_names_pin: Optional[List[str]] = None
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

    def _get_pinocchio_essentials_from_blackboard(self) -> bool:
        """Reads Pinocchio model, data, and relevant IDs/names from blackboard."""
        if self._pinocchio_ready:  # Avoid re-reading if already successful
            return True
        try:
            self._pin_model = self.blackboard_get("pinocchio_model")
            self._pin_data = self.blackboard_get("pinocchio_data")
            self._jaco_joint_names_pin = self.blackboard_get("jaco_joint_names_pin")
            self._jaco_ee_frame_id_pin = self.blackboard_get("jaco_ee_frame_id_pin")
            self._articutool_joint_names_pin = self.blackboard_get(
                "articutool_joint_names_pin"
            )
            self._pitch_limits_rad = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits_rad = self.blackboard_get("articutool_roll_limits_rad")

            if (
                not isinstance(self._pin_model, pin.Model)
                or not isinstance(self._pin_data, pin.Data)
                or not isinstance(self._jaco_joint_names_pin, list)
                or not isinstance(self._jaco_ee_frame_id_pin, int)
                or not isinstance(self._articutool_joint_names_pin, list)
                or not (
                    isinstance(self._pitch_limits_rad, tuple)
                    and len(self._pitch_limits_rad) == 2
                )
                or not (
                    isinstance(self._roll_limits_rad, tuple)
                    and len(self._roll_limits_rad) == 2
                )
            ):
                self.logger.error(
                    f"[{self.name}] One or more Pinocchio-related inputs from blackboard are invalid type/format."
                )
                return False
            self._pinocchio_ready = True
            return True
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Failed to get Pinocchio info from blackboard: {e}. Ensure LoadPinocchioModel ran."
            )
            return False

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_benchmark_style(
        self, target_y_axis_in_jaco_hand_frame: np.ndarray
    ) -> List[Tuple[float, float]]:
        vx, vy, vz = target_y_axis_in_jaco_hand_frame
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

    @override
    def initialise(self) -> None:
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("articutool_is_orientation_feasible", None)
        self._pinocchio_ready = False

    @override
    def update(self) -> Status:
        if not self.node:
            self.feedback_message = "Node not initialized"
            return Status.FAILURE
        if (
            not self._pinocchio_ready
        ):  # Attempt to load Pinocchio essentials if not already done
            if not self._get_pinocchio_essentials_from_blackboard():
                self.feedback_message = (
                    "Failed to get Pinocchio essentials from blackboard."
                )
                return Status.FAILURE
        # Now, self._pin_model, self._pin_data etc. should be populated if _get_pinocchio_essentials_from_blackboard was successful

        try:
            trajectory_input: Union[RobotTrajectory, JointTrajectory] = (
                self.blackboard_get("jaco_trajectory")
            )
            desired_orientation_msg: QuaternionMsg = self.blackboard_get(
                "desired_tool_tip_world_orientation"
            )
            num_points_to_check: int = self.blackboard_get(
                "num_trajectory_points_to_check"
            )

            if not isinstance(desired_orientation_msg, QuaternionMsg):
                self.feedback_message = (
                    "Input 'desired_tool_tip_world_orientation' is not Quaternion."
                )
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            R_World_TipTarget = Rotation.from_quat(
                [
                    desired_orientation_msg.x,
                    desired_orientation_msg.y,
                    desired_orientation_msg.z,
                    desired_orientation_msg.w,
                ]
            )
            y_axis_TipTarget_InWorld = R_World_TipTarget.apply(
                np.array([0.0, 1.0, 0.0])
            )

            jaco_ee_world_poses: List[Pose] = []

            if isinstance(trajectory_input, RobotTrajectory):
                if trajectory_input.multi_dof_joint_trajectory.points:
                    self.logger.debug(
                        f"[{self.name}] Using multi_dof_joint_trajectory from RobotTrajectory."
                    )
                    for mjt_point in trajectory_input.multi_dof_joint_trajectory.points:
                        if mjt_point.transforms:
                            transform = mjt_point.transforms[0]
                            pose = Pose()
                            pose.position.x = transform.translation.x
                            pose.position.y = transform.translation.y
                            pose.position.z = transform.translation.z
                            pose.orientation = transform.rotation
                            jaco_ee_world_poses.append(pose)
                elif trajectory_input.joint_trajectory.points:
                    self.logger.debug(
                        f"[{self.name}] Using joint_trajectory from RobotTrajectory. Performing FK."
                    )
                    jt_points = trajectory_input.joint_trajectory.points
                    jt_joint_names = trajectory_input.joint_trajectory.joint_names
                    # Map jt_joint_names to the order expected by self._jaco_joint_names_pin for FK
                    # This part needs care if orders differ or if jt_joint_names is not just Jaco

                    q_robot = pin.neutral(self._pin_model)  # Full robot configuration
                    for point_data in jt_points:
                        # Populate q_robot based on point_data.positions, jt_joint_names
                        # and neutral/current for other joints (Articutool, virtual base)

                        # Simplified: Assume jt_joint_names are ONLY Jaco joints and match _jaco_joint_names_pin order
                        # And Articutool joints remain at their neutral/current state from q_robot initialization
                        # A more robust solution would take current full robot state and update Jaco parts.

                        # Map positions from trajectory to the correct Jaco joints in Pinocchio model
                        temp_jaco_map = {
                            name: point_data.positions[i]
                            for i, name in enumerate(jt_joint_names)
                        }

                        for (
                            j_name_pin
                        ) in (
                            self._jaco_joint_names_pin
                        ):  # Iterate through Pinocchio's Jaco joint names
                            if j_name_pin in temp_jaco_map:
                                joint_id = self._pin_model.getJointId(j_name_pin)
                                joint_obj = self._pin_model.joints[joint_id]
                                theta = temp_jaco_map[j_name_pin]
                                if joint_obj.nq == 2 and joint_obj.nv == 1:  # Revolute
                                    q_robot[joint_obj.idx_q] = math.cos(theta)
                                    q_robot[joint_obj.idx_q + 1] = math.sin(theta)
                                elif (
                                    joint_obj.nq == 1 and joint_obj.nv == 1
                                ):  # Prismatic
                                    q_robot[joint_obj.idx_q] = theta

                        # Articutool joints (assume they stay at neutral or a fixed known state for FK of Jaco EE)
                        # For simplicity, let's assume they remain neutral as set by pin.neutral(self._pin_model)
                        # If Articutool moves with Jaco, its state should also be in trajectory or known.

                        pin.forwardKinematics(self._pin_model, self._pin_data, q_robot)
                        pin.updateFramePlacements(self._pin_model, self._pin_data)
                        jaco_ee_transform_pin: pin.SE3 = self._pin_data.oMf[
                            self._jaco_ee_frame_id_pin
                        ]

                        pose = Pose()
                        pose.position.x, pose.position.y, pose.position.z = (
                            jaco_ee_transform_pin.translation
                        )
                        quat_xyzw = Rotation.from_matrix(
                            jaco_ee_transform_pin.rotation
                        ).as_quat()
                        (
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        ) = quat_xyzw
                        jaco_ee_world_poses.append(pose)

                else:  # RobotTrajectory is empty
                    self.logger.warn(
                        f"[{self.name}] Input RobotTrajectory is empty. Assuming feasible."
                    )
                    self.blackboard_set("articutool_is_orientation_feasible", True)
                    return Status.SUCCESS

            elif isinstance(trajectory_input, JointTrajectory):
                self.logger.debug(
                    f"[{self.name}] Input is JointTrajectory. Performing FK."
                )
                # Similar FK logic as above for RobotTrajectory.joint_trajectory
                jt_points = trajectory_input.points
                jt_joint_names = trajectory_input.joint_names
                q_robot = pin.neutral(self._pin_model)
                for point_data in jt_points:
                    temp_jaco_map = {
                        name: point_data.positions[i]
                        for i, name in enumerate(jt_joint_names)
                    }
                    for j_name_pin in self._jaco_joint_names_pin:
                        if j_name_pin in temp_jaco_map:
                            joint_id = self._pin_model.getJointId(j_name_pin)
                            joint_obj = self._pin_model.joints[joint_id]
                            theta = temp_jaco_map[j_name_pin]
                            if joint_obj.nq == 2 and joint_obj.nv == 1:
                                q_robot[joint_obj.idx_q] = math.cos(theta)
                                q_robot[joint_obj.idx_q + 1] = math.sin(theta)
                            elif joint_obj.nq == 1 and joint_obj.nv == 1:
                                q_robot[joint_obj.idx_q] = theta
                    pin.forwardKinematics(self._pin_model, self._pin_data, q_robot)
                    pin.updateFramePlacements(self._pin_model, self._pin_data)
                    jaco_ee_transform_pin: pin.SE3 = self._pin_data.oMf[
                        self._jaco_ee_frame_id_pin
                    ]
                    pose = Pose()
                    pose.position.x, pose.position.y, pose.position.z = (
                        jaco_ee_transform_pin.translation
                    )
                    quat_xyzw = Rotation.from_matrix(
                        jaco_ee_transform_pin.rotation
                    ).as_quat()
                    (
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ) = quat_xyzw
                    jaco_ee_world_poses.append(pose)
            else:
                self.feedback_message = f"Input 'jaco_trajectory' is of unexpected type: {type(trajectory_input)}."
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            if not jaco_ee_world_poses:
                self.logger.warn(
                    f"[{self.name}] No Jaco EE poses to check. Assuming feasible."
                )
                self.blackboard_set("articutool_is_orientation_feasible", True)
                return Status.SUCCESS

            indices_to_check = []
            if len(jaco_ee_world_poses) == 1:
                indices_to_check = [0]
            elif num_points_to_check >= len(jaco_ee_world_poses):
                indices_to_check = list(range(len(jaco_ee_world_poses)))
            elif num_points_to_check > 0:
                indices_to_check = np.linspace(
                    0, len(jaco_ee_world_poses) - 1, num_points_to_check, dtype=int
                )

            # Check the length of indices_to_check, as it can be a list or numpy array
            if len(indices_to_check) == 0 and num_points_to_check > 0:
                self.logger.warn(
                    f"[{self.name}] No indices to check (num_points_to_check={num_points_to_check}, len_poses={len(jaco_ee_world_poses)}). Assuming feasible."
                )
                self.blackboard_set("articutool_is_orientation_feasible", True)
                return Status.SUCCESS

            for traj_idx in indices_to_check:
                jaco_ee_pose_msg: Pose = jaco_ee_world_poses[traj_idx]
                R_World_JacoEE = Rotation.from_quat(
                    [
                        jaco_ee_pose_msg.orientation.x,
                        jaco_ee_pose_msg.orientation.y,
                        jaco_ee_pose_msg.orientation.z,
                        jaco_ee_pose_msg.orientation.w,
                    ]
                )
                target_y_for_ik_in_atool_base = R_World_JacoEE.inv().apply(
                    y_axis_TipTarget_InWorld
                )
                norm_target_y = np.linalg.norm(target_y_for_ik_in_atool_base)
                if norm_target_y < self.EPSILON:
                    self.feedback_message = (
                        f"Target Y-axis for IK near zero at traj point {traj_idx}."
                    )
                    self.logger.error(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_orientation_feasible", False)
                    return Status.FAILURE
                target_y_for_ik_in_atool_base /= norm_target_y

                ik_solutions = self._solve_articutool_ik_benchmark_style(
                    target_y_for_ik_in_atool_base
                )

                found_valid_solution_for_waypoint = False
                for theta_p_sol, theta_r_sol in ik_solutions:
                    if (
                        self._pitch_limits_rad[0] - self.EPSILON
                        <= theta_p_sol
                        <= self._pitch_limits_rad[1] + self.EPSILON
                        and self._roll_limits_rad[0] - self.EPSILON
                        <= theta_r_sol
                        <= self._roll_limits_rad[1] + self.EPSILON
                    ):
                        found_valid_solution_for_waypoint = True
                        break

                if not found_valid_solution_for_waypoint:
                    self.feedback_message = (
                        f"Articutool IK failed or solution out of limits at traj point {traj_idx}. "
                        f"Target Y_AtoolBase: {np.round(target_y_for_ik_in_atool_base, 3)}. "
                        f"Raw IK solutions (tp,tr): {[(round(s[0], 3), round(s[1], 3)) for s in ik_solutions]}"
                    )
                    self.logger.warn(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_orientation_feasible", False)
                    return Status.FAILURE

                self.logger.debug(
                    f"[{self.name}] Pt {traj_idx}: Articutool IK feasible."
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
