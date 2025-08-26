# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the CheckArticutoolPathLevelingFeasibility behavior.

This behavior checks if the Articutool can maintain a "level" orientation
(tool_tip Y-axis aligned against gravity) throughout a given trajectory
of the Jaco arm. It iterates through the trajectory, and for each waypoint,
it calculates if a valid, within-limits IK solution exists for the
Articutool to achieve leveling.
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
from py_trees.common import Status
import rclpy.node
import pinocchio as pin

# Local imports
# Assuming these are in your project structure
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class CheckArticutoolPathLevelingFeasibility(BlackboardBehavior):
    """
    Checks if the Articutool can remain level throughout a Jaco trajectory.

    This behavior iterates through a specified number of points along a given
    Jaco arm trajectory. For each point, it calculates the corresponding
    Jaco end-effector pose and then solves the inverse kinematics for the
    Articutool to determine if there's a joint configuration that would keep
    the tool_tip's Y-axis pointing upwards (level), within the Articutool's
    joint limits.

    If a valid leveling solution exists for all checked points, the behavior
    returns SUCCESS. If any point is found to be infeasible, it immediately
    returns FAILURE. This is crucial for validating transport motions where
    spillage must be prevented.
    """

    EPSILON = 1e-6
    # The desired "up" vector for the tool tip's Y-axis in the world frame.
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])

    def blackboard_inputs(
        self,
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        jaco_joint_names_pin: Union[BlackboardKey, List[str]],
        jaco_ee_frame_id_pin: Union[BlackboardKey, int],
        jaco_trajectory: Union[BlackboardKey, RobotTrajectory, JointTrajectory],
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 10,
    ) -> None:
        """Define blackboard inputs."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_is_leveling_feasible: Optional[BlackboardKey],
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
        self._jaco_ee_frame_id_pin: Optional[int] = None
        self._pitch_limits_rad: Optional[Tuple[float, float]] = None
        self._roll_limits_rad: Optional[Tuple[float, float]] = None
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
            self._jaco_ee_frame_id_pin = self.blackboard_get("jaco_ee_frame_id_pin")
            self._pitch_limits_rad = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits_rad = self.blackboard_get("articutool_roll_limits_rad")

            if not all(
                [
                    isinstance(self._pin_model, pin.Model),
                    isinstance(self._pin_data, pin.Data),
                    isinstance(self._jaco_joint_names_pin, list),
                    isinstance(self._jaco_ee_frame_id_pin, int),
                    isinstance(self._pitch_limits_rad, tuple)
                    and len(self._pitch_limits_rad) == 2,
                    isinstance(self._roll_limits_rad, tuple)
                    and len(self._roll_limits_rad) == 2,
                ]
            ):
                self.logger.error(
                    f"[{self.name}] One or more Pinocchio-related inputs from blackboard are invalid."
                )
                return False
            self._pinocchio_ready = True
            return True
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Failed to get Pinocchio info from blackboard: {e}."
            )
            return False

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_for_leveling(
        self, target_y_axis_in_atool_base: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Analytical IK solver based on the corrected kinematic model.
        Solves for Articutool (pitch, roll) to align its tool_tip Y-axis
        with the given target vector expressed in the Articutool's base frame.
        """
        vx, vy, vz = target_y_axis_in_atool_base
        solutions: List[Tuple[float, float]] = []

        # From sin(theta_r) = -vx (as per corrected model)
        asin_arg_for_tr = -vx
        if not (-1.0 - self.EPSILON <= asin_arg_for_tr <= 1.0 + self.EPSILON):
            return []  # No real solution for theta_r

        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = [theta_r_sol1]
        if not math.isclose(theta_r_sol1, theta_r_sol2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_sol2)

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            # Handle singularity when cos(theta_r) is near zero
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                # If singular, vy and vz must also be near zero for a solution to exist.
                if math.isclose(vy, 0.0, abs_tol=self.EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=self.EPSILON
                ):
                    # theta_p is indeterminate, choose a convention (e.g., 0)
                    solutions.append((0.0, self._normalize_angle(theta_r)))
                continue

            # Standard case: theta_p = atan2(vz, vy) as per corrected model
            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
        return solutions

    def _get_jaco_ee_poses_from_trajectory(
        self, trajectory_input: Union[RobotTrajectory, JointTrajectory]
    ) -> Optional[List[Pose]]:
        """Extract or compute Jaco EE poses from a trajectory message via FK."""
        jaco_ee_world_poses: List[Pose] = []

        # Determine the source of joint trajectory points
        if isinstance(trajectory_input, RobotTrajectory):
            # This implementation assumes the trajectory is defined in joint space.
            # A multi_dof_joint_trajectory (Cartesian) is not used for FK.
            if trajectory_input.multi_dof_joint_trajectory.points:
                self.logger.error(
                    f"[{self.name}] Multi-DOF trajectories are not supported for FK-based checks."
                )
                return None
            jt = trajectory_input.joint_trajectory
        elif isinstance(trajectory_input, JointTrajectory):
            jt = trajectory_input
        else:
            self.logger.error(
                f"[{self.name}] Input 'jaco_trajectory' has unexpected type: {type(trajectory_input)}."
            )
            return None

        # Perform Forward Kinematics for each joint trajectory point
        q_robot = pin.neutral(self._pin_model)
        for point in jt.points:
            # Map positions from trajectory to the correct joints in Pinocchio model
            # This assumes jt.joint_names corresponds to the Jaco arm joints
            traj_joint_map = {
                name: point.positions[i] for i, name in enumerate(jt.joint_names)
            }
            for j_name_pin in self._jaco_joint_names_pin:
                if j_name_pin in traj_joint_map:
                    joint_id = self._pin_model.getJointId(j_name_pin)
                    joint_obj = self._pin_model.joints[joint_id]
                    theta = traj_joint_map[j_name_pin]

                    # Assuming revolute joints for Jaco. Update q based on Pinocchio's model.
                    if (
                        joint_obj.nq == 2 and joint_obj.nv == 1
                    ):  # Revolute joint represented by cos/sin
                        q_robot[joint_obj.idx_q] = math.cos(theta)
                        q_robot[joint_obj.idx_q + 1] = math.sin(theta)
                    elif (
                        joint_obj.nq == 1 and joint_obj.nv == 1
                    ):  # Revolute or Prismatic
                        q_robot[joint_obj.idx_q] = theta

            pin.forwardKinematics(self._pin_model, self._pin_data, q_robot)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            ee_transform: pin.SE3 = self._pin_data.oMf[self._jaco_ee_frame_id_pin]

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = ee_transform.translation
            quat_xyzw = Rotation.from_matrix(ee_transform.rotation).as_quat()
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ) = quat_xyzw
            jaco_ee_world_poses.append(pose)

        return jaco_ee_world_poses

    @override
    def update(self) -> Status:
        """Execute the behavior's logic."""
        if not self.node or not self._get_pinocchio_essentials_from_blackboard():
            self.feedback_message = "Behavior not properly initialized."
            self.blackboard_set("articutool_is_leveling_feasible", False)
            return Status.FAILURE

        try:
            trajectory_input = self.blackboard_get("jaco_trajectory")
            num_points = self.blackboard_get("num_trajectory_points_to_check")

            jaco_ee_poses = self._get_jaco_ee_poses_from_trajectory(trajectory_input)

            if jaco_ee_poses is None:
                self.feedback_message = "Failed to get Jaco EE poses from trajectory."
                self.blackboard_set("articutool_is_leveling_feasible", False)
                return Status.FAILURE

            if not jaco_ee_poses:
                self.logger.warn(
                    f"[{self.name}] Trajectory is empty. Assuming feasible."
                )
                self.blackboard_set("articutool_is_leveling_feasible", True)
                return Status.SUCCESS

            # Select a subset of points to check for efficiency
            indices = (
                np.linspace(0, len(jaco_ee_poses) - 1, num_points, dtype=int)
                if num_points < len(jaco_ee_poses)
                else range(len(jaco_ee_poses))
            )

            for idx in indices:
                ee_pose: Pose = jaco_ee_poses[idx]
                R_World_JacoEE = Rotation.from_quat(
                    [
                        ee_pose.orientation.x,
                        ee_pose.orientation.y,
                        ee_pose.orientation.z,
                        ee_pose.orientation.w,
                    ]
                )

                # Transform the world "up" vector into the Articutool's base frame
                target_y_in_atool_base = R_World_JacoEE.inv().apply(
                    self.WORLD_Z_UP_VECTOR
                )

                ik_solutions = self._solve_articutool_ik_for_leveling(
                    target_y_in_atool_base
                )

                # Check if any solution is within joint limits
                is_feasible_at_point = any(
                    self._pitch_limits_rad[0] - self.EPSILON
                    <= sol[0]
                    <= self._pitch_limits_rad[1] + self.EPSILON
                    and self._roll_limits_rad[0] - self.EPSILON
                    <= sol[1]
                    <= self._roll_limits_rad[1] + self.EPSILON
                    for sol in ik_solutions
                )

                if not is_feasible_at_point:
                    self.feedback_message = f"Path is infeasible. No leveling solution at trajectory point {idx}."
                    self.logger.warn(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_leveling_feasible", False)
                    return Status.FAILURE

            # If all checked points are feasible
            self.feedback_message = (
                "Articutool can maintain leveling throughout the trajectory."
            )
            self.logger.info(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_leveling_feasible", True)
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_leveling_feasible", False)
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_leveling_feasible", False)
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")

    @override
    def initialise(self) -> None:
        """Reset blackboard output on initialization."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("articutool_is_leveling_feasible", None)
        # Do not reset pinocchio_ready here to avoid re-reading on every tick
