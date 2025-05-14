#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script is used to benchmark planner performance for reaching
pre-defined joint configurations. It has been updated to include:
- Kinematic feasibility checks for a 2-DOF Articutool attached to the end-effector.
- Saving of Articutool's per-waypoint IK solutions (pitch, roll) within the
  trajectory file for visualization and detailed analysis.
- Detailed metrics related to Articutool performance and joint utilization.
- Physical movement to start states for pairwise planning tasks.
"""

# Standard imports
from collections import defaultdict, namedtuple
from datetime import datetime
import os
import sys
import time
from threading import Thread, Lock
from typing import Optional, List, Dict, Tuple, Any
import json
import math  # For atan2, asin, cos, sin, pi, isclose

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import (
    Duration as RCLPYDuration,
)  # Alias to avoid conflict with local Duration
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from geometry_msgs.msg import Quaternion, PoseStamped, Pose
import moveit_msgs.msg
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
from rosidl_runtime_py import message_to_ordereddict

# --- Logging and Path Length Utility ---
_LOGGER_INSTANCE = None


def _get_logger():
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        if rclpy.ok() and rclpy.utilities.get_default_context().ok():
            _LOGGER_INSTANCE = rclpy.logging.get_logger("planner_benchmark_script")
        else:

            class PrintLogger:
                def info(self, msg):
                    print(f"INFO: {msg}")

                def warn(self, msg):
                    print(f"WARN: {msg}")

                def error(self, msg):
                    print(f"ERROR: {msg}")

                def debug(self, msg):
                    print(f"DEBUG: {msg}")

            _LOGGER_INSTANCE = PrintLogger()
    return _LOGGER_INSTANCE


try:
    from ada_feeding.behaviors.moveit2.moveit2_plan import (
        MoveIt2Plan as AdaMoveIt2PlanUtil,
    )

    GET_PATH_LEN_METHOD = AdaMoveIt2PlanUtil.get_path_len
except ImportError:
    _get_logger().warn(
        "Could not import AdaMoveIt2PlanUtil.get_path_len. Using fallback."
    )

    def fallback_get_path_len(
        trajectory: JointTrajectory, exclude_j6_name: Optional[str] = "j2n6s200_joint_6"
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        if not trajectory or not trajectory.points or len(trajectory.points) < 1:
            return None, None
        if len(trajectory.points) < 2:
            return 0.0, {name: 0.0 for name in trajectory.joint_names}
        total_len_l2_no_j6 = 0.0
        joint_lens = {name: 0.0 for name in trajectory.joint_names}
        j6_idx = -1
        if exclude_j6_name and exclude_j6_name in trajectory.joint_names:
            try:
                j6_idx = trajectory.joint_names.index(exclude_j6_name)
            except ValueError:
                j6_idx = -1
        prev_positions = np.array(trajectory.points[0].positions)
        for point_idx in range(1, len(trajectory.points)):
            point = trajectory.points[point_idx]
            curr_positions = np.array(point.positions)
            if len(curr_positions) != len(prev_positions):
                _get_logger().warn(
                    f"Path length calc: Mismatch joint count at point {point_idx}. Skipping."
                )
                prev_positions = curr_positions
                continue
            diff = np.abs(curr_positions - prev_positions)
            diff = np.minimum(diff, 2 * np.pi - diff)
            current_segment_geom_dist_sq_no_j6 = 0.0
            for i in range(len(diff)):
                joint_lens[trajectory.joint_names[i]] += diff[i]
                if i != j6_idx:
                    current_segment_geom_dist_sq_no_j6 += diff[i] ** 2
            total_len_l2_no_j6 += np.sqrt(current_segment_geom_dist_sq_no_j6)
            prev_positions = curr_positions
        return total_len_l2_no_j6, joint_lens

    GET_PATH_LEN_METHOD = fallback_get_path_len

BenchmarkNamedConfig = namedtuple("BenchmarkNamedConfig", ["name", "joint_values"])
PlanningTask = namedtuple("PlanningTask", ["start_config", "goal_config"])

ArticutoolWaypointSolution = namedtuple(
    "ArticutoolWaypointSolution",
    [
        "waypoint_feasible",  # bool: Is Articutool feasible at this specific waypoint?
        "pitch_solution_rad",  # Optional[float]: Solved pitch angle (theta_p)
        "roll_solution_rad",  # Optional[float]: Solved roll angle (theta_r)
    ],
)

ArticutoolMetrics = namedtuple(
    "ArticutoolMetrics",
    [
        "path_feasible",
        "min_pitch_rad",
        "max_pitch_rad",
        "avg_pitch_abs_rad",
        "pitch_range_used_percent",
        "min_roll_rad",
        "max_roll_rad",
        "avg_roll_abs_rad",
        "roll_range_used_percent",
        "num_infeasible_points",
    ],
)

PlanResult = namedtuple(
    "PlanResult",
    [
        "trajectory",  # Original JointTrajectory from planner
        "elapsed_time",
        "path_length",
        "joint_path_lengths",
        "max_jaco_hand_roll_deviation",
        "jaco_plan_success",
        "trajectory_filename",  # Filename of the *enhanced* JSON trajectory
        "articutool_metrics",  # Aggregate metrics for the Articutool over the path
        "articutool_solutions_per_waypoint",  # List[ArticutoolWaypointSolution]
    ],
)


class PlannerBenchmark:
    NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ = (0.5, 0.5, 0.5, 0.5)
    NO_ROLL_CONSTRAINT_TOLERANCE_XYZ_ABS = (
        np.pi * 1.99,
        np.pi * 1.99,
        np.deg2rad(90.0),
    )
    NO_ROLL_CONSTRAINT_WEIGHT = 1.0
    NO_ROLL_PARAMETERIZATION = 0
    R_JACO_HAND_TO_ATOOL_BASE_SCIPY = R.from_euler("z", np.pi / 2)
    ARTICUTOOL_PITCH_LIMITS_RAD = (-np.pi / 2, np.pi / 2)
    ARTICUTOOL_ROLL_LIMITS_RAD = (-np.pi, np.pi)
    WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])
    EPSILON = 1e-6

    def __init__(
        self,
        node: Node,
        moveit2_interface: MoveIt2,
        hardcoded_target_configs: Dict[str, List[float]],
        planners_to_test: List[str],
        initial_joint_config: List[float],
        planning_group: str,
        end_effector_link: str,
        base_link: str,
        joint_names_for_group: List[str],
        planning_timeout_sec: float = 10.0,
        trajectory_save_dir: Optional[str] = None,
        joint_state_topic: str = "/joint_states",
        continuous_joint_indices: Optional[List[int]] = None,
        use_naive_jaco_hand_constraint: bool = True,
    ):
        self.node = node
        self.logger = _get_logger()  # Use the global logger
        self.moveit2_interface = moveit2_interface
        self.planners_to_test = planners_to_test
        self.initial_joint_config = initial_joint_config
        self.planning_group = planning_group
        self.end_effector_link = end_effector_link
        self.base_link = base_link
        self.joint_names_for_group = joint_names_for_group
        self.planning_timeout_sec = planning_timeout_sec
        self.joint_state_topic_name = joint_state_topic
        self.use_naive_jaco_hand_constraint = use_naive_jaco_hand_constraint

        if continuous_joint_indices is None:
            self.continuous_joint_indices = []
            default_continuous = [
                "j2n6s200_joint_1",
                "j2n6s200_joint_4",
                "j2n6s200_joint_5",
                "j2n6s200_joint_6",
            ]
            for i, name in enumerate(self.joint_names_for_group):
                if name in default_continuous:
                    self.continuous_joint_indices.append(i)
            self.logger.info(
                f"Auto-detected continuous joint indices: {self.continuous_joint_indices}"
            )
        else:
            self.continuous_joint_indices = continuous_joint_indices
            self.logger.info(
                f"Using provided continuous joint indices: {self.continuous_joint_indices}"
            )

        self.all_named_configurations = self._process_hardcoded_configurations(
            hardcoded_target_configs
        )
        self.planning_tasks = self._generate_planning_tasks(
            self.all_named_configurations
        )
        self.num_tasks = len(self.planning_tasks)
        self.results: Dict[Tuple[str, str], Dict[str, PlanResult]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.rate = self.node.create_rate(10)
        self.naive_jaco_hand_constraint_kwargs = (
            self._get_naive_jaco_hand_constraint_kwargs()
        )
        self.trajectory_save_dir = trajectory_save_dir
        if self.trajectory_save_dir:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            self.logger.info(
                f"Successful trajectories will be saved in: {self.trajectory_save_dir}"
            )

        self._latest_joint_state_msg: Optional[JointState] = None
        self._joint_state_lock = Lock()
        self.joint_state_sub = self.node.create_subscription(
            JointState, self.joint_state_topic_name, self._joint_state_callback, 10
        )
        self.logger.info(
            f"Subscribed to '{self.joint_state_topic_name}' for current joint state."
        )
        time.sleep(0.5)
        self.logger.info(
            f"Benchmark initialized for group '{self.planning_group}' and EE '{self.end_effector_link}'."
        )
        self.logger.info(
            f"Generated {self.num_tasks} planning tasks from {len(self.all_named_configurations)} unique configurations."
        )
        self.logger.info(f"Testing with planners: {self.planners_to_test}")
        if self.use_naive_jaco_hand_constraint:
            self.logger.info(
                f"Using NAIVE Jaco hand constraint: Target Quat (xyzw)={self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ}, Tol(xyz_abs)={self.NO_ROLL_CONSTRAINT_TOLERANCE_XYZ_ABS}"
            )
        else:
            self.logger.info(
                "NOT using naive Jaco hand orientation constraint during planning."
            )
        self.logger.info(
            f"Articutool Pitch Limits (theta_p): {self.ARTICUTOOL_PITCH_LIMITS_RAD} rad"
        )
        self.logger.info(
            f"Articutool Roll Limits (theta_r): {self.ARTICUTOOL_ROLL_LIMITS_RAD} rad"
        )

    def _joint_state_callback(self, msg: JointState):
        # ... (implementation from previous version) ...
        if not self.joint_names_for_group:
            return
        msg_joint_names_set = set(msg.name)
        all_required_joints_present = all(
            req_joint_name in msg_joint_names_set
            for req_joint_name in self.joint_names_for_group
        )
        if all_required_joints_present:
            with self._joint_state_lock:
                self._latest_joint_state_msg = msg

    def get_current_joint_positions(
        self, timeout_sec: float = 1.0
    ) -> Optional[Dict[str, float]]:
        # ... (implementation from previous version) ...
        start_time = self.node.get_clock().now()
        latest_msg_to_process: Optional[JointState] = None
        while rclpy.ok() and (self.node.get_clock().now() - start_time) < RCLPYDuration(
            seconds=timeout_sec
        ):
            with self._joint_state_lock:
                if self._latest_joint_state_msg is not None:
                    latest_msg_to_process = self._latest_joint_state_msg
                    break
            self.node.get_clock().sleep_for(RCLPYDuration(seconds=0.02))

        if latest_msg_to_process:
            current_positions_dict: Dict[str, float] = {}
            name_to_pos_map = {
                name: latest_msg_to_process.position[i]
                for i, name in enumerate(latest_msg_to_process.name)
                if i < len(latest_msg_to_process.position)
            }
            all_planning_group_joints_found = True
            for req_name in self.joint_names_for_group:
                if req_name in name_to_pos_map:
                    current_positions_dict[req_name] = name_to_pos_map[req_name]
                else:
                    all_planning_group_joints_found = False
                    break
            if all_planning_group_joints_found:
                return current_positions_dict
        self.logger.warn(
            f"Timed out or no relevant JointState msg on '{self.joint_state_topic_name}' for get_current_joint_positions."
        )
        return None

    def _process_hardcoded_configurations(
        self, hardcoded_configs: Dict[str, List[float]]
    ) -> List[BenchmarkNamedConfig]:
        # ... (implementation from previous version) ...
        configs = []
        self.logger.info(f"Processing hardcoded named configurations...")
        num_expected_joints = len(self.joint_names_for_group)
        for name, values in hardcoded_configs.items():
            if isinstance(values, list) and len(values) == num_expected_joints:
                configs.append(BenchmarkNamedConfig(name, values))
            else:
                self.logger.warn(
                    f"  Skipping hardcoded config '{name}'. Expected {num_expected_joints}, got {len(values) if isinstance(values, list) else type(values)}."
                )
        if not configs or len(configs) < 2:
            self.logger.error(
                f"Not enough valid named configurations provided (found {len(configs)}, need at least 2). Exiting."
            )
            sys.exit(1)
        return configs

    def _generate_planning_tasks(
        self, named_configs: List[BenchmarkNamedConfig]
    ) -> List[PlanningTask]:
        # ... (implementation from previous version) ...
        tasks = []
        for start_config in named_configs:
            for goal_config in named_configs:
                if start_config.name != goal_config.name:
                    tasks.append(PlanningTask(start_config, goal_config))
        self.logger.info(
            f"Generated {len(tasks)} planning tasks (all pairs, start != goal)."
        )
        return tasks

    def _get_naive_jaco_hand_constraint_kwargs(self) -> Dict[str, Any]:
        # ... (implementation from previous version) ...
        return {
            "quat_xyzw": Quaternion(
                x=self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ[0],
                y=self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ[1],
                z=self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ[2],
                w=self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ[3],
            ),
            "frame_id": self.base_link,
            "target_link": self.end_effector_link,
            "tolerance": self.NO_ROLL_CONSTRAINT_TOLERANCE_XYZ_ABS,
            "weight": self.NO_ROLL_CONSTRAINT_WEIGHT,
            "parameterization": self.NO_ROLL_PARAMETERIZATION,
        }

    def _normalize_angle(self, angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _solve_articutool_ik(
        self, target_vector_in_jaco_hand_frame: np.ndarray
    ) -> List[Tuple[float, float]]:
        # ... (implementation from previous version, using math module) ...
        vx, vy, vz = target_vector_in_jaco_hand_frame
        solutions: List[Tuple[float, float]] = []
        if not (-1.0 - self.EPSILON <= -vx <= 1.0 + self.EPSILON):
            return []
        asin_arg = np.clip(-vx, -1.0, 1.0)
        theta_r_cand1 = math.asin(asin_arg)
        theta_r_cand2 = self._normalize_angle(math.pi - theta_r_cand1)
        candidate_thetas_r = [theta_r_cand1]
        if not math.isclose(theta_r_cand1, theta_r_cand2, abs_tol=self.EPSILON):
            candidate_thetas_r.append(theta_r_cand2)

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            theta_p_sol: float = 0.0
            if math.isclose(cos_theta_r, 0.0, abs_tol=self.EPSILON):
                if (
                    math.isclose(abs(vx), 1.0, abs_tol=self.EPSILON)
                    and math.isclose(vy, 0.0, abs_tol=self.EPSILON)
                    and math.isclose(vz, 0.0, abs_tol=self.EPSILON)
                ):
                    solutions.append((theta_p_sol, theta_r))
            else:
                theta_p_sol = math.atan2(vz / cos_theta_r, vy / cos_theta_r)
                solutions.append((theta_p_sol, theta_r))
        return solutions

    def _check_articutool_feasibility_at_waypoint(
        self, R_world_jaco_hand: R
    ) -> ArticutoolWaypointSolution:
        target_vector_in_jaco_hand = R_world_jaco_hand.inv().apply(self.WORLD_UP_VECTOR)
        ik_solutions = self._solve_articutool_ik(target_vector_in_jaco_hand)
        valid_solutions_in_limits = []
        for theta_p, theta_r in ik_solutions:
            theta_p_norm = self._normalize_angle(theta_p)
            theta_r_norm = self._normalize_angle(theta_r)
            if (
                self.ARTICUTOOL_PITCH_LIMITS_RAD[0] - self.EPSILON
                <= theta_p_norm
                <= self.ARTICUTOOL_PITCH_LIMITS_RAD[1] + self.EPSILON
                and self.ARTICUTOOL_ROLL_LIMITS_RAD[0] - self.EPSILON
                <= theta_r_norm
                <= self.ARTICUTOOL_ROLL_LIMITS_RAD[1] + self.EPSILON
            ):
                valid_solutions_in_limits.append((theta_p_norm, theta_r_norm))
        if not valid_solutions_in_limits:
            return ArticutoolWaypointSolution(False, None, None)
        best_sol = min(valid_solutions_in_limits, key=lambda s: s[0] ** 2 + s[1] ** 2)
        return ArticutoolWaypointSolution(True, best_sol[0], best_sol[1])

    def _analyze_trajectory_for_articutool(
        self, trajectory: Optional[JointTrajectory]
    ) -> Tuple[ArticutoolMetrics, List[ArticutoolWaypointSolution]]:
        # Default return values
        default_metrics = ArticutoolMetrics(False, *([np.nan] * 8), 0)
        default_solutions_per_wp: List[ArticutoolWaypointSolution] = []

        if not trajectory or not trajectory.points:
            return default_metrics, default_solutions_per_wp

        required_pitches_rad: List[float] = []
        required_rolls_rad: List[float] = []
        path_is_articutool_feasible = True
        num_infeasible_wps = 0
        per_waypoint_solutions: List[ArticutoolWaypointSolution] = []

        for point_idx, point in enumerate(trajectory.points):
            current_wp_solution = ArticutoolWaypointSolution(False, None, None)
            if len(point.positions) != len(self.joint_names_for_group):
                self.logger.warn(
                    f"Articutool Metrics: Mismatch joint count at point {point_idx}."
                )
                path_is_articutool_feasible = False
                num_infeasible_wps += 1
            else:
                jaco_joint_positions = list(point.positions)
                try:
                    fk_results: Optional[List[PoseStamped]] = (
                        self.moveit2_interface.compute_fk(
                            joint_state=jaco_joint_positions,
                            fk_link_names=[self.end_effector_link],
                        )
                    )
                    if not fk_results or not fk_results[0]:
                        self.logger.warn(
                            f"Articutool Metrics: FK failed for Jaco hand at point {point_idx}."
                        )
                        path_is_articutool_feasible = False
                        num_infeasible_wps += 1
                    else:
                        jaco_hand_pose_msg: Pose = fk_results[0].pose
                        R_world_jaco_hand = R.from_quat(
                            [
                                jaco_hand_pose_msg.orientation.x,
                                jaco_hand_pose_msg.orientation.y,
                                jaco_hand_pose_msg.orientation.z,
                                jaco_hand_pose_msg.orientation.w,
                            ]
                        )
                        current_wp_solution = (
                            self._check_articutool_feasibility_at_waypoint(
                                R_world_jaco_hand
                            )
                        )
                        if not current_wp_solution.waypoint_feasible:
                            path_is_articutool_feasible = False
                            num_infeasible_wps += 1
                        else:
                            if current_wp_solution.pitch_solution_rad is not None:
                                required_pitches_rad.append(
                                    current_wp_solution.pitch_solution_rad
                                )
                            if current_wp_solution.roll_solution_rad is not None:
                                required_rolls_rad.append(
                                    current_wp_solution.roll_solution_rad
                                )
                except Exception as e:
                    self.logger.error(
                        f"Articutool Metrics: Error at point {point_idx}: {e}"
                    )
                    path_is_articutool_feasible = False
                    num_infeasible_wps += 1
            per_waypoint_solutions.append(current_wp_solution)

        p_min_lim, p_max_lim = self.ARTICUTOOL_PITCH_LIMITS_RAD
        r_min_lim, r_max_lim = self.ARTICUTOOL_ROLL_LIMITS_RAD
        pitch_joint_range = p_max_lim - p_min_lim
        roll_joint_range = r_max_lim - r_min_lim
        min_p_val, max_p_val, avg_abs_p_val, pitch_range_used_val = [np.nan] * 4
        if required_pitches_rad:
            min_p_val, max_p_val = (
                np.min(required_pitches_rad),
                np.max(required_pitches_rad),
            )
            avg_abs_p_val = np.mean(np.abs(required_pitches_rad))
            pitch_range_used_val = (
                ((max_p_val - min_p_val) / pitch_joint_range * 100)
                if pitch_joint_range > self.EPSILON
                else 0.0
            )
        min_r_val, max_r_val, avg_abs_r_val, roll_range_used_val = [np.nan] * 4
        if required_rolls_rad:
            min_r_val, max_r_val = (
                np.min(required_rolls_rad),
                np.max(required_rolls_rad),
            )
            avg_abs_r_val = np.mean(np.abs(required_rolls_rad))
            roll_range_used_val = (
                ((max_r_val - min_r_val) / roll_joint_range * 100)
                if roll_joint_range > self.EPSILON
                else 0.0
            )

        aggregate_metrics = ArticutoolMetrics(
            path_is_articutool_feasible,
            min_p_val,
            max_p_val,
            avg_abs_p_val,
            pitch_range_used_val,
            min_r_val,
            max_r_val,
            avg_abs_r_val,
            roll_range_used_val,
            num_infeasible_wps,
        )
        return aggregate_metrics, per_waypoint_solutions

    def _calculate_max_jaco_hand_roll_deviation(
        self, trajectory: Optional[JointTrajectory]
    ) -> Optional[float]:
        # ... (implementation from previous version, ensure it uses self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ) ...
        if trajectory is None or not trajectory.points:
            return None
        max_abs_roll_deviation = 0.0
        target_quat_xyzw_array = np.array(
            self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ
        )
        target_ee_orientation_scipy = R.from_quat(target_quat_xyzw_array)
        for point_idx, point in enumerate(trajectory.points):
            if len(point.positions) != len(self.joint_names_for_group):
                continue
            joint_positions_list = list(point.positions)
            try:
                result_poses: Optional[List[PoseStamped]] = (
                    self.moveit2_interface.compute_fk(
                        joint_state=joint_positions_list,
                        fk_link_names=[self.end_effector_link],
                    )
                )
                if not result_poses or not result_poses[0]:
                    continue
                q_msg = result_poses[0].pose.orientation
                actual_ee_orientation_scipy = R.from_quat(
                    [q_msg.x, q_msg.y, q_msg.z, q_msg.w]
                )
                diff_rotation = (
                    target_ee_orientation_scipy.inv() * actual_ee_orientation_scipy
                )
                euler_angles_of_diff = diff_rotation.as_euler("xyz", degrees=False)
                current_roll_deviation = self._normalize_angle(euler_angles_of_diff[2])
                max_abs_roll_deviation = max(
                    max_abs_roll_deviation, abs(current_roll_deviation)
                )
            except Exception as e:
                self.logger.error(
                    f"Jaco Hand Roll Dev FK: Error at point {point_idx}: {e}"
                )
                return float("inf")
        return max_abs_roll_deviation

    def _get_path_length_stats(
        self, trajectory: Optional[JointTrajectory]
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        # ... (implementation from previous version) ...
        if trajectory is None or not trajectory.points:
            return None, None
        return GET_PATH_LEN_METHOD(trajectory)

    def _save_trajectory_to_file(
        self,
        original_trajectory: JointTrajectory,  # Original Jaco trajectory
        articutool_solutions_per_wp: List[
            ArticutoolWaypointSolution
        ],  # Per-waypoint solutions
        start_config_name: str,
        goal_config_name: str,
        planner_id_str: str,
    ) -> Optional[str]:
        if not self.trajectory_save_dir:
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            safe_start_name = start_config_name.replace(" ", "_").replace("/", "_")
            safe_goal_name = goal_config_name.replace(" ", "_").replace("/", "_")
            safe_planner_id_str = planner_id_str.replace(" ", "_").replace("/", "_")
            filename = f"{safe_start_name}_to_{safe_goal_name}_{safe_planner_id_str}_{timestamp}_enhanced.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)

            enhanced_trajectory_data: Dict[str, Any] = {
                "jaco_joint_names": list(original_trajectory.joint_names),
                "waypoints": [],
            }

            num_jaco_points = len(original_trajectory.points)
            num_articutool_solutions = len(articutool_solutions_per_wp)

            if num_jaco_points != num_articutool_solutions:
                self.logger.warn(
                    f"Mismatch between Jaco trajectory points ({num_jaco_points}) and "
                    f"Articutool solutions ({num_articutool_solutions}) for {filename}. "
                    f"Saving only up to the minimum length."
                )

            min_len = min(num_jaco_points, num_articutool_solutions)

            for i in range(min_len):
                jaco_point = original_trajectory.points[i]
                at_solution = articutool_solutions_per_wp[i]

                waypoint_data = {
                    "time_from_start_sec": jaco_point.time_from_start.sec
                    + jaco_point.time_from_start.nanosec * 1e-9,
                    "jaco_positions_rad": list(jaco_point.positions),
                    "jaco_velocities_rad_per_sec": (
                        list(jaco_point.velocities) if jaco_point.velocities else []
                    ),
                    "jaco_accelerations_rad_per_sec2": (
                        list(jaco_point.accelerations)
                        if jaco_point.accelerations
                        else []
                    ),
                    "articutool_waypoint_feasible": at_solution.waypoint_feasible,
                    "articutool_pitch_solution_rad": (
                        at_solution.pitch_solution_rad
                        if at_solution.pitch_solution_rad is not None
                        else None
                    ),  # Ensure None is JSON null
                    "articutool_roll_solution_rad": (
                        at_solution.roll_solution_rad
                        if at_solution.roll_solution_rad is not None
                        else None
                    ),
                }
                enhanced_trajectory_data["waypoints"].append(waypoint_data)

            with open(filepath, "w") as f:
                json.dump(enhanced_trajectory_data, f, indent=2)
            self.logger.info(
                f"    Successfully saved ENHANCED trajectory to: {filepath}"
            )
            return filename
        except Exception as e:
            self.logger.error(
                f"    Failed to save ENHANCED trajectory for {start_config_name} to {goal_config_name} ({planner_id_str}): {e}",
                exc_info=True,
            )
            return None

    def plan_to_target_configuration(
        self,
        goal_joints: List[float],
        planner_id_str: str,
        path_constraints_kwargs_to_apply: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[JointTrajectory], float]:
        # ... (implementation from previous version, ensure self.rate.sleep() is time.sleep() or handled by executor) ...
        self.logger.info(
            f"  Attempting to plan with: {planner_id_str} to GOAL: {np.round(goal_joints, 3).tolist()}"
        )
        if planner_id_str.lower() == "chomp":
            self.moveit2_interface.pipeline_id = "chomp"
            self.moveit2_interface.planner_id = "chomp"
        elif planner_id_str.lower() == "stomp":
            self.moveit2_interface.pipeline_id = "stomp"
            self.moveit2_interface.planner_id = ""
        else:
            self.moveit2_interface.pipeline_id = "ompl"
            self.moveit2_interface.planner_id = planner_id_str
        self.moveit2_interface.allowed_planning_time = self.planning_timeout_sec
        self.moveit2_interface.clear_goal_constraints()
        self.moveit2_interface.clear_path_constraints()
        try:
            self.moveit2_interface.set_joint_goal(
                joint_positions=goal_joints,
                joint_names=self.joint_names_for_group,
                tolerance=0.01,
                weight=1.0,
            )
        except Exception as e:
            self.logger.error(f"    Failed to set joint goal: {e}")
            return None, 0.0
        if path_constraints_kwargs_to_apply:
            try:
                self.moveit2_interface.set_path_orientation_constraint(
                    **path_constraints_kwargs_to_apply
                )
            except Exception as e:
                self.logger.error(f"    Failed to set path orientation constraint: {e}")
        start_time_ros = self.node.get_clock().now()
        future = self.moveit2_interface.plan_async(start_joint_state=None)
        joint_trajectory_for_analysis: Optional[JointTrajectory] = None
        timeout_duration_rclpy = RCLPYDuration(seconds=self.planning_timeout_sec + 5.0)
        wait_start_time_rclpy = self.node.get_clock().now()
        while rclpy.ok() and not future.done():
            elapsed_wait = self.node.get_clock().now() - wait_start_time_rclpy
            if elapsed_wait >= timeout_duration_rclpy:
                self.logger.warn(
                    f"    Planner {planner_id_str} timed out after {elapsed_wait.nanoseconds / 1e9:.2f}s waiting for future."
                )
                if (
                    hasattr(future, "cancel")
                    and callable(future.cancel)
                    and not future.cancelled()
                ):
                    future.cancel()
                break
            time.sleep(0.05)  # Yield for other ROS processing
        if future.done() and not future.cancelled():
            try:
                plan_result_srv_response = future.result()
                if (
                    plan_result_srv_response.motion_plan_response.error_code.val
                    == moveit_msgs.msg.MoveItErrorCodes.SUCCESS
                ):
                    traj_msg = plan_result_srv_response.motion_plan_response.trajectory
                    if traj_msg and traj_msg.joint_trajectory.points:
                        joint_trajectory_for_analysis = traj_msg.joint_trajectory
                        self.logger.info(f"    Planner {planner_id_str} succeeded.")
                    else:
                        self.logger.warn(
                            f"    Planner {planner_id_str} succeeded but returned empty trajectory."
                        )
                else:
                    self.logger.warn(
                        f"    Planner {planner_id_str} failed with MoveItErrorCode: {plan_result_srv_response.motion_plan_response.error_code.val}"
                    )
            except Exception as e:
                self.logger.error(
                    f"    Exception getting plan result for {planner_id_str}: {e}"
                )
        elif future.cancelled():
            self.logger.warn(f"    Planning for {planner_id_str} was cancelled.")
        elapsed_time_sec = (
            self.node.get_clock().now() - start_time_ros
        ).nanoseconds / 1e9
        self.moveit2_interface.clear_path_constraints()
        self.moveit2_interface.clear_goal_constraints()
        return joint_trajectory_for_analysis, elapsed_time_sec

    def _are_angles_close(
        self, angle1: float, angle2: float, tolerance: float, is_continuous: bool
    ) -> bool:
        # ... (implementation from previous version) ...
        if is_continuous:
            diff = self._normalize_angle(angle1 - angle2)
            return abs(diff) <= tolerance
        else:
            return abs(angle1 - angle2) <= tolerance

    def _move_to_config_blocking(
        self,
        target_config_name: str,
        target_joints: List[float],
        verification_timeout_sec: float = 10.0,
        verification_poll_interval_sec: float = 1.0,
        verification_tolerance: float = 0.05,
        max_move_attempts: int = 3,
    ):
        # ... (implementation from previous version) ...
        self.logger.info(
            f"Attempting to move to config: '{target_config_name}' target: {np.round(target_joints, 4).tolist()}"
        )
        original_planner, original_pipeline, original_timeout = (
            self.moveit2_interface.planner_id,
            self.moveit2_interface.pipeline_id,
            self.moveit2_interface.allowed_planning_time,
        )
        self.moveit2_interface.pipeline_id = "ompl"
        self.moveit2_interface.planner_id = "RRTConnectkConfigDefault"
        self.moveit2_interface.allowed_planning_time = max(
            10.0, self.planning_timeout_sec
        )
        achieved_target = False
        last_known_joints_str = "N/A"
        last_exception_detail = "No attempts."
        for attempt in range(max_move_attempts):
            self.logger.info(
                f"Move attempt {attempt + 1}/{max_move_attempts} to '{target_config_name}'."
            )
            try:
                self.moveit2_interface.clear_goal_constraints()
                self.moveit2_interface.clear_path_constraints()
                self.moveit2_interface.move_to_configuration(
                    joint_positions=target_joints,
                    joint_names=self.joint_names_for_group,
                    tolerance=0.01,
                )
                self.logger.info(
                    f"  Move_to_configuration call for '{target_config_name}' completed. Verifying..."
                )
                # Verification loop
                verif_start_time = self.node.get_clock().now()
                achieved_this_attempt = False
                num_verif_checks = max(
                    3,
                    int(verification_timeout_sec / verification_poll_interval_sec) + 1,
                )
                for v_idx in range(num_verif_checks):
                    current_joints_dict = self.get_current_joint_positions(
                        timeout_sec=0.5
                    )
                    if current_joints_dict:
                        current_joints_list = [
                            current_joints_dict[name]
                            for name in self.joint_names_for_group
                            if name in current_joints_dict
                        ]
                        if len(current_joints_list) != len(self.joint_names_for_group):
                            last_known_joints_str = "Partial state."
                            self.node.get_clock().sleep_for(
                                RCLPYDuration(seconds=verification_poll_interval_sec)
                            )
                            continue
                        last_known_joints_str = str(
                            np.round(current_joints_list, 4).tolist()
                        )
                        all_close = True
                        for j_idx, (tgt, act) in enumerate(
                            zip(target_joints, current_joints_list)
                        ):
                            if not self._are_angles_close(
                                tgt,
                                act,
                                verification_tolerance,
                                j_idx in self.continuous_joint_indices,
                            ):
                                all_close = False
                                break
                        if all_close:
                            self.logger.info(
                                f"  Successfully verified robot reached '{target_config_name}'."
                            )
                            achieved_target = True
                            achieved_this_attempt = True
                            break
                    else:
                        last_known_joints_str = "No joint state."
                    if (
                        self.node.get_clock().now() - verif_start_time
                    ) >= RCLPYDuration(seconds=verification_timeout_sec):
                        if not achieved_this_attempt:
                            self.logger.warn(
                                f"  Verification timeout for '{target_config_name}'."
                            )
                            break
                    if not achieved_this_attempt:
                        self.node.get_clock().sleep_for(
                            RCLPYDuration(seconds=verification_poll_interval_sec)
                        )
                if achieved_target:
                    break
                if not achieved_this_attempt:
                    last_exception_detail = (
                        f"Verification failed. Last state: {last_known_joints_str}"
                    )
            except Exception as e:
                self.logger.warn(
                    f"  Move attempt {attempt + 1} to '{target_config_name}' failed during move: {e}"
                )
                last_exception_detail = f"Error in move: {e}"
            if achieved_target:
                break
            if attempt < max_move_attempts - 1:
                self.node.get_clock().sleep_for(RCLPYDuration(seconds=1.5))
        (
            self.moveit2_interface.planner_id,
            self.moveit2_interface.pipeline_id,
            self.moveit2_interface.allowed_planning_time,
        ) = (
            original_planner,
            original_pipeline,
            original_timeout,
        )
        if not achieved_target:
            final_error_msg = f"CRITICAL: Failed to move AND verify robot at '{target_config_name}' after {max_move_attempts} attempts. Last state: {last_known_joints_str}. Detail: {last_exception_detail}"
            self.logger.error(final_error_msg)
            raise RuntimeError(final_error_msg)
        self.logger.info(
            f"Successfully moved to and verified configuration '{target_config_name}'."
        )

    def move_to_initial_config(self):
        self._move_to_config_blocking("InitialScriptSetup", self.initial_joint_config)

    def run_benchmark_planning_task(self, task: PlanningTask):
        start_name, goal_name, goal_joints = (
            task.start_config.name,
            task.goal_config.name,
            task.goal_config.joint_values,
        )
        self.logger.info(f"--- Testing Task: FROM '{start_name}' TO '{goal_name}' ---")
        self.logger.info(f"  Goal Joints: {np.round(goal_joints, 3).tolist()}")
        task_key = (start_name, goal_name)

        for planner_id_str in self.planners_to_test:
            path_constraints_to_apply = (
                self.naive_jaco_hand_constraint_kwargs
                if self.use_naive_jaco_hand_constraint
                else None
            )

            jaco_trajectory, elapsed_time = self.plan_to_target_configuration(
                goal_joints, planner_id_str, path_constraints_to_apply
            )

            jaco_plan_success = jaco_trajectory is not None and bool(
                jaco_trajectory.points
            )
            path_len_total, joint_path_lens_map = self._get_path_length_stats(
                jaco_trajectory
            )
            max_jaco_roll_dev = self._calculate_max_jaco_hand_roll_deviation(
                jaco_trajectory
            )

            aggregate_at_metrics: ArticutoolMetrics
            per_wp_at_solutions: List[ArticutoolWaypointSolution]

            if jaco_plan_success:
                aggregate_at_metrics, per_wp_at_solutions = (
                    self._analyze_trajectory_for_articutool(jaco_trajectory)
                )
            else:
                num_potential_wps = len(
                    goal_joints
                )  # A rough estimate if trajectory is None
                aggregate_at_metrics = ArticutoolMetrics(
                    False, *([np.nan] * 8), num_potential_wps
                )
                per_wp_at_solutions = [
                    ArticutoolWaypointSolution(False, None, None)
                ] * num_potential_wps

            trajectory_filename = None
            if jaco_plan_success and self.trajectory_save_dir:
                trajectory_filename = self._save_trajectory_to_file(
                    jaco_trajectory,
                    per_wp_at_solutions,
                    start_name,
                    goal_name,
                    planner_id_str,
                )

            self.results[task_key][planner_id_str] = PlanResult(
                jaco_trajectory,
                elapsed_time,
                path_len_total,
                joint_path_lens_map,
                max_jaco_roll_dev,
                jaco_plan_success,
                trajectory_filename,
                aggregate_at_metrics,
                per_wp_at_solutions,  # Storing per-wp solutions in PlanResult
            )

            # Pre-format strings for logging to avoid f-string errors with None/NaN
            path_len_str = (
                f"{path_len_total:.3f}"
                if path_len_total is not None
                and not np.isnan(path_len_total)
                and not np.isinf(path_len_total)
                else "N/A"
            )
            jaco_roll_dev_str = (
                f"{max_jaco_roll_dev:.3f}"
                if max_jaco_roll_dev is not None
                and not np.isnan(max_jaco_roll_dev)
                and not np.isinf(max_jaco_roll_dev)
                else "N/A"
            )

            at_feasible_str = str(aggregate_at_metrics.path_feasible)
            at_infeasible_pts_str = str(aggregate_at_metrics.num_infeasible_points)
            at_p_range_str = (
                f"{aggregate_at_metrics.pitch_range_used_percent:.1f}"
                if not np.isnan(aggregate_at_metrics.pitch_range_used_percent)
                else "N/A"
            )
            at_r_range_str = (
                f"{aggregate_at_metrics.roll_range_used_percent:.1f}"
                if not np.isnan(aggregate_at_metrics.roll_range_used_percent)
                else "N/A"
            )

            articutool_log = (
                f"| Articutool Feasible: {at_feasible_str} (Infeasible Pts: {at_infeasible_pts_str}, "
                f"P-Range%: {at_p_range_str}, R-Range%: {at_r_range_str})"
            )
            log_suffix = (
                f"| TrajFile: {trajectory_filename}" if trajectory_filename else ""
            )

            if jaco_plan_success:
                self.logger.info(
                    f"  Planner: {planner_id_str} | Jaco Success: True  | Time: {elapsed_time:.3f}s | PathLen: {path_len_str} | JacoHandRollDev: {jaco_roll_dev_str} rad {articutool_log} {log_suffix}"
                )
            else:
                self.logger.warn(
                    f"  Planner: {planner_id_str} | Jaco Success: False | Time: {elapsed_time:.3f}s {log_suffix}"
                )

    def run_all_tests(self, log_summary_every_n_tasks: Optional[int] = None):
        # ... (implementation from previous version, ensure default ArticutoolMetrics and empty list for per_wp_solutions on failure) ...
        self.logger.info(
            "=== Starting Benchmark: Moving to Initial Script Configuration ==="
        )
        try:
            self.move_to_initial_config()
        except RuntimeError as e:
            self.logger.error(
                f"CRITICAL FAILURE: Could not move to initial config. Benchmark aborted. Error: {e}"
            )
            return

        for i, planning_task in enumerate(self.planning_tasks):
            self.logger.info(
                f"\n=== Running Planning Task {i + 1}/{self.num_tasks}: FROM '{planning_task.start_config.name}' TO '{planning_task.goal_config.name}' ==="
            )
            try:
                self._move_to_config_blocking(
                    planning_task.start_config.name,
                    planning_task.start_config.joint_values,
                )
            except RuntimeError as e:
                self.logger.error(
                    f"Skipping task FROM '{planning_task.start_config.name}' TO '{planning_task.goal_config.name}' due to failure to reach start state: {e}"
                )
                task_key = (
                    planning_task.start_config.name,
                    planning_task.goal_config.name,
                )
                default_at_metrics = ArticutoolMetrics(False, *([np.nan] * 8), 0)
                empty_at_solutions: List[ArticutoolWaypointSolution] = []
                for planner_id_str_fail in self.planners_to_test:
                    self.results[task_key][planner_id_str_fail] = PlanResult(
                        None,
                        0.0,
                        None,
                        None,
                        None,
                        False,
                        f"ERROR_MOVE_TO_START_FAILED:{str(e)[:50]}",
                        default_at_metrics,
                        empty_at_solutions,
                    )
                continue
            self.run_benchmark_planning_task(planning_task)
            if log_summary_every_n_tasks and (i + 1) % log_summary_every_n_tasks == 0:
                self.log_summary_results()
        self.logger.info("=== Benchmark Run Completed ===")

    def get_csv_header(self) -> List[str]:
        # ... (implementation from previous version, ensure names match PlanResult) ...
        header = [
            "start_config_name",
            *[f"start_{name}" for name in self.joint_names_for_group],
            "goal_config_name",
            *[f"goal_{name}" for name in self.joint_names_for_group],
            "planner_id",
            "elapsed_time_s",
            "jaco_plan_success",
            "path_length_total",
            "max_jaco_hand_roll_deviation_rad",
            "trajectory_filename",
            "articutool_path_feasible",
            "articutool_num_infeasible_points",
            "articutool_min_pitch_rad",
            "articutool_max_pitch_rad",
            "articutool_avg_pitch_abs_rad",
            "articutool_pitch_range_used_percent",
            "articutool_min_roll_rad",
            "articutool_max_roll_rad",
            "articutool_avg_roll_abs_rad",
            "articutool_roll_range_used_percent",
            *[f"path_length_{name}" for name in self.joint_names_for_group],
        ]
        return header

    def write_results_to_csv(self, filename: str):
        # ... (implementation from previous version, ensure all fields from PlanResult.articutool_metrics are written) ...
        self.logger.info(f"Writing results to {filename}")
        with open(filename, "w", newline="") as f:
            import csv

            csv_writer = csv.writer(f)
            csv_writer.writerow(self.get_csv_header())
            for task_key in sorted(self.results.keys()):
                start_name, goal_name = task_key
                start_cfg_obj = next(
                    (c for c in self.all_named_configurations if c.name == start_name),
                    None,
                )
                goal_cfg_obj = next(
                    (c for c in self.all_named_configurations if c.name == goal_name),
                    None,
                )
                if not start_cfg_obj or not goal_cfg_obj:
                    continue
                if task_key in self.results:
                    for planner_id_str in self.planners_to_test:
                        if planner_id_str in self.results[task_key]:
                            result = self.results[task_key][planner_id_str]
                            row = [
                                start_cfg_obj.name,
                                *map(str, start_cfg_obj.joint_values),
                                goal_cfg_obj.name,
                                *map(str, goal_cfg_obj.joint_values),
                                planner_id_str,
                                f"{result.elapsed_time:.4f}",
                                str(1 if result.jaco_plan_success else 0),
                                (
                                    f"{result.path_length:.4f}"
                                    if result.path_length is not None
                                    else ""
                                ),
                                (
                                    f"{result.max_jaco_hand_roll_deviation:.4f}"
                                    if result.max_jaco_hand_roll_deviation is not None
                                    else ""
                                ),
                                (
                                    result.trajectory_filename
                                    if result.trajectory_filename
                                    else ""
                                ),
                            ]
                            atm = result.articutool_metrics
                            if atm:
                                row.extend(
                                    [
                                        str(1 if atm.path_feasible else 0),
                                        str(atm.num_infeasible_points),
                                        (
                                            f"{atm.min_pitch_rad:.4f}"
                                            if not np.isnan(atm.min_pitch_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.max_pitch_rad:.4f}"
                                            if not np.isnan(atm.max_pitch_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.avg_pitch_abs_rad:.4f}"
                                            if not np.isnan(atm.avg_pitch_abs_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.pitch_range_used_percent:.2f}"
                                            if not np.isnan(
                                                atm.pitch_range_used_percent
                                            )
                                            else ""
                                        ),
                                        (
                                            f"{atm.min_roll_rad:.4f}"
                                            if not np.isnan(atm.min_roll_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.max_roll_rad:.4f}"
                                            if not np.isnan(atm.max_roll_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.avg_roll_abs_rad:.4f}"
                                            if not np.isnan(atm.avg_roll_abs_rad)
                                            else ""
                                        ),
                                        (
                                            f"{atm.roll_range_used_percent:.2f}"
                                            if not np.isnan(atm.roll_range_used_percent)
                                            else ""
                                        ),
                                    ]
                                )
                            else:
                                row.extend(["ERROR_NO_AT_METRICS"] * 10)
                            if result.joint_path_lengths:
                                row.extend(
                                    [
                                        (
                                            f"{result.joint_path_lengths.get(name, ''):.4f}"
                                            if result.joint_path_lengths.get(name)
                                            is not None
                                            else ""
                                        )
                                        for name in self.joint_names_for_group
                                    ]
                                )
                            else:
                                row.extend([""] * len(self.joint_names_for_group))
                            csv_writer.writerow(row)
        self.logger.info(f"Results successfully written to {filename}")

    def log_summary_results(self):
        # ... (implementation from previous version, ensure it uses updated PlanResult fields) ...
        self.logger.info("--- BENCHMARK SUMMARY (ALL COMPLETED TASKS) ---")
        for planner_id_str in self.planners_to_test:
            results_for_planner = [
                res[planner_id_str]
                for res in self.results.values()
                if planner_id_str in res
            ]
            if not results_for_planner:
                self.logger.info(
                    f"--- Planner: {planner_id_str} ---\n  No tasks recorded."
                )
                continue

            successful_jaco_plans = sum(
                1 for r in results_for_planner if r.jaco_plan_success
            )
            total_tasks = len(results_for_planner)
            jaco_success_rate = (
                (successful_jaco_plans / total_tasks * 100) if total_tasks > 0 else 0
            )

            self.logger.info(f"--- Planner: {planner_id_str} ---")
            self.logger.info(
                f"  Jaco Plan Success Rate: {jaco_success_rate:.2f}% ({successful_jaco_plans}/{total_tasks})"
            )
            # ... (other Jaco metrics as before) ...

            articutool_feasible_paths_count = sum(
                1
                for r in results_for_planner
                if r.jaco_plan_success
                and r.articutool_metrics
                and r.articutool_metrics.path_feasible
            )
            if successful_jaco_plans > 0:
                articutool_path_success_rate = (
                    articutool_feasible_paths_count / successful_jaco_plans
                ) * 100
                self.logger.info(
                    f"  Articutool Path Feasibility Rate (of Jaco succ. plans): {articutool_path_success_rate:.2f}% ({articutool_feasible_paths_count}/{successful_jaco_plans})"
                )
            else:
                self.logger.info(
                    "  Articutool Path Feasibility Rate: N/A (no Jaco successful plans)"
                )
        self.logger.info("-------------------------")

    @staticmethod
    def _summary_stats_str(data: List[float]) -> str:
        # ... (implementation from previous version) ...
        if not data:
            return "N/A"
        valid_data = [x for x in data if isinstance(x, (int, float)) and np.isfinite(x)]
        if not valid_data:
            return "N/A (no valid numeric data)"
        return (
            f"Mean {np.mean(valid_data):.3f}, Median {np.median(valid_data):.3f}, "
            f"Min {np.min(valid_data):.3f}, Max {np.max(valid_data):.3f}, "
            f"Std {np.std(valid_data):.3f}, Count {len(valid_data)}"
        )


def main(output_dir_arg: Optional[str], use_naive_constraint_arg: bool):
    # ... (implementation from previous version, ensure PlannerBenchmark is instantiated correctly) ...
    rclpy.init()
    node = Node("planner_benchmark_pairwise_physical_moves")
    global _LOGGER_INSTANCE
    _LOGGER_INSTANCE = node.get_logger()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    _LOGGER_INSTANCE.info(
        "Benchmark node spinning. Waiting for services (approx 5s)..."
    )
    time.sleep(5.0)

    hardcoded_configs = {
        "MoveAbovePlate": [-2.4538, 3.0797, 1.8320, 4.0961, -2.0034, -3.2123],
        "RestingAcquireFood": [-1.9467, 2.5126, 0.3565, -4.7650, 5.9999, 4.9955],
        "StagingConfig": [-2.3252, 4.4562, 4.1676, 1.5326, -2.1835, -2.1952],
        "StowLocation": [-1.5210, 2.6009, 0.3281, -4.0001, 0.2283, 3.8788],
    }
    planners = ["RRTConnectkConfigDefault", "RRTstarkConfigDefault", "CHOMP"]
    initial_script_setup_config = (
        list(hardcoded_configs.values())[0]
        if hardcoded_configs
        else [0.0] * len(kinova.joint_names())
    )
    planning_group_name, ee_link_name, base_link_name, group_joint_names = (
        "jaco_arm",
        kinova.end_effector_name(),
        kinova.base_link_name(),
        kinova.joint_names(),
    )
    planning_timeout = 15.0
    base_output_dir = (
        output_dir_arg
        if output_dir_arg
        else os.path.join(os.getcwd(), "benchmark_results")
    )
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    trajectory_save_location = os.path.join(
        base_output_dir, "saved_trajectories_enhanced"
    )  # New subdir
    if not os.path.exists(trajectory_save_location):
        os.makedirs(trajectory_save_location)

    callback_group = ReentrantCallbackGroup()
    moveit2_interface = MoveIt2(
        node=node,
        joint_names=group_joint_names,
        base_link_name=base_link_name,
        end_effector_name=ee_link_name,
        group_name=planning_group_name,
        callback_group=callback_group,
    )
    moveit2_interface.allowed_planning_time = planning_timeout
    moveit2_interface.max_velocity_scaling_factor = 0.5
    moveit2_interface.max_acceleration_scaling_factor = 0.5

    benchmark_runner = PlannerBenchmark(
        node=node,
        moveit2_interface=moveit2_interface,
        hardcoded_target_configs=hardcoded_configs,
        planners_to_test=planners,
        initial_joint_config=initial_script_setup_config,
        planning_group=planning_group_name,
        end_effector_link=ee_link_name,
        base_link=base_link_name,
        joint_names_for_group=group_joint_names,
        planning_timeout_sec=planning_timeout,
        trajectory_save_dir=trajectory_save_location,
        joint_state_topic="/joint_states",
        use_naive_jaco_hand_constraint=use_naive_constraint_arg,
    )
    try:
        num_total_tasks = benchmark_runner.num_tasks
        log_interval = max(1, num_total_tasks // 5 if num_total_tasks > 0 else 1)
        benchmark_runner.run_all_tests(log_summary_every_n_tasks=log_interval)
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        constraint_label = (
            "naive_constraint" if use_naive_constraint_arg else "no_hand_constraint"
        )
        csv_filename = os.path.join(
            base_output_dir,
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + f"_benchmark_{constraint_label}.csv",
        )
        benchmark_runner.write_results_to_csv(csv_filename)
        benchmark_runner.log_summary_results()
    except Exception as e:
        _LOGGER_INSTANCE.error(f"Benchmark run failed: {e}", exc_info=True)
    finally:
        _LOGGER_INSTANCE.info("Shutting down benchmark node.")
        rclpy.shutdown()
        executor_thread.join()
        _LOGGER_INSTANCE.info("Executor joined. Script finished.")


if __name__ == "__main__":
    default_out_dir = os.path.join(os.getcwd(), "planner_benchmark_output_v3")
    out_dir_arg = default_out_dir
    use_naive_arg_str = "true"  # Default

    if len(sys.argv) == 2:  # Only one arg
        if sys.argv[1].lower() in ["true", "use_naive", "naive"]:
            use_naive_arg_str = "true"
        elif sys.argv[1].lower() in ["false", "no_constraint", "none"]:
            use_naive_arg_str = "false"
        else:
            out_dir_arg = os.path.expanduser(sys.argv[1])  # Assume it's output dir
    elif len(sys.argv) > 2:  # Two or more args
        out_dir_arg = os.path.expanduser(sys.argv[1])
        if sys.argv[2].lower() in ["true", "use_naive", "naive"]:
            use_naive_arg_str = "true"
        elif sys.argv[2].lower() in ["false", "no_constraint", "none"]:
            use_naive_arg_str = "false"

    use_naive_constraint_bool = use_naive_arg_str == "true"
    _get_logger().info(f"Output directory set to: {out_dir_arg}")
    _get_logger().info(f"Using naive Jaco hand constraint: {use_naive_constraint_bool}")
    main(out_dir_arg, use_naive_constraint_bool)
