#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script is used to benchmark planner performance for reaching
pre-defined joint configurations while enforcing a path-wide orientation
constraint on the end-effector. It now also saves successful trajectories.
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

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from geometry_msgs.msg import Quaternion, PoseStamped
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
                _get_logger().warn(
                    f"Joint {exclude_j6_name} for exclusion not found in trajectory joint_names."
                )
                j6_idx = -1

        prev_positions = np.array(trajectory.points[0].positions)
        for point_idx in range(1, len(trajectory.points)):
            point = trajectory.points[point_idx]
            curr_positions = np.array(point.positions)
            if len(curr_positions) != len(prev_positions):
                _get_logger().warn(
                    f"Path length calc: Mismatch in joint count at point {point_idx}. Skipping."
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

BenchmarkCondition = namedtuple("BenchmarkCondition", ["name", "joint_values"])
PlanResult = namedtuple(
    "PlanResult",
    [
        "trajectory",
        "elapsed_time",
        "path_length",
        "joint_path_lengths",
        "max_roll_deviation",
        "success",
        "trajectory_filename",
    ],
)


class PlannerBenchmark:
    NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ = (
        0.5,
        0.5,
        0.5,
        0.5,
    )
    NO_ROLL_CONSTRAINT_TOLERANCE_XYZ_ABS = (
        np.pi * 1.99,
        np.pi * 1.99,
        np.deg2rad(90.0),
    )
    NO_ROLL_CONSTRAINT_WEIGHT = 1.0
    NO_ROLL_PARAMETERIZATION = 0

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
    ):
        self.node = node
        self.logger = self.node.get_logger()
        self.moveit2_interface = moveit2_interface
        self.planners_to_test = planners_to_test
        self.initial_joint_config = initial_joint_config
        self.planning_group = planning_group
        self.end_effector_link = end_effector_link
        self.base_link = base_link
        self.joint_names_for_group = joint_names_for_group
        self.planning_timeout_sec = planning_timeout_sec

        self.target_configurations = self._process_hardcoded_configurations(
            hardcoded_target_configs
        )
        self.num_conditions = len(self.target_configurations)

        self.results: Dict[str, Dict[str, PlanResult]] = defaultdict(dict)
        self.rate = self.node.create_rate(10)

        self.no_roll_path_constraint_kwargs = self._get_no_roll_constraint_kwargs()

        self.trajectory_save_dir = trajectory_save_dir
        if self.trajectory_save_dir:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            self.logger.info(
                f"Successful trajectories will be saved in: {self.trajectory_save_dir}"
            )

        self.logger.info(
            f"Benchmark initialized for group '{self.planning_group}' and EE '{self.end_effector_link}'."
        )
        self.logger.info(
            f"Testing {self.num_conditions} target configurations with planners: {self.planners_to_test}"
        )
        self.logger.info(
            f"Using 'no roll' constraint: Target Quat (xyzw)={self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ}, Tol(xyz_abs)={self.NO_ROLL_CONSTRAINT_TOLERANCE_XYZ_ABS}"
        )

    def _process_hardcoded_configurations(
        self, hardcoded_configs: Dict[str, List[float]]
    ) -> List[BenchmarkCondition]:
        configs = []
        self.logger.info(f"Processing hardcoded target configurations...")
        num_expected_joints = len(self.joint_names_for_group)
        for name, values in hardcoded_configs.items():
            if isinstance(values, list) and len(values) == num_expected_joints:
                configs.append(BenchmarkCondition(name, values))
                self.logger.info(
                    f"  Loaded '{name}' with {num_expected_joints} joints."
                )
            else:
                self.logger.warn(
                    f"  Skipping hardcoded config '{name}'. Expected {num_expected_joints} joint values, got {len(values) if isinstance(values, list) else type(values)}."
                )

        if not configs:
            self.logger.error(
                f"No valid target configurations provided from hardcoded list. Exiting."
            )
            sys.exit(1)
        return configs

    def _get_no_roll_constraint_kwargs(self) -> Dict[str, Any]:
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

    def _calculate_max_roll_deviation(
        self, trajectory: Optional[JointTrajectory]
    ) -> Optional[float]:
        if trajectory is None or not trajectory.points:
            return None

        max_abs_roll_deviation = 0.0

        for point_idx, point in enumerate(trajectory.points):
            if len(point.positions) != len(self.joint_names_for_group):
                self.logger.warn(
                    f"FK Calc: Mismatch in joint count at point {point_idx}. Expected {len(self.joint_names_for_group)}, got {len(point.positions)}. Skipping."
                )
                continue

            joint_positions_list = list(point.positions)

            try:
                result_poses: Optional[List[PoseStamped]] = (
                    self.moveit2_interface.compute_fk(
                        joint_state=joint_positions_list,
                        fk_link_names=[self.end_effector_link],
                    )
                )

                if (
                    not result_poses
                    or not isinstance(result_poses, list)
                    or not result_poses[0]
                ):
                    self.logger.warn(
                        f"FK returned None or invalid format for point {point_idx}."
                    )
                    continue

                pose_stamped_ee = result_poses[0]

                if pose_stamped_ee.header.frame_id.lstrip("/") != self.base_link.lstrip(
                    "/"
                ):
                    self.logger.warn(
                        f"FK pose for {self.end_effector_link} is in frame '{pose_stamped_ee.header.frame_id}', "
                        f"expected '{self.base_link}'. Ensure consistency or implement TF."
                    )

                q = pose_stamped_ee.pose.orientation

                # Convert the actual EE orientation to SciPy Rotation
                actual_ee_orientation_scipy = R.from_quat([q.x, q.y, q.z, q.w])

                # Convert the target "no roll" orientation (defined by NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ)
                # to SciPy Rotation. This quaternion is already w.r.t. base_link.
                target_quat_params = self.NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ
                target_ee_orientation_scipy = R.from_quat(target_quat_params)

                # Calculate the rotational difference: R_diff = R_target_inv * R_actual
                # This R_diff represents the rotation needed to get from target to actual.
                # Its Euler angles (especially around the Z-axis of the *target* frame)
                # can indicate roll deviation.
                diff_rotation = (
                    target_ee_orientation_scipy.inv() * actual_ee_orientation_scipy
                )

                # Get Euler angles of this difference. The Z-angle of this difference,
                # if using 'xyz' for example, would represent the roll *after* aligning X and Y.
                # Choose an Euler sequence that makes sense for your definition of "roll".
                # 'zyx' is common, where the first angle (z) is yaw, second (y) is pitch, third (x) is roll.
                # If your constraint is primarily about the EE's Z-axis pointing, then 'xyz' might be more intuitive
                # where the last angle (z) is the rotation around the new Z axis.
                # For this 'no roll' constraint, we are interested in rotation around the target EE's Z-axis.
                # Let's use the target frame's Z-axis as the roll axis.
                # Euler angles of R_diff in target frame's basis:
                euler_angles_of_diff_in_target_basis = diff_rotation.as_euler(
                    "xyz", degrees=False
                )  # or 'ZYX', etc.

                # The roll deviation would be the rotation around the axis that corresponds to "roll"
                # in your chosen Euler sequence. For 'xyz', euler_angles_of_diff_in_target_basis[2] is about the new Z.
                # This should correspond to the Z-axis of the `NO_ROLL_CONSTRAINT_TARGET_QUATERNION_XYWZ`
                current_roll_deviation = euler_angles_of_diff_in_target_basis[2]

                # Normalize to [-pi, pi]
                while current_roll_deviation > np.pi:
                    current_roll_deviation -= 2 * np.pi
                while current_roll_deviation < -np.pi:
                    current_roll_deviation += 2 * np.pi

                max_abs_roll_deviation = max(
                    max_abs_roll_deviation, abs(current_roll_deviation)
                )

            except Exception as e:
                self.logger.error(
                    f"Error during FK for roll deviation at point {point_idx}: {e}",
                )
                return float("inf")
        return max_abs_roll_deviation

    def _get_path_length_stats(
        self, trajectory: Optional[JointTrajectory]
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        if trajectory is None or not trajectory.points:
            return None, None
        j6_name = (
            "j2n6s200_joint_6"
            if "j2n6s200_joint_6" in self.joint_names_for_group
            else None
        )
        return GET_PATH_LEN_METHOD(trajectory)

    def _save_trajectory_to_file(
        self, trajectory: JointTrajectory, condition_name: str, planner_id_str: str
    ) -> Optional[str]:
        if not self.trajectory_save_dir:
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            # Sanitize condition_name and planner_id_str for filename
            safe_condition_name = condition_name.replace(" ", "_").replace("/", "_")
            safe_planner_id_str = planner_id_str.replace(" ", "_").replace("/", "_")

            filename = f"{safe_condition_name}_{safe_planner_id_str}_{timestamp}.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)

            traj_dict = message_to_ordereddict(trajectory)
            with open(filepath, "w") as f:
                json.dump(traj_dict, f, indent=2)
            self.logger.info(f"    Successfully saved trajectory to: {filepath}")
            return filename
        except Exception as e:
            self.logger.error(
                f"    Failed to save trajectory for {condition_name} ({planner_id_str}): {e}",
            )
            return None

    def plan_to_target_configuration(
        self,
        goal_joints: List[float],
        planner_id_str: str,
        path_constraints_kwargs_to_apply: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[JointTrajectory], float]:
        self.logger.info(
            f"  Attempting to plan with: {planner_id_str} to {goal_joints}"
        )

        if planner_id_str.lower() == "chomp":
            self.moveit2_interface.planning_pipeline_id = "chomp"
            self.moveit2_interface.planner_id = "chomp"
        elif planner_id_str.lower() == "stomp":
            self.moveit2_interface.planning_pipeline_id = "stomp"
            self.moveit2_interface.planner_id = ""
        else:  # OMPL
            self.moveit2_interface.planning_pipeline_id = "ompl"
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
            self.logger.info(f"    Set joint goal for {planner_id_str}.")
        except Exception as e:
            self.logger.error(f"    Failed to set joint goal: {e}")
            return None, 0.0

        if path_constraints_kwargs_to_apply:
            try:
                self.moveit2_interface.set_path_orientation_constraint(
                    **path_constraints_kwargs_to_apply
                )
                self.logger.info(
                    f"    Applied 'no roll' path constraint for {planner_id_str}."
                )
            except Exception as e:
                self.logger.error(f"    Failed to set path orientation constraint: {e}")

        start_time_ros = self.node.get_clock().now()
        future = self.moveit2_interface.plan_async(start_joint_state=None)
        joint_trajectory_for_analysis: Optional[JointTrajectory] = None
        timeout_duration_rclpy = Duration(seconds=self.planning_timeout_sec + 2.0)
        wait_start_time_rclpy = self.node.get_clock().now()

        while rclpy.ok() and not future.done():
            elapsed_wait = self.node.get_clock().now() - wait_start_time_rclpy
            if elapsed_wait >= timeout_duration_rclpy:
                self.logger.warn(
                    f"    Planner {planner_id_str} timed out after {elapsed_wait.nanoseconds / 1e9:.2f}s while waiting for future.done()."
                )
                if (
                    hasattr(future, "cancel")
                    and callable(future.cancel)
                    and not future.cancelled()
                ):
                    future.cancel()
                break
            self.rate.sleep()

        if future.done() and not future.cancelled():
            try:
                plan_result_srv_response = future.result()
                if (
                    plan_result_srv_response.motion_plan_response.error_code.val
                    == moveit_msgs.msg.MoveItErrorCodes.SUCCESS
                ):
                    trajectory_msg = (
                        plan_result_srv_response.motion_plan_response.trajectory
                    )
                    if trajectory_msg and trajectory_msg.joint_trajectory.points:
                        joint_trajectory_for_analysis = trajectory_msg.joint_trajectory
                        self.logger.info(f"    Planner {planner_id_str} succeeded.")
                    else:
                        self.logger.warn(
                            f"    Planner {planner_id_str} succeeded but returned an empty joint_trajectory."
                        )
                else:
                    self.logger.warn(
                        f"    Planner {planner_id_str} failed with MoveItErrorCode: {plan_result_srv_response.motion_plan_response.error_code.val}"
                    )
            except Exception as e:
                self.logger.error(
                    f"    Exception while getting plan result for {planner_id_str}: {e}",
                )
        elif future.cancelled():
            self.logger.warn(
                f"    Planning for {planner_id_str} was cancelled (likely due to timeout)."
            )

        elapsed_time = (self.node.get_clock().now() - start_time_ros).nanoseconds / 1e9
        self.moveit2_interface.clear_path_constraints()
        self.moveit2_interface.clear_goal_constraints()
        return joint_trajectory_for_analysis, elapsed_time

    def move_to_initial_config(self):
        self.logger.info(
            f"Moving to initial configuration: {self.initial_joint_config}"
        )

        original_planner = self.moveit2_interface.planner_id
        original_pipeline = self.moveit2_interface.pipeline_id
        original_timeout = self.moveit2_interface.allowed_planning_time

        self.moveit2_interface.planning_pipeline_id = "ompl"
        self.moveit2_interface.planner_id = "RRTConnectkConfigDefault"
        self.moveit2_interface.allowed_planning_time = 10.0
        self.moveit2_interface.clear_goal_constraints()
        self.moveit2_interface.clear_path_constraints()

        try:
            self.logger.info(
                f"    Commanding move to configuration with {len(self.initial_joint_config)} joints for group '{self.planning_group}'"
            )
            self.moveit2_interface.move_to_configuration(
                joint_positions=self.initial_joint_config,
                joint_names=self.joint_names_for_group,
            )
            self.moveit2_interface.wait_until_executed()
            self.logger.info("Reached initial configuration.")
        except Exception as e:
            self.logger.error(f"Failed to move to initial configuration: {e}")
            raise
        finally:
            self.moveit2_interface.planner_id = original_planner
            self.moveit2_interface.planning_pipeline_id = original_pipeline
            self.moveit2_interface.allowed_planning_time = original_timeout

    def run_benchmark_condition(self, condition: BenchmarkCondition):
        self.logger.info(f"--- Testing Configuration: {condition.name} ---")
        self.logger.info(f"  Target Joints: {condition.joint_values}")

        for planner_id_str in self.planners_to_test:
            joint_trajectory, elapsed_time = self.plan_to_target_configuration(
                goal_joints=condition.joint_values,
                planner_id_str=planner_id_str,
                path_constraints_kwargs_to_apply=self.no_roll_path_constraint_kwargs,
            )

            success = joint_trajectory is not None and bool(joint_trajectory.points)
            path_len_total, joint_path_lens_map = self._get_path_length_stats(
                joint_trajectory
            )
            max_roll_dev = self._calculate_max_roll_deviation(joint_trajectory)

            trajectory_filename = None
            if success and self.trajectory_save_dir:
                trajectory_filename = self._save_trajectory_to_file(
                    joint_trajectory, condition.name, planner_id_str
                )

            if condition.name not in self.results:
                self.results[condition.name] = {}

            self.results[condition.name][planner_id_str] = PlanResult(
                joint_trajectory,
                elapsed_time,
                path_len_total,
                joint_path_lens_map,
                max_roll_dev,
                success,
                trajectory_filename,
            )

            log_msg_suffix = (
                f"| TrajFile: {trajectory_filename}" if trajectory_filename else ""
            )
            if success:
                self.logger.info(
                    f"  Planner: {planner_id_str} | Success: True  | Time: {elapsed_time:.3f}s "
                    f"| PathLen: {path_len_total if path_len_total is not None else 'N/A':.3f} "
                    f"| MaxRollDev: {max_roll_dev if max_roll_dev is not None else 'N/A':.3f} rad {log_msg_suffix}"
                )
            else:
                self.logger.warn(
                    f"  Planner: {planner_id_str} | Success: False | Time: {elapsed_time:.3f}s {log_msg_suffix}"
                )

    def run_all_tests(self, log_summary_every_n_conditions: Optional[int] = None):
        self.move_to_initial_config()

        for i, condition in enumerate(self.target_configurations):
            self.logger.info(
                f"=== Running Condition {i + 1}/{self.num_conditions}: {condition.name} ==="
            )
            self.run_benchmark_condition(condition)

            if (
                log_summary_every_n_conditions
                and (i + 1) % log_summary_every_n_conditions == 0
            ):
                self.log_summary_results()

        self.logger.info("=== Benchmark Run Completed ===")

    def get_csv_header(self) -> List[str]:
        header = [f"start_{name}" for name in self.joint_names_for_group]
        header.extend(
            [
                "goal_config_name",
                *[f"goal_{name}" for name in self.joint_names_for_group],
                "planner_id",
                "elapsed_time_s",
                "success",
                "path_length_total",
                "max_roll_deviation_rad",
                "trajectory_filename",
            ]
        )
        header.extend([f"path_length_{name}" for name in self.joint_names_for_group])
        return header

    def write_results_to_csv(self, filename: str):
        self.logger.info(f"Writing results to {filename}")
        start_config_for_csv = self.initial_joint_config

        with open(filename, "w", newline="") as f:
            import csv

            csv_writer = csv.writer(f)
            header = self.get_csv_header()
            csv_writer.writerow(header)

            for target_condition in self.target_configurations:
                config_name = target_condition.name
                if config_name in self.results:
                    planner_runs = self.results[config_name]
                    for planner_id_str in self.planners_to_test:
                        if planner_id_str in planner_runs:
                            result = planner_runs[planner_id_str]
                            row = []
                            row.extend(map(str, start_config_for_csv))
                            row.append(str(target_condition.name))
                            row.extend(map(str, target_condition.joint_values))
                            row.append(str(planner_id_str))
                            row.append(f"{result.elapsed_time:.4f}")
                            row.append(str(1 if result.success else 0))
                            row.append(
                                f"{result.path_length:.4f}"
                                if result.path_length is not None
                                else ""
                            )
                            row.append(
                                f"{result.max_roll_deviation:.4f}"
                                if result.max_roll_deviation is not None
                                else ""
                            )
                            row.append(
                                result.trajectory_filename
                                if result.trajectory_filename
                                else ""
                            )

                            if result.joint_path_lengths:
                                for joint_name in self.joint_names_for_group:
                                    length_val = result.joint_path_lengths.get(
                                        joint_name
                                    )
                                    row.append(
                                        f"{length_val:.4f}"
                                        if length_val is not None
                                        else ""
                                    )
                            else:
                                row.extend([""] * len(self.joint_names_for_group))
                            csv_writer.writerow(row)
                        else:
                            self.logger.warn(
                                f"No result for planner '{planner_id_str}' in condition '{config_name}'. Skipping CSV row."
                            )
                else:
                    self.logger.warn(
                        f"No results found for configuration: {config_name} in self.results. Skipping CSV rows for this condition."
                    )

        self.logger.info(f"Results successfully written to {filename}")

    def log_summary_results(self):
        self.logger.info("--- BENCHMARK SUMMARY ---")
        for planner_id_str in self.planners_to_test:
            results_for_planner = [
                run_results[planner_id_str]
                for run_results in self.results.values()
                if planner_id_str in run_results
            ]

            total_plans = len(results_for_planner)
            if total_plans == 0:
                self.logger.info(f"--- Planner: {planner_id_str} ---")
                self.logger.info("  No plans attempted or recorded for this planner.")
                continue

            successful_plans = sum(1 for r in results_for_planner if r.success)
            total_planning_time = sum(r.elapsed_time for r in results_for_planner)

            path_lengths_list = [
                r.path_length
                for r in results_for_planner
                if r.success and r.path_length is not None
            ]
            roll_deviations_list = [
                r.max_roll_deviation
                for r in results_for_planner
                if r.success and r.max_roll_deviation is not None
            ]

            self.logger.info(f"--- Planner: {planner_id_str} ---")
            success_rate = (
                (successful_plans / total_plans) * 100 if total_plans > 0 else 0
            )
            avg_planning_time_all = (
                total_planning_time / total_plans if total_plans > 0 else 0
            )

            successful_planning_times = [
                r.elapsed_time for r in results_for_planner if r.success
            ]
            avg_planning_time_succ = (
                sum(successful_planning_times) / len(successful_planning_times)
                if successful_planning_times
                else 0
            )

            self.logger.info(
                f"  Success Rate: {success_rate:.2f}% ({successful_plans}/{total_plans})"
            )
            self.logger.info(
                f"  Avg. Planning Time (all attempts): {avg_planning_time_all:.3f}s"
            )
            if successful_planning_times:
                self.logger.info(
                    f"  Avg. Planning Time (successful attempts): {avg_planning_time_succ:.3f}s"
                )

            if path_lengths_list:
                self.logger.info(
                    f"  Path Lengths (successful plans): {PlannerBenchmark._summary_stats_str(path_lengths_list)}"
                )
            else:
                self.logger.info("  Path Lengths (successful plans): N/A")

            valid_roll_deviations = [
                r
                for r in roll_deviations_list
                if r is not None and not (isinstance(r, float) and np.isinf(r))
            ]
            if valid_roll_deviations:
                self.logger.info(
                    f"  Max Roll Deviations (rad, successful plans): {PlannerBenchmark._summary_stats_str(valid_roll_deviations)}"
                )
            else:
                self.logger.info(
                    "  Max Roll Deviations (rad, successful plans): N/A (or all FK failed/no valid deviations)"
                )
        self.logger.info("-------------------------")

    @staticmethod
    def _summary_stats_str(data: List[float]) -> str:
        if not data:
            return "N/A"
        valid_data = [x for x in data if isinstance(x, (int, float)) and np.isfinite(x)]
        if not valid_data:
            return "N/A (no valid data points)"
        return (
            f"Mean {np.mean(valid_data):.3f}, Median {np.median(valid_data):.3f}, "
            f"Min {np.min(valid_data):.3f}, Max {np.max(valid_data):.3f}, Std {np.std(valid_data):.3f}, "
            f"Count {len(valid_data)}"
        )


def main(output_dir_arg: Optional[str]):
    rclpy.init()
    node = Node("planner_benchmark_joint_config_constrained")

    global _LOGGER_INSTANCE
    _LOGGER_INSTANCE = node.get_logger()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True, args=())
    executor_thread.start()

    _LOGGER_INSTANCE.info(
        "Benchmark node spinning. Waiting for services (approx 5s)..."
    )
    time.sleep(5.0)

    hardcoded_targets = {
        "MoveAbovePlate": [
            -2.4538579336877304,
            3.07974419938212,
            1.8320725365979,
            4.096143890468605,
            -2.003422584820525,
            -3.2123560395465063,
        ],
        "RestingAcquireFood": [-1.94672, 2.51268, 0.35653, -4.76501, 5.99991, 4.99555],
        "MoveToRestingPosition": [
            -1.94672,
            2.51268,
            0.35653,
            -4.76501,
            5.99991,
            4.99555,
        ],
        "MoveToStagingConfiguration": [
            -2.32526,
            4.456298,
            4.16769,
            1.53262,
            -2.18359,
            -2.19525,
        ],
        "MoveToStowLocation": [-1.52101, 2.60098, 0.32811, -4.00012, 0.22831, 3.87886],
    }
    _LOGGER_INSTANCE.info("Using hardcoded target configurations.")

    planners = [
        "RRTConnectkConfigDefault",
        "RRTstarkConfigDefault",
        "CHOMP",
    ]
    initial_config = [
        -2.4538579336877304,
        3.07974419938212,
        1.8320725365979,
        4.096143890468605,
        -2.003422584820525,
        -3.2123560395465063,
    ]
    planning_group_name = "jaco_arm"
    ee_link_name = "j2n6s200_end_effector"
    base_link_name = kinova.base_link_name()
    group_joint_names = kinova.joint_names()
    planning_timeout = 15.0

    # --- Trajectory Save Directory ---
    base_output_dir = output_dir_arg if output_dir_arg else "."
    # Create the base output directory here if it's specified and doesn't exist
    if output_dir_arg and not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        _LOGGER_INSTANCE.info(f"Created base output directory: {base_output_dir}")

    trajectory_save_location = os.path.join(base_output_dir, "saved_trajectories")

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
    _LOGGER_INSTANCE.info(
        f"MoveIt2 interface initialized for benchmark with group: '{planning_group_name}'"
    )

    benchmark_runner = PlannerBenchmark(
        node=node,
        moveit2_interface=moveit2_interface,
        hardcoded_target_configs=hardcoded_targets,
        planners_to_test=planners,
        initial_joint_config=initial_config,
        planning_group=planning_group_name,
        end_effector_link=ee_link_name,
        base_link=base_link_name,
        joint_names_for_group=group_joint_names,
        planning_timeout_sec=planning_timeout,
        trajectory_save_dir=trajectory_save_location,
    )

    try:
        benchmark_runner.run_all_tests(log_summary_every_n_conditions=1)

        # Ensure base_output_dir exists before writing CSV
        if not os.path.exists(base_output_dir) and base_output_dir != ".":
            os.makedirs(base_output_dir)
            _LOGGER_INSTANCE.info(
                f"Created output directory for CSV: {base_output_dir}"
            )
        elif base_output_dir == "." and not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        csv_filename = os.path.join(
            base_output_dir,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_joint_config_constrained_benchmark.csv",
        )
        benchmark_runner.write_results_to_csv(csv_filename)
        benchmark_runner.log_summary_results()

    except Exception as e:
        _LOGGER_INSTANCE.error(f"Benchmark run failed: {e}")
    finally:
        _LOGGER_INSTANCE.info("Shutting down benchmark node.")
        rclpy.shutdown()
        executor_thread.join()


if __name__ == "__main__":
    out_dir_arg = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
    main(out_dir_arg)
