#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script runs a benchmark to evaluate a "constraint-driven planning"
methodology for the Articutool system. It features comprehensive, single-file
logging designed for easy data analysis and plotting for publications.

This version is a PURE SIMULATION. It does not move the robot. Each planning
task starts from the end-state of the previously computed plan.

It can be run in two modes:
1.  Position-Only Goals (default): Plans to a random (x, y, z) position.
2.  Position + Yaw Goals (`--constrain-goal-yaw`): Plans to a random
    (x, y, z) position AND a random final yaw orientation.

In both modes, a "smart" orientation constraint (lenient pitch, free yaw,
strict roll) is applied to the ENTIRE PATH to ensure the Articutool can
always maintain a level configuration.
"""

# Standard imports
from collections import namedtuple
from datetime import datetime
import os
import time
from threading import Thread
from typing import Optional, List, Dict, Tuple, Any
import json
import math
import subprocess
import sys

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
import moveit_msgs.msg
from geometry_msgs.msg import Quaternion, Point
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# --- Constants and Named Tuples ---
LOGGER = rclpy.logging.get_logger("constrained_task_space_benchmark")
PLANNING_GROUP = "jaco_arm"
JOINT_NAMES = kinova.joint_names()
BASE_LINK = kinova.base_link_name()
END_EFFECTOR_LINK = kinova.end_effector_name()
EPSILON = 1e-6

# Define Articutool joint limits in radians
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])

# --- Parameters for the "Smart" PATH Orientation Constraint ---
PATH_CONSTRAINT_QUAT_XYZW = (0.707, 0.0, 0.0, 0.707)
PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (1.5, 3.14, 0.8)

# --- Parameters for the optional GOAL Yaw Orientation Constraint ---
GOAL_YAW_CONSTRAINT_TOLERANCE_XYZ_RAD = (math.pi, math.pi, 0.1)

# --- Define a reasonable workspace for sampling random positions ---
WORKSPACE_OUTER_RADIUS = 0.7
WORKSPACE_INNER_RADIUS = 0.3

# --- Data Structures for Clarity ---
TrajectoryMetrics = namedtuple(
    "TrajectoryMetrics",
    [
        "duration_s",
        "joint_space_path_length_rad",
        "final_joint_positions",
        "articutool_pitch_stats_rad",
        "articutool_roll_stats_rad",
        "waypoints_data",
    ],
)


class ConstrainedTaskSpaceBenchmark:
    """Manages the task-space benchmark planning process."""

    def __init__(
        self,
        node: Node,
        moveit2: MoveIt2,
        xacro_file_path: str,
        num_tasks: int = 100,
        planning_timeout: float = 5.0,
        planner_id: str = "RRTConnectkConfigDefault",
        output_dir: Optional[str] = None,
        constrain_goal_yaw: bool = False,
    ):
        self.node = node
        self.moveit2 = moveit2
        self.xacro_file_path = xacro_file_path
        self.num_tasks = num_tasks
        self.planning_timeout = planning_timeout
        self.planner_id = planner_id
        self.output_dir = output_dir
        self.constrain_goal_yaw = constrain_goal_yaw
        self.results: List[Dict[str, Any]] = []

        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self.joint_name_to_pinocchio_id: Dict[str, int] = {}
        self.jaco_vel_indices: List[int] = []
        self._initialize_pinocchio()
        self.joint_limits = self._get_joint_limits()
        self.debug_printed = False  # Flag to print debug logs only once

        LOGGER.info("Benchmark simulator initialized.")
        if self.constrain_goal_yaw:
            LOGGER.info("Planning Mode: Position + Yaw Goals.")
        else:
            LOGGER.info("Planning Mode: Position-Only Goals.")
        LOGGER.info(f"Applying 'smart' orientation constraint to all paths.")

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            LOGGER.info(f"Comprehensive results will be saved to: {self.output_dir}")

    def _initialize_pinocchio(self):
        """Loads the robot model from a XACRO file into Pinocchio."""
        try:
            process = subprocess.run(
                ["ros2", "run", "xacro", "xacro", self.xacro_file_path],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_xml_string = process.stdout
            self.pinocchio_model = pin.buildModelFromXML(urdf_xml_string)
            self.pinocchio_data = self.pinocchio_model.createData()
            self.jaco_ee_frame_id_pin = self.pinocchio_model.getFrameId(
                END_EFFECTOR_LINK
            )
            for name in JOINT_NAMES:
                if self.pinocchio_model.existJointName(name):
                    joint_id = self.pinocchio_model.getJointId(name)
                    self.joint_name_to_pinocchio_id[name] = joint_id
                    # Store the velocity index for this joint
                    self.jaco_vel_indices.append(
                        self.pinocchio_model.joints[joint_id].idx_v
                    )

            LOGGER.info("Pinocchio model loaded successfully.")
        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            LOGGER.error(f"Failed to initialize Pinocchio model: {e}")
            self.pinocchio_model = None

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Retrieves joint limits from the Pinocchio model."""
        if self.pinocchio_model is None:
            return [(-math.pi, math.pi)] * len(JOINT_NAMES)
        limits = []
        for name in JOINT_NAMES:
            if self.pinocchio_model.existJointName(name):
                joint_id = self.pinocchio_model.getJointId(name)
                idx = self.pinocchio_model.joints[joint_id].idx_q
                limits.append(
                    (
                        self.pinocchio_model.lowerPositionLimit[idx],
                        self.pinocchio_model.upperPositionLimit[idx],
                    )
                )
        return limits

    def generate_random_position_in_workspace(self) -> np.ndarray:
        """Generates a random (x, y, z) position from a half-spherical shell."""
        r = (
            np.random.uniform(WORKSPACE_INNER_RADIUS**3, WORKSPACE_OUTER_RADIUS**3)
        ) ** (1 / 3)
        theta = np.random.uniform(0, 2 * math.pi)
        cos_phi = np.random.uniform(0, 1)
        phi = math.acos(cos_phi)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * cos_phi
        return np.array([x, y, z])

    def generate_random_yaw_quaternion(self) -> Tuple[float, float, float, float]:
        """Generates a quaternion representing a random yaw."""
        random_yaw_angle = np.random.uniform(-math.pi, math.pi)
        quat_xyzw = R.from_euler("z", random_yaw_angle).as_quat()
        return tuple(quat_xyzw)

    def _get_articutool_jacobian(self, pitch: float, roll: float) -> np.ndarray:
        """
        Computes the analytical Jacobian for the Articutool's 'up' vector (y-axis).
        This relates Articutool joint velocities to the angular velocity of its y-axis.
        J = [∂y/∂θp, ∂y/∂θr]
        """
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        # Partial derivatives of the y-axis vector [-sr, cp*cr, sp*cr]
        dydp = np.array([0, -sp * cr, cp * cr])
        dydr = np.array([-cr, -cp * sr, -sp * sr])
        return np.vstack([dydp, dydr]).T

    def _solve_articutool_ik_for_leveling(
        self, target_y_axis_in_atool_base: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Analytical IK solver for the Articutool."""
        vx, vy, vz = target_y_axis_in_atool_base
        solutions: List[Tuple[float, float]] = []
        asin_arg_for_tr = -vx
        if not (-1.0 - EPSILON <= asin_arg_for_tr <= 1.0 + EPSILON):
            return []
        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = (math.pi - theta_r_sol1 + math.pi) % (2 * math.pi) - math.pi
        candidate_thetas_r = list(set([theta_r_sol1, theta_r_sol2]))
        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            if math.isclose(cos_theta_r, 0.0, abs_tol=EPSILON):
                if math.isclose(vy, 0.0, abs_tol=EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=EPSILON
                ):
                    solutions.append(
                        (0.0, (theta_r + math.pi) % (2 * math.pi) - math.pi)
                    )
                continue
            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (
                    (theta_p_sol + math.pi) % (2 * math.pi) - math.pi,
                    (theta_r + math.pi) % (2 * math.pi) - math.pi,
                )
            )
        return solutions

    def _calculate_trajectory_metrics(
        self, trajectory: JointTrajectory
    ) -> TrajectoryMetrics:
        """Calculates detailed metrics for a given trajectory."""
        if self.pinocchio_model is None or not trajectory.points:
            return TrajectoryMetrics(0.0, 0.0, [], {}, {}, [])

        joint_space_path_length = 0.0
        waypoints_data = []
        q = pin.neutral(self.pinocchio_model)
        prev_positions = np.array(trajectory.points[0].positions)
        all_pitches, all_rolls = [], []

        for i, point in enumerate(trajectory.points):
            # Update Pinocchio model for current waypoint
            for j, name in enumerate(JOINT_NAMES):
                if name in self.joint_name_to_pinocchio_id:
                    joint_id = self.joint_name_to_pinocchio_id[name]
                    joint_obj = self.pinocchio_model.joints[joint_id]
                    theta = point.positions[j]
                    if joint_obj.nq == 2:
                        q[joint_obj.idx_q : joint_obj.idx_q + 2] = [
                            math.cos(theta),
                            math.sin(theta),
                        ]
                    else:
                        q[joint_obj.idx_q] = theta

            pin.computeAllTerms(
                self.pinocchio_model,
                self.pinocchio_data,
                q,
                np.zeros(self.pinocchio_model.nv),
            )
            pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

            # Get EE Pose and Articutool IK solution
            ee_transform = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]
            R_world_ee = R.from_matrix(ee_transform.rotation)
            target_up = R_world_ee.inv().apply(WORLD_UP_VECTOR)
            solutions = self._solve_articutool_ik_for_leveling(target_up)

            at_solution, at_velocities = None, None
            if solutions:
                valid_sols = [
                    s
                    for s in solutions
                    if ARTICUTOOL_PITCH_LIMITS_RAD[0]
                    <= s[0]
                    <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
                    and ARTICUTOOL_ROLL_LIMITS_RAD[0]
                    <= s[1]
                    <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
                ]
                if valid_sols:
                    pitch, roll = min(valid_sols, key=lambda s: s[0] ** 2 + s[1] ** 2)
                    at_solution = {"pitch": pitch, "roll": roll}
                    all_pitches.append(pitch)
                    all_rolls.append(roll)

                    # Calculate required Articutool velocities
                    if i > 0:
                        dt = (
                            point.time_from_start.sec
                            + point.time_from_start.nanosec * 1e-9
                        ) - (
                            trajectory.points[i - 1].time_from_start.sec
                            + trajectory.points[i - 1].time_from_start.nanosec * 1e-9
                        )
                        if dt > EPSILON:
                            q_dot_jaco = (
                                np.array(point.positions)
                                - np.array(trajectory.points[i - 1].positions)
                            ) / dt

                            J_jaco_full = pin.getFrameJacobian(
                                self.pinocchio_model,
                                self.pinocchio_data,
                                self.jaco_ee_frame_id_pin,
                                pin.ReferenceFrame.LOCAL,
                            )
                            J_jaco_arm = J_jaco_full[:, self.jaco_vel_indices]

                            v_ee = J_jaco_arm @ q_dot_jaco
                            omega_disturbance_local = v_ee[
                                3:
                            ]  # Angular velocity portion

                            J_atool = self._get_articutool_jacobian(pitch, roll)
                            J_atool_inv = np.linalg.pinv(J_atool)
                            q_dot_atool = -J_atool_inv @ omega_disturbance_local
                            at_velocities = {
                                "pitch_vel": q_dot_atool[0],
                                "roll_vel": q_dot_atool[1],
                            }

                            # --- START DEBUG LOGS ---
                            if not self.debug_printed and i < 5:
                                LOGGER.info(f"--- DEBUG WP {i} ---")
                                LOGGER.info(f"  dt: {dt:.4f}s")
                                LOGGER.info(f"  q_dot_jaco: {np.round(q_dot_jaco, 3)}")
                                LOGGER.info(f"  J_jaco_full shape: {J_jaco_full.shape}")
                                LOGGER.info(f"  J_jaco_arm shape: {J_jaco_arm.shape}")
                                LOGGER.info(f"  v_ee (local): {np.round(v_ee, 3)}")
                                LOGGER.info(
                                    f"  omega_disturbance_local: {np.round(omega_disturbance_local, 3)}"
                                )
                                LOGGER.info(f"  J_atool:\n{np.round(J_atool, 3)}")
                                LOGGER.info(
                                    f"  q_dot_atool (req'd): {np.round(q_dot_atool, 3)}"
                                )
                            # --- END DEBUG LOGS ---

            waypoints_data.append(
                {
                    "time_from_start_sec": point.time_from_start.sec
                    + point.time_from_start.nanosec * 1e-9,
                    "jaco_positions_rad": list(point.positions),
                    "ee_pose_world": {
                        "position": ee_transform.translation.tolist(),
                        "quat_xyzw": R.from_matrix(ee_transform.rotation)
                        .as_quat()
                        .tolist(),
                    },
                    "articutool_solution_rad": at_solution,
                    "articutool_velocities_rad_per_sec": at_velocities,
                }
            )
            joint_space_path_length += np.linalg.norm(
                np.array(point.positions) - prev_positions
            )
            prev_positions = np.array(point.positions)

        self.debug_printed = (
            True  # Ensure debug logs only print for the first trajectory
        )
        duration = (
            trajectory.points[-1].time_from_start.sec
            + trajectory.points[-1].time_from_start.nanosec * 1e-9
        )
        pitch_stats = (
            {
                "min": min(all_pitches),
                "max": max(all_pitches),
                "mean": np.mean(all_pitches),
                "std_dev": np.std(all_pitches),
            }
            if all_pitches
            else {}
        )
        roll_stats = (
            {
                "min": min(all_rolls),
                "max": max(all_rolls),
                "mean": np.mean(all_rolls),
                "std_dev": np.std(all_rolls),
            }
            if all_rolls
            else {}
        )

        return TrajectoryMetrics(
            duration,
            joint_space_path_length,
            list(trajectory.points[-1].positions),
            pitch_stats,
            roll_stats,
            waypoints_data,
        )

    def run(self):
        """Main benchmark execution loop."""
        if not self.pinocchio_model:
            LOGGER.error("Aborting benchmark: Pinocchio model failed to initialize.")
            return

        for i in range(self.num_tasks):
            LOGGER.info(f"--- Running Task {i + 1}/{self.num_tasks} ---")

            # Get the starting joint state for this trial from the robot's current position
            start_joint_positions = self.moveit2.joint_state.position

            # Generate task goals
            target_position = self.generate_random_position_in_workspace()
            target_quat_xyzw = (
                self.generate_random_yaw_quaternion()
                if self.constrain_goal_yaw
                else None
            )

            # Set up and execute planning request
            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()
            self.moveit2.set_position_goal(
                position=target_position.tolist(),
                frame_id=BASE_LINK,
                target_link=END_EFFECTOR_LINK,
                tolerance=0.01,
            )
            if self.constrain_goal_yaw:
                self.moveit2.set_orientation_goal(
                    quat_xyzw=Quaternion(
                        x=target_quat_xyzw[0],
                        y=target_quat_xyzw[1],
                        z=target_quat_xyzw[2],
                        w=target_quat_xyzw[3],
                    ),
                    target_link=END_EFFECTOR_LINK,
                    tolerance=GOAL_YAW_CONSTRAINT_TOLERANCE_XYZ_RAD,
                    parameterization=1,
                )
            self.moveit2.set_path_orientation_constraint(
                quat_xyzw=Quaternion(
                    x=PATH_CONSTRAINT_QUAT_XYZW[0],
                    y=PATH_CONSTRAINT_QUAT_XYZW[1],
                    z=PATH_CONSTRAINT_QUAT_XYZW[2],
                    w=PATH_CONSTRAINT_QUAT_XYZW[3],
                ),
                frame_id=BASE_LINK,
                target_link=END_EFFECTOR_LINK,
                tolerance=PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
                weight=1.0,
                parameterization=1,
            )
            self.moveit2.planner_id = self.planner_id
            self.moveit2.allowed_planning_time = self.planning_timeout

            planning_start_time = time.perf_counter()
            plan_future = self.moveit2.plan_async()
            while rclpy.ok() and not plan_future.done():
                time.sleep(0.01)
            trajectory = self.moveit2.get_trajectory(plan_future)
            planning_time = time.perf_counter() - planning_start_time

            plan_success = trajectory is not None and bool(trajectory.points)
            trial_data = {
                "task_id": i,
                "planning_mode": "pos_yaw" if self.constrain_goal_yaw else "pos_only",
                "start_joint_positions": list(start_joint_positions),
                "target_position": target_position.tolist(),
                "target_yaw_quat": target_quat_xyzw,
                "plan_success": plan_success,
                "planning_time_s": planning_time,
                "verification_success": None,
                "trajectory_metrics": None,
            }

            if plan_success:
                LOGGER.info(f"  Jaco plan SUCCEEDED in {planning_time:.4f}s.")
                metrics = self._calculate_trajectory_metrics(trajectory)
                is_feasible = all(
                    wp["articutool_solution_rad"] is not None
                    for wp in metrics.waypoints_data
                )
                trial_data["verification_success"] = is_feasible
                trial_data["trajectory_metrics"] = metrics._asdict()
                LOGGER.info(f"  Verification Result: Path Feasible = {is_feasible}")
            else:
                LOGGER.warn(f"  Jaco plan FAILED in {planning_time:.4f}s.")

            self.results.append(trial_data)

    def save_results(self):
        """Saves the comprehensive benchmark results to a single JSON file."""
        if not self.output_dir:
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode = "pos_yaw" if self.constrain_goal_yaw else "pos_only"
        filename = os.path.join(
            self.output_dir, f"benchmark_results_{mode}_{timestamp}.json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            LOGGER.info(f"Comprehensive benchmark results saved to {filename}")
        except Exception as e:
            LOGGER.error(f"Failed to save results: {e}")
        total_tasks, jaco_successes, verified_successes = (
            len(self.results),
            sum(1 for r in self.results if r["plan_success"]),
            sum(1 for r in self.results if r.get("verification_success", False)),
        )
        jaco_success_rate = (
            (jaco_successes / total_tasks * 100) if total_tasks > 0 else 0
        )
        LOGGER.info(f"\n--- TASK-SPACE BENCHMARK SUMMARY ({mode.upper()}) ---")
        LOGGER.info(f"Total Tasks Attempted: {total_tasks}")
        LOGGER.info(
            f"Jaco Planner Success Rate: {jaco_success_rate:.2f}% ({jaco_successes}/{total_tasks})"
        )
        if jaco_successes > 0:
            LOGGER.info(
                f"Verification Success Rate (of successful plans): {(verified_successes / jaco_successes * 100):.2f}%"
            )
        LOGGER.info("-------------------------------------\n")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 constrained_task_space_benchmark.py /path/to/robot.urdf.xacro [--constrain-goal-yaw]"
        )
        sys.exit(1)
    xacro_file_arg, constrain_goal_yaw_arg = (
        sys.argv[1],
        "--constrain-goal-yaw" in sys.argv,
    )
    if not os.path.exists(xacro_file_arg):
        print(f"Error: XACRO file not found at '{xacro_file_arg}'")
        sys.exit(1)

    rclpy.init()
    node = Node("constrained_task_space_benchmark_node")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=END_EFFECTOR_LINK,
        group_name=PLANNING_GROUP,
        callback_group=ReentrantCallbackGroup(),
    )

    NUM_TASKS, PLANNING_TIMEOUT, PLANNER_ID, OUTPUT_DIR = (
        100,
        10.0,
        "RRTConnectkConfigDefault",
        os.path.join(os.getcwd(), "constrained_task_space_benchmark_output"),
    )

    benchmark = ConstrainedTaskSpaceBenchmark(
        node,
        moveit2,
        xacro_file_arg,
        NUM_TASKS,
        PLANNING_TIMEOUT,
        PLANNER_ID,
        OUTPUT_DIR,
        constrain_goal_yaw_arg,
    )

    try:
        if benchmark.pinocchio_model is not None:
            # This benchmark is now a pure simulation. It starts from the robot's
            # current state and each subsequent plan starts from the end of the
            # previous one, without actually moving the robot.
            LOGGER.info("Ready to begin benchmark from current robot configuration.")
            benchmark.run()
            benchmark.save_results()
        else:
            LOGGER.error("Benchmark cannot run because Pinocchio model failed to load.")
    except Exception as e:
        LOGGER.error(f"An error occurred during the benchmark: {e}", exc_info=True)
    finally:
        LOGGER.info("Shutting down.")
        rclpy.shutdown()
        executor_thread.join()


if __name__ == "__main__":
    main()
