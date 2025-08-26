#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script runs a simulation-based benchmark to evaluate the "plan-then-verify"
control methodology for the Articutool, as described in the accompanying paper.

It operates in a simulation-only mode:
1. Generates a specified number of random start and goal joint configurations for
   the Jaco arm.
2. For each pair, it uses MoveIt2 to plan a trajectory.
3. If a plan is found, it performs a post-hoc feasibility analysis on the
   trajectory. For each waypoint, it calculates the required Articutool pitch and
   roll angles to keep the end-effector level with respect to gravity.
4. It checks if these required angles are within the Articutool's joint limits.
5. It saves an "enhanced" trajectory file (.json) containing the per-waypoint
   Articutool solutions, which can then be used by `visualize_trajectory.py`.
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
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# --- Constants and Named Tuples ---
LOGGER = rclpy.logging.get_logger("articutool_benchmark_sim")
PLANNING_GROUP = "jaco_arm"
JOINT_NAMES = kinova.joint_names()
BASE_LINK = kinova.base_link_name()
END_EFFECTOR_LINK = kinova.end_effector_name()
EPSILON = 1e-6

# Define Articutool joint limits in radians
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)  # (-90, 90) degrees
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)  # (-180, 180) degrees

# Define the world's "up" vector (positive Z axis)
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])

# --- Data Structures for Clarity ---
PlanningTask = namedtuple("PlanningTask", ["start_config", "goal_config"])
ArticutoolWaypointSolution = namedtuple(
    "ArticutoolWaypointSolution",
    ["waypoint_feasible", "pitch_solution_rad", "roll_solution_rad"],
)
ArticutoolMetrics = namedtuple(
    "ArticutoolMetrics",
    [
        "path_feasible",
        "num_infeasible_points",
        "pitch_range_used_percent",
        "roll_range_used_percent",
        "avg_pitch_abs_rad",
        "avg_roll_abs_rad",
    ],
)


class ArticutoolBenchmarkSimulator:
    """Manages the simulation-based benchmarking process."""

    def __init__(
        self,
        node: Node,
        moveit2: MoveIt2,
        xacro_file_path: str,
        num_tasks: int = 100,
        planning_timeout: float = 5.0,
        planner_id: str = "RRTConnectkConfigDefault",
        trajectory_save_dir: Optional[str] = None,
    ):
        self.node = node
        self.moveit2 = moveit2
        self.xacro_file_path = xacro_file_path
        self.num_tasks = num_tasks
        self.planning_timeout = planning_timeout
        self.planner_id = planner_id
        self.trajectory_save_dir = trajectory_save_dir
        self.results: List[Dict[str, Any]] = []

        # Pinocchio model for forward kinematics
        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self.joint_name_to_pinocchio_id: Dict[str, int] = {}
        self._initialize_pinocchio()

        self.joint_limits = self._get_joint_limits()

        LOGGER.info("Benchmark simulator initialized.")
        LOGGER.info(f"Testing with planner: {self.planner_id}")
        LOGGER.info(f"Number of random tasks to generate: {self.num_tasks}")
        if self.trajectory_save_dir:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            LOGGER.info(
                f"Enhanced trajectories will be saved to: {self.trajectory_save_dir}"
            )

    def _initialize_pinocchio(self):
        """
        Loads the robot model from a XACRO file into Pinocchio.
        This is a more robust method that explicitly handles XACRO to URDF conversion
        and ensures Pinocchio can find all mesh files.
        """
        LOGGER.info(f"Processing Xacro file: {self.xacro_file_path}")
        try:
            # Convert XACRO to a URDF string
            process = subprocess.run(
                ["ros2", "run", "xacro", "xacro", self.xacro_file_path],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_xml_string = process.stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            LOGGER.error(f"Failed to process XACRO file: {e}")
            LOGGER.error(
                "Make sure ROS 2 environment is sourced and xacro is installed."
            )
            self.pinocchio_model = None
            return

        # Load the model from the URDF string
        try:
            self.pinocchio_model = pin.buildModelFromXML(urdf_xml_string)
            self.pinocchio_data = self.pinocchio_model.createData()
            self.jaco_ee_frame_id_pin = self.pinocchio_model.getFrameId(
                END_EFFECTOR_LINK
            )
            # Map joint names to their Pinocchio model IDs for quick access
            for name in JOINT_NAMES:
                if self.pinocchio_model.existJointName(name):
                    self.joint_name_to_pinocchio_id[name] = (
                        self.pinocchio_model.getJointId(name)
                    )
            LOGGER.info("Pinocchio model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to build Pinocchio model from URDF string: {e}")
            self.pinocchio_model = None

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Retrieves joint limits from the Pinocchio model in the correct order."""
        if self.pinocchio_model is None:
            LOGGER.warn("Pinocchio model not available, using default joint limits.")
            return [(-math.pi, math.pi)] * len(JOINT_NAMES)

        limits = []
        for joint_name in JOINT_NAMES:
            if self.pinocchio_model.existJointName(joint_name):
                joint_id = self.pinocchio_model.getJointId(joint_name)
                joint = self.pinocchio_model.joints[joint_id]
                # Get the index in the configuration vector 'q' for this joint
                limit_idx = joint.idx_q
                lower = self.pinocchio_model.lowerPositionLimit[limit_idx]
                upper = self.pinocchio_model.upperPositionLimit[limit_idx]
                limits.append((lower, upper))
            else:
                # Fallback for safety, though this shouldn't happen with a correct URDF
                limits.append((-math.pi, math.pi))
        return limits

    def generate_random_config(self) -> List[float]:
        """Generates a random joint configuration within the robot's limits."""
        return [
            (
                np.random.uniform(low, high)
                if np.isfinite(low) and np.isfinite(high)
                else np.random.uniform(-np.pi, np.pi)
            )
            for low, high in self.joint_limits
        ]

    def _normalize_angle(self, angle: float) -> float:
        """Normalizes an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _solve_articutool_ik_for_leveling(
        self, target_y_axis_in_atool_base: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Analytical Inverse Kinematics (IK) solver for the 2-DOF Articutool.
        """
        vx, vy, vz = target_y_axis_in_atool_base
        solutions: List[Tuple[float, float]] = []

        # From the kinematic model: sin(theta_r) = -vx
        asin_arg_for_tr = -vx
        if not (-1.0 - EPSILON <= asin_arg_for_tr <= 1.0 + EPSILON):
            return []

        asin_arg_for_tr_clipped = np.clip(asin_arg_for_tr, -1.0, 1.0)
        theta_r_sol1 = math.asin(asin_arg_for_tr_clipped)
        theta_r_sol2 = self._normalize_angle(math.pi - theta_r_sol1)

        candidate_thetas_r = list(set([theta_r_sol1, theta_r_sol2]))

        for theta_r in candidate_thetas_r:
            cos_theta_r = math.cos(theta_r)
            if math.isclose(cos_theta_r, 0.0, abs_tol=EPSILON):
                if math.isclose(vy, 0.0, abs_tol=EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=EPSILON
                ):
                    solutions.append((0.0, self._normalize_angle(theta_r)))
                continue

            theta_p_sol = math.atan2(vz, vy)
            solutions.append(
                (self._normalize_angle(theta_p_sol), self._normalize_angle(theta_r))
            )
        return solutions

    def _analyze_trajectory_for_articutool_leveling(
        self, trajectory: JointTrajectory
    ) -> Tuple[ArticutoolMetrics, List[ArticutoolWaypointSolution]]:
        """
        Performs the core "plan-then-verify" analysis on a trajectory.
        """
        if self.pinocchio_model is None:
            LOGGER.error(
                "Cannot perform feasibility check: Pinocchio model not loaded."
            )
            return (
                ArticutoolMetrics(
                    False, len(trajectory.points), np.nan, np.nan, np.nan, np.nan
                ),
                [],
            )

        required_pitches, required_rolls = [], []
        num_infeasible_wps = 0
        per_waypoint_solutions: List[ArticutoolWaypointSolution] = []
        q = pin.neutral(self.pinocchio_model)

        for point in trajectory.points:
            # ----- KINEMATIC FIX START -----
            # Correctly populate the Pinocchio configuration vector 'q'.
            # Some joints (especially revolute) are represented by Pinocchio
            # using 2 values (cos(theta), sin(theta)) in the 'q' vector for a
            # 1-DOF joint. A simple assignment q[idx] = theta is incorrect.
            # We must check the model's representation for each joint.
            for i, name in enumerate(JOINT_NAMES):
                if name in self.joint_name_to_pinocchio_id:
                    joint_id = self.joint_name_to_pinocchio_id[name]
                    joint_obj = self.pinocchio_model.joints[joint_id]
                    theta = point.positions[i]

                    # nq is the size of the joint's representation in 'q'
                    # nv is the number of degrees of freedom (velocity size)
                    if joint_obj.nq == 2 and joint_obj.nv == 1:
                        # This is a revolute joint represented by (cos, sin)
                        q[joint_obj.idx_q] = math.cos(theta)
                        q[joint_obj.idx_q + 1] = math.sin(theta)
                    elif joint_obj.nq == 1 and joint_obj.nv == 1:
                        # This is a simple 1-to-1 mapping (e.g., prismatic or simple revolute)
                        q[joint_obj.idx_q] = theta
                    else:
                        # Fallback for other joint types, though not expected for Jaco
                        q[joint_obj.idx_q] = theta
            # ----- KINEMATIC FIX END -----

            pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, q)
            pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

            ee_transform: pin.SE3 = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]
            R_world_ee = R.from_matrix(ee_transform.rotation)

            target_up_in_ee_frame = R_world_ee.inv().apply(WORLD_UP_VECTOR)

            # Call the IK solver with the correctly calculated target vector.
            ik_solutions = self._solve_articutool_ik_for_leveling(target_up_in_ee_frame)

            valid_solutions = [
                s
                for s in ik_solutions
                if (
                    ARTICUTOOL_PITCH_LIMITS_RAD[0] - EPSILON
                    <= s[0]
                    <= ARTICUTOOL_PITCH_LIMITS_RAD[1] + EPSILON
                    and ARTICUTOOL_ROLL_LIMITS_RAD[0] - EPSILON
                    <= s[1]
                    <= ARTICUTOOL_ROLL_LIMITS_RAD[1] + EPSILON
                )
            ]

            if not valid_solutions:
                num_infeasible_wps += 1
                per_waypoint_solutions.append(
                    ArticutoolWaypointSolution(False, None, None)
                )
            else:
                best_sol = min(valid_solutions, key=lambda s: s[0] ** 2 + s[1] ** 2)
                required_pitches.append(best_sol[0])
                required_rolls.append(best_sol[1])
                per_waypoint_solutions.append(
                    ArticutoolWaypointSolution(True, best_sol[0], best_sol[1])
                )

        path_feasible = num_infeasible_wps == 0
        p_range = (
            (np.max(required_pitches) - np.min(required_pitches))
            if required_pitches
            else 0
        )
        r_range = (
            (np.max(required_rolls) - np.min(required_rolls)) if required_rolls else 0
        )
        total_p_range = ARTICUTOOL_PITCH_LIMITS_RAD[1] - ARTICUTOOL_PITCH_LIMITS_RAD[0]
        total_r_range = ARTICUTOOL_ROLL_LIMITS_RAD[1] - ARTICUTOOL_ROLL_LIMITS_RAD[0]

        return (
            ArticutoolMetrics(
                path_feasible=path_feasible,
                num_infeasible_points=num_infeasible_wps,
                pitch_range_used_percent=(
                    (p_range / total_p_range * 100) if total_p_range > EPSILON else 0
                ),
                roll_range_used_percent=(
                    (r_range / total_r_range * 100) if total_r_range > EPSILON else 0
                ),
                avg_pitch_abs_rad=(
                    np.mean(np.abs(required_pitches)) if required_pitches else 0.0
                ),
                avg_roll_abs_rad=(
                    np.mean(np.abs(required_rolls)) if required_rolls else 0.0
                ),
            ),
            per_waypoint_solutions,
        )

    def _save_enhanced_trajectory_to_file(
        self,
        original_trajectory: JointTrajectory,
        articutool_solutions: List[ArticutoolWaypointSolution],
        task_id: int,
    ) -> Optional[str]:
        """Saves the trajectory data along with Articutool solutions to a JSON file."""
        if not self.trajectory_save_dir:
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            filename = f"task_{task_id}_{self.planner_id}_{timestamp}.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)

            waypoints_data = []
            for i, jaco_point in enumerate(original_trajectory.points):
                at_solution = (
                    articutool_solutions[i]
                    if i < len(articutool_solutions)
                    else ArticutoolWaypointSolution(False, None, None)
                )
                waypoints_data.append(
                    {
                        "time_from_start_sec": jaco_point.time_from_start.sec
                        + jaco_point.time_from_start.nanosec * 1e-9,
                        "jaco_positions_rad": list(jaco_point.positions),
                        "articutool_waypoint_feasible": at_solution.waypoint_feasible,
                        "articutool_pitch_solution_rad": at_solution.pitch_solution_rad,
                        "articutool_roll_solution_rad": at_solution.roll_solution_rad,
                    }
                )

            enhanced_data = {
                "jaco_joint_names": list(original_trajectory.joint_names),
                "waypoints": waypoints_data,
            }

            with open(filepath, "w") as f:
                json.dump(enhanced_data, f, indent=2)
            return filename
        except Exception as e:
            LOGGER.error(f"Failed to save enhanced trajectory for task {task_id}: {e}")
            return None

    def run(self):
        """Main benchmark execution loop."""
        if not self.pinocchio_model:
            LOGGER.error("Aborting benchmark: Pinocchio model failed to initialize.")
            return

        LOGGER.info("Generating random planning tasks...")
        tasks = [
            PlanningTask(self.generate_random_config(), self.generate_random_config())
            for _ in range(self.num_tasks)
        ]
        LOGGER.info(f"Successfully generated {len(tasks)} tasks.")

        for i, task in enumerate(tasks):
            total_task_start_time = time.perf_counter()
            LOGGER.info(f"--- Running Task {i + 1}/{len(tasks)} ---")

            start_state_msg = moveit_msgs.msg.RobotState()
            start_state_msg.joint_state.name = JOINT_NAMES
            start_state_msg.joint_state.position = list(task.start_config)
            self.moveit2._MoveIt2__move_action_goal.request.start_state = (
                start_state_msg
            )

            self.moveit2.set_joint_goal(list(task.goal_config))
            self.moveit2.planner_id = self.planner_id
            self.moveit2.allowed_planning_time = self.planning_timeout

            planning_start_time = time.perf_counter()
            plan_future = self.moveit2.plan_async()

            while rclpy.ok() and not plan_future.done():
                time.sleep(0.01)

            trajectory = self.moveit2.get_trajectory(plan_future)
            planning_time = time.perf_counter() - planning_start_time

            plan_success = trajectory is not None and bool(trajectory.points)
            articutool_metrics = None
            feasibility_check_time = 0.0
            trajectory_filename = None

            if plan_success:
                LOGGER.info(f"  Jaco plan SUCCEEDED in {planning_time:.4f}s.")
                feasibility_start_time = time.perf_counter()
                articutool_metrics, solutions = (
                    self._analyze_trajectory_for_articutool_leveling(trajectory)
                )
                feasibility_check_time = time.perf_counter() - feasibility_start_time
                LOGGER.info(
                    f"  Articutool Leveling Feasibility: {articutool_metrics.path_feasible} "
                    f"({articutool_metrics.num_infeasible_points} infeasible points) "
                    f"(checked in {feasibility_check_time:.4f}s)"
                )
                trajectory_filename = self._save_enhanced_trajectory_to_file(
                    trajectory, solutions, i
                )
            else:
                LOGGER.warn(f"  Jaco plan FAILED in {planning_time:.4f}s.")

            total_task_time = time.perf_counter() - total_task_start_time
            self.results.append(
                {
                    "task_id": i,
                    "jaco_plan_success": plan_success,
                    "planning_time_s": planning_time,
                    "feasibility_check_time_s": feasibility_check_time,
                    "total_task_time_s": total_task_time,
                    "trajectory_filename": trajectory_filename,
                    **(articutool_metrics._asdict() if articutool_metrics else {}),
                }
            )

    def save_results(self, output_dir: str):
        """Saves the benchmark results to a JSON file and prints a summary."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")

        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        LOGGER.info(f"Benchmark results saved to {filename}")

        total_tasks = len(self.results)
        jaco_successes = sum(1 for r in self.results if r["jaco_plan_success"])
        articutool_successes = sum(
            1 for r in self.results if r.get("path_feasible", False)
        )

        jaco_success_rate = (
            (jaco_successes / total_tasks * 100) if total_tasks > 0 else 0
        )
        rejection_rate = (
            ((jaco_successes - articutool_successes) / jaco_successes * 100)
            if jaco_successes > 0
            else 0
        )

        LOGGER.info("\n--- BENCHMARK SUMMARY ---")
        LOGGER.info(f"Total Tasks Attempted: {total_tasks}")
        LOGGER.info(
            f"Jaco Planner Success Rate: {jaco_success_rate:.2f}% ({jaco_successes}/{total_tasks})"
        )
        LOGGER.info(
            f"Articutool Feasibility Rejection Rate (of successful Jaco plans): {rejection_rate:.2f}%"
        )

        feasible_results = [r for r in self.results if r.get("path_feasible")]
        if feasible_results:
            avg_plan_time = np.mean([r["planning_time_s"] for r in feasible_results])
            avg_check_time = np.mean(
                [r["feasibility_check_time_s"] for r in feasible_results]
            )
            LOGGER.info(f"Avg. Planning Time (on feasible paths): {avg_plan_time:.4f}s")
            LOGGER.info(
                f"Avg. Feasibility Check Time (on feasible paths): {avg_check_time:.4f}s"
            )
        LOGGER.info("-------------------------\n")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 articutool_feasibility_benchmark.py /path/to/your/robot.urdf.xacro"
        )
        sys.exit(1)
    xacro_file_arg = sys.argv[1]
    if not os.path.exists(xacro_file_arg):
        print(f"Error: XACRO file not found at '{xacro_file_arg}'")
        sys.exit(1)

    rclpy.init()
    node = Node("articutool_benchmark_node")

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=kinova.base_link_name(),
        end_effector_name=END_EFFECTOR_LINK,
        group_name=PLANNING_GROUP,
        callback_group=ReentrantCallbackGroup(),
    )

    NUM_TASKS = 200
    PLANNING_TIMEOUT = 5.0
    PLANNER_ID = "RRTConnectkConfigDefault"
    OUTPUT_DIR = os.path.join(os.getcwd(), "articutool_benchmark_output")

    benchmark = ArticutoolBenchmarkSimulator(
        node=node,
        moveit2=moveit2,
        xacro_file_path=xacro_file_arg,
        num_tasks=NUM_TASKS,
        planning_timeout=PLANNING_TIMEOUT,
        planner_id=PLANNER_ID,
        trajectory_save_dir=OUTPUT_DIR,
    )

    try:
        if benchmark.pinocchio_model is not None:
            benchmark.run()
            benchmark.save_results(OUTPUT_DIR)
        else:
            LOGGER.error("Benchmark cannot run because Pinocchio model failed to load.")
    except Exception as e:
        LOGGER.error(f"An error occurred during the benchmark: {e}")
    finally:
        LOGGER.info("Shutting down.")
        rclpy.shutdown()
        executor_thread.join()


if __name__ == "__main__":
    main()
