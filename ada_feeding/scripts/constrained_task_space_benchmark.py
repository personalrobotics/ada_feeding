#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script runs a benchmark to evaluate a "constraint-driven planning"
methodology for the Articutool system.

It can be run in two modes:
1.  Position-Only Goals (default): Plans to a random (x, y, z) position,
    leaving the final orientation unconstrained.
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
PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (
    1.5,
    3.14,
    0.8,
)  # (Pitch, Yaw, Roll) - Roll relaxed to 0.8

# --- Parameters for the optional GOAL Yaw Orientation Constraint ---
GOAL_YAW_CONSTRAINT_TOLERANCE_XYZ_RAD = (
    math.pi,
    math.pi,
    0.1,
)  # Loose Pitch/Roll, Tight Yaw

# --- Define a reasonable workspace for sampling random positions ---
WORKSPACE_OUTER_RADIUS = 0.7  # Max reach in meters
WORKSPACE_INNER_RADIUS = 0.3  # Min distance to avoid singularity at base

# --- Data Structures for Clarity ---
ArticutoolWaypointSolution = namedtuple(
    "ArticutoolWaypointSolution",
    ["waypoint_feasible", "pitch_solution_rad", "roll_solution_rad"],
)
ArticutoolMetrics = namedtuple(
    "ArticutoolMetrics",
    [
        "path_feasible",
        "num_infeasible_points",
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
        trajectory_save_dir: Optional[str] = None,
        constrain_goal_yaw: bool = False,
    ):
        self.node = node
        self.moveit2 = moveit2
        self.xacro_file_path = xacro_file_path
        self.num_tasks = num_tasks
        self.planning_timeout = planning_timeout
        self.planner_id = planner_id
        self.trajectory_save_dir = trajectory_save_dir
        self.constrain_goal_yaw = constrain_goal_yaw
        self.results: List[Dict[str, Any]] = []

        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self.joint_name_to_pinocchio_id: Dict[str, int] = {}
        self._initialize_pinocchio()

        LOGGER.info("Benchmark simulator initialized.")
        if self.constrain_goal_yaw:
            LOGGER.info("Planning Mode: Position + Yaw Goals.")
        else:
            LOGGER.info("Planning Mode: Position-Only Goals.")
        LOGGER.info(f"Applying 'smart' orientation constraint to all paths.")

        if self.trajectory_save_dir:
            os.makedirs(self.trajectory_save_dir, exist_ok=True)
            LOGGER.info(f"Trajectories will be saved to: {self.trajectory_save_dir}")

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
                    self.joint_name_to_pinocchio_id[name] = (
                        self.pinocchio_model.getJointId(name)
                    )
            LOGGER.info("Pinocchio model loaded successfully.")
        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            LOGGER.error(f"Failed to initialize Pinocchio model: {e}")
            self.pinocchio_model = None

    def generate_random_position_in_workspace(self) -> np.ndarray:
        """Generates a random (x, y, z) position from a half-spherical shell."""
        r_outer, r_inner = WORKSPACE_OUTER_RADIUS, WORKSPACE_INNER_RADIUS
        r = (np.random.uniform(r_inner**3, r_outer**3)) ** (1 / 3)
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
        # Scipy handles conversion from Euler angles (Z-axis for yaw) to quaternion
        quat_xyzw = R.from_euler("z", random_yaw_angle).as_quat()
        return tuple(quat_xyzw)

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

    def _verify_trajectory_feasibility(
        self, trajectory: JointTrajectory
    ) -> Tuple[ArticutoolMetrics, List[ArticutoolWaypointSolution]]:
        """Verifies if a given trajectory is feasible for the Articutool."""
        if self.pinocchio_model is None:
            return (ArticutoolMetrics(False, len(trajectory.points)), [])

        num_infeasible_wps = 0
        per_waypoint_solutions: List[ArticutoolWaypointSolution] = []
        q = pin.neutral(self.pinocchio_model)

        for point in trajectory.points:
            for i, name in enumerate(JOINT_NAMES):
                if name in self.joint_name_to_pinocchio_id:
                    joint_id = self.joint_name_to_pinocchio_id[name]
                    joint_obj = self.pinocchio_model.joints[joint_id]
                    theta = point.positions[i]
                    if joint_obj.nq == 2 and joint_obj.nv == 1:
                        q[joint_obj.idx_q] = math.cos(theta)
                        q[joint_obj.idx_q + 1] = math.sin(theta)
                    else:
                        q[joint_obj.idx_q] = theta

            pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, q)
            pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)
            ee_transform: pin.SE3 = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]
            R_world_ee = R.from_matrix(ee_transform.rotation)
            target_up_in_ee_frame = R_world_ee.inv().apply(WORLD_UP_VECTOR)
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
                per_waypoint_solutions.append(
                    ArticutoolWaypointSolution(True, best_sol[0], best_sol[1])
                )

        return (
            ArticutoolMetrics(
                path_feasible=(num_infeasible_wps == 0),
                num_infeasible_points=num_infeasible_wps,
            ),
            per_waypoint_solutions,
        )

    def _save_trajectory_to_file(
        self,
        original_trajectory: JointTrajectory,
        articutool_solutions: List[ArticutoolWaypointSolution],
        task_id: int,
    ) -> Optional[str]:
        """Saves the trajectory data to a JSON file."""
        if not self.trajectory_save_dir:
            return None
        try:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            mode = "pos_yaw" if self.constrain_goal_yaw else "pos_only"
            filename = f"task_{mode}_{task_id}_{timestamp}.json"
            filepath = os.path.join(self.trajectory_save_dir, filename)
            waypoints_data = [
                {
                    "time_from_start_sec": p.time_from_start.sec
                    + p.time_from_start.nanosec * 1e-9,
                    "jaco_positions_rad": list(p.positions),
                    "articutool_waypoint_feasible": s.waypoint_feasible,
                    "articutool_pitch_solution_rad": s.pitch_solution_rad,
                    "articutool_roll_solution_rad": s.roll_solution_rad,
                }
                for p, s in zip(original_trajectory.points, articutool_solutions)
            ]
            enhanced_data = {
                "jaco_joint_names": list(original_trajectory.joint_names),
                "waypoints": waypoints_data,
            }
            with open(filepath, "w") as f:
                json.dump(enhanced_data, f, indent=2)
            return filename
        except Exception as e:
            LOGGER.error(f"Failed to save trajectory for task {task_id}: {e}")
            return None

    def run(self):
        """Main benchmark execution loop."""
        if not self.pinocchio_model:
            LOGGER.error("Aborting benchmark: Pinocchio model failed to initialize.")
            return

        for i in range(self.num_tasks):
            LOGGER.info(f"--- Running Task {i + 1}/{self.num_tasks} ---")

            # 1. Generate a random target position and optional yaw
            target_position = self.generate_random_position_in_workspace()
            target_quat_xyzw = None
            if self.constrain_goal_yaw:
                target_quat_xyzw = self.generate_random_yaw_quaternion()
                LOGGER.info(
                    f"  Target Position (x,y,z): {np.round(target_position, 3).tolist()}"
                )
                LOGGER.info(
                    f"  Target Goal Yaw Quat (x,y,z,w): {np.round(target_quat_xyzw, 3).tolist()}"
                )
            else:
                LOGGER.info(
                    f"  Target Position (x,y,z): {np.round(target_position, 3).tolist()}"
                )

            # 2. Set up the planning request
            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()

            # Goal Constraint(s)
            self.moveit2.set_position_goal(
                position=target_position.tolist(),
                frame_id=BASE_LINK,
                target_link=END_EFFECTOR_LINK,
                tolerance=0.01,
            )
            if self.constrain_goal_yaw and target_quat_xyzw is not None:
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

            # Path is always governed by our "smart" OrientationConstraint
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

            # 3. Plan the motion
            planning_start_time = time.perf_counter()
            plan_future = self.moveit2.plan_async()
            while rclpy.ok() and not plan_future.done():
                time.sleep(0.01)
            trajectory = self.moveit2.get_trajectory(plan_future)
            planning_time = time.perf_counter() - planning_start_time

            plan_success = trajectory is not None and bool(trajectory.points)
            articutool_metrics = None
            if plan_success:
                LOGGER.info(f"  Jaco plan SUCCEEDED in {planning_time:.4f}s.")
                articutool_metrics, solutions = self._verify_trajectory_feasibility(
                    trajectory
                )
                LOGGER.info(
                    f"  Verification Result: Path Feasible = {articutool_metrics.path_feasible} ({articutool_metrics.num_infeasible_points} infeasible points)"
                )
                self._save_trajectory_to_file(trajectory, solutions, i)
            else:
                LOGGER.warn(f"  Jaco plan FAILED in {planning_time:.4f}s.")

            self.results.append(
                {
                    "task_id": i,
                    "target_position": target_position.tolist(),
                    "target_yaw_quat": target_quat_xyzw,
                    "plan_success": plan_success,
                    "planning_time_s": planning_time,
                    **(articutool_metrics._asdict() if articutool_metrics else {}),
                }
            )

    def save_results(self, output_dir: str):
        """Saves the benchmark results to a JSON file and prints a summary."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        mode = "pos_yaw" if self.constrain_goal_yaw else "pos_only"
        filename = os.path.join(
            output_dir, f"benchmark_results_{mode}_{timestamp}.json"
        )
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        LOGGER.info(f"Benchmark results saved to {filename}")

        total_tasks = len(self.results)
        jaco_successes = sum(1 for r in self.results if r["plan_success"])
        verified_successes = sum(
            1 for r in self.results if r.get("path_feasible", False)
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
            verification_rate = verified_successes / jaco_successes * 100
            LOGGER.info(
                f"Verification Success Rate (of successful plans): {verification_rate:.2f}%"
            )
        LOGGER.info("-------------------------------------\n")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python3 constrained_task_space_benchmark.py /path/to/robot.urdf.xacro [--constrain-goal-yaw]"
        )
        sys.exit(1)
    xacro_file_arg = sys.argv[1]
    if not os.path.exists(xacro_file_arg):
        print(f"Error: XACRO file not found at '{xacro_file_arg}'")
        sys.exit(1)

    constrain_goal_yaw_arg = "--constrain-goal-yaw" in sys.argv

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

    NUM_TASKS = 100
    PLANNING_TIMEOUT = 10.0  # Increased slightly for more complex constrained planning
    PLANNER_ID = "RRTConnectkConfigDefault"
    OUTPUT_DIR = os.path.join(os.getcwd(), "constrained_task_space_benchmark_output")

    benchmark = ConstrainedTaskSpaceBenchmark(
        node=node,
        moveit2=moveit2,
        xacro_file_path=xacro_file_arg,
        num_tasks=NUM_TASKS,
        planning_timeout=PLANNING_TIMEOUT,
        planner_id=PLANNER_ID,
        trajectory_save_dir=OUTPUT_DIR,
        constrain_goal_yaw=constrain_goal_yaw_arg,
    )

    try:
        if benchmark.pinocchio_model is not None:
            benchmark.run()
            benchmark.save_results(OUTPUT_DIR)
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
