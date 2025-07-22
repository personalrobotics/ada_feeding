#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script provides a holistic benchmark to compare multiple planning
methodologies for the Jaco + Articutool system. Its goal is to generate a
comprehensive dataset to empirically evaluate the trade-offs between planner
success, motion flexibility, and guaranteed kinematic/dynamic feasibility.

This version uses a collision-checking service to find a valid, random start
state for each trial, ensuring a robust and diverse evaluation.

The script can be run in one of four modes via the `--mode` flag:
1.  `joint_unconstrained`: The baseline. Plans between random joint-space goals
    with no path constraints.
2.  `task_pos_unconstrained`: Plans to a random task-space position goal with
    no path constraints.
3.  `task_pos_constrained`: Plans to a random task-space position goal WITH the
    "smart" orientation path constraint.
4.  `task_pos_yaw_constrained`: Plans to a random task-space position AND yaw
    goal, WITH the "smart" orientation path constraint.
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
import argparse

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Quaternion
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# --- Constants ---
LOGGER = rclpy.logging.get_logger("holistic_benchmark")
PLANNING_GROUP = "jaco_arm"
JOINT_NAMES = kinova.joint_names()
BASE_LINK = kinova.base_link_name()
END_EFFECTOR_LINK = kinova.end_effector_name()
EPSILON = 1e-6
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])
PATH_CONSTRAINT_QUAT_XYZW = (0.707, 0.0, 0.0, 0.707)
PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (1.5, 3.14, 0.8)
GOAL_YAW_CONSTRAINT_TOLERANCE_XYZ_RAD = (math.pi, math.pi, 0.1)
WORKSPACE_OUTER_RADIUS = 0.7
WORKSPACE_INNER_RADIUS = 0.3

# --- Data Structures ---
TrajectoryMetrics = namedtuple(
    "TrajectoryMetrics",
    [
        "duration_s",
        "joint_space_path_length_rad",
        "final_joint_positions",
        "waypoints_data",
    ],
)


class HolisticBenchmark:
    """Manages the holistic benchmark planning process."""

    def __init__(
        self,
        node: Node,
        moveit2: MoveIt2,
        xacro_file_path: str,
        mode: str,
        num_tasks: int = 100,
        planning_timeout: float = 10.0,
        planner_id: str = "RRTConnectkConfigDefault",
        output_dir: Optional[str] = None,
    ):
        self.node = node
        self.moveit2 = moveit2
        self.xacro_file_path = xacro_file_path
        self.mode = mode
        self.num_tasks = num_tasks
        self.planning_timeout = planning_timeout
        self.planner_id = planner_id
        self.output_dir = output_dir
        self.results: List[Dict[str, Any]] = []

        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self.jaco_vel_indices: List[int] = []
        self._initialize_pinocchio()
        self.joint_limits = self._get_joint_limits()

        self.service_callback_group = ReentrantCallbackGroup()
        self.state_validity_client = self.node.create_client(
            GetStateValidity,
            "/check_state_validity",
            callback_group=self.service_callback_group,
        )
        if not self.state_validity_client.wait_for_service(timeout_sec=5.0):
            LOGGER.error(
                "Could not connect to /check_state_validity service. Cannot guarantee valid start states."
            )
            self.state_validity_client = None

        LOGGER.info("Holistic Benchmark Initialized.")
        LOGGER.info(f"  - Planning Mode: {self.mode}")
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            LOGGER.info(f"  - Results will be saved to: {self.output_dir}")

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
                    self.jaco_vel_indices.append(
                        self.pinocchio_model.joints[joint_id].idx_v
                    )
            LOGGER.info("Pinocchio model loaded successfully.")
        except Exception as e:
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

    def generate_random_joint_config(self) -> List[float]:
        """Generates a random joint configuration within limits."""
        return [np.random.uniform(low, high) for low, high in self.joint_limits]

    def is_state_valid(self, joint_positions: List[float]) -> bool:
        """Checks if a joint configuration is collision-free using MoveIt's service."""
        if (
            not self.state_validity_client
            or not self.state_validity_client.service_is_ready()
        ):
            LOGGER.warn("State validity client not available. Assuming state is valid.")
            return True

        # --- FIX: Build a complete RobotState to avoid ambiguity ---
        # Get the current full state of the robot
        full_joint_state = self.moveit2.joint_state
        joint_state_map = {
            name: pos
            for name, pos in zip(full_joint_state.name, full_joint_state.position)
        }

        # Overwrite the Jaco arm joints with the configuration we want to test
        for i, name in enumerate(JOINT_NAMES):
            joint_state_map[name] = joint_positions[i]

        # Build the RobotState message with all joints
        robot_state = RobotState()
        robot_state.joint_state.name = list(joint_state_map.keys())
        robot_state.joint_state.position = list(joint_state_map.values())

        req = GetStateValidity.Request()
        req.group_name = PLANNING_GROUP
        req.robot_state = robot_state

        future = self.state_validity_client.call_async(req)

        timeout_sec = 2.0
        start_time = self.node.get_clock().now()
        while (
            rclpy.ok()
            and (self.node.get_clock().now() - start_time).nanoseconds / 1e9
            < timeout_sec
        ):
            if future.done():
                try:
                    response = future.result()
                    return response.valid if response is not None else False
                except Exception as e:
                    LOGGER.error(
                        f"Exception getting result from /check_state_validity: {e}"
                    )
                    return False
            time.sleep(0.01)

        LOGGER.error(f"Service call to /check_state_validity timed out.")
        return False

    def generate_valid_random_joint_config(
        self, max_attempts=100
    ) -> Optional[List[float]]:
        """Generates a random, collision-free joint configuration."""
        for attempt in range(max_attempts):
            config = self.generate_random_joint_config()
            if self.is_state_valid(config):
                LOGGER.info(
                    f"  Found valid start configuration on attempt {attempt + 1}."
                )
                return config
        LOGGER.error(
            f"Failed to find a valid random joint configuration after {max_attempts} attempts."
        )
        return None

    def generate_random_position_in_workspace(self) -> np.ndarray:
        """Generates a random (x, y, z) position from a half-spherical shell."""
        r = (
            np.random.uniform(WORKSPACE_INNER_RADIUS**3, WORKSPACE_OUTER_RADIUS**3)
        ) ** (1 / 3)
        theta = np.random.uniform(0, 2 * math.pi)
        cos_phi = np.random.uniform(0, 1)
        phi = math.acos(cos_phi)
        return np.array(
            [
                r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * cos_phi,
            ]
        )

    def generate_random_yaw_quaternion(self) -> Tuple[float, float, float, float]:
        """Generates a quaternion representing a random yaw."""
        return tuple(R.from_euler("z", np.random.uniform(-math.pi, math.pi)).as_quat())

    def _get_articutool_jacobian(self, pitch: float, roll: float) -> np.ndarray:
        """Computes the analytical Jacobian for the Articutool."""
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        dydp = np.array([0, -sp * cr, cp * cr])
        dydr = np.array([-cr, -cp * sr, -sp * sr])
        return np.vstack([dydp, dydr]).T

    def _solve_articutool_ik(
        self, target_vector: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Analytical IK solver for the Articutool."""
        vx, vy, vz = target_vector
        solutions = []
        asin_arg = -vx
        if not (-1.0 - EPSILON <= asin_arg <= 1.0 + EPSILON):
            return []
        theta_r1 = math.asin(np.clip(asin_arg, -1.0, 1.0))
        theta_r2 = (math.pi - theta_r1 + math.pi) % (2 * math.pi) - math.pi
        for theta_r in list(set([theta_r1, theta_r2])):
            cos_tr = math.cos(theta_r)
            if math.isclose(cos_tr, 0.0, abs_tol=EPSILON):
                if math.isclose(vy, 0.0, abs_tol=EPSILON) and math.isclose(
                    vz, 0.0, abs_tol=EPSILON
                ):
                    solutions.append(
                        (0.0, (theta_r + math.pi) % (2 * math.pi) - math.pi)
                    )
                continue
            theta_p = math.atan2(vz, vy)
            solutions.append(
                (
                    (theta_p + math.pi) % (2 * math.pi) - math.pi,
                    (theta_r + math.pi) % (2 * math.pi) - math.pi,
                )
            )
        return solutions

    def _calculate_trajectory_metrics(
        self, trajectory: JointTrajectory
    ) -> TrajectoryMetrics:
        """Calculates detailed metrics for a given trajectory."""
        if self.pinocchio_model is None or not trajectory.points:
            return TrajectoryMetrics(0.0, 0.0, [], [])

        q = pin.neutral(self.pinocchio_model)
        prev_pos = np.array(trajectory.points[0].positions)
        waypoints_data, path_len = [], 0.0

        for i, point in enumerate(trajectory.points):
            for j, name in enumerate(JOINT_NAMES):
                joint_id = self.pinocchio_model.getJointId(name)
                joint_obj = self.pinocchio_model.joints[joint_id]
                if joint_obj.nq == 2:
                    q[joint_obj.idx_q : joint_obj.idx_q + 2] = [
                        math.cos(point.positions[j]),
                        math.sin(point.positions[j]),
                    ]
                else:
                    q[joint_obj.idx_q] = point.positions[j]

            pin.computeAllTerms(
                self.pinocchio_model,
                self.pinocchio_data,
                q,
                np.zeros(self.pinocchio_model.nv),
            )
            ee_transform = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]
            target_up = (
                R.from_matrix(ee_transform.rotation).inv().apply(WORLD_UP_VECTOR)
            )
            solutions = self._solve_articutool_ik(target_up)

            at_sol, at_vel = None, None
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
                    at_sol = {"pitch": pitch, "roll": roll}
                    if i > 0:
                        dt = (
                            point.time_from_start.sec
                            + 1e-9 * point.time_from_start.nanosec
                        ) - (
                            trajectory.points[i - 1].time_from_start.sec
                            + 1e-9 * trajectory.points[i - 1].time_from_start.nanosec
                        )
                        if dt > EPSILON:
                            q_dot = (
                                np.array(point.positions)
                                - np.array(trajectory.points[i - 1].positions)
                            ) / dt
                            J_full = pin.getFrameJacobian(
                                self.pinocchio_model,
                                self.pinocchio_data,
                                self.jaco_ee_frame_id_pin,
                                pin.ReferenceFrame.LOCAL,
                            )
                            J_arm = J_full[:, self.jaco_vel_indices]
                            omega = (J_arm @ q_dot)[3:]
                            J_atool_inv = np.linalg.pinv(
                                self._get_articutool_jacobian(pitch, roll)
                            )
                            q_dot_atool = -J_atool_inv @ omega
                            at_vel = {
                                "pitch_vel": q_dot_atool[0],
                                "roll_vel": q_dot_atool[1],
                            }

            waypoints_data.append(
                {
                    "jaco_positions_rad": list(point.positions),
                    "ee_pose_world": {
                        "position": ee_transform.translation.tolist(),
                        "quat_xyzw": R.from_matrix(ee_transform.rotation)
                        .as_quat()
                        .tolist(),
                    },
                    "articutool_solution_rad": at_sol,
                    "articutool_velocities_rad_per_sec": at_vel,
                }
            )
            path_len += np.linalg.norm(np.array(point.positions) - prev_pos)
            prev_pos = np.array(point.positions)

        duration = (
            trajectory.points[-1].time_from_start.sec
            + 1e-9 * trajectory.points[-1].time_from_start.nanosec
        )
        return TrajectoryMetrics(
            duration, path_len, list(trajectory.points[-1].positions), waypoints_data
        )

    def run(self):
        """Main benchmark execution loop."""
        if not self.pinocchio_model:
            return

        for i in range(self.num_tasks):
            LOGGER.info(f"--- Running Task {i + 1}/{self.num_tasks} ---")

            LOGGER.info("  Searching for a valid random start configuration...")
            start_pos = self.generate_valid_random_joint_config()
            if start_pos is None:
                LOGGER.warn("  Skipping task, could not find a valid start state.")
                continue

            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()

            target_pos, target_quat, goal_joint_pos = None, None, None
            if self.mode == "joint_unconstrained":
                goal_joint_pos = self.generate_valid_random_joint_config()
                if goal_joint_pos is None:
                    LOGGER.warn("  Skipping task, could not find a valid goal state.")
                    continue
                self.moveit2.set_joint_goal(joint_positions=goal_joint_pos)
            else:
                target_pos = self.generate_random_position_in_workspace()
                self.moveit2.set_position_goal(
                    position=target_pos.tolist(),
                    frame_id=BASE_LINK,
                    target_link=END_EFFECTOR_LINK,
                    tolerance=0.01,
                )
                if self.mode == "task_pos_yaw_constrained":
                    target_quat = self.generate_random_yaw_quaternion()
                    self.moveit2.set_orientation_goal(
                        quat_xyzw=Quaternion(
                            x=target_quat[0],
                            y=target_quat[1],
                            z=target_quat[2],
                            w=target_quat[3],
                        ),
                        target_link=END_EFFECTOR_LINK,
                        tolerance=GOAL_YAW_CONSTRAINT_TOLERANCE_XYZ_RAD,
                        parameterization=1,
                    )
                if "constrained" in self.mode:
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

            planning_start = time.perf_counter()
            future = self.moveit2.plan_async(start_joint_state=start_pos)
            while rclpy.ok() and not future.done():
                time.sleep(0.01)
            traj = self.moveit2.get_trajectory(future)
            planning_time = time.perf_counter() - planning_start

            success = traj is not None and bool(traj.points)
            trial_data = {
                "task_id": i,
                "planning_mode": self.mode,
                "start_joint_positions": list(start_pos),
                "plan_success": success,
                "planning_time_s": planning_time,
            }
            if goal_joint_pos:
                trial_data["target_joint_positions"] = goal_joint_pos
            if target_pos is not None:
                trial_data["target_position"] = target_pos.tolist()
            if target_quat:
                trial_data["target_yaw_quat"] = target_quat

            if success:
                LOGGER.info(f"  Plan SUCCEEDED in {planning_time:.4f}s.")
                metrics = self._calculate_trajectory_metrics(traj)
                is_feasible = all(
                    wp["articutool_solution_rad"] is not None
                    for wp in metrics.waypoints_data
                )
                trial_data["verification_success"] = is_feasible
                trial_data["trajectory_metrics"] = metrics._asdict()
                LOGGER.info(f"  Verification Result: Feasible = {is_feasible}")
            else:
                LOGGER.warn(f"  Plan FAILED in {planning_time:.4f}s.")

            self.results.append(trial_data)

    def save_results(self):
        """Saves the comprehensive benchmark results to a single JSON file."""
        if not self.output_dir:
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(
            self.output_dir, f"holistic_benchmark_results_{self.mode}_{timestamp}.json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            LOGGER.info(f"Comprehensive results saved to {filename}")
        except Exception as e:
            LOGGER.error(f"Failed to save results: {e}")

        total = len(self.results)
        successes = sum(1 for r in self.results if r["plan_success"])
        verified = sum(1 for r in self.results if r.get("verification_success", False))
        success_rate = (successes / total * 100) if total > 0 else 0
        LOGGER.info(f"\n--- HOLISTIC BENCHMARK SUMMARY ({self.mode.upper()}) ---")
        LOGGER.info(
            f"Total Tasks: {total}, Planner Success Rate: {success_rate:.2f}% ({successes}/{total})"
        )
        if successes > 0:
            LOGGER.info(
                f"Verification Success Rate (of successful plans): {(verified / successes * 100):.2f}%"
            )
        LOGGER.info("---------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Holistic benchmark for Articutool planning methodologies."
    )
    parser.add_argument(
        "xacro_file", type=str, help="Path to the robot URDF/XACRO file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "joint_unconstrained",
            "task_pos_unconstrained",
            "task_pos_constrained",
            "task_pos_yaw_constrained",
        ],
        help="The planning methodology to benchmark.",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=100, help="Number of random tasks to generate."
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Planning timeout in seconds."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="holistic_benchmark_output",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.xacro_file):
        print(f"Error: XACRO file not found at '{args.xacro_file}'")
        sys.exit(1)

    rclpy.init()
    node = Node("holistic_benchmark_node")
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

    benchmark = HolisticBenchmark(
        node,
        moveit2,
        args.xacro_file,
        args.mode,
        args.num_tasks,
        args.timeout,
        "RRTConnectkConfigDefault",
        args.output_dir,
    )

    try:
        if benchmark.pinocchio_model is not None:
            LOGGER.info(
                "Ready to begin benchmark. Each trial will start from a new random, collision-free state."
            )
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
