#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This script implements the End-to-End Assistive Feeding Benchmark.

It evaluates a set of generalizable motion primitives across a complete,
simulated assistive feeding cycle. The benchmark operates within a variety of
procedurally generated, robot-centric planning scenes to test the
generalization and robustness of different control strategies.
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
from enum import Enum

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# --- Constants ---
LOGGER = rclpy.logging.get_logger("end_to_end_benchmark")
PLANNING_GROUP_JACO = "jaco_arm"
PLANNING_GROUP_ATOOL = "articutool"
PLANNING_GROUP_FULL = "jaco_arm_with_articutool"
JOINT_NAMES_JACO = [f"j2n6s200_joint_{i + 1}" for i in range(6)]
JOINT_NAMES_ATOOL = [f"atool_joint_{i + 1}" for i in range(2)]
JOINT_NAMES_FULL = JOINT_NAMES_JACO + JOINT_NAMES_ATOOL
BASE_LINK_JACO = "j2n6s200_link_base"
BASE_LINK_ATOOL = "atool_link_base"
BASE_LINK_FULL = BASE_LINK_JACO
END_EFFECTOR_LINK_JACO = "j2n6s200_end_effector"
END_EFFECTOR_LINK_ATOOL = "tool_tip"
END_EFFECTOR_LINK_FULL = END_EFFECTOR_LINK_ATOOL
EPSILON = 1e-6
ARTICUTOOL_PITCH_LIMITS_RAD = (-math.pi / 2, math.pi / 2)
ARTICUTOOL_ROLL_LIMITS_RAD = (-math.pi, math.pi)
WORLD_UP_VECTOR = np.array([0.0, 0.0, 1.0])
PATH_CONSTRAINT_QUAT_XYZW = (0.707, 0.0, 0.0, 0.707)
PATH_CONSTRAINT_TOLERANCE_XYZ_RAD = (1.5, 3.14, 0.8)


# --- Data Structures ---
class TrialStatus(Enum):
    SUCCESS = "Success"
    IK_FAILURE = "IK Failure"
    PLANNER_FAILURE = "Planner Failure"
    VERIFICATION_FAILURE = "Path Verification Failure"
    SKIPPED = "Skipped"


class EndToEndBenchmark:
    """Manages the end-to-end benchmark planning process."""

    def __init__(
        self,
        node: Node,
        moveit2_jaco: MoveIt2,
        moveit2_atool: MoveIt2,
        moveit2_full: MoveIt2,
        xacro_file_path: str,
        num_trials: int = 100,
        planning_timeout: float = 5.0,
        output_dir: Optional[str] = None,
    ):
        self.node = node
        self.moveit2_jaco = moveit2_jaco
        self.moveit2_atool = moveit2_atool
        self.moveit2_full = moveit2_full
        self.xacro_file_path = xacro_file_path
        self.num_trials = num_trials
        self.planning_timeout = planning_timeout
        self.output_dir = output_dir
        self.results: List[Dict[str, Any]] = []

        # Pinocchio model for feasibility checks
        self.pinocchio_model: Optional[pin.Model] = None
        self.pinocchio_data: Optional[pin.Data] = None
        self.jaco_ee_frame_id_pin: Optional[int] = None
        self._initialize_pinocchio()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

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
                END_EFFECTOR_LINK_FULL
            )
            LOGGER.info("Pinocchio model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Pinocchio model: {e}", exc_info=True)
            self.pinocchio_model = None

    # --- Scene Generation ---
    def _generate_scene(self, base_z_offset: float) -> Dict[str, Any]:
        """Generates a randomized, robot-centric planning scene."""
        scene = {"base_z_offset": base_z_offset}

        # Sample food and mouth poses
        scene["food_pose"] = self._sample_pose_in_spherical_shell()
        scene["mouth_pose"] = self._sample_pose_in_spherical_shell()

        # Define derived poses
        scene["home_config"] = [-1.47568, 2.92779, 1.00845, -2.0847, 1.43588, 1.32575]
        scene["above_plate_pose"] = self._calculate_above_plate_pose(scene["food_pose"])
        scene["move_above_pose"] = self._calculate_move_above_pose(scene["food_pose"])
        scene["move_into_pose"] = self._calculate_move_into_pose(
            scene["food_pose"], scene["move_above_pose"]
        )
        scene["staging_pose"] = self._calculate_staging_pose(scene["mouth_pose"])
        scene["resting_pose"] = Pose(position=Point(x=0.4, y=-0.4, z=0.3))

        return scene

    def _sample_pose_in_spherical_shell(self) -> Pose:
        """Samples a random pose within a spherical shell in front of the robot."""
        inner_radius, outer_radius = 0.3, 0.7

        # Sample position
        r = np.random.uniform(inner_radius**3, outer_radius**3) ** (1 / 3)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi / 2)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        position = Point(x=x, y=y, z=z)

        # Enforce "look-at-robot" constraint for orientation
        direction_vector = -np.array([x, y, z])
        direction_vector /= np.linalg.norm(direction_vector)

        # Create rotation that aligns a frame's X-axis with this vector
        # With some variability
        rand_axis = np.random.randn(3)
        rand_axis /= np.linalg.norm(rand_axis)
        rand_angle = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))
        variability_rot = R.from_rotvec(rand_angle * rand_axis)

        # Main rotation to look at origin
        up_vector = np.array([0, 0, 1])
        x_axis = direction_vector
        y_axis = np.cross(up_vector, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        main_rot = R.from_matrix(rotation_matrix)

        final_rot = main_rot * variability_rot
        quat = final_rot.as_quat()

        return Pose(
            position=position,
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )

    def _calculate_above_plate_pose(self, food_pose: Pose) -> Pose:
        """Calculate a camera pose that looks down at the food."""
        # For now, a simplified version. A real implementation would be more complex.
        p = food_pose.position
        pose = Pose()
        pose.position = Point(x=p.x, y=p.y, z=p.z + 0.3)
        # Orientation looking down with some variability
        r = R.from_euler("y", np.deg2rad(-90 + np.random.uniform(-15, 15)))
        q = r.as_quat()
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    def _calculate_move_above_pose(self, food_pose: Pose) -> Pose:
        """Calculate the pre-acquisition pose based on ADA action schema."""
        # Parameterize with approach vector (polar and azimuthal angles)
        polar_angle = np.random.uniform(np.deg2rad(30), np.deg2rad(60))
        azimuthal_angle = np.random.uniform(-np.deg2rad(45), np.deg2rad(45))
        offset_dist = 0.1

        # Calculate offset in a frame aligned with the food pose
        x_offset = offset_dist * np.sin(polar_angle) * np.cos(azimuthal_angle)
        y_offset = offset_dist * np.sin(polar_angle) * np.sin(azimuthal_angle)
        z_offset = offset_dist * np.cos(polar_angle)

        # Create rotation from angles
        rot = R.from_euler("zy", [-azimuthal_angle, -polar_angle])

        # Apply this transform to the food pose
        food_rot = R.from_quat(
            [
                food_pose.orientation.x,
                food_pose.orientation.y,
                food_pose.orientation.z,
                food_pose.orientation.w,
            ]
        )
        final_rot = food_rot * rot
        q = final_rot.as_quat()

        p = food_pose.position
        final_pos = Point(x=p.x - x_offset, y=p.y - y_offset, z=p.z + z_offset)

        return Pose(
            position=final_pos, orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        )

    def _calculate_move_into_pose(self, food_pose: Pose, move_above_pose: Pose) -> Pose:
        """The MoveInto pose has the same orientation as MoveAbove."""
        return Pose(
            position=food_pose.position, orientation=move_above_pose.orientation
        )

    def _calculate_staging_pose(self, mouth_pose: Pose) -> Pose:
        """Calculate the staging pose relative to the mouth."""
        offset_dist = 0.15

        p = mouth_pose.position
        q = mouth_pose.orientation
        mouth_rot = R.from_quat([q.x, q.y, q.z, q.w])

        # Offset is along the mouth's forward-facing X-axis
        offset_vec = mouth_rot.apply([offset_dist, 0, 0])

        staged_pos = Point(
            x=p.x - offset_vec[0], y=p.y - offset_vec[1], z=p.z - offset_vec[2]
        )

        # Here we would add the complex dual-orientation constraint logic
        # For now, we use the same orientation as the mouth
        return Pose(position=staged_pos, orientation=q)

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

    # --- Feasibility Checking ---
    def _is_config_kinematically_feasible(self, jaco_joint_config: List[float]) -> bool:
        """
        Checks if the Articutool can maintain leveling at a single Jaco configuration.
        This is our "oracle" for testing membership in the true feasibility manifold.
        """
        if (
            self.pinocchio_model is None
            or self.pinocchio_data is None
            or self.jaco_ee_frame_id_pin is None
        ):
            return False

        q = pin.neutral(self.pinocchio_model)
        for j, name in enumerate(JOINT_NAMES_FULL):
            if self.pinocchio_model.existJointName(name):
                joint_id = self.pinocchio_model.getJointId(name)
                joint_obj = self.pinocchio_model.joints[joint_id]
                if joint_obj.nq == 2 and not joint_obj.shortname().startswith(
                    "JointModelRX"
                ):
                    q[joint_obj.idx_q : joint_obj.idx_q + 2] = [
                        math.cos(jaco_joint_config[j]),
                        math.sin(jaco_joint_config[j]),
                    ]
                else:
                    q[joint_obj.idx_q] = jaco_joint_config[j]

        pin.forwardKinematics(self.pinocchio_model, self.pinocchio_data, q)
        pin.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)

        ee_transform = self.pinocchio_data.oMf[self.jaco_ee_frame_id_pin]
        target_up_in_ee_frame = (
            R.from_matrix(ee_transform.rotation).inv().apply(WORLD_UP_VECTOR)
        )
        solutions = self._solve_articutool_ik(target_up_in_ee_frame)

        if not solutions:
            return False

        return any(
            ARTICUTOOL_PITCH_LIMITS_RAD[0] <= pitch <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
            and ARTICUTOOL_ROLL_LIMITS_RAD[0] <= roll <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
            for pitch, roll in solutions
        )

    def _verify_trajectory(self, trajectory: JointTrajectory) -> float:
        """Verifies a trajectory and returns the percentage of feasible waypoints."""
        if not trajectory or not trajectory.points:
            return 0.0

        feasible_waypoints = 0
        for point in trajectory.points:
            if self._is_config_kinematically_feasible(point.positions):
                feasible_waypoints += 1

        return (feasible_waypoints / len(trajectory.points)) * 100.0

    # --- Planning Primitive Placeholders ---
    def _plan_s1_unconstrained(
        self, goal_pose: Pose, start_state: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory]]:
        LOGGER.info("  Planning with S1 (6-DOF Unconstrained)...")
        self.moveit2_jaco.clear_goal_constraints()
        self.moveit2_jaco.clear_path_constraints()

        self.moveit2_jaco.set_pose_goal(goal_pose, END_EFFECTOR_LINK_JACO)

        future = self.moveit2_jaco.plan_async(start_joint_state=start_state)
        rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=self.planning_timeout
        )
        traj = self.moveit2_jaco.get_trajectory(future)

        if not traj or not traj.points:
            return TrialStatus.PLANNER_FAILURE, None

        return TrialStatus.SUCCESS, traj

    def _plan_s2_guided(
        self, goal_pose: Pose, start_state: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        LOGGER.info("  Planning with S2 (6-DOF Guided)...")
        self.moveit2_jaco.clear_goal_constraints()
        self.moveit2_jaco.clear_path_constraints()

        self.moveit2_jaco.set_pose_goal(goal_pose, END_EFFECTOR_LINK_JACO)
        self.moveit2_jaco.set_path_orientation_constraint(
            quat_xyzw=Quaternion(
                x=PATH_CONSTRAINT_QUAT_XYZW[0],
                y=PATH_CONSTRAINT_QUAT_XYZW[1],
                z=PATH_CONSTRAINT_QUAT_XYZW[2],
                w=PATH_CONSTRAINT_QUAT_XYZW[3],
            ),
            target_link=END_EFFECTOR_LINK_JACO,
            tolerance=PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
            weight=1.0,
        )

        future = self.moveit2_jaco.plan_async(
            start_joint_state=start_state,
        )
        rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=self.planning_timeout
        )
        traj = self.moveit2_jaco.get_trajectory(future)

        if not traj or not traj.points:
            return TrialStatus.PLANNER_FAILURE, None, 0.0

        feasibility_percent = self._verify_trajectory(traj)
        if feasibility_percent < 99.0:
            return TrialStatus.VERIFICATION_FAILURE, traj, feasibility_percent

        return TrialStatus.SUCCESS, traj, feasibility_percent

    def _plan_s3_coordinated(
        self, goal_pose: Pose, start_state: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory]]:
        LOGGER.info("  Planning with S3 (8-DOF Coordinated)...")
        # TODO: Implement 8-DOF IK solve and sequential planning
        return TrialStatus.SKIPPED, None

    def _plan_s4_cartesian(
        self, goal_pose: Pose, start_state: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory]]:
        LOGGER.info("  Planning with S4 (6-DOF Cartesian)...")
        # TODO: Implement MoveIt2 call for Cartesian planning
        return TrialStatus.SKIPPED, None

    # --- Main Benchmark Loop ---
    def run(self):
        """Main benchmark execution loop."""
        if not self.pinocchio_model:
            LOGGER.error("Benchmark cannot run because Pinocchio model failed to load.")
            return

        for i in range(self.num_trials):
            LOGGER.info(f"--- Running Trial {i + 1}/{self.num_trials} ---")

            # 1. Generate a new scene
            scene = self._generate_scene(base_z_offset=0.1)
            LOGGER.info(
                f"""
                --- Generated Scene Parameters for Trial {i + 1} ---
                - Robot Base Z Offset: {scene["base_z_offset"]:.3f}m

                - Initial State:
                  - Home Config: [{", ".join(f"{j:.4f}" for j in scene["home_config"])}]

                - Core Sampled Poses:
                  - Food Pose:
                      Position:    [x={scene["food_pose"].position.x:.3f}, y={scene["food_pose"].position.y:.3f}, z={scene["food_pose"].position.z:.3f}]
                      Orientation: [x={scene["food_pose"].orientation.x:.3f}, y={scene["food_pose"].orientation.y:.3f}, z={scene["food_pose"].orientation.z:.3f}, w={scene["food_pose"].orientation.w:.3f}]
                  - Mouth Pose:
                      Position:    [x={scene["mouth_pose"].position.x:.3f}, y={scene["mouth_pose"].position.y:.3f}, z={scene["mouth_pose"].position.z:.3f}]
                      Orientation: [x={scene["mouth_pose"].orientation.x:.3f}, y={scene["mouth_pose"].orientation.y:.3f}, z={scene["mouth_pose"].orientation.z:.3f}, w={scene["mouth_pose"].orientation.w:.3f}]

                - Derived Poses for Feeding Cycle:
                  - Above Plate Pose:
                      Position:    [x={scene["above_plate_pose"].position.x:.3f}, y={scene["above_plate_pose"].position.y:.3f}, z={scene["above_plate_pose"].position.z:.3f}]
                      Orientation: [x={scene["above_plate_pose"].orientation.x:.3f}, y={scene["above_plate_pose"].orientation.y:.3f}, z={scene["above_plate_pose"].orientation.z:.3f}, w={scene["above_plate_pose"].orientation.w:.3f}]
                  - Move Above Pose:
                      Position:    [x={scene["move_above_pose"].position.x:.3f}, y={scene["move_above_pose"].position.y:.3f}, z={scene["move_above_pose"].position.z:.3f}]
                      Orientation: [x={scene["move_above_pose"].orientation.x:.3f}, y={scene["move_above_pose"].orientation.y:.3f}, z={scene["move_above_pose"].orientation.z:.3f}, w={scene["move_above_pose"].orientation.w:.3f}]
                  - Move Into Pose:
                      Position:    [x={scene["move_into_pose"].position.x:.3f}, y={scene["move_into_pose"].position.y:.3f}, z={scene["move_into_pose"].position.z:.3f}]
                      Orientation: [x={scene["move_into_pose"].orientation.x:.3f}, y={scene["move_into_pose"].orientation.y:.3f}, z={scene["move_into_pose"].orientation.z:.3f}, w={scene["move_into_pose"].orientation.w:.3f}]
                  - Staging Pose:
                      Position:    [x={scene["staging_pose"].position.x:.3f}, y={scene["staging_pose"].position.y:.3f}, z={scene["staging_pose"].position.z:.3f}]
                      Orientation: [x={scene["staging_pose"].orientation.x:.3f}, y={scene["staging_pose"].orientation.y:.3f}, z={scene["staging_pose"].orientation.z:.3f}, w={scene["staging_pose"].orientation.w:.3f}]
                  - Resting Pose:
                      Position:    [x={scene["resting_pose"].position.x:.3f}, y={scene["resting_pose"].position.y:.3f}, z={scene["resting_pose"].position.z:.3f}]
                      Orientation: [x={scene["resting_pose"].orientation.x:.3f}, y={scene["resting_pose"].orientation.y:.3f}, z={scene["resting_pose"].orientation.z:.3f}, w={scene["resting_pose"].orientation.w:.3f}]
                -------------------------------------------------
            """
            )

            # 2. Simulate the feeding cycle state machine
            food_on_tool = False
            current_jaco_state = scene["home_config"]

            # Stage 1: Home -> AbovePlate (P1)
            LOGGER.info("Stage 1: Home -> AbovePlate")
            above_plate_config = self.moveit2_jaco.compute_ik(
                position=scene["move_above_pose"].position,
                quat_xyzw=scene["move_above_pose"].orientation,
                start_joint_state=current_jaco_state,
            )
            ik_status = (
                TrialStatus.SUCCESS if above_plate_config else TrialStatus.IK_FAILURE
            )
            self.results.append(
                {
                    "trial_id": i,
                    "stage": "HomeToAbovePlate",
                    "primitive": "IK",
                    "status": ik_status.value,
                }
            )
            if ik_status != TrialStatus.SUCCESS:
                continue
            current_jaco_state = list(above_plate_config.position)

            # Stage 2: AbovePlate -> MoveAbove (P2, P4)
            LOGGER.info("Stage 2: AbovePlate -> MoveAbove")
            status_s1, traj_s1 = self._plan_s1_unconstrained(
                scene["move_above_pose"], current_jaco_state
            )
            self.results.append(
                {
                    "trial_id": i,
                    "stage": "AbovePlateToMoveAbove",
                    "primitive": "S2",
                    "status": status_s1.value,
                }
            )
            if status_s1 != TrialStatus.SUCCESS:
                continue  # End trial on failure
            current_jaco_state = list(traj_s1.points[-1].positions)

            # Stage 3: MoveToStaging
            LOGGER.info("Stage 3: MoveToStaging (Food on tool!)")
            food_on_tool = True
            status_s2, traj_s2, feasibility_percent = self._plan_s2_guided(
                scene["staging_pose"], current_jaco_state
            )

            self.results.append(
                {
                    "trial_id": i,
                    "stage": "AcquiredToStaging",
                    "primitive": "S2",
                    "status": status_s2.value,
                    "leveling_feasibility": feasibility_percent,
                }
            )
            if status_s2 != TrialStatus.SUCCESS:
                continue
            current_jaco_state = list(traj_s2.points[-1].positions)

        self.save_results()

    def save_results(self):
        """Saves the comprehensive benchmark results to a single JSON file."""
        if not self.output_dir:
            return
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(
            self.output_dir, f"end_to_end_benchmark_{timestamp}.json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            LOGGER.info(f"Benchmark results saved to {filename}")
        except Exception as e:
            LOGGER.error(f"Failed to save results: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the end-to-end feeding benchmark."
    )
    parser.add_argument(
        "--xacro_file",
        type=str,
        default="/home/regulus/ada_ws/src/ada_ros2/ada_moveit/config/ada.urdf.xacro",
        help="Path to the robot URDF/XACRO file.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of full feeding trials to run.",
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Planning timeout in seconds."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="e2e_benchmark_output",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("end_to_end_benchmark_node")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Initialize MoveIt2 for the Jaco arm, Articutool, and Full
    moveit2_jaco = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES_JACO,
        base_link_name=BASE_LINK_JACO,
        end_effector_name=END_EFFECTOR_LINK_JACO,
        group_name=PLANNING_GROUP_JACO,
        callback_group=ReentrantCallbackGroup(),
    )
    moveit2_atool = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES_ATOOL,
        base_link_name=BASE_LINK_ATOOL,
        end_effector_name=END_EFFECTOR_LINK_ATOOL,
        group_name=PLANNING_GROUP_ATOOL,
        callback_group=ReentrantCallbackGroup(),
    )
    moveit2_full = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES_FULL,
        base_link_name=BASE_LINK_FULL,
        end_effector_name=END_EFFECTOR_LINK_FULL,
        group_name=PLANNING_GROUP_FULL,
        callback_group=ReentrantCallbackGroup(),
    )

    benchmark = EndToEndBenchmark(
        node,
        moveit2_jaco,
        moveit2_atool,
        moveit2_full,
        args.xacro_file,
        args.num_trials,
        args.timeout,
        args.output_dir,
    )

    try:
        benchmark.run()
    except KeyboardInterrupt:
        LOGGER.info("Benchmark interrupted by user.")
    except Exception as e:
        LOGGER.error(f"An unhandled error occurred: {e}")
    finally:
        LOGGER.info("Shutting down.")
        rclpy.shutdown()
        executor_thread.join()


if __name__ == "__main__":
    main()
