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
from enum import Enum, auto
from dataclasses import dataclass

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
from moveit_msgs.msg import PlanningScene, AllowedCollisionEntry, AllowedCollisionMatrix
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

# --- Constants ---
LOGGER = rclpy.logging.get_logger("end_to_end_benchmark")
PLANNING_GROUP_JACO = "jaco_arm"
PLANNING_GROUP_ATOOL = "articutool"
PLANNING_GROUP_FULL = "jaco_arm_with_articutool"
JOINT_NAMES_JACO = [f"j2n6s200_joint_{i + 1}" for i in range(6)]
JOINT_NAMES_ATOOL = [f"atool_joint{i + 1}" for i in range(2)]
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
ARTICUTOOL_LENGTH_M = 0.14


# --- Data Structures ---
class TrialStatus(Enum):
    SUCCESS = "Success"
    IK_FAILURE = "IK Failure"
    PLANNER_FAILURE = "Planner Failure"
    VERIFICATION_FAILURE = "Path Verification Failure"
    SKIPPED = "Skipped"


# --- Semantic Schema for Acquisition Actions ---
class AcquisitionStrategy(Enum):
    """High-level choice of physical interaction"""

    SKEWER = auto()
    SCOOP = auto()
    CUT = auto()


class MotionAxis(Enum):
    """The primary axis for the arm's linear motion"""

    MAJOR_AXIS = auto()
    MINOR_AXIS = auto()
    VERTICAL = auto()


class ToolAlignment(Enum):
    """How the tool is oriented relative to the food's structure"""

    PARALLEL = auto()
    PERPENDICULAR = auto()


@dataclass
class ActionRecipe:
    """Holds the semantic parameters for a single acquisition action"""

    strategy: AcquisitionStrategy
    motion_axis: MotionAxis
    tool_alignment: ToolAlignment

    def __str__(self):
        return (
            f"{self.strategy.name}-{self.motion_axis.name}-{self.tool_alignment.name}"
        )


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

    def add_articutool_bounding_cylinder(self):
        """
        Adds a cylindrical collision object to represent the Articutool
        and updates the ACM to prevent spurious collisions.
        """
        # Create a publisher to the planning scene topic if it doesn't exist
        if not hasattr(self, "planning_scene_publisher"):
            self.planning_scene_publisher = self.node.create_publisher(
                PlanningScene, "/planning_scene", 10
            )

        # --- Step 1: Define and add the CollisionObject ---
        collision_object = CollisionObject()
        collision_object.header.frame_id = (
            END_EFFECTOR_LINK_JACO  # Attach to Jaco wrist
        )
        collision_object.id = "articutool_bounding_cylinder"
        collision_object.operation = CollisionObject.ADD

        # Define the cylinder's dimensions and pose relative to the wrist link
        cylinder = SolidPrimitive()
        cylinder.type = SolidPrimitive.CYLINDER
        cylinder.dimensions = [
            0.04,  # height
            ARTICUTOOL_LENGTH_M,  # radius
        ]
        cylinder_pose = Pose()
        cylinder_pose.position = Point(x=0.0, y=0.0, z=0.132)
        cylinder_pose.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707)

        collision_object.primitives.append(cylinder)
        collision_object.primitive_poses.append(cylinder_pose)

        # Create a PlanningScene message to add the object
        planning_scene_msg = PlanningScene()
        planning_scene_msg.world.collision_objects.append(collision_object)
        planning_scene_msg.is_diff = True
        self.planning_scene_publisher.publish(planning_scene_msg)
        self.node.get_logger().info("Published bounding cylinder to planning scene.")

        # Short delay to ensure the scene is updated
        time.sleep(1.0)

        # --- Step 2: Modify the Allowed Collision Matrix (ACM) ---
        # Create a client to get the planning scene
        scene_client = self.node.create_client(GetPlanningScene, "/get_planning_scene")
        while not scene_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                'Service "/get_planning_scene" not available, waiting...'
            )

        # Request the full scene, including the ACM
        req = GetPlanningScene.Request()
        req.components.components = req.components.ALLOWED_COLLISION_MATRIX
        future = scene_client.call_async(req)

        # Add a callback to the future, which will be executed by the executor thread
        future.add_done_callback(self._update_acm_callback)

    def _update_acm_callback(self, future):
        """
        This callback is executed once the GetPlanningScene service returns a result.
        """
        try:
            result = future.result()
            if result is None:
                self.node.get_logger().error("Failed to get planning scene")
                return

            acm = result.scene.allowed_collision_matrix
            object_name = "articutool_bounding_cylinder"
            links_to_allow = [
                "j2n6s200_link_6",
                "atool_base",
                "atool_electronics_holder_bottom_plate",
                "atool_electronics_holder_upper_plate",
                "atool_electronics_holder_wire_guard",
                "atool_ft_adapter",
                "atool_handle",
                "atool_handle_cover",
                "atool_link1",
                "atool_link2",
                "atool_motor_link",
                "atool_u2d2",
                "tool",
                "ft",
                "j2n6s200_link_finger_1",
                "j2n6s200_link_finger_2",
                "j2n6s200_link_finger_tip_1",
                "j2n6s200_link_finger_tip_2",
            ]

            if object_name not in acm.entry_names:
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_names.append(object_name)
                new_row = AllowedCollisionEntry()
                new_row.enabled = [False] * len(acm.entry_names)
                acm.entry_values.append(new_row)

            obj_idx = acm.entry_names.index(object_name)
            link_indices = []
            for name in links_to_allow:
                try:
                    link_indices.append(acm.entry_names.index(name))
                except ValueError:
                    self.node.get_logger().warning(
                        f"Link '{name}' not in ACM, skipping."
                    )

            for link_idx in link_indices:
                acm.entry_values[obj_idx].enabled[link_idx] = True
                acm.entry_values[link_idx].enabled[obj_idx] = True

            scene_update = PlanningScene(is_diff=True)
            scene_update.allowed_collision_matrix = acm
            self.planning_scene_publisher.publish(scene_update)
            self.node.get_logger().info("Successfully updated ACM via callback.")

        except Exception as e:
            self.node.get_logger().error(f"Error in ACM callback: {e}")

    # --- Scene Generation ---
    def _generate_scene(self) -> Dict[str, Any]:
        """Generates a randomized, robot-centric planning scene."""
        scene = {}

        # Sample food and mouth poses
        scene["food_pose"] = self._sample_pose_in_cylindrical_shell(
            inner_radius=0.4,
            outer_radius=0.7,
            min_height=0.0,
            max_height=0.3,
        )
        scene["mouth_pose"] = self._sample_pose_in_cylindrical_shell(
            inner_radius=0.3,
            outer_radius=0.6,
            min_height=0.0,
            max_height=0.6,
        )

        # Define derived poses
        scene["home_config"] = [-1.47568, 2.92779, 1.00845, -2.0847, 1.43588, 1.32575]
        scene["above_plate_pose"] = self._calculate_above_plate_pose(scene["food_pose"])
        scene["move_into_pose"], approach_vector = self._calculate_move_into_pose(
            food_pose=scene["food_pose"],
            recipe=ActionRecipe(
                AcquisitionStrategy.SKEWER,
                MotionAxis.VERTICAL,
                ToolAlignment.PERPENDICULAR,
            ),
        )
        scene["move_above_pose"] = self._calculate_move_above_pose(
            move_into_pose=scene["move_into_pose"], approach_vector=approach_vector
        )
        scene["staging_pose"] = self._calculate_staging_pose(scene["mouth_pose"])
        scene["resting_pose"] = Pose(position=Point(x=0.4, y=-0.4, z=0.3))

        return scene

    def _sample_pose_in_cylindrical_shell(
        self,
        inner_radius: float,
        outer_radius: float,
        min_height: float,
        max_height: float,
    ) -> Pose:
        """
        Samples a random pose within a cylindrical shell

        The frame's Z-axis is constrained to be world up, and its X-axis is
        oriented to point towards the robot base with some random variability.
        """
        # --- Position Sampling in a Cylindrical Shell ---
        # 1. Sample the radius and angle
        radius = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2))
        theta = np.random.uniform(0, 2 * np.pi)

        # 2. Convert to Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.random.uniform(min_height, max_height)
        position = Point(x=x, y=y, z=z)

        # --- Orientation Calculation  ---
        z_axis = np.array([0.0, 0.0, 1.0])
        look_at_vector = -np.array([x, y, 0.0])  # Project to XY plane
        if np.linalg.norm(look_at_vector) < 1e-6:
            look_at_vector = np.array([1.0, 0.0, 0.0])
        x_axis_direction = look_at_vector / np.linalg.norm(look_at_vector)

        y_axis = np.cross(z_axis, x_axis_direction)
        rotation_matrix = np.array([x_axis_direction, y_axis, z_axis]).T
        main_rot = R.from_matrix(rotation_matrix)

        rand_yaw_angle = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))
        variability_rot = R.from_euler("z", rand_yaw_angle)

        final_rot = main_rot * variability_rot
        quat = final_rot.as_quat()

        return Pose(
            position=position,
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )

    def _sample_pose_in_spherical_shell(self) -> Pose:
        """
        Samples a random pose within a spherical shell in front of the robot.
        The frame's Z-axis is constrained to be world up, and its X-axis is
        oriented to point towards the robot base with some random variability.
        """
        inner_radius, outer_radius = 0.3, 0.7

        # Sample position
        r = np.random.uniform(inner_radius**3, outer_radius**3) ** (1 / 3)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi / 2)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        position = Point(x=x, y=y, z=z)

        # --- Corrected Orientation Logic ---
        # 1. Define the primary Z-axis constraint (world up).
        z_axis = np.array([0.0, 0.0, 1.0])

        # 2. Define the vector pointing from the sampled point to the robot base (origin).
        look_at_vector = -np.array([x, y, z])

        # 3. Project the look-at vector onto the XY plane to get the direction for the X-axis.
        #    This ensures the final X-axis will be perpendicular to the world Z-axis.
        x_axis_direction = np.array([look_at_vector[0], look_at_vector[1], 0.0])

        # Normalize the direction vector. Handle the case where the point is directly above the origin.
        if np.linalg.norm(x_axis_direction) < 1e-6:
            x_axis_direction = np.array(
                [1.0, 0.0, 0.0]
            )  # Default to pointing along world X
        x_axis_direction /= np.linalg.norm(x_axis_direction)

        # 4. Create a base rotation where the X-axis points in the desired direction and Z is up.
        #    The Y-axis is derived from the cross product to form a right-handed frame.
        y_axis = np.cross(z_axis, x_axis_direction)
        rotation_matrix = np.array([x_axis_direction, y_axis, z_axis]).T
        main_rot = R.from_matrix(rotation_matrix)

        # 5. Add a small random rotational variability around the Z-axis.
        rand_yaw_angle = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))
        variability_rot = R.from_euler("z", rand_yaw_angle)

        # 6. Combine the main rotation with the variability.
        final_rot = main_rot * variability_rot
        quat = final_rot.as_quat()

        return Pose(
            position=position,
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )

    def _calculate_above_plate_pose(self, food_pose: Pose) -> Pose:
        """
        Calculates a camera pose for the Jaco end-effector that looks at the food,
        with a yaw constraint that keeps the arm aligned with the robot base.
        """
        food_position = np.array(
            [food_pose.position.x, food_pose.position.y, food_pose.position.z]
        )

        # --- Position Calculation with Yaw Constraint ---
        # 1. Determine the base yaw angle from the robot's origin to the food's XY position.
        base_yaw_angle = np.arctan2(food_position[1], food_position[0])

        # 2. Add random variability to this base angle.
        yaw_variability = np.random.uniform(-np.deg2rad(45), np.deg2rad(45))
        final_azimuthal_angle = base_yaw_angle + yaw_variability + np.pi

        # 3. Use spherical coordinates relative to the food pose to find the camera position.
        radial_distance = 0.3  # Constant distance from the food
        polar_angle = np.random.uniform(0, np.deg2rad(45))  # Angle from vertical

        # Calculate the offset from the food pose.
        x_offset = radial_distance * np.sin(polar_angle) * np.cos(final_azimuthal_angle)
        y_offset = radial_distance * np.sin(polar_angle) * np.sin(final_azimuthal_angle)
        z_offset = radial_distance * np.cos(polar_angle)

        # The final camera position is the food position plus this offset.
        camera_position = food_position + np.array([x_offset, y_offset, z_offset])

        # --- Orientation Calculation (Look-at with no roll) ---
        # 1. The end-effector's Z-axis must point from its position to the food's origin.
        z_axis = food_position - camera_position
        z_axis /= np.linalg.norm(z_axis)

        # 2. The end-effector's Y-axis should be aligned with the world's "up" to prevent roll.
        world_up = np.array([0.0, 0.0, 1.0])

        # 3. Calculate the X-axis (left) and handle the singularity when looking straight down.
        if np.abs(np.dot(z_axis, world_up)) > 0.999:
            # Looking straight down, define "left" relative to the world frame.
            x_axis = np.array([0.0, 1.0, 0.0])
        else:
            x_axis = np.cross(world_up, z_axis)
            x_axis /= np.linalg.norm(x_axis)

        # 4. Re-calculate the Y-axis to ensure the frame is perfectly orthonormal.
        y_axis = np.cross(z_axis, x_axis)

        # 5. Construct the final rotation matrix from the basis vectors.
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()

        # Create and return the final Pose message
        final_pose = Pose()
        final_pose.position = Point(
            x=camera_position[0], y=camera_position[1], z=camera_position[2]
        )
        final_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        return final_pose

    # --- Semantic Pose Calculation ---
    def _calculate_move_into_pose(
        self, food_pose: Pose, recipe: ActionRecipe
    ) -> Tuple[Pose, np.ndarray]:
        """
        Calculates the MoveInto tool tip pose based on a semantic ActionRecipe.
        Returns the pose and the calculated approach vector (the tool's Z-axis).
        """
        # Default to identity rotation and a vertical approach vector
        final_rotation = R.identity()
        approach_vector = np.array([0.0, 0.0, -1.0])

        # --- SKEWER Strategy Logic ---
        if recipe.strategy == AcquisitionStrategy.SKEWER:
            # Get the food's principal axes from its orientation
            food_orientation = R.from_quat(
                [
                    food_pose.orientation.x,
                    food_pose.orientation.y,
                    food_pose.orientation.z,
                    food_pose.orientation.w,
                ]
            )
            major_axis_food = food_orientation.apply(
                [1.0, 0.0, 0.0]
            )  # Food's X (longer)
            minor_axis_food = food_orientation.apply(
                [0.0, 1.0, 0.0]
            )  # Food's Y (shorter)

            # 1. Align Tines: The tool's X-axis (tines) must align with the
            #    food's minor axis for a stable skewer.
            tool_x_final = minor_axis_food

            # 2. Define Approach & Tilt: The approach is along the food's major axis,
            #    tilted by a random polar angle. We find the tool's Z-axis by rotating
            #    a downward vector around the tool's new X-axis.
            sampled_polar_angle = np.random.uniform(
                0, np.deg2rad(60)
            )  # Angle from vertical
            rotation = R.from_rotvec(sampled_polar_angle * tool_x_final)
            tool_z_final = rotation.apply(
                np.array([0.0, 0.0, -1.0])
            )  # Rotate a downward vector

            # 3. Complete the Frame: The tool's Y-axis is derived from the cross product.
            tool_y_final = np.cross(tool_z_final, tool_x_final)

            # 4. The final rotation is constructed from these basis vectors.
            rotation_matrix = np.array([tool_x_final, tool_y_final, tool_z_final]).T
            final_rotation = R.from_matrix(rotation_matrix)
            approach_vector = tool_z_final

        # --- SCOOP Strategy Logic (Placeholder) ---
        elif recipe.strategy == AcquisitionStrategy.SCOOP:
            # This logic will be implemented next.
            base_rotation = R.from_euler("y", -60, degrees=True)
            final_rotation = base_rotation
            # TODO: Add alignment logic based on food's principal axes.

        # --- Construct the final pose ---
        q = final_rotation.as_quat()
        move_into_pose = Pose()
        move_into_pose.position = food_pose.position
        move_into_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        return move_into_pose, approach_vector

    def _calculate_move_above_pose(
        self, move_into_pose: Pose, approach_vector: np.ndarray
    ) -> Pose:
        """
        Calculates the MoveAbove pose by offsetting from MoveInto along the
        calculated approach vector.
        """
        # 1. Inherit the orientation directly from the target pose
        final_orientation = move_into_pose.orientation

        # 2. The motion vector for the linear path IS the approach vector (tool's Z-axis)
        motion_vector = approach_vector

        # 3. Calculate the offset position
        offset_dist = 0.1  # 10 cm
        p = move_into_pose.position
        # We add the offset because the approach_vector is already pointing "down".
        # To get the "above" pose, we move in the opposite direction of the approach.
        offset = motion_vector * -offset_dist
        final_position = Point(x=p.x + offset[0], y=p.y + offset[1], z=p.z + offset[2])

        return Pose(position=final_position, orientation=final_orientation)

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

    def _serialize_pose(self, pose: Pose) -> Dict[str, List[float]]:
        """Converts a Pose message to a JSON-serializable dictionary."""
        return {
            "position": [pose.position.x, pose.position.y, pose.position.z],
            "orientation_xyzw": [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
        }

    def _serialize_trajectory(
        self, trajectory: JointTrajectory
    ) -> Optional[Dict[str, Any]]:
        """Converts a JointTrajectory message to a JSON-serializable dictionary."""
        if not trajectory or not trajectory.points:
            return None

        serialized_points = []
        for point in trajectory.points:
            serialized_points.append(
                {
                    "positions": list(point.positions),
                    "velocities": list(point.velocities),
                    "accelerations": list(point.accelerations),
                    "time_from_start_sec": point.time_from_start.sec,
                    "time_from_start_nanosec": point.time_from_start.nanosec,
                }
            )

        return {
            "joint_names": list(trajectory.joint_names),
            "points": serialized_points,
        }

    # --- Planning Primitive Placeholders ---
    def _plan_to_joint_config(
        self,
        moveit_planner: MoveIt2,
        target_config: List[float],
        start_config: List[float],
    ) -> Optional[JointTrajectory]:
        """Plans a joint-space trajectory for a given planner and target."""
        moveit_planner.set_joint_goal(target_config)
        future = moveit_planner.plan_async(start_joint_state=start_config)
        rclpy.spin_until_future_complete(
            self.node, future, timeout_sec=self.planning_timeout
        )
        return moveit_planner.get_trajectory(future)

    def _plan_to_move_above(
        self,
        move_above_pose: Pose,
        start_state_jaco: List[float],
        start_state_atool: List[float],
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], Optional[JointTrajectory]]:
        """
        Plans to the MoveAbove configuration by solving 8-DOF IK and then
        planning for each subgroup separately.
        """
        LOGGER.info("  Solving 8-DOF IK for MoveAbove pose...")
        start_state_full = start_state_jaco + start_state_atool

        # 1. Solve 8-DOF IK for the target pose
        ik_solution = self.moveit2_full.compute_ik(
            position=move_above_pose.position,
            quat_xyzw=move_above_pose.orientation,
            start_joint_state=start_state_full,
        )

        if not ik_solution:
            LOGGER.warning("  IK solution for 8-DOF system not found.")
            return TrialStatus.IK_FAILURE, None, None

        LOGGER.info("  8-DOF IK solution found. Planning for subgroups.")
        target_jaco_config = ik_solution[:6]
        target_atool_config = ik_solution[6:]

        # 2. Plan for Jaco arm (6-DOF)
        traj_jaco = self._plan_to_joint_config(
            self.moveit2_jaco, target_jaco_config, start_state_jaco
        )
        if not traj_jaco or not traj_jaco.points:
            LOGGER.warning("  Jaco arm planning failed.")
            return TrialStatus.PLANNER_FAILURE, None, None

        # 3. Plan for Articutool wrist (2-DOF)
        traj_atool = self._plan_to_joint_config(
            self.moveit2_atool, target_atool_config, start_state_atool
        )
        if not traj_atool or not traj_atool.points:
            LOGGER.warning("  Articutool planning failed.")
            return TrialStatus.PLANNER_FAILURE, traj_jaco, None

        LOGGER.info("  Subgroup planning successful.")
        return TrialStatus.SUCCESS, traj_jaco, traj_atool

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
        self.add_articutool_bounding_cylinder()
        for i in range(self.num_trials):
            LOGGER.info(f"--- Running Trial {i + 1}/{self.num_trials} ---")

            # 1. Generate a new scene
            scene = self._generate_scene()
            LOGGER.info(
                f"""
                --- Generated Scene Parameters for Trial {i + 1} ---

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
            current_atool_state = [0.0, 0.0]  # Assume atool starts at zero

            # --- Stage 1: Home -> AbovePlate (P1 with S1) ---
            LOGGER.info("Stage 1: Home -> AbovePlate")

            status, trajectory = self._plan_s1_unconstrained(
                scene["above_plate_pose"], current_jaco_state
            )

            # Create a dictionary of all generated poses for this trial
            scene_poses = {
                "food_pose": self._serialize_pose(scene["food_pose"]),
                "mouth_pose": self._serialize_pose(scene["mouth_pose"]),
                "above_plate_pose": self._serialize_pose(scene["above_plate_pose"]),
                "move_above_pose": self._serialize_pose(scene["move_above_pose"]),
                "move_into_pose": self._serialize_pose(scene["move_into_pose"]),
                "staging_pose": self._serialize_pose(scene["staging_pose"]),
                "resting_pose": self._serialize_pose(scene["resting_pose"]),
            }

            # --- Result Logging for Stage 1 ---
            # Consolidate all metrics for this stage into a single record.
            self.results.append(
                {
                    "trial_id": i,
                    "stage": "HomeToAbovePlate",
                    "primitive": "S1",
                    "status": status.value,
                    "trajectory": self._serialize_trajectory(trajectory),
                    "scene_poses": scene_poses,
                }
            )

            if status != TrialStatus.SUCCESS:
                LOGGER.error(
                    f"  Stage 1 failed with status: {status.value}. Skipping trial."
                )
                continue

            # Update state for the next stage
            current_jaco_state = list(trajectory.points[-1].positions)
            LOGGER.info("  Stage 1 successful.")

            # --- Stage 2: AbovePlate -> MoveAbove (P2 with SX) ---
            LOGGER.info("Stage 2: AbovePlate -> MoveAbove")

            # Plan to the MoveAbove configuration
            status, traj_jaco_to_above, traj_atool_to_above = self._plan_to_move_above(
                scene["move_above_pose"], current_jaco_state, current_atool_state
            )

            # Log the results for this stage and recipe
            self.results.append(
                {
                    "trial_id": i,
                    "stage": "AbovePlateToMoveAbove",
                    "status": status.value,
                }
            )

            if status != TrialStatus.SUCCESS:
                LOGGER.error(
                    f"  Stage 2 failed with status: {status.value}. Skipping trial."
                )
                continue

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
