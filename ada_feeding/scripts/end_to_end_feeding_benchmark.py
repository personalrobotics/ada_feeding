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
from threading import Thread, Lock
from typing import Optional, List, Dict, Tuple, Any
import json
import math
import subprocess
import sys
import argparse
from enum import Enum, auto
from dataclasses import dataclass, asdict

# Third-party imports
import numpy as np
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from moveit_msgs.msg import PlanningScene, AllowedCollisionEntry, AllowedCollisionMatrix
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import tf2_ros

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


class ExecutionMode(Enum):
    """Defines how trajectories for a stage should be interpreted."""

    SEQUENTIAL = "Sequential"  # Jaco and Articutool move one after the other.
    SYNCHRONOUS = "Synchronous"  # Jaco and Articutool move at the same time.
    JACO_ONLY = "Jaco Only"  # Only a Jaco trajectory exists for this stage.
    ATOOL_ONLY = (
        "Articutool Only"  # Only an Articutool trajectory exists for this stage.
    )


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


@dataclass
class CylindricalSamplingParams:
    """Parameters for sampling a pose within a cylindrical shell."""

    name: str
    inner_radius: float
    outer_radius: float
    min_height: float
    max_height: float


@dataclass
class SphericalSamplingParams:
    """Parameters for sampling a pose within a spherical shell."""

    name: str
    inner_radius: float
    outer_radius: float
    # Angles in radians for theta (XY plane) and phi (from Z-axis)
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    min_height: Optional[float] = None
    max_height: Optional[float] = None


@dataclass
class SceneGenerationParams:
    """Holds all parameters that define the random scene generation."""

    food_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="food",
        inner_radius=0.4,
        outer_radius=1.04,  # Corresponds to the 8-DOF workspace
        theta_range=(0.0, 2 * math.pi),
        phi_range=(0.0, math.pi / 2),  # Upper hemisphere
        min_height=0.1,
        max_height=0.4,
    )
    mouth_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="mouth",
        inner_radius=0.4,
        outer_radius=1.04,  # Corresponds to the 8-DOF workspace
        theta_range=(0.0, 2 * math.pi),
        phi_range=(0.0, math.pi / 2),
        min_height=0.3,
        max_height=0.5,
    )
    resting_sampling: SphericalSamplingParams = SphericalSamplingParams(
        name="resting",
        inner_radius=0.5,
        outer_radius=0.9,  # Corresponds to the 6-DOF workspace
        theta_range=(0.0, 2 * math.pi),
        phi_range=(0, math.pi / 2),
        min_height=0.2,
        max_height=0.4,
    )
    above_plate_radial_dist: float = 0.3
    above_plate_polar_angle_rad_max: float = math.pi / 3
    above_plate_yaw_variability_rad: float = math.pi / 8
    skewer_polar_angle_rad_max: float = math.pi / 2
    in_food_tool_roll_angle_deg: float = 180.0
    above_food_offset_dist: float = 0.1  # 10 cm
    staging_offset_dist: float = 0.15  # 15 cm


# --- Constraint Helpers ---
class MoveIt2ConstraintType(Enum):
    JOINT = "joint"
    POSITION = "position"
    ORIENTATION = "orientation"
    POSE = "pose"


def create_pose_constraint(
    pose: Pose, tolerance_position: float = 0.001, tolerance_orientation: float = 0.001
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a standard pose goal constraint."""
    return (
        MoveIt2ConstraintType.POSE,
        {
            "pose": pose,
            "tolerance_position": tolerance_position,
            "tolerance_orientation": tolerance_orientation,
        },
    )


def create_orientation_path_constraint(
    quat_xyzw: Tuple, tolerance_rad: Tuple
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates an orientation path constraint, useful for keeping the tool level."""
    return (
        MoveIt2ConstraintType.ORIENTATION,
        {
            "quat_xyzw": Quaternion(
                x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3]
            ),
            "tolerance": tolerance_rad,
            "weight": 1.0,
        },
    )


def create_joint_constraint(
    joint_positions: List[float],
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a joint goal constraint."""
    return (MoveIt2ConstraintType.JOINT, {"joint_positions": joint_positions})


def create_position_constraint(
    position: Point, tolerance_position: float = 0.001
) -> Tuple[MoveIt2ConstraintType, Dict]:
    """Creates a standard position goal constraint."""
    return (
        MoveIt2ConstraintType.POSITION,
        {
            "position": position,
            "tolerance": tolerance_position,
            "weight": 1.0,
        },
    )


class SceneGenerator:
    """A dedicated class for procedurally generating planning scenes."""

    def __init__(self, params: SceneGenerationParams):
        self.params = params

    def generate(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Generates a randomized scene and returns the scene dict and sampled parameters.
        """
        scene = {}
        scene_characteristics = {}

        food_position, food_params = self._sample_pose_in_spherical_shell(
            self.params.food_sampling
        )
        food_orientation = self._calculate_base_facing_orientation(food_position)
        scene["food_pose"] = Pose(position=food_position, orientation=food_orientation)
        scene_characteristics.update(food_params)

        mouth_position, mouth_params = self._sample_pose_in_spherical_shell(
            self.params.mouth_sampling
        )
        mouth_orientation = self._calculate_base_facing_orientation(mouth_position)
        scene["mouth_pose"] = Pose(
            position=mouth_position, orientation=mouth_orientation
        )
        scene_characteristics.update(mouth_params)

        resting_position, resting_params = self._sample_pose_in_spherical_shell(
            self.params.resting_sampling
        )
        resting_orientation = self._calculate_jaco_ee_orientation(resting_position)
        scene["resting_pose"] = Pose(
            position=resting_position, orientation=resting_orientation
        )
        scene_characteristics.update(resting_params)
        scene_characteristics.update(
            self._characterize_pose(scene["resting_pose"], "resting_pose")
        )

        scene["home_config"] = [-1.47568, 2.92779, 1.00845, -2.0847, 1.43588, 1.32575]

        scene["above_plate_pose"], above_plate_params = (
            self._calculate_above_plate_pose(scene["food_pose"])
        )
        scene_characteristics.update(above_plate_params)
        scene_characteristics.update(
            self._characterize_pose(scene["above_plate_pose"], "above_plate_pose")
        )

        (
            scene["in_food_pose"],
            approach_vector,
            in_food_params,
        ) = self._calculate_in_food_pose(
            food_pose=scene["food_pose"],
            recipe=ActionRecipe(
                AcquisitionStrategy.SKEWER,
                MotionAxis.VERTICAL,
                ToolAlignment.PERPENDICULAR,
            ),
        )
        scene_characteristics.update(in_food_params)
        scene_characteristics.update(
            self._characterize_pose(scene["in_food_pose"], "in_food_pose")
        )

        scene["above_food_pose"] = self._calculate_above_food_pose(
            scene["in_food_pose"], approach_vector
        )
        scene_characteristics.update(
            self._characterize_pose(scene["above_food_pose"], "above_food_pose")
        )
        scene["staging_pose"] = self._calculate_staging_pose(scene["mouth_pose"])
        scene_characteristics.update(
            self._characterize_pose(scene["staging_pose"], "staging_pose")
        )

        # Add derived characteristics for analysis
        food_pos = scene["food_pose"].position
        mouth_pos = scene["mouth_pose"].position
        resting_pos = scene["resting_pose"].position
        scene_characteristics["food_mouth_distance_m"] = math.sqrt(
            (food_pos.x - mouth_pos.x) ** 2
            + (food_pos.y - mouth_pos.y) ** 2
            + (food_pos.z - mouth_pos.z) ** 2
        )
        scene_characteristics["food_resting_distance_m"] = math.sqrt(
            (food_pos.x - resting_pos.x) ** 2
            + (food_pos.y - resting_pos.y) ** 2
            + (food_pos.z - resting_pos.z) ** 2
        )

        return scene, scene_characteristics

    def _characterize_pose(self, pose: Pose, pose_name: str) -> Dict[str, float]:
        """Extracts key kinematic characteristics from a pose."""
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        rot = R.from_quat(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )

        # For a Jaco EE frame, the Z-axis is "forward"
        forward_vec = rot.apply([0.0, 0.0, 1.0])

        characteristics = {
            f"{pose_name}_dist_3d": np.linalg.norm(pos),
            f"{pose_name}_dist_2d": np.linalg.norm(pos[:2]),
            f"{pose_name}_global_yaw_rad": np.arctan2(forward_vec[1], forward_vec[0]),
            f"{pose_name}_global_pitch_rad": np.arcsin(forward_vec[2]),
        }
        return characteristics

    def _sample_pose_in_cylindrical_shell(
        self, sampling_params: CylindricalSamplingParams
    ) -> Tuple[Point, Dict[str, float]]:
        """
        A general function to sample a POSITION within the given cylindrical parameters.
        Returns a Point and the sampled characteristic values.
        """
        radius = np.sqrt(
            np.random.uniform(
                sampling_params.inner_radius**2, sampling_params.outer_radius**2
            )
        )
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(sampling_params.min_height, sampling_params.max_height)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        position = Point(x=x, y=y, z=z)

        prefix = sampling_params.name
        sampled_values = {
            f"{prefix}_sampled_radius": radius,
            f"{prefix}_sampled_theta_rad": theta,
            f"{prefix}_sampled_z": z,
        }
        return position, sampled_values

    def _sample_pose_in_spherical_shell(
        self, sampling_params: SphericalSamplingParams
    ) -> Tuple[Point, Dict[str, float]]:
        """
        A general function to sample a POSITION within a spherical shell.
        Returns a Point and the sampled characteristic values.
        Uses rejection sampling if min/max height is specified.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            r = np.random.uniform(
                sampling_params.inner_radius**3, sampling_params.outer_radius**3
            ) ** (1 / 3)
            theta = np.random.uniform(*sampling_params.theta_range)
            phi = np.random.uniform(*sampling_params.phi_range)

            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)

            if (
                sampling_params.min_height is None or z >= sampling_params.min_height
            ) and (
                sampling_params.max_height is None or z <= sampling_params.max_height
            ):
                position = Point(x=x, y=y, z=z)
                prefix = sampling_params.name
                sampled_values = {
                    f"{prefix}_sampled_radius": r,
                    f"{prefix}_sampled_theta_rad": theta,
                    f"{prefix}_sampled_phi_rad": phi,
                }
                return position, sampled_values

    def _calculate_base_facing_orientation(self, position: Point) -> Quaternion:
        """
        Calculates an orientation that is upright (Z-up) and has its X-axis
        pointing towards the robot base (origin).
        """
        z_axis = np.array([0.0, 0.0, 1.0])
        look_at_vector = -np.array([position.x, position.y, 0.0])
        if np.linalg.norm(look_at_vector) < 1e-6:
            look_at_vector = np.array([1.0, 0.0, 0.0])
        x_axis = look_at_vector / np.linalg.norm(look_at_vector)
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        # Add random yaw variability
        rand_yaw = np.random.uniform(-np.deg2rad(30), np.deg2rad(30))
        final_rot = R.from_matrix(rotation_matrix) * R.from_euler("z", rand_yaw)
        quat = final_rot.as_quat()

        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _calculate_jaco_ee_orientation(self, position: Point) -> Quaternion:
        """
        Calculates an orientation matching the Jaco EE convention (Y-up, Z-forward)
        at the given position.
        """
        # (Y-up) is the world's Z-axis
        target_y_axis = np.array([0.0, 0.0, 1.0])

        # (Z-forward) points horizontally away from the robot's base
        target_z_axis_dir = np.array([position.x, position.y, 0.0])
        if np.linalg.norm(target_z_axis_dir) < 1e-6:
            target_z_axis_dir = np.array([1.0, 0.0, 0.0])
        target_z_axis = target_z_axis_dir / np.linalg.norm(target_z_axis_dir)

        # (X-left) is derived from the cross product
        target_x_axis = np.cross(target_y_axis, target_z_axis)

        rotation_matrix = np.array([target_x_axis, target_y_axis, target_z_axis]).T
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()

        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _calculate_above_plate_pose(
        self, food_pose: Pose
    ) -> Tuple[Pose, Dict[str, float]]:
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
        yaw_variability = np.random.uniform(
            -self.params.above_plate_yaw_variability_rad,
            self.params.above_plate_yaw_variability_rad,
        )
        final_azimuthal_angle = base_yaw_angle + yaw_variability + np.pi

        # 3. Use spherical coordinates relative to the food pose to find the camera position.
        radial_distance = self.params.above_plate_radial_dist
        polar_angle = np.random.uniform(0, self.params.above_plate_polar_angle_rad_max)

        # Calculate the offset from the food pose.
        x_offset = radial_distance * np.sin(polar_angle) * np.cos(final_azimuthal_angle)
        y_offset = radial_distance * np.sin(polar_angle) * np.sin(final_azimuthal_angle)
        z_offset = radial_distance * np.cos(polar_angle)

        # The final camera position is the food position plus this offset.
        camera_position = food_position + np.array([x_offset, y_offset, z_offset])

        # --- Orientation Calculation (Look-at with no roll) ---
        # (This logic remains the same)
        z_axis = food_position - camera_position
        z_axis /= np.linalg.norm(z_axis)
        world_up = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(z_axis, world_up)) > 0.999:
            x_axis = np.array([0.0, 1.0, 0.0])
        else:
            x_axis = np.cross(world_up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()

        # Create and return the final Pose message
        final_pose = Pose()
        final_pose.position = Point(
            x=camera_position[0], y=camera_position[1], z=camera_position[2]
        )
        final_pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        sampled_params = {
            "above_plate_sampled_yaw_variability_rad": yaw_variability,
            "above_plate_sampled_polar_angle_rad": polar_angle,
        }
        return final_pose, sampled_params

    def _calculate_in_food_pose(
        self, food_pose: Pose, recipe: ActionRecipe
    ) -> Tuple[Pose, np.ndarray, Dict[str, float]]:
        """
        Calculates the InFood tool tip pose based on a semantic ActionRecipe.
        Returns the pose, the approach vector, and sampled parameters.
        """
        # Default to identity rotation and a vertical approach vector
        final_rotation = R.identity()
        approach_vector = np.array([0.0, 0.0, -1.0])
        sampled_polar_angle = 0.0

        # --- SKEWER Strategy Logic ---
        if recipe.strategy == AcquisitionStrategy.SKEWER:
            # (Logic for finding food axes remains the same)
            food_orientation = R.from_quat(
                [
                    food_pose.orientation.x,
                    food_pose.orientation.y,
                    food_pose.orientation.z,
                    food_pose.orientation.w,
                ]
            )
            minor_axis_food = food_orientation.apply([0.0, 1.0, 0.0])
            tool_x_final = minor_axis_food

            # Use parameter from the params object
            sampled_polar_angle = np.random.uniform(
                0, self.params.skewer_polar_angle_rad_max
            )
            rotation = R.from_rotvec(sampled_polar_angle * tool_x_final)
            tool_z_final = rotation.apply(np.array([0.0, 0.0, -1.0]))

            tool_y_final = np.cross(tool_z_final, tool_x_final)

            base_rotation_matrix = np.array(
                [tool_x_final, tool_y_final, tool_z_final]
            ).T
            base_orientation = R.from_matrix(base_rotation_matrix)
            roll_rotation = R.from_rotvec(
                np.deg2rad(self.params.in_food_tool_roll_angle_deg) * tool_z_final
            )
            final_rotation = roll_rotation * base_orientation
            approach_vector = tool_z_final

        # --- SCOOP Strategy Logic (Placeholder) ---
        elif recipe.strategy == AcquisitionStrategy.SCOOP:
            # ...
            pass

        # --- Construct the final pose ---
        q = final_rotation.as_quat()
        in_food_pose = Pose()
        in_food_pose.position = food_pose.position
        in_food_pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        sampled_params = {"in_food_sampled_polar_angle_rad": sampled_polar_angle}
        return in_food_pose, approach_vector, sampled_params

    def _calculate_above_food_pose(
        self, in_food_pose: Pose, approach_vector: np.ndarray
    ) -> Pose:
        """
        Calculates the AboveFood pose by offsetting from InFood along the
        calculated approach vector.
        """
        # 1. Inherit the orientation directly from the target pose
        final_orientation = in_food_pose.orientation

        # 2. The motion vector for the linear path IS the approach vector (tool's Z-axis)
        motion_vector = approach_vector

        # 3. Calculate the offset position using parameter from the params object
        offset_dist = self.params.above_food_offset_dist
        p = in_food_pose.position
        offset = motion_vector * -offset_dist
        final_position = Point(x=p.x + offset[0], y=p.y + offset[1], z=p.z + offset[2])

        return Pose(position=final_position, orientation=final_orientation)

    def _calculate_staging_pose(self, mouth_pose: Pose) -> Pose:
        """Calculate the staging pose relative to the mouth."""
        offset_dist = self.params.staging_offset_dist

        p = mouth_pose.position
        q = mouth_pose.orientation
        mouth_rot = R.from_quat([q.x, q.y, q.z, q.w])

        # Offset is along the mouth's forward-facing X-axis
        offset_vec = mouth_rot.apply([offset_dist, 0, 0])

        staged_pos = Point(
            x=p.x - offset_vec[0], y=p.y - offset_vec[1], z=p.z - offset_vec[2]
        )

        return Pose(position=staged_pos, orientation=q)

    def _calculate_resting_pose(self, food_pose: Pose, mouth_pose: Pose) -> Pose:
        """
        Calculates a dynamic "Resting" pose for the Jaco end-effector.

        This method implements the "Angular Standoff" strategy. It positions
        the resting pose at a fixed radial distance from the robot base, but with
        an angular offset from the food's position. This offset is directed
        away from the mouth's position, placing the arm in a safe, clear, and
        context-aware staging area.

        Args:
            food_pose: The 6D pose of the food item.
            mouth_pose: The 6D pose of the user's mouth.

        Returns:
            The calculated 6D resting pose.
        """
        # --- Extract XY positions and create 2D vectors from the origin ---
        v_food = np.array([food_pose.position.x, food_pose.position.y])
        v_mouth = np.array([mouth_pose.position.x, mouth_pose.position.y])

        # --- 1. Determine the direction of angular offset ---
        # Use the 2D cross product to find the sign of the angle between vectors.
        # This tells us if the mouth is clockwise or counter-clockwise from the food.
        cross_product_z = np.cross(v_food, v_mouth)

        # We apply the offset in the direction that moves away from the mouth.
        angle = (
            -np.deg2rad(self.params.resting_angular_offset_deg)
            if cross_product_z > 0
            else np.deg2rad(self.params.resting_angular_offset_deg)
        )

        # --- 2. Calculate the new position ---
        # Normalize the food vector to get its direction
        v_food_dir = v_food / np.linalg.norm(v_food)

        # Create a 2D rotation matrix and apply it to the food's direction
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array(((c, -s), (s, c)))
        v_rest_dir = rotation_matrix @ v_food_dir

        # Scale the new direction by the fixed radial distance for the final XY position
        rest_position_xy = v_rest_dir * self.params.resting_radial_dist
        rest_position_z = food_pose.position.z + self.params.resting_vertical_offset

        # --- 3. Calculate the new orientation (upright and facing outward) ---
        # The z-axis (forward) points horizontally from the base to the new position
        z_axis = np.array([rest_position_xy[0], rest_position_xy[1], 0.0])
        z_axis /= np.linalg.norm(z_axis)

        # The y-axis (up) is aligned with the world's Z-axis
        y_axis = np.array([0.0, 0.0, 1.0])

        # The x-axis (left) is the cross product, forming an orthonormal frame
        x_axis = np.cross(y_axis, z_axis)

        # Construct the final rotation matrix and convert to a quaternion
        rotation_matrix_3d = np.array([x_axis, y_axis, z_axis]).T
        rotation = R.from_matrix(rotation_matrix_3d)
        quat = rotation.as_quat()

        # --- 4. Assemble and return the final Pose message ---
        return Pose(
            position=Point(
                x=rest_position_xy[0], y=rest_position_xy[1], z=rest_position_z
            ),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )


# --- Core Benchmark Classes ---
class MotionPlanner:
    """A wrapper for MoveIt2 to provide a seamless, synchronous API for planning."""

    def __init__(
        self,
        node: Node,
        moveit2_jaco: MoveIt2,
        moveit2_atool: MoveIt2,
        moveit2_full: MoveIt2,
        tf_buffer: tf2_ros.Buffer,
        planning_timeout: float,
    ):
        self._node = node
        self._planning_timeout = planning_timeout

        self._tf_buffer = tf_buffer

        # Manages all moveit2 objects and the shared lock
        self._moveit2_objects = {
            PLANNING_GROUP_JACO: moveit2_jaco,
            PLANNING_GROUP_ATOOL: moveit2_atool,
            PLANNING_GROUP_FULL: moveit2_full,
        }
        self._lock = Lock()

    def _get_planner(self, group_name: str) -> MoveIt2:
        """Selects the correct moveit2 object based on group name."""
        if group_name not in self._moveit2_objects:
            raise ValueError(f"Unknown planning group: {group_name}")
        return self._moveit2_objects[group_name]

    @staticmethod
    def _scale_cartesian_trajectory_velocity(
        traj: JointTrajectory, scale_factor: float
    ):
        """Scales the velocity of a Cartesian trajectory"""
        for point in traj.points:
            nsec = (point.time_from_start.sec * 1e9) + point.time_from_start.nanosec
            nsec /= scale_factor
            sec = int(math.floor(nsec / 1e9))
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = int(nsec - (sec * 1e9))
            for i in range(len(point.velocities)):
                point.velocities[i] *= scale_factor
            for i in range(len(point.accelerations)):
                point.accelerations[i] *= scale_factor**2

    def _transform_goal_to_base_link(
        self, planner: MoveIt2, constraint: Tuple[MoveIt2ConstraintType, Dict]
    ):
        """Transforms a pose-based goal constraint to the robot's base frame."""
        constraint_type, kwargs = constraint

        # Only transform pose constraints and only if a frame_id is specified
        if (
            constraint_type != MoveIt2ConstraintType.POSE
            or "frame_id" not in kwargs
            or kwargs["frame_id"] is None
        ):
            return

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = kwargs["frame_id"]
        pose_stamped.pose = kwargs["pose"]

        transformed_pose = self._tf_buffer.transform(
            pose_stamped, planner.base_link_name
        )

        kwargs["pose"] = transformed_pose.pose
        kwargs["frame_id"] = planner.base_link_name

    def compute_ik(
        self, group_name: str, target_pose: Pose, start_joint_state: List[float]
    ) -> Optional[JointState]:
        """
        Computes Inverse Kinematics for a given group and target pose.

        Returns:
            A JointState message on success, None on failure.
        """
        planner = self._get_planner(group_name)
        ik_solution = None

        with self._lock:
            ik_solution = planner.compute_ik(
                position=target_pose.position,
                quat_xyzw=target_pose.orientation,
                start_joint_state=start_joint_state,
            )

        return ik_solution

    def compute_fk(
        self, group_name: str, joint_state: JointState, fk_link_names: List[str]
    ) -> Optional[List[PoseStamped]]:
        """
        Computes Forward Kinematics for a given set of links.

        Returns:
            A list of PoseStamped messages, one for each requested link.
        """
        planner = self._get_planner(group_name)
        fk_poses = None

        with self._lock:
            fk_poses = planner.compute_fk(
                joint_state=joint_state, fk_link_names=fk_link_names
            )

        return fk_poses

    def plan(
        self,
        group_name: str,
        start_state: Optional[List[float]],
        goal_constraints: List[Tuple[MoveIt2ConstraintType, Dict]],
        path_constraints: Optional[List[Tuple[MoveIt2ConstraintType, Dict]]] = None,
        cartesian: bool = False,
        cartesian_max_step: float = 0.001,
        cartesian_jump_threshold: float = 5.0,
        cartesian_fraction_threshold: float = 0.92,
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        """
        Plans a trajectory and measures the planning time.

        Returns:
            A status, the resulting trajectory, and the planning time in seconds.
        """
        planner = self._get_planner(group_name)
        future = None

        if not goal_constraints:
            LOGGER.error("Planning failed: At least one goal constraint is required.")
            return TrialStatus.IK_FAILURE, None, 0.0

        if cartesian:
            planner.cartesian_jump_threshold = cartesian_jump_threshold
            # Note: Other properties like prismatic/revolute jump thresholds could also be set here

        with self._lock:
            planner.clear_goal_constraints()
            planner.clear_path_constraints()

            # If cartesian, transform pose goals to the base link frame
            if cartesian:
                try:
                    for constraint in goal_constraints:
                        self._transform_goal_to_base_link(planner, constraint)
                except Exception as e:
                    LOGGER.error(
                        f"Failed to transform Cartesian goal to base link: {e}"
                    )
                    return TrialStatus.IK_FAILURE, None, 0.0

            # --- Process Goal Constraints ---
            for constraint_type, kwargs in goal_constraints:
                if constraint_type == MoveIt2ConstraintType.JOINT:
                    planner.set_joint_goal(**kwargs)
                elif constraint_type == MoveIt2ConstraintType.POSITION:
                    planner.set_position_goal(**kwargs)
                elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                    planner.set_orientation_goal(**kwargs)
                elif constraint_type == MoveIt2ConstraintType.POSE:
                    planner.set_pose_goal(**kwargs)

            # --- Process Path Constraints ---
            if path_constraints:
                for constraint_type, kwargs in path_constraints:
                    if constraint_type == MoveIt2ConstraintType.JOINT:
                        planner.set_path_joint_constraint(**kwargs)
                    elif constraint_type == MoveIt2ConstraintType.POSITION:
                        planner.set_path_position_constraint(**kwargs)
                    elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                        planner.set_path_orientation_constraint(**kwargs)

            # --- Initiate Asynchronous Planning ---
            future = planner.plan_async(
                start_joint_state=start_state,
                cartesian=cartesian,
                max_step=cartesian_max_step,
            )

        # --- Start timing the planning operation ---
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > self._planning_timeout:
                future.cancel()
                return TrialStatus.PLANNER_FAILURE, None, self._planning_timeout
            time.sleep(0.1)
        planning_time = time.time() - start_time
        # --- End timing ---

        traj = planner.get_trajectory(
            future,
            cartesian=cartesian,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )
        if not traj or not traj.points:
            return TrialStatus.PLANNER_FAILURE, None, planning_time

        # Scale velocity for cartesian plans, as pymoveit2 doesn't do this automatically
        if cartesian and planner.max_velocity > 0.0:
            MotionPlanner._scale_cartesian_trajectory_velocity(
                traj, planner.max_velocity
            )

        return TrialStatus.SUCCESS, traj, planning_time


class PinocchioModel:
    """A wrapper for Pinocchio to provide a seamless interface for kinematic queries."""

    def __init__(self, xacro_file_path: str):
        """Loads the robot model from a XACRO file."""
        self.model: Optional[pin.Model] = None
        self.data: Optional[pin.Data] = None
        self._is_ready = False

        try:
            # Convert XACRO to URDF string
            process = subprocess.run(
                ["ros2", "run", "xacro", "xacro", xacro_file_path],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_xml_string = process.stdout

            # Load model from string
            self.model = pin.buildModelFromXML(urdf_xml_string)
            self.data = self.model.createData()
            self._is_ready = True
            LOGGER.info("Pinocchio model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Pinocchio model: {e}", exc_info=True)

    def is_ready(self) -> bool:
        """Returns True if the model was loaded successfully."""
        return self._is_ready

    def _update_configuration(
        self, jaco_joints: List[float], atool_joints: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Populates Pinocchio's configuration vector `q` from joint lists.
        Handles the sin/cos representation for revolute joints correctly using the
        python `math` library to ensure type compatibility with Pinocchio's bindings.
        """
        q = pin.neutral(self.model)

        all_joints = jaco_joints + (atool_joints if atool_joints is not None else [])
        all_names = JOINT_NAMES_JACO + (
            JOINT_NAMES_ATOOL if atool_joints is not None else []
        )

        for i, name in enumerate(all_names):
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                joint_obj = self.model.joints[joint_id]
                angle = all_joints[i]

                idx = joint_obj.idx_q
                if joint_obj.nq == 2:
                    q[idx : idx + 2] = [math.cos(angle), math.sin(angle)]
                else:
                    q[idx] = angle
        return q

    def get_frame_transform(
        self,
        frame_name: str,
        jaco_joints: List[float],
        atool_joints: Optional[List[float]] = None,
    ) -> Optional[pin.SE3]:
        """
        Performs Forward Kinematics to get the transform of a specific frame.

        Returns:
            A Pinocchio SE3 transform object, or None on failure.
        """
        if not self.is_ready():
            return None

        try:
            # Update the model's configuration vector
            q = self._update_configuration(jaco_joints, atool_joints)

            # Run Forward Kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Get and return the transform
            frame_id = self.model.getFrameId(frame_name)
            return self.data.oMf[frame_id]
        except Exception as e:
            LOGGER.error(f"Pinocchio FK failed for frame '{frame_name}': {e}")
            return None


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
        self.num_trials = num_trials
        self.output_filename = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Use the .jsonl extension for JSON Lines format
            self.output_filename = os.path.join(
                output_dir, f"end_to_end_benchmark_{timestamp}.jsonl"
            )

        self.motion_planner = MotionPlanner(
            node,
            moveit2_jaco,
            moveit2_atool,
            moveit2_full,
            tf2_ros.Buffer(),
            planning_timeout,
        )
        # Pinocchio model for feasibility checks
        self.kinematics_model = PinocchioModel(xacro_file_path)

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
        scene["in_food_pose"], approach_vector = self._calculate_in_food_pose(
            food_pose=scene["food_pose"],
            recipe=ActionRecipe(
                AcquisitionStrategy.SKEWER,
                MotionAxis.VERTICAL,
                ToolAlignment.PERPENDICULAR,
            ),
        )
        scene["above_food_pose"] = self._calculate_above_food_pose(
            in_food_pose=scene["in_food_pose"], approach_vector=approach_vector
        )
        scene["staging_pose"] = self._calculate_staging_pose(scene["mouth_pose"])
        scene["resting_pose"] = self._calculate_resting_pose(
            scene["food_pose"], scene["mouth_pose"]
        )

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

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _test_articutool_ik_solver(self):
        """
        Performs a round-trip test of the Articutool's analytical IK solver.
        """
        LOGGER.info("--- Running Articutool IK Solver Test ---")

        test_cases_deg = {
            "Zero": (0, 0),
            "Pitch_Positive": (30, 0),
            "Pitch_Negative": (-45, 0),
            "Roll_Positive": (0, 30),
            "Roll_Negative": (0, -45),
            "Combined_1": (30, 45),
            "Combined_2": (-20, -60),
            "Limit_Pitch": (90, 0),  # A singularity case
            "Limit_Roll": (0, 90),
        }

        all_passed = True
        for name, (pitch_deg, roll_deg) in test_cases_deg.items():
            # --- 1. START: Convert test case to radians ---
            theta_p = np.deg2rad(pitch_deg)
            theta_r = np.deg2rad(roll_deg)

            # --- 2. FORWARD KINEMATICS: Calculate the target Y-axis vector ---
            # Note the swapped sin/cos for y_y due to kinematic conventions
            y_x = np.cos(theta_p) * np.cos(theta_r)
            y_y = np.sin(theta_r)
            y_z = np.sin(theta_p) * np.cos(theta_r)
            target_y_in_wrist_frame = np.array([y_x, y_y, y_z])

            # --- 3. PRE-ROTATION: Apply the necessary R_z(+pi/2) rotation ---
            # This is the crucial step that mimics what our trajectory generator does.
            ik_input_vector = np.array(
                [
                    -target_y_in_wrist_frame[1],
                    target_y_in_wrist_frame[0],
                    target_y_in_wrist_frame[2],
                ]
            )

            # --- 4. INVERSE KINEMATICS: Call the solver ---
            ik_solutions_rad = self._solve_articutool_ik(ik_input_vector)

            # --- 5. VERIFICATION: Check if the original angles are in the solution set ---
            found_match = False
            for sol_p, sol_r in ik_solutions_rad:
                # Check if a solution is close to the original input, accounting for angle wrapping
                if np.isclose(
                    self._normalize_angle(sol_p), self._normalize_angle(theta_p)
                ) and np.isclose(
                    self._normalize_angle(sol_r), self._normalize_angle(theta_r)
                ):
                    found_match = True
                    break

            status = "✅ SUCCESS" if found_match else "❌ FAILURE"
            if not found_match:
                all_passed = False

            print(f"\n- Test Case: {name} ({pitch_deg}°, {roll_deg}°)")
            print(
                f"  - FK Target Vector (y_axis): {np.round(target_y_in_wrist_frame, 3)}"
            )
            print(f"  - IK Input Vector (pre-rotated): {np.round(ik_input_vector, 3)}")
            print(
                f"  - IK Solutions Found (deg): {[(np.rad2deg(p), np.rad2deg(r)) for p, r in ik_solutions_rad]}"
            )
            print(f"  - Result: {status}")

        LOGGER.info("--- IK Solver Test Finished ---")
        if all_passed:
            LOGGER.info("✅ All test cases passed!")
        else:
            LOGGER.error("❌ One or more test cases failed!")

    def _generate_orientation_holding_atool_trajectory(
        self, traj_jaco: JointTrajectory, desired_tool_tip_world_orientation: Quaternion
    ) -> Optional[JointTrajectory]:
        """
        Generates a synchronized Articutool trajectory that maintains a fixed world
        orientation, including the necessary kinematic pre-rotation for the IK solver.
        """
        if not traj_jaco or not traj_jaco.points:
            return None

        R_World_TipTarget = R.from_quat(
            [
                desired_tool_tip_world_orientation.x,
                desired_tool_tip_world_orientation.y,
                desired_tool_tip_world_orientation.z,
                desired_tool_tip_world_orientation.w,
            ]
        )
        y_axis_TipTarget_InWorld = R_World_TipTarget.apply(np.array([0.0, 1.0, 0.0]))

        atool_solutions = []
        last_valid_solution = None

        for point in traj_jaco.points:
            jaco_points = list(point.positions)
            jaco_wrist_transform = self.kinematics_model.get_frame_transform(
                frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_points
            )
            if jaco_wrist_transform is None:
                LOGGER.error("  Pinocchio FK failed for a waypoint.")
                return None
            R_World_JacoEE = R.from_matrix(jaco_wrist_transform.rotation)

            target_y_in_wrist_frame = R_World_JacoEE.inv().apply(
                y_axis_TipTarget_InWorld
            )
            ik_solutions = self._solve_articutool_ik(target_y_in_wrist_frame)

            valid_solutions = [
                np.array(sol)
                for sol in ik_solutions
                if (
                    ARTICUTOOL_PITCH_LIMITS_RAD[0]
                    <= sol[0]
                    <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
                    and ARTICUTOOL_ROLL_LIMITS_RAD[0]
                    <= sol[1]
                    <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
                )
            ]

            if not valid_solutions:
                LOGGER.warning(
                    "  Failed to find valid IK solution for an Articutool waypoint."
                )
                return None

            if last_valid_solution is None:
                chosen_solution = valid_solutions[0]
            else:
                distances = [
                    np.linalg.norm(sol - last_valid_solution) for sol in valid_solutions
                ]
                chosen_solution = valid_solutions[np.argmin(distances)]

            atool_solutions.append(chosen_solution)
            last_valid_solution = chosen_solution

        traj_atool = JointTrajectory()
        traj_atool.joint_names = JOINT_NAMES_ATOOL
        for i, pos in enumerate(atool_solutions):
            point = JointTrajectoryPoint()
            point.positions = pos.tolist()
            point.time_from_start = traj_jaco.points[i].time_from_start
            traj_atool.points.append(point)

        return traj_atool

    def _generate_leveling_atool_trajectory(
        self, traj_jaco: JointTrajectory
    ) -> Optional[JointTrajectory]:
        """
        Generates a synchronized Articutool trajectory that maintains a level-to-gravity
        orientation during a Jaco arm motion, ensuring solution continuity.
        """
        if not traj_jaco or not traj_jaco.points:
            return None

        atool_solutions = []
        last_valid_solution = None

        # 1. Iterate through each waypoint of the Jaco trajectory
        for point in traj_jaco.points:
            jaco_joint_config = list(point.positions)

            # 2. For each arm configuration, find the pose of the wrist using FK
            jaco_wrist_transform = self.kinematics_model.get_frame_transform(
                frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_joint_config
            )
            if jaco_wrist_transform is None:
                LOGGER.error(
                    "  Pinocchio FK failed for a waypoint while generating leveling trajectory."
                )
                return None

            p = jaco_wrist_transform.translation
            q = R.from_matrix(jaco_wrist_transform.rotation).as_quat()
            jaco_wrist_pose = Pose(
                position=Point(x=p[0], y=p[1], z=p[2]),
                orientation=Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
            )

            # 3. Calculate all possible Articutool IK solutions for leveling
            R_world_jacoee = R.from_quat([q[0], q[1], q[2], q[3]])
            target_up_in_wrist_frame = R_world_jacoee.inv().apply(
                np.array([0.0, 0.0, 1.0])
            )
            ik_solutions = self._solve_articutool_ik(target_up_in_wrist_frame)

            # 4. Filter for solutions that are within joint limits
            valid_solutions = [
                np.array(sol)
                for sol in ik_solutions
                if (
                    ARTICUTOOL_PITCH_LIMITS_RAD[0]
                    <= sol[0]
                    <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
                    and ARTICUTOOL_ROLL_LIMITS_RAD[0]
                    <= sol[1]
                    <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
                )
            ]

            if not valid_solutions:
                LOGGER.warning(
                    "  Failed to find any valid leveling IK solution for a waypoint."
                )
                # As a fallback, try to re-use the last known good solution
                if last_valid_solution is not None:
                    chosen_solution = last_valid_solution
                else:
                    return None  # Cannot proceed if the first waypoint has no solution
            elif last_valid_solution is None:
                # For the first waypoint, just pick the first valid solution
                chosen_solution = valid_solutions[0]
            else:
                # For subsequent waypoints, choose the solution closest to the previous one
                distances = [
                    np.linalg.norm(sol - last_valid_solution) for sol in valid_solutions
                ]
                chosen_solution = valid_solutions[np.argmin(distances)]

            atool_solutions.append(chosen_solution)
            last_valid_solution = chosen_solution

        # 5. Compile the continuous solutions into a JointTrajectory message
        traj_atool = JointTrajectory()
        traj_atool.joint_names = JOINT_NAMES_ATOOL
        for i, pos in enumerate(atool_solutions):
            point_msg = JointTrajectoryPoint()
            point_msg.positions = pos.tolist()
            point_msg.time_from_start = traj_jaco.points[i].time_from_start
            traj_atool.points.append(point_msg)

        return traj_atool

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

    def _compute_leveling_joints(self, jaco_wrist_pose: Pose) -> Optional[List[float]]:
        """
        Calculates the Articutool joint angles required to point the tool's Y-axis up.
        This now uses the original `_solve_articutool_ik` method.
        """
        # 1. Get the rotation of the Jaco wrist in the world frame
        R_world_jacoee = R.from_quat(
            [
                jaco_wrist_pose.orientation.x,
                jaco_wrist_pose.orientation.y,
                jaco_wrist_pose.orientation.z,
                jaco_wrist_pose.orientation.w,
            ]
        )

        # 2. Transform the world "up" vector into the wrist's frame
        world_z_up_vector = np.array([0.0, 0.0, 1.0])
        target_up_in_wrist_frame = R_world_jacoee.inv().apply(world_z_up_vector)

        # 3. Solve the Articutool IK for the transformed vector using the original solver
        #    The solver's math is correct for this specific leveling task.
        ik_solutions = self._solve_articutool_ik(target_up_in_wrist_frame)

        if not ik_solutions:
            return None

        # 4. Find the first valid solution within joint limits
        for theta_p, theta_r in ik_solutions:
            if (
                ARTICUTOOL_PITCH_LIMITS_RAD[0]
                <= theta_p
                <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
                and ARTICUTOOL_ROLL_LIMITS_RAD[0]
                <= theta_r
                <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
            ):
                LOGGER.info(
                    f"  Found valid leveling solution: Pitch={math.degrees(theta_p):.1f}, Roll={math.degrees(theta_r):.1f}"
                )
                return [theta_p, theta_r]

        return None

    def _calculate_cartesian_path_length(
        self, trajectory: Optional[JointTrajectory], group_name: str
    ) -> float:
        """Calculates the Cartesian path length of the end-effector for a trajectory."""
        if (
            not trajectory
            or not self.kinematics_model.is_ready()
            or len(trajectory.points) < 2
        ):
            return 0.0

        total_length = 0.0
        last_position = None
        joint_names = trajectory.joint_names

        # Determine the correct end-effector link for the planning group
        if group_name == PLANNING_GROUP_JACO:
            ee_link = END_EFFECTOR_LINK_JACO
        else:
            ee_link = END_EFFECTOR_LINK_FULL

        for point in trajectory.points:
            joint_map = dict(zip(joint_names, point.positions))
            jaco_config = [joint_map.get(name, 0.0) for name in JOINT_NAMES_JACO]
            atool_config = [joint_map.get(name, 0.0) for name in JOINT_NAMES_ATOOL]

            transform = self.kinematics_model.get_frame_transform(
                ee_link, jaco_config, atool_config
            )
            if transform is None:
                continue

            current_position = transform.translation
            if last_position is not None:
                total_length += np.linalg.norm(current_position - last_position)
            last_position = current_position

        return total_length

    # --- Feasibility Checking ---
    def _is_config_kinematically_feasible(self, jaco_joint_config: List[float]) -> bool:
        """
        Checks if the Articutool can maintain leveling at a single Jaco configuration
        by using the PinocchioModel to perform forward kinematics.
        """
        if not self.kinematics_model.is_ready():
            return False

        # Get the Jaco end-effector's transform for the given joint configuration.
        # We don't need to specify atool_joints, as they don't affect the Jaco wrist's pose.
        ee_transform = self.kinematics_model.get_frame_transform(
            frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_joint_config
        )

        if ee_transform is None:
            # The FK calculation failed.
            return False

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
            jaco_joint_config = list(point.positions)
            if self._is_config_kinematically_feasible(jaco_joint_config):
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

    # --- Planning Primitive ---
    def _plan_to_above_plate(
        self, above_plate_pose: Pose, start_state_jaco: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        LOGGER.info("  Planning to AbovePlate pose...")
        goal_constraints = [
            create_pose_constraint(above_plate_pose, tolerance_orientation=0.2)
        ]

        return self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
        )

    def _plan_to_above_food(
        self,
        above_food_pose: Pose,
        start_state_jaco: List[float],
        start_state_atool: List[float],
    ) -> Tuple[
        TrialStatus, Optional[JointTrajectory], Optional[JointTrajectory], float
    ]:
        LOGGER.info("  Solving 8-DOF IK for AboveFood pose...")

        # 1. Compute IK using the planner
        ik_solution = self.motion_planner.compute_ik(
            group_name=PLANNING_GROUP_FULL,
            target_pose=above_food_pose,
            start_joint_state=(start_state_jaco + start_state_atool),
        )

        if not ik_solution:
            LOGGER.warning("  IK solution for 8-DOF system not found.")
            return TrialStatus.IK_FAILURE, None, None, 0.0

        LOGGER.info(f"  8-DOF IK solution found. Planning for subgroups.")
        solution_map = dict(zip(ik_solution.name, ik_solution.position))
        target_jaco_config = [solution_map[name] for name in JOINT_NAMES_JACO]
        target_atool_config = [solution_map[name] for name in JOINT_NAMES_ATOOL]

        # 2. Plan for the Jaco arm
        jaco_goal_constraints = [create_joint_constraint(target_jaco_config)]
        jaco_path_constraints = [
            create_orientation_path_constraint(
                quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                tolerance_rad=(np.pi, 2 * np.pi, 2 * np.pi),
            )
        ]
        status_jaco, traj_jaco, time_jaco = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=jaco_goal_constraints,
            path_constraints=jaco_path_constraints,
        )

        if status_jaco != TrialStatus.SUCCESS:
            LOGGER.warning("  Jaco arm planning failed.")
            return TrialStatus.PLANNER_FAILURE, None, None, time_jaco

        # 3. Plan for the Articutool
        atool_goal_constraints = [create_joint_constraint(target_atool_config)]
        status_atool, traj_atool, time_atool = self.motion_planner.plan(
            group_name=PLANNING_GROUP_ATOOL,
            start_state=start_state_atool,
            goal_constraints=atool_goal_constraints,
        )

        total_time = time_jaco + time_atool
        if status_atool != TrialStatus.SUCCESS:
            LOGGER.warning("  Articutool planning failed.")
            return TrialStatus.PLANNER_FAILURE, traj_jaco, None, total_time

        return TrialStatus.SUCCESS, traj_jaco, traj_atool, total_time

    def _plan_to_in_food(
        self,
        in_food_pose: Pose,
        start_state_jaco: List[float],
        start_state_atool: List[float],
    ) -> Tuple[
        TrialStatus,
        Optional[JointTrajectory],
        Optional[JointTrajectory],
        Optional[Pose],
        float,
    ]:
        LOGGER.info("  Solving 8-DOF IK for InFood pose...")

        # 1. Compute the 8-DOF IK solution for the target tool tip pose
        ik_solution = self.motion_planner.compute_ik(
            group_name=PLANNING_GROUP_FULL,
            target_pose=in_food_pose,
            start_joint_state=(start_state_jaco + start_state_atool),
        )

        if not ik_solution:
            LOGGER.warning("  IK solution for 8-DOF system not found.")
            return TrialStatus.IK_FAILURE, None, None, None, 0.0

        LOGGER.info(f"  8-DOF IK solution found. Calculating wrist pose via FK.")

        # 2. Use FK to find the Jaco wrist pose from the 8-DOF solution
        fk_poses = self.motion_planner.compute_fk(
            group_name=PLANNING_GROUP_FULL,
            joint_state=ik_solution,
            fk_link_names=[END_EFFECTOR_LINK_JACO],
        )

        if not fk_poses:
            LOGGER.warning("  FK calculation for Jaco wrist failed.")
            return TrialStatus.IK_FAILURE, None, None, None, 0.0

        # The goal for the Jaco arm is the calculated pose of its wrist
        jaco_wrist_goal_pose = fk_poses[0].pose

        # 3. Plan a Cartesian motion for the Jaco arm to the wrist pose
        jaco_goal_constraints = [create_pose_constraint(jaco_wrist_goal_pose)]

        # 4. Plan a Cartesian motion for the Jaco arm to the wrist pose
        status_jaco, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=jaco_goal_constraints,
            cartesian=True,
        )

        if status_jaco != TrialStatus.SUCCESS:
            LOGGER.warning("  Jaco arm planning failed.")
            return TrialStatus.PLANNER_FAILURE, None, None, None, planning_time

        # 4. Generate the corresponding synchronous Articutool trajectory
        LOGGER.info("  Generating synchronous Articutool trajectory...")
        traj_atool = self._generate_orientation_holding_atool_trajectory(
            traj_jaco, in_food_pose.orientation
        )

        if traj_atool is None:
            LOGGER.error("  Failed to generate synchronous Articutool trajectory.")
            return TrialStatus.IK_FAILURE, traj_jaco, None, None, planning_time

        return (
            TrialStatus.SUCCESS,
            traj_jaco,
            traj_atool,
            jaco_wrist_goal_pose,
            planning_time,
        )

    def _plan_to_level_articutool(
        self,
        jaco_wrist_pose: Pose,
        start_state_atool: List[float],
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        """
        Calculates the required joint angles for the Articutool to achieve a level
        pose and plans a trajectory to that configuration.
        """
        LOGGER.info("  Calculating Articutool leveling configuration...")

        # 1. Compute the target joint angles for leveling based on the wrist's pose
        target_leveling_joints = self._compute_leveling_joints(jaco_wrist_pose)

        if target_leveling_joints is None:
            LOGGER.warning("  Could not find an IK solution for Articutool leveling.")
            return TrialStatus.IK_FAILURE, None, 0.0

        # 2. Create a joint goal constraint for the Articutool
        goal_constraints = [create_joint_constraint(target_leveling_joints)]

        # 3. Plan a joint-space motion for the Articutool group
        status, trajectory, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_ATOOL,
            start_state=start_state_atool,
            goal_constraints=goal_constraints,
        )

        if status != TrialStatus.SUCCESS:
            LOGGER.warning("  Articutool leveling plan failed.")
            return TrialStatus.PLANNER_FAILURE, None, planning_time

        LOGGER.info("  Articutool leveling plan successful.")
        return TrialStatus.SUCCESS, trajectory, planning_time

    def _plan_to_resting(
        self,
        resting_wrist_pose: Pose,
        start_state_jaco: List[float],
    ) -> Tuple[
        TrialStatus,
        Optional[JointTrajectory],
        Optional[JointTrajectory],
        float,
        float,
    ]:
        """
        Plans a 6-DOF guided motion for the Jaco arm to a resting wrist pose
        and measures its leveling feasibility.
        """
        LOGGER.info("  Planning to Resting pose (S2-Heuristic)...")

        # 1. Define goal and path constraints for the Jaco arm's wrist
        goal_constraints = [create_position_constraint(resting_wrist_pose.position)]
        path_constraints = [
            create_orientation_path_constraint(
                quat_xyzw=PATH_CONSTRAINT_QUAT_XYZW,
                tolerance_rad=PATH_CONSTRAINT_TOLERANCE_XYZ_RAD,
            )
        ]

        # 2. Plan the guided 6-DOF trajectory for the Jaco arm
        status, traj_jaco, planning_time = self.motion_planner.plan(
            group_name=PLANNING_GROUP_JACO,
            start_state=start_state_jaco,
            goal_constraints=goal_constraints,
            path_constraints=path_constraints,
        )
        if status != TrialStatus.SUCCESS:
            return TrialStatus.PLANNER_FAILURE, None, None, 0.0, planning_time

        # 3. VERIFY the Jaco trajectory for leveling feasibility
        feasibility_percent = self._verify_trajectory(traj_jaco)
        if feasibility_percent < 99.0:
            LOGGER.warning(
                f"  Path to Resting failed verification ({feasibility_percent:.1f}% feasible)."
            )
            return (
                TrialStatus.VERIFICATION_FAILURE,
                traj_jaco,
                None,
                feasibility_percent,
                planning_time,
            )

        # 4. Generate the corresponding synchronous Articutool trajectory
        LOGGER.info("  Generating synchronous Articutool trajectory...")
        traj_atool = self._generate_leveling_atool_trajectory(traj_jaco)

        if traj_atool is None:
            LOGGER.error("  Failed to generate synchronous Articutool trajectory.")
            return (
                TrialStatus.IK_FAILURE,
                traj_jaco,
                None,
                feasibility_percent,
                planning_time,
            )
        return (
            TrialStatus.SUCCESS,
            traj_jaco,
            traj_atool,
            feasibility_percent,
            planning_time,
        )

    def _plan_s2_guided(
        self, goal_pose: Pose, start_state: Any
    ) -> Tuple[TrialStatus, Optional[JointTrajectory], float]:
        LOGGER.info("  Planning with S2 (6-DOF Guided)...")

        future = None
        # --- Lock the entire MoveIt2 configuration and planning block ---
        with self.moveit2_lock:
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

        # Wait for future outside the lock
        start_time = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_time > self.planning_timeout:
                future.cancel()
                return TrialStatus.PLANNER_FAILURE, None, 0.0
            time.sleep(0.1)

        traj = self.moveit2_jaco.get_trajectory(future)

        if not traj or not traj.points:
            return TrialStatus.PLANNER_FAILURE, None, 0.0

        feasibility_percent = self._verify_trajectory(traj)
        if feasibility_percent < 99.0:
            return TrialStatus.VERIFICATION_FAILURE, traj, feasibility_percent

        return TrialStatus.SUCCESS, traj, feasibility_percent

    # --- Main Benchmark Loop ---
    def run(self):
        """Main benchmark execution loop with granular metric collection."""
        # TODO: Decide whether we should keep the bounding cylinder active
        # self.add_articutool_bounding_cylinder()

        for i in range(self.num_trials):
            LOGGER.info(f"--- Running Trial {i + 1}/{self.num_trials} ---")

            # 1. Create the parameter set and the generator for this trial
            generation_params = SceneGenerationParams()
            scene_generator = SceneGenerator(generation_params)

            # 2. Generate the scene and characteristics with a single, clean call
            scene, scene_characteristics = scene_generator.generate()

            params_dict = asdict(generation_params)
            params_dict["path_constraint_tolerance_xyz_rad"] = list(
                PATH_CONSTRAINT_TOLERANCE_XYZ_RAD
            )
            params_dict["articutool_pitch_limits_rad"] = list(
                ARTICUTOOL_PITCH_LIMITS_RAD
            )
            params_dict["articutool_roll_limits_rad"] = list(ARTICUTOOL_ROLL_LIMITS_RAD)

            trial_data = {
                "trial_id": i,
                "scene_poses": {
                    "food_pose": self._serialize_pose(scene["food_pose"]),
                    "mouth_pose": self._serialize_pose(scene["mouth_pose"]),
                    "above_plate_pose": self._serialize_pose(scene["above_plate_pose"]),
                    "above_food_pose": self._serialize_pose(scene["above_food_pose"]),
                    "in_food_pose": self._serialize_pose(scene["in_food_pose"]),
                    "staging_pose": self._serialize_pose(scene["staging_pose"]),
                    "resting_pose": self._serialize_pose(scene["resting_pose"]),
                },
                "scene_characteristics": scene_characteristics,
                "parameters": params_dict,
                "stages": [],
                "end_to_end_success": False,  # Default to False
            }

            # 3. Initialize the robot's state for the trial
            current_jaco_state = scene["home_config"]
            current_atool_state = [0.0, 0.0]
            trial_failed = False

            # --- Stage 1: Home -> AbovePlate ---
            if not trial_failed:
                LOGGER.info("Stage 1: Home -> AbovePlate")
                status, traj_jaco, planning_time = self._plan_to_above_plate(
                    scene["above_plate_pose"], current_jaco_state
                )
                path_length = self._calculate_cartesian_path_length(
                    traj_jaco, PLANNING_GROUP_JACO
                )
                trial_data["stages"].append(
                    {
                        "stage_name": "HomeToAbovePlate",
                        "target_frame": END_EFFECTOR_LINK_JACO,
                        "status": status.value,
                        "execution_mode": ExecutionMode.JACO_ONLY.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": path_length,
                        "custom_metrics": {},
                        "traj_jaco": self._serialize_trajectory(traj_jaco),
                        "traj_atool": None,
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 1 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)

            # --- Stage 2: AbovePlate -> AboveFood ---
            if not trial_failed:
                LOGGER.info("Stage 2: AbovePlate -> AboveFood")
                (
                    status,
                    traj_jaco,
                    traj_atool,
                    planning_time,
                ) = self._plan_to_above_food(
                    scene["above_food_pose"], current_jaco_state, current_atool_state
                )
                path_length_jaco = self._calculate_cartesian_path_length(
                    traj_jaco, PLANNING_GROUP_JACO
                )
                path_length_atool = self._calculate_cartesian_path_length(
                    traj_atool, PLANNING_GROUP_ATOOL
                )
                trial_data["stages"].append(
                    {
                        "stage_name": "AbovePlateToAboveFood",
                        "target_frame": END_EFFECTOR_LINK_FULL,
                        "status": status.value,
                        "execution_mode": ExecutionMode.SEQUENTIAL.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": path_length_jaco
                        + path_length_atool,
                        "custom_metrics": {},
                        "traj_jaco": self._serialize_trajectory(traj_jaco),
                        "traj_atool": self._serialize_trajectory(traj_atool),
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 2 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)
                    current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 3: AboveFood -> InFood ---
            if not trial_failed:
                LOGGER.info("Stage 3: AboveFood -> InFood (Cartesian)")
                (
                    status,
                    traj_jaco,
                    traj_atool,
                    jaco_wrist_pose,
                    planning_time,
                ) = self._plan_to_in_food(
                    scene["in_food_pose"], current_jaco_state, current_atool_state
                )
                path_length = self._calculate_cartesian_path_length(
                    traj_jaco, PLANNING_GROUP_JACO
                )
                trial_data["stages"].append(
                    {
                        "stage_name": "AboveFoodToInFood",
                        "target_frame": END_EFFECTOR_LINK_FULL,
                        "status": status.value,
                        "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": path_length,
                        "custom_metrics": {},
                        "traj_jaco": self._serialize_trajectory(traj_jaco),
                        "traj_atool": self._serialize_trajectory(traj_atool),
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 3 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)
                    current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 4: InFood -> LevelArticutool ---
            if not trial_failed:
                LOGGER.info("Stage 4: Level Articutool")
                status, traj_atool, planning_time = self._plan_to_level_articutool(
                    jaco_wrist_pose, current_atool_state
                )
                path_length = self._calculate_cartesian_path_length(
                    traj_atool, PLANNING_GROUP_ATOOL
                )
                trial_data["stages"].append(
                    {
                        "stage_name": "LevelArticutool",
                        "target_frame": END_EFFECTOR_LINK_ATOOL,
                        "status": status.value,
                        "execution_mode": ExecutionMode.ATOOL_ONLY.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": path_length,
                        "custom_metrics": {},
                        "traj_jaco": None,
                        "traj_atool": self._serialize_trajectory(traj_atool),
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 4 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_atool_state = list(traj_atool.points[-1].positions)

            # --- Stage 5: LevelArticutool -> Resting ---
            if not trial_failed:
                LOGGER.info("Stage 5: Resting")
                (
                    status,
                    traj_jaco,
                    traj_atool,
                    leveling_feasibility,
                    planning_time,
                ) = self._plan_to_resting(scene["resting_pose"], current_jaco_state)
                path_length = self._calculate_cartesian_path_length(
                    traj_jaco, PLANNING_GROUP_JACO
                )
                trial_data["stages"].append(
                    {
                        "stage_name": "Resting",
                        "target_frame": END_EFFECTOR_LINK_JACO,
                        "status": status.value,
                        "execution_mode": ExecutionMode.SYNCHRONOUS.value,
                        "planning_time_sec": planning_time,
                        "trajectory_path_length_m": path_length,
                        "custom_metrics": {
                            "leveling_feasibility_percent": leveling_feasibility
                        },
                        "traj_jaco": self._serialize_trajectory(traj_jaco),
                        "traj_atool": self._serialize_trajectory(traj_atool),
                    }
                )
                if status != TrialStatus.SUCCESS:
                    LOGGER.error(f"  Stage 5 failed. Skipping trial.")
                    trial_failed = True
                else:
                    current_jaco_state = list(traj_jaco.points[-1].positions)
                    current_atool_state = list(traj_atool.points[-1].positions)

            # --- Finalize Trial ---
            if not trial_failed:
                trial_data["end_to_end_success"] = True

            self._save_trial_data(trial_data)
            LOGGER.info(f"Trial {i} data saved to {self.output_filename}")

        LOGGER.info("Benchmark finished.")

    def _save_trial_data(self, trial_data: Dict[str, Any]):
        """
        Saves a single trial's data by appending it as a new line
        to the output file.
        """
        if not self.output_filename:
            return
        try:
            # Open the file in append mode ('a')
            with open(self.output_filename, "a") as f:
                # Use json.dumps to serialize the single trial dictionary
                json_string = json.dumps(trial_data)
                # Write the string followed by a newline character
                f.write(json_string + "\n")
        except Exception as e:
            LOGGER.error(
                f"Failed to save trial data for trial {trial_data.get('trial_id', 'N/A')}: {e}"
            )


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
