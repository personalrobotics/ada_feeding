# Standard imports
import math
from typing import Tuple, Dict, Any

# Third-party imports
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion

# Local application imports
from .data_structures import (
    SceneGenerationParams,
    SphericalSamplingParams,
    ActionRecipe,
    AcquisitionStrategy,
    MotionAxis,
    ToolAlignment,
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

        initial_mouth_position, mouth_params = self._sample_pose_in_spherical_shell(
            self.params.mouth_sampling
        )
        scene_characteristics.update(mouth_params)

        initial_mouth_orientation = self._calculate_tangential_facing_orientation(
            initial_mouth_position,
            food_position=scene["food_pose"].position,
        )

        R_initial_mouth = R.from_quat(
            [
                initial_mouth_orientation.x,
                initial_mouth_orientation.y,
                initial_mouth_orientation.z,
                initial_mouth_orientation.w,
            ]
        )
        x_axis_initial_mouth = R_initial_mouth.apply([1.0, 0.0, 0.0])

        scoot_dist = (
            self.params.staging_offset_dist * self.params.staging_scoot_back_factor
        )
        p_initial_mouth = np.array(
            [
                initial_mouth_position.x,
                initial_mouth_position.y,
                initial_mouth_position.z,
            ]
        )
        p_final_mouth = p_initial_mouth - (x_axis_initial_mouth * scoot_dist)

        scene["mouth_pose"] = Pose(
            position=Point(x=p_final_mouth[0], y=p_final_mouth[1], z=p_final_mouth[2]),
            orientation=initial_mouth_orientation,
        )
        scene_characteristics.update(
            self._characterize_pose(scene["mouth_pose"], "mouth_pose")
        )

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
        scene["presentation_pose"], scene["staging_pose"] = (
            self._calculate_presentation_and_staging_poses(scene["mouth_pose"])
        )
        scene_characteristics.update(
            self._characterize_pose(scene["presentation_pose"], "presentation_pose")
        )
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
        forward_vec = rot.apply([0.0, 0.0, 1.0])
        characteristics = {
            f"{pose_name}_dist_3d": np.linalg.norm(pos),
            f"{pose_name}_dist_2d": np.linalg.norm(pos[:2]),
            f"{pose_name}_global_yaw_rad": np.arctan2(forward_vec[1], forward_vec[0]),
            f"{pose_name}_global_pitch_rad": np.arcsin(forward_vec[2]),
        }
        return characteristics

    def _sample_pose_in_spherical_shell(
        self, sampling_params: SphericalSamplingParams
    ) -> Tuple[Point, Dict[str, float]]:
        """
        A general function to sample a POSITION within a spherical shell.
        """
        max_attempts = 1000
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
                sampling_params.max_height is None
                or z <= sampling_params.max_height
                and (
                    sampling_params.min_horizontal_radius is None
                    or np.sqrt(x**2 + y**2) >= sampling_params.min_horizontal_radius
                )
            ):
                position = Point(x=x, y=y, z=z)
                prefix = sampling_params.name
                sampled_values = {
                    f"{prefix}_sampled_radius": r,
                    f"{prefix}_sampled_theta_rad": theta,
                    f"{prefix}_sampled_phi_rad": phi,
                }
                return position, sampled_values
        raise RuntimeError(
            f"Failed to sample a valid pose for '{sampling_params.name}' "
            f"after {max_attempts} attempts. Your constraints may be too strict."
        )

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
        rand_yaw = np.random.uniform(-np.deg2rad(45), np.deg2rad(45))
        final_rot = R.from_matrix(rotation_matrix) * R.from_euler("z", rand_yaw)
        quat = final_rot.as_quat()
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _calculate_jaco_ee_orientation(self, position: Point) -> Quaternion:
        """
        Calculates an orientation matching the Jaco EE convention (Y-up, Z-forward)
        at the given position.
        """
        target_y_axis = np.array([0.0, 0.0, 1.0])
        target_z_axis_dir = np.array([position.x, position.y, 0.0])
        if np.linalg.norm(target_z_axis_dir) < 1e-6:
            target_z_axis_dir = np.array([1.0, 0.0, 0.0])
        target_z_axis = target_z_axis_dir / np.linalg.norm(target_z_axis_dir)
        target_x_axis = np.cross(target_y_axis, target_z_axis)
        rotation_matrix = np.array([target_x_axis, target_y_axis, target_z_axis]).T
        rotation = R.from_matrix(rotation_matrix)
        quat = rotation.as_quat()
        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    def _calculate_above_plate_pose(
        self, food_pose: Pose
    ) -> Tuple[Pose, Dict[str, float]]:
        """
        Calculates a camera pose for the Jaco end-effector that looks at the food.
        """
        food_position = np.array(
            [food_pose.position.x, food_pose.position.y, food_pose.position.z]
        )
        base_yaw_angle = np.arctan2(food_position[1], food_position[0])
        yaw_variability = np.random.uniform(
            -self.params.above_plate_yaw_variability_rad,
            self.params.above_plate_yaw_variability_rad,
        )
        final_azimuthal_angle = base_yaw_angle + yaw_variability + np.pi
        radial_distance = self.params.above_plate_radial_dist
        polar_angle = np.random.uniform(0, self.params.above_plate_polar_angle_rad_max)
        x_offset = radial_distance * np.sin(polar_angle) * np.cos(final_azimuthal_angle)
        y_offset = radial_distance * np.sin(polar_angle) * np.sin(final_azimuthal_angle)
        z_offset = radial_distance * np.cos(polar_angle)
        camera_position = food_position + np.array([x_offset, y_offset, z_offset])

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
        """
        final_rotation = R.identity()
        approach_vector = np.array([0.0, 0.0, -1.0])
        sampled_polar_angle = 0.0
        if recipe.strategy == AcquisitionStrategy.SKEWER:
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
        Calculates the AboveFood pose by offsetting from InFood along the approach vector.
        """
        final_orientation = in_food_pose.orientation
        motion_vector = approach_vector
        offset_dist = self.params.above_food_offset_dist
        p = in_food_pose.position
        offset = motion_vector * -offset_dist
        final_position = Point(x=p.x + offset[0], y=p.y + offset[1], z=p.z + offset[2])
        return Pose(position=final_position, orientation=final_orientation)

    def _calculate_presentation_and_staging_poses(
        self,
        mouth_pose: Pose,
    ) -> Tuple[Pose, Pose]:
        """
        Calculates the final Presentation (tool tip) and Staging (wrist) poses.
        """
        y_axis_new = np.array([0.0, 0.0, 1.0])
        mouth_rot = R.from_quat(
            [
                mouth_pose.orientation.x,
                mouth_pose.orientation.y,
                mouth_pose.orientation.z,
                mouth_pose.orientation.w,
            ]
        )
        mouth_x_axis = mouth_rot.apply([1.0, 0.0, 0.0])
        z_axis_new = -mouth_x_axis
        x_axis_new = np.cross(y_axis_new, z_axis_new)
        rotation_matrix = np.array([x_axis_new, y_axis_new, z_axis_new]).T
        q = R.from_matrix(rotation_matrix).as_quat()
        final_orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        mouth_pos = np.array(
            [mouth_pose.position.x, mouth_pose.position.y, mouth_pose.position.z]
        )
        presentation_pos = mouth_pos + (
            mouth_x_axis * self.params.presentation_offset_dist
        )
        staging_pos = mouth_pos + (mouth_x_axis * self.params.staging_offset_dist)
        presentation_pose = Pose(
            position=Point(
                x=presentation_pos[0], y=presentation_pos[1], z=presentation_pos[2]
            ),
            orientation=final_orientation,
        )
        staging_pose = Pose(
            position=Point(x=staging_pos[0], y=staging_pos[1], z=staging_pos[2]),
            orientation=final_orientation,
        )
        return presentation_pose, staging_pose

    def _calculate_tangential_facing_orientation(
        self,
        mouth_position: Point,
        food_position: Point,
        yaw_variability_rad: float = 0.0,
    ) -> Quaternion:
        """
        Calculates a tangential orientation that is deterministically chosen
        to face towards the food position.
        """
        # Vector from base to the mouth
        v_radial = np.array([mouth_position.x, mouth_position.y, 0.0])
        if np.linalg.norm(v_radial) < 1e-6:
            v_radial = np.array([1.0, 0.0, 0.0])

        # Define one of the two possible tangential vectors
        v_tangent_candidate = np.array([-v_radial[1], v_radial[0], 0.0])
        if np.linalg.norm(v_tangent_candidate) < 1e-6:
            v_tangent_candidate = np.array([0.0, 1.0, 0.0])
        v_tangent_candidate /= np.linalg.norm(v_tangent_candidate)

        # Vector from the mouth to the food
        v_mouth_to_food = np.array(
            [
                food_position.x - mouth_position.x,
                food_position.y - mouth_position.y,
                0.0,
            ]
        )
        if np.linalg.norm(v_mouth_to_food) > 1e-6:
            v_mouth_to_food /= np.linalg.norm(v_mouth_to_food)

        # Use the dot product to choose the tangent vector that faces the food
        if np.dot(v_tangent_candidate, v_mouth_to_food) >= 0:
            chosen_tangent = v_tangent_candidate
        else:
            chosen_tangent = -v_tangent_candidate  # The opposite direction

        # Apply random yaw variability to the chosen direction
        yaw_offset = np.random.uniform(-yaw_variability_rad, yaw_variability_rad)
        R_variability = R.from_euler("z", yaw_offset)
        x_axis = R_variability.apply(chosen_tangent)

        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        quat = R.from_matrix(rotation_matrix).as_quat()

        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
