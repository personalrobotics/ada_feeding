# Standard imports
import math
from typing import Optional, List, Dict, Tuple
import numpy as np

# ROS 2 message imports
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation as R

# Local application imports
from .constants import (
    LOGGER,
    EPSILON,
    ARTICUTOOL_PITCH_LIMITS_RAD,
    ARTICUTOOL_ROLL_LIMITS_RAD,
    WORLD_UP_VECTOR,
    END_EFFECTOR_LINK_JACO,
    END_EFFECTOR_LINK_ATOOL,
    PLANNING_GROUP_JACO,
)
from .kinematics import PinocchioModel
from .motion_planner import MotionPlanner


def solve_articutool_ik(target_vector: np.ndarray) -> List[Tuple[float, float]]:
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
                solutions.append((0.0, (theta_r + math.pi) % (2 * math.pi) - math.pi))
            continue
        theta_p = math.atan2(vz, vy)
        solutions.append(
            (
                (theta_p + math.pi) % (2 * math.pi) - math.pi,
                (theta_r + math.pi) % (2 * math.pi) - math.pi,
            )
        )
    return solutions


def compute_leveling_joints(
    jaco_wrist_pose: Pose,
) -> Optional[List[float]]:
    """Calculates the Articutool joint angles required to point the tool's Y-axis up."""
    R_world_jacoee = R.from_quat(
        [
            jaco_wrist_pose.orientation.x,
            jaco_wrist_pose.orientation.y,
            jaco_wrist_pose.orientation.z,
            jaco_wrist_pose.orientation.w,
        ]
    )
    target_up_in_wrist_frame = R_world_jacoee.inv().apply(np.array([0.0, 0.0, 1.0]))
    ik_solutions = solve_articutool_ik(target_up_in_wrist_frame)
    if not ik_solutions:
        return None
    for theta_p, theta_r in ik_solutions:
        if (
            ARTICUTOOL_PITCH_LIMITS_RAD[0] <= theta_p <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
            and ARTICUTOOL_ROLL_LIMITS_RAD[0]
            <= theta_r
            <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
        ):
            return [theta_p, theta_r]
    return None


def is_config_kinematically_feasible(
    jaco_joint_config: List[float], kinematics_model: PinocchioModel
) -> bool:
    """Checks if the Articutool can maintain leveling at a single Jaco configuration."""
    if not kinematics_model.is_ready():
        return False
    ee_transform = kinematics_model.get_frame_transform(
        frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_joint_config
    )
    if ee_transform is None:
        return False
    target_up_in_ee_frame = (
        R.from_matrix(ee_transform.rotation).inv().apply(WORLD_UP_VECTOR)
    )
    solutions = solve_articutool_ik(target_up_in_ee_frame)
    if not solutions:
        return False
    return any(
        ARTICUTOOL_PITCH_LIMITS_RAD[0] <= pitch <= ARTICUTOOL_PITCH_LIMITS_RAD[1]
        and ARTICUTOOL_ROLL_LIMITS_RAD[0] <= roll <= ARTICUTOOL_ROLL_LIMITS_RAD[1]
        for pitch, roll in solutions
    )


def is_elbow_up_configuration(
    jaco_joint_config: List[float], kinematics_model: PinocchioModel
) -> bool:
    """
    Checks if a Jaco arm configuration is "elbow-up" by ensuring the normal
    vector of the arm's plane has a positive component in the world's
    up direction.
    """
    try:
        # Get the positions of the key links that define the arm plane.
        # Link 2 is the shoulder, 3 is the elbow, and 4 is the wrist.
        p_shoulder = kinematics_model.get_frame_transform(
            "j2n6s200_link_2", jaco_joint_config
        ).translation
        p_elbow = kinematics_model.get_frame_transform(
            "j2n6s200_link_3", jaco_joint_config
        ).translation
        p_wrist = kinematics_model.get_frame_transform(
            "j2n6s200_link_4", jaco_joint_config
        ).translation

        # Create vectors for the upper arm and forearm.
        v_upper_arm = p_elbow - p_shoulder
        v_forearm = p_wrist - p_elbow

        # The cross product gives a vector normal to the arm's plane.
        # This vector's direction indicates if the elbow is "up" or "down".
        elbow_normal = np.cross(v_upper_arm, v_forearm)

        # The dot product with the world's up vector directly checks the "up-ness".
        # A positive result means the elbow normal has an upward component.
        return np.dot(elbow_normal, WORLD_UP_VECTOR) > 0

    except Exception as e:
        LOGGER.warning(f"  Elbow-up check failed with exception: {e}")
        return False


def find_optimal_skewer_config(
    above_food_tool_pose: Pose,
    in_food_tool_pose: Pose,
    skewer_polar_angle: float,
    kinematics_model: PinocchioModel,
    motion_planner: MotionPlanner,
) -> Tuple[Optional[Pose], Optional[Pose], Optional[Dict[str, float]]]:
    """Finds an optimal pair of Jaco EE poses for the skewer motion."""
    LOGGER.info("  Searching for optimal skewer configuration...")
    candidate_tilts_rad = [np.deg2rad(t) for t in np.arange(0.0, 50.0, 5.0)]

    for jaco_pitch_tilt in candidate_tilts_rad:
        try:
            required_atool_pitch = (
                (math.pi / 2.0) - skewer_polar_angle - jaco_pitch_tilt
            )
            if not (
                ARTICUTOOL_PITCH_LIMITS_RAD[0] - EPSILON
                <= required_atool_pitch
                <= ARTICUTOOL_PITCH_LIMITS_RAD[1] + EPSILON
            ):
                continue

            atool_config = [required_atool_pitch, 0.0]
            T_wrist_tip = kinematics_model.get_relative_transform(
                parent_frame=END_EFFECTOR_LINK_JACO,
                child_frame=END_EFFECTOR_LINK_ATOOL,
                jaco_joints=[0.0] * 6,
                atool_joints=atool_config,
            )
            if T_wrist_tip is None:
                continue

            q_tip_in_food = in_food_tool_pose.orientation
            R_world_tip = R.from_quat(
                [q_tip_in_food.x, q_tip_in_food.y, q_tip_in_food.z, q_tip_in_food.w]
            )
            R_wrist_tip = R.from_matrix(T_wrist_tip.rotation)
            R_world_wrist_untilted = R_world_tip * R_wrist_tip.inv()
            R_tilt = R.from_euler("x", jaco_pitch_tilt)
            R_world_wrist_target = R_world_wrist_untilted * R_tilt

            p_world_tip_infood = np.array(
                [
                    in_food_tool_pose.position.x,
                    in_food_tool_pose.position.y,
                    in_food_tool_pose.position.z,
                ]
            )
            p_world_wrist_infood = p_world_tip_infood - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )

            q_target = R_world_wrist_target.as_quat()
            final_orientation = Quaternion(
                x=q_target[0], y=q_target[1], z=q_target[2], w=q_target[3]
            )

            in_food_wrist_pose = Pose(
                position=Point(
                    x=p_world_wrist_infood[0],
                    y=p_world_wrist_infood[1],
                    z=p_world_wrist_infood[2],
                ),
                orientation=final_orientation,
            )

            p_world_tip_above = np.array(
                [
                    above_food_tool_pose.position.x,
                    above_food_tool_pose.position.y,
                    above_food_tool_pose.position.z,
                ]
            )
            p_world_wrist_above = p_world_tip_above - R_world_wrist_target.apply(
                T_wrist_tip.translation
            )

            above_food_wrist_pose = Pose(
                position=Point(
                    x=p_world_wrist_above[0],
                    y=p_world_wrist_above[1],
                    z=p_world_wrist_above[2],
                ),
                orientation=final_orientation,
            )

            for _ in range(2):
                ik_solution_msg = motion_planner.compute_ik(
                    group_name=PLANNING_GROUP_JACO, target_pose=above_food_wrist_pose
                )
                if ik_solution_msg and is_elbow_up_configuration(
                    list(ik_solution_msg.position), kinematics_model
                ):
                    winning_angles = {
                        "skewer_polar_angle_rad": skewer_polar_angle,
                        "chosen_jaco_pitch_tilt_rad": jaco_pitch_tilt,
                        "calculated_atool_pitch_rad": required_atool_pitch,
                    }
                    return above_food_wrist_pose, in_food_wrist_pose, winning_angles

        except Exception as e:
            LOGGER.warning(
                f"Exception while checking tilt {np.rad2deg(jaco_pitch_tilt)}: {e}"
            )
            continue

    LOGGER.error("Failed to find any valid skewer configuration.")
    return None, None, None
