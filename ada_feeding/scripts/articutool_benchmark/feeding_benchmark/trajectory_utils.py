# Standard imports
from typing import Optional, List, Tuple
import numpy as np

# ROS 2 message imports
from geometry_msgs.msg import Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from scipy.spatial.transform import Rotation as R

# Local application imports
from .constants import (
    LOGGER,
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    END_EFFECTOR_LINK_JACO,
    END_EFFECTOR_LINK_ATOOL,
    ARTICUTOOL_PITCH_LIMITS_RAD,
    ARTICUTOOL_ROLL_LIMITS_RAD,
)
from .kinematics import PinocchioModel
from .kinematic_solvers import solve_articutool_ik


def generate_orientation_holding_atool_trajectory(
    traj_jaco: JointTrajectory,
    desired_tool_tip_world_orientation: Quaternion,
    kinematics_model: PinocchioModel,
) -> Optional[JointTrajectory]:
    """
    Generates a synchronized Articutool trajectory that maintains a fixed world orientation.
    This version includes an FK check to resolve IK ambiguity and prevent flipped solutions.
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
    # The "up" vector (local Y) of the target tool pose, expressed in the world frame.
    y_axis_TipTarget_InWorld = R_World_TipTarget.apply(np.array([0.0, 1.0, 0.0]))

    atool_solutions = []
    last_valid_solution = None

    for point in traj_jaco.points:
        jaco_config = list(point.positions)
        jaco_wrist_transform = kinematics_model.get_frame_transform(
            frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_config
        )
        if jaco_wrist_transform is None:
            LOGGER.error("FK failed for Jaco wrist at a waypoint.")
            return None

        R_World_JacoEE = R.from_matrix(jaco_wrist_transform.rotation)
        target_y_in_wrist_frame = R_World_JacoEE.inv().apply(y_axis_TipTarget_InWorld)
        ik_solutions = solve_articutool_ik(target_y_in_wrist_frame)

        limit_valid_solutions = [
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

        if not limit_valid_solutions:
            LOGGER.warning("No IK solution found within joint limits for a waypoint.")
            return None

        # --- Forward Kinematics Validation to Resolve Ambiguity ---
        orientation_valid_solutions = []
        for sol in limit_valid_solutions:
            T_JacoEE_Tip = kinematics_model.get_relative_transform(
                parent_frame=END_EFFECTOR_LINK_JACO,
                child_frame=END_EFFECTOR_LINK_ATOOL,
                jaco_joints=jaco_config,
                atool_joints=sol.tolist(),
            )
            if T_JacoEE_Tip is None:
                continue

            R_World_Tip_Candidate = R_World_JacoEE * R.from_matrix(
                T_JacoEE_Tip.rotation
            )
            y_axis_candidate_InWorld = R_World_Tip_Candidate.apply(
                np.array([0.0, 1.0, 0.0])
            )

            if np.dot(y_axis_candidate_InWorld, y_axis_TipTarget_InWorld) > 0:
                orientation_valid_solutions.append(sol)

        if not orientation_valid_solutions:
            LOGGER.warning("No correctly-oriented IK solution found for a waypoint.")
            return None

        if last_valid_solution is None:
            chosen_solution = orientation_valid_solutions[0]
        else:
            distances = [
                np.linalg.norm(sol - last_valid_solution)
                for sol in orientation_valid_solutions
            ]
            chosen_solution = orientation_valid_solutions[np.argmin(distances)]

        atool_solutions.append(chosen_solution)
        last_valid_solution = chosen_solution

    traj_atool = JointTrajectory()
    traj_atool.joint_names = JOINT_NAMES_ATOOL
    for i, pos in enumerate(atool_solutions):
        point = JointTrajectoryPoint(
            positions=pos.tolist(), time_from_start=traj_jaco.points[i].time_from_start
        )
        traj_atool.points.append(point)
    return traj_atool


def generate_leveling_atool_trajectory(
    traj_jaco: JointTrajectory, kinematics_model: PinocchioModel
) -> Optional[JointTrajectory]:
    """
    Generates a synchronized Articutool trajectory that keeps the tool's y-axis
    level with gravity (aligned with world Z). This version includes an FK
    check to resolve IK ambiguity.
    """
    if not traj_jaco or not traj_jaco.points:
        return None

    # The target "up" vector for leveling is the world's Z-axis (anti-gravity).
    y_axis_Target_InWorld = np.array([0.0, 0.0, 1.0])
    atool_solutions = []
    last_valid_solution = None

    for point in traj_jaco.points:
        jaco_config = list(point.positions)
        jaco_wrist_transform = kinematics_model.get_frame_transform(
            frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_config
        )
        if jaco_wrist_transform is None:
            return None

        R_World_JacoEE = R.from_matrix(jaco_wrist_transform.rotation)
        target_up_in_wrist_frame = R_World_JacoEE.inv().apply(y_axis_Target_InWorld)
        ik_solutions = solve_articutool_ik(target_up_in_wrist_frame)

        limit_valid_solutions = [
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

        if not limit_valid_solutions:
            LOGGER.warning("No leveling solution within joint limits for waypoint.")
            return None

        # --- Forward Kinematics Validation to Resolve Ambiguity ---
        orientation_valid_solutions = []
        for sol in limit_valid_solutions:
            T_JacoEE_Tip = kinematics_model.get_relative_transform(
                parent_frame=END_EFFECTOR_LINK_JACO,
                child_frame=END_EFFECTOR_LINK_ATOOL,
                jaco_joints=jaco_config,
                atool_joints=sol.tolist(),
            )
            if T_JacoEE_Tip is None:
                continue

            R_World_Tip_Candidate = R_World_JacoEE * R.from_matrix(
                T_JacoEE_Tip.rotation
            )
            y_axis_candidate_InWorld = R_World_Tip_Candidate.apply(
                np.array([0.0, 1.0, 0.0])
            )

            if np.dot(y_axis_candidate_InWorld, y_axis_Target_InWorld) > 0:
                orientation_valid_solutions.append(sol)

        if not orientation_valid_solutions:
            chosen_solution = (
                last_valid_solution if last_valid_solution is not None else None
            )
            if chosen_solution is None:
                LOGGER.warning("No valid leveling solution found for initial waypoint.")
                return None
        elif last_valid_solution is None:
            chosen_solution = orientation_valid_solutions[0]
        else:
            distances = [
                np.linalg.norm(sol - last_valid_solution)
                for sol in orientation_valid_solutions
            ]
            chosen_solution = orientation_valid_solutions[np.argmin(distances)]

        atool_solutions.append(chosen_solution)
        last_valid_solution = chosen_solution

    traj_atool = JointTrajectory()
    traj_atool.joint_names = JOINT_NAMES_ATOOL
    for i, pos in enumerate(atool_solutions):
        point_msg = JointTrajectoryPoint(
            positions=pos.tolist(), time_from_start=traj_jaco.points[i].time_from_start
        )
        traj_atool.points.append(point_msg)
    return traj_atool


def generate_hold_trajectory(
    reference_traj: JointTrajectory,
    joint_names: List[str],
    joint_positions: List[float],
) -> JointTrajectory:
    """Creates a trajectory that holds a fixed position."""
    hold_traj = JointTrajectory()
    hold_traj.joint_names = joint_names
    num_joints = len(joint_names)
    for point in reference_traj.points:
        new_point = JointTrajectoryPoint(
            positions=joint_positions,
            velocities=[0.0] * num_joints,
            accelerations=[0.0] * num_joints,
            time_from_start=point.time_from_start,
        )
        hold_traj.points.append(new_point)
    return hold_traj


def split_full_trajectory(
    traj_full: Optional[JointTrajectory],
) -> Tuple[Optional[JointTrajectory], Optional[JointTrajectory]]:
    """Splits an 8-DOF trajectory into Jaco and Articutool trajectories."""
    if traj_full is None:
        return None, None

    traj_jaco = JointTrajectory(joint_names=JOINT_NAMES_JACO)
    traj_atool = JointTrajectory(joint_names=JOINT_NAMES_ATOOL)

    try:
        full_joint_names = traj_full.joint_names
        jaco_indices = [full_joint_names.index(name) for name in JOINT_NAMES_JACO]
        atool_indices = [full_joint_names.index(name) for name in JOINT_NAMES_ATOOL]
    except ValueError as e:
        LOGGER.error(f"Error mapping joint names during trajectory split: {e}")
        return None, None

    for point_full in traj_full.points:
        point_jaco = JointTrajectoryPoint(time_from_start=point_full.time_from_start)
        point_atool = JointTrajectoryPoint(time_from_start=point_full.time_from_start)
        point_jaco.positions = [point_full.positions[i] for i in jaco_indices]
        point_atool.positions = [point_full.positions[i] for i in atool_indices]
        if point_full.velocities:
            point_jaco.velocities = [point_full.velocities[i] for i in jaco_indices]
            point_atool.velocities = [point_full.velocities[i] for i in atool_indices]
        if point_full.accelerations:
            point_jaco.accelerations = [
                point_full.accelerations[i] for i in jaco_indices
            ]
            point_atool.accelerations = [
                point_full.accelerations[i] for i in atool_indices
            ]
        traj_jaco.points.append(point_jaco)
        traj_atool.points.append(point_atool)

    return traj_jaco, traj_atool
