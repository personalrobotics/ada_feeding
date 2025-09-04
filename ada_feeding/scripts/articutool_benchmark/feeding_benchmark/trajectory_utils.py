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
    """Generates a synchronized Articutool trajectory that maintains a fixed world orientation."""
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
        jaco_wrist_transform = kinematics_model.get_frame_transform(
            frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_points
        )
        if jaco_wrist_transform is None:
            return None
        R_World_JacoEE = R.from_matrix(jaco_wrist_transform.rotation)
        target_y_in_wrist_frame = R_World_JacoEE.inv().apply(y_axis_TipTarget_InWorld)
        ik_solutions = solve_articutool_ik(target_y_in_wrist_frame)
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
        point = JointTrajectoryPoint(
            positions=pos.tolist(), time_from_start=traj_jaco.points[i].time_from_start
        )
        traj_atool.points.append(point)
    return traj_atool


def generate_leveling_atool_trajectory(
    traj_jaco: JointTrajectory, kinematics_model: PinocchioModel
) -> Optional[JointTrajectory]:
    """Generates a synchronized Articutool trajectory that maintains a level orientation."""
    if not traj_jaco or not traj_jaco.points:
        return None

    atool_solutions = []
    last_valid_solution = None

    for point in traj_jaco.points:
        jaco_joint_config = list(point.positions)
        jaco_wrist_transform = kinematics_model.get_frame_transform(
            frame_name=END_EFFECTOR_LINK_JACO, jaco_joints=jaco_joint_config
        )
        if jaco_wrist_transform is None:
            return None
        q = R.from_matrix(jaco_wrist_transform.rotation).as_quat()
        R_world_jacoee = R.from_quat([q[0], q[1], q[2], q[3]])
        target_up_in_wrist_frame = R_world_jacoee.inv().apply(np.array([0.0, 0.0, 1.0]))
        ik_solutions = solve_articutool_ik(target_up_in_wrist_frame)
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
            chosen_solution = (
                last_valid_solution if last_valid_solution is not None else None
            )
            if chosen_solution is None:
                return None
        elif last_valid_solution is None:
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
