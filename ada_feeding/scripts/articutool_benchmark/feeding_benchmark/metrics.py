# Standard imports
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# ROS 2 message imports
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory
from scipy.spatial.transform import Rotation as R

# Local application imports
from .constants import (
    LOGGER,
    PLANNING_GROUP_JACO,
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    END_EFFECTOR_LINK_FULL,
)
from .kinematics import PinocchioModel
from .kinematic_solvers import is_config_kinematically_feasible
from .motion_planner import MoveIt2ConstraintType


def calculate_cartesian_path_length(
    trajectory: Optional[JointTrajectory],
    group_name: str,
    kinematics_model: PinocchioModel,
    ee_link_name_jaco: str,
) -> float:
    """Calculates the Cartesian path length of the end-effector for a trajectory."""
    if not trajectory or not trajectory.points or len(trajectory.points) < 2:
        return 0.0

    ee_link = (
        ee_link_name_jaco
        if group_name == PLANNING_GROUP_JACO
        else END_EFFECTOR_LINK_FULL
    )

    if not kinematics_model.is_ready() or not kinematics_model.model.existFrame(
        ee_link
    ):
        LOGGER.error(f"Cannot calculate path length for frame '{ee_link}'.")
        return 0.0

    total_length = 0.0
    last_position = None
    joint_names = trajectory.joint_names

    for point in trajectory.points:
        joint_map = dict(zip(joint_names, point.positions))
        jaco_config = [joint_map.get(name, 0.0) for name in JOINT_NAMES_JACO]
        atool_config = [joint_map.get(name, 0.0) for name in JOINT_NAMES_ATOOL]
        transform = kinematics_model.get_frame_transform(
            ee_link, jaco_config, atool_config
        )
        if transform is None:
            continue
        current_position = transform.translation
        if last_position is not None:
            total_length += np.linalg.norm(current_position - last_position)
        last_position = current_position

    return total_length


def calculate_total_joint_travel(trajectory: Optional[JointTrajectory]) -> float:
    """Calculates the sum of absolute angular distance traveled by all joints."""
    if not trajectory or not trajectory.points or len(trajectory.points) < 2:
        return 0.0

    total_travel = 0.0
    last_positions = np.array(trajectory.points[0].positions)

    for i in range(1, len(trajectory.points)):
        current_positions = np.array(trajectory.points[i].positions)
        delta = np.sum(np.abs(current_positions - last_positions))
        total_travel += delta
        last_positions = current_positions

    return total_travel


def verify_trajectory(
    trajectory: JointTrajectory, kinematics_model: PinocchioModel
) -> Dict[str, Any]:
    """
    Verifies a trajectory and returns a dictionary of feasibility metrics.
    """
    if not trajectory or not trajectory.points:
        return {
            "feasible_percent": 0.0,
            "kinematic_failures": 0,
            "total_waypoints": 0,
        }

    feasible_waypoints = 0
    for point in trajectory.points:
        jaco_joint_config = list(point.positions)
        if is_config_kinematically_feasible(jaco_joint_config, kinematics_model):
            feasible_waypoints += 1

    total_waypoints = len(trajectory.points)
    feasible_percent = (
        (feasible_waypoints / total_waypoints) * 100.0 if total_waypoints > 0 else 0.0
    )

    return {
        "feasible_percent": feasible_percent,
        "kinematic_failures": total_waypoints - feasible_waypoints,
        "total_waypoints": total_waypoints,
    }


def compute_moveit_orientation_error(
    current_pose: Pose,
    orientation_constraint: Tuple[Any, Dict[str, Any]],
) -> Dict[str, Any]:
    """Computes the orientation error as interpreted by a MoveIt2 path constraint."""
    constraint_type, constraint_dict = orientation_constraint
    if constraint_type != MoveIt2ConstraintType.ORIENTATION:
        raise ValueError("Provided constraint is not an orientation constraint.")

    q_current = current_pose.orientation
    R_current = R.from_quat([q_current.x, q_current.y, q_current.z, q_current.w])
    q_target = constraint_dict["quat_xyzw"]
    R_target = R.from_quat([q_target.x, q_target.y, q_target.z, q_target.w])
    R_error = R_target.inv() * R_current
    euler_error_rad = R_error.as_euler("xyz")
    pitch_error_rad, yaw_error_rad, roll_error_rad = euler_error_rad
    tolerances_rad = constraint_dict["tolerance"]
    pitch_tol_rad, yaw_tol_rad, roll_tol_rad = tolerances_rad
    pitch_ok = abs(pitch_error_rad) <= pitch_tol_rad
    yaw_ok = abs(yaw_error_rad) <= yaw_tol_rad
    roll_ok = abs(roll_error_rad) <= roll_tol_rad

    return {
        "is_satisfied": pitch_ok and yaw_ok and roll_ok,
        "details": {
            "pitch (x)": {
                "error_rad": pitch_error_rad,
                "tolerance_rad": pitch_tol_rad,
                "satisfied": pitch_ok,
            },
            "yaw (y)": {
                "error_rad": yaw_error_rad,
                "tolerance_rad": yaw_tol_rad,
                "satisfied": yaw_ok,
            },
            "roll (z)": {
                "error_rad": roll_error_rad,
                "tolerance_rad": roll_tol_rad,
                "satisfied": roll_ok,
            },
        },
    }
