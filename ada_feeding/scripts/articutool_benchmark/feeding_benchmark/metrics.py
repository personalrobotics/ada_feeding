# Standard imports
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pinocchio as pin

# ROS 2 message imports
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory
from scipy.spatial.transform import Rotation as R

# Local application imports
from .constants import (
    LOGGER,
    PLANNING_GROUP_JACO,
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    END_EFFECTOR_LINK_JACO,
    END_EFFECTOR_LINK_FULL,
    ARTICUTOOL_MAX_VELOCITY_RAD_S,
)
from .kinematics import PinocchioModel
from .kinematic_solvers import (
    is_config_kinematically_feasible,
    compute_leveling_joints,
    get_articutool_jacobian,
)
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
    traj_jaco: JointTrajectory, kinematics_model: PinocchioModel
) -> Dict:
    """
    Verifies a Jaco trajectory for both kinematic and dynamic feasibility
    for the Articutool to maintain a level orientation.
    """
    results = {
        "feasible_percent": 0.0,
        "is_kinematically_feasible": False,
        "is_dynamically_feasible": False,
        "max_required_velocity_rad_s": 0.0,
    }
    if not traj_jaco or not traj_jaco.points:
        return results

    kinematically_feasible_waypoints = 0
    max_required_velocity = 0.0
    dynamic_check_possible = True

    for point in traj_jaco.points:
        jaco_config = list(point.positions)
        jaco_wrist_transform = kinematics_model.get_frame_transform(
            END_EFFECTOR_LINK_JACO, jaco_config
        )
        if jaco_wrist_transform is None:
            continue

        # Convert the pin.SE3 object to a geometry_msgs.msg.Pose object
        translation = jaco_wrist_transform.translation
        rotation_quat = R.from_matrix(jaco_wrist_transform.rotation).as_quat()

        jaco_wrist_pose = Pose()
        jaco_wrist_pose.position = Point(
            x=translation[0], y=translation[1], z=translation[2]
        )
        jaco_wrist_pose.orientation = Quaternion(
            x=rotation_quat[0],
            y=rotation_quat[1],
            z=rotation_quat[2],
            w=rotation_quat[3],
        )
        # 1. Kinematic Check (can it be level at this pose?)
        leveling_solution = compute_leveling_joints(jaco_wrist_pose)
        if leveling_solution is None:
            continue
        kinematically_feasible_waypoints += 1
        pitch, roll = leveling_solution

        # 2. Dynamic Check (can it move fast enough to stay level?)
        if not point.velocities:
            dynamic_check_possible = False
            continue

        q_dot_jaco = np.array(point.velocities)
        J_local = kinematics_model.get_frame_jacobian(
            frame_name=END_EFFECTOR_LINK_JACO,
            jaco_joints=jaco_config,
            group=PLANNING_GROUP_JACO,
            reference_frame=pin.ReferenceFrame.LOCAL,
        )
        if J_local is None:
            dynamic_check_possible = False
            continue

        # Angular velocity disturbance in the Jaco EE's local frame
        omega_disturbance_local = J_local[3:6, :] @ q_dot_jaco

        # Solve for required Articutool velocities
        J_atool_inv = np.linalg.pinv(get_articutool_jacobian(pitch, roll))
        q_dot_atool = -J_atool_inv @ omega_disturbance_local

        required_velocity_norm = np.linalg.norm(q_dot_atool)
        max_required_velocity = max(max_required_velocity, required_velocity_norm)

    # Finalize results
    total_waypoints = len(traj_jaco.points)
    results["feasible_percent"] = (
        kinematically_feasible_waypoints / total_waypoints
    ) * 100
    results["is_kinematically_feasible"] = bool(results["feasible_percent"] > 99.0)

    if not dynamic_check_possible:
        LOGGER.warn(
            "Could not perform dynamic check; trajectory missing velocity data."
        )
        results["is_dynamically_feasible"] = False
    else:
        results["is_dynamically_feasible"] = bool(
            max_required_velocity <= ARTICUTOOL_MAX_VELOCITY_RAD_S
        )

    results["max_required_velocity_rad_s"] = max_required_velocity

    return results


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
