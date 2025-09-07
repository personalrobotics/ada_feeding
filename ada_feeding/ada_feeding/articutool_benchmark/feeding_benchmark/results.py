# Standard imports
import os
import json
from typing import Optional, List, Dict, Any

# ROS 2 message imports
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory

# Local application imports
from .constants import LOGGER


def serialize_pose(pose: Pose) -> Dict[str, List[float]]:
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


def serialize_trajectory(
    trajectory: Optional[JointTrajectory],
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


def save_trial_data(trial_data: Dict[str, Any], output_filename: str):
    """
    Saves a single trial's data by appending it as a new line
    to the output file.
    """
    if not output_filename:
        return
    try:
        with open(output_filename, "a") as f:
            json_string = json.dumps(trial_data)
            f.write(json_string + "\n")
    except Exception as e:
        LOGGER.error(
            f"Failed to save trial data for trial {trial_data.get('trial_id', 'N/A')}: {e}"
        )
