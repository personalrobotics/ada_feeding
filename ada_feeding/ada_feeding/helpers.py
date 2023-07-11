"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from typing import Any, List, Optional

# Third-party imports
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class DistanceToGoal(object):
    """
    The DistanceToGoal class is used to determine how much of the trajectory
    the robot arm has yet to execute.

    In practice, it keeps track of what joint state along the trajectory the
    robot is currently in, and returns the number of remaining joint states. As
    a result, this is not technically a measure of either distance or time, but
    should give some intuition of how much of the trajectory is left to execute.
    """

    def __init__(self):
        """
        Initializes the DistanceToGoal class.
        """
        self.joint_names = None
        self.aligned_joint_indices = None

        self.trajectory = None

    def set_joint_names(self, joint_names: List[str]) -> None:
        """
        This function stores the robot's joint names.

        Parameters
        ----------
        joint_names: The names of the joints that the robot arm is moving.
        """
        self.joint_names = joint_names

    def set_trajectory(self, trajectory: JointTrajectory) -> float:
        """
        This function takes in the robot's trajectory and returns the initial
        distance to goal e.g., the distance between the starting and ending
        joint state. In practice, this returns the length of the trajectory.
        """
        self.trajectory = trajectory
        self.curr_joint_state_i = 0
        return float(len(self.trajectory.points))

    def joint_state_callback(self, msg: JointState) -> None:
        """
        This function stores the robot's current joint state, and
        """
        self.curr_joint_state = msg

        if (
            self.aligned_joint_indices is None
            and self.joint_names is not None
            and self.trajectory is not None
        ):
            # Align the joint names between the JointState and JointTrajectory
            # messages.
            self.aligned_joint_indices = []
            for joint_name in self.joint_names:
                if joint_name in msg.name and joint_name in self.trajectory.joint_names:
                    joint_state_i = msg.name.index(joint_name)
                    joint_traj_i = self.trajectory.joint_names.index(joint_name)
                    self.aligned_joint_indices.append(
                        (joint_name, joint_state_i, joint_traj_i)
                    )

    def get_distance(self) -> Optional[float]:
        """
        This function determines where in the trajectory the robot is. It does
        this by computing the distance (L1 distance across the joint positions)
        between the current joint state and the upcoming joint states in the
        trajectory, and selecting the nearest local min.

        This function assumes the joint names are aligned between the JointState
        and JointTrajectory messages.
        """
        # If we haven't yet received a joint state message to the trajectory,
        # immediately return
        if self.aligned_joint_indices is None:
            if self.trajectory is None:
                return None
            else:
                return float(len(self.trajectory.points) - self.curr_joint_state_i)

        # Else, determine how much remaining the robot has of the trajectory
        prev_dist = None
        for i in range(self.curr_joint_state_i, len(self.trajectory.points)):
            # Compute the distance between the current joint state and the
            # ujoint state at index i.
            traj_joint_state = self.trajectory.points[i]
            dist = sum(
                [
                    abs(
                        self.curr_joint_state.position[joint_state_i]
                        - traj_joint_state.positions[joint_traj_i]
                    )
                    for (_, joint_state_i, joint_traj_i) in self.aligned_joint_indices
                ]
            )

            # If the distance is increasing, we've found the local min.
            if prev_dist is not None:
                if dist >= prev_dist:
                    self.curr_joint_state_i = i - 1
                    return float(len(self.trajectory.points) - self.curr_joint_state_i)

            prev_dist = dist

        # If the distance never increased, we are nearing the final waypoint.
        # Because the robot may still have slight motion even after this point,
        # we conservatively return 1.0.
        return 1.0


def import_from_string(import_string: str) -> Any:
    """
    Imports a module from a string.

    Parameters
    ----------
    import_string: The string to import, e.g., "ada_feeding_msgs.action.MoveTo"
        This should be in the format "package.module.class"

    Returns
    -------
    module: The imported module.

    Raises
    ------
    NameError: If the import string is invalid.
    ImportError: If the module cannot be imported.
    """
    try:
        import_package, import_module, import_class = import_string.split(".", 3)
    except Exception as exc:
        raise NameError(
            'Invalid import string %s. Except "package.module.class" e.g., "ada_feeding_msgs.action.MoveTo"'
            % import_string
        ) from exc
    try:
        return getattr(
            getattr(
                __import__("%s.%s" % (import_package, import_module)),
                import_module,
            ),
            import_class,
        )
    except Exception as exc:
        raise ImportError("Error importing %s" % (import_string)) from exc
