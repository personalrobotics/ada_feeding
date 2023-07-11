"""
This module defines a number of helper functions that are reused throughout the
Ada Feeding project.
"""

# Standard imports
from typing import Any, Optional

# Third-party imports
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

class CartesianDistanceToGoal(object):
    """
    The CartesianDistanceToGoal class is used to compute the cartesian distance
    between the robot end effector's current position and it's goal position.
    """

    def __init__(self):
        """
        Initializes the CartesianDistanceToGoal class.
        """
        pass

    def set_trajectory(self, trajectory: JointTrajectory) -> None:
        """
        This function takes in a trajectory and initiates an asynchronous call
        to compute the forward kinematics of the trajectory's goal position.
        """
        pass

    def get_goal_end_effector_pose(self) -> Optional[PoseStamped]:
        """
        This function returns the end effector position at the goal of the
        trajectory, if it has been computed, and None otherwise.
        """
        pass

    def joint_state_callback(self, msg: JointState) -> None:
        """
        This function takes in the robot's current joint state. If there is
        an unfinished asynchrcall to forward kinematics, it ignores the current
        joint state. Otherwise, it stores the results of the previous forward
        kinematics call and starts a new asynchronous forward kinematics call.
        """
        pass

    def get_latest_end_effector_pose(self) -> Optional[PoseStamped]:
        """
        This function returns the end effector position at the latest joint
        state, if it has been computed, and None otherwise.
        """
        pass

    def get_distance(self) -> float:
        """
        This function returns the cartesian distance between the latest end
        effector pose and its goal pose.
        """
        pass


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
