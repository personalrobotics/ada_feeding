# Standard imports
from typing import Union, Optional
from overrides import override

# Third-party imports
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Vector3
import py_trees
from py_trees.common import Status
from scipy.spatial.transform import Rotation

# Local BT Imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class ComputeLevelPose(BlackboardBehavior):
    """
    Computes a new goal pose by first leveling the orientation of an initial
    pose (zeroing roll and pitch while preserving yaw) and then applying a
    Cartesian offset within the new, level frame.
    """

    def blackboard_inputs(
        self,
        current_pose: Union[BlackboardKey, PoseStamped],
        offset: Union[BlackboardKey, Vector3],
    ) -> None:
        """Define blackboard inputs for this behavior."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        goal_pose: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs for this behavior."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> Status:
        """
        Perform the pose computation.
        """
        try:
            # 1. Get inputs using the BlackboardBehavior helper
            current_pose_stamped: PoseStamped = self.blackboard_get("current_pose")
            offset: Vector3 = self.blackboard_get("offset")

            # Extract the current orientation from the input pose
            q_current = current_pose_stamped.pose.orientation

            # 2. Compute the "level" orientation
            # Convert the orientation quaternion to a scipy Rotation object
            rot_current = Rotation.from_quat(
                [q_current.x, q_current.y, q_current.z, q_current.w]
            )
            # Decompose into Euler angles to isolate the yaw
            _roll, _pitch, yaw = rot_current.as_euler("xyz", degrees=False)

            # Create the Z-up orientation with the correct yaw
            rot_z_up_level = Rotation.from_euler("xyz", [0, 0, yaw], degrees=False)

            # Create the fixed corrective rotation to tilt to a Y-up convention (+90 deg about local X-axis)
            rot_correction = Rotation.from_euler("x", 90, degrees=True)

            # Compose them to get the final, correct level orientation
            rot_level = rot_z_up_level * rot_correction

            # 3. Apply the Cartesian offset relative to the new level frame
            # Convert the offset from a ROS Vector3 to a NumPy array
            offset_vec = np.array([offset.x, offset.y, offset.z])
            # Rotate the offset vector by the new level orientation to get the world-frame translation
            rotated_offset = rot_level.apply(offset_vec)

            # Extract the current position
            p_current = current_pose_stamped.pose.position
            # Calculate the new goal position by adding the rotated offset
            p_goal_np = (
                np.array([p_current.x, p_current.y, p_current.z]) + rotated_offset
            )

            # 4. Assemble the final goal pose
            goal_pose = Pose()
            goal_pose.position.x = p_goal_np[0]
            goal_pose.position.y = p_goal_np[1]
            goal_pose.position.z = p_goal_np[2]

            q_level_xyzw = rot_level.as_quat()
            goal_pose.orientation.x = q_level_xyzw[0]
            goal_pose.orientation.y = q_level_xyzw[1]
            goal_pose.orientation.z = q_level_xyzw[2]
            goal_pose.orientation.w = q_level_xyzw[3]

            # Write the result using the BlackboardBehavior helper
            self.blackboard_set("goal_pose", goal_pose)
            self.feedback_message = "Successfully computed level pose."

            return Status.SUCCESS

        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to compute pose: {e}")
            return Status.FAILURE
