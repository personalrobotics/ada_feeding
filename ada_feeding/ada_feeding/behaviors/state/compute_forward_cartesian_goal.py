# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the ComputeForwardCartesianGoal behavior.
It computes a new goal pose by translating an input pose along its own
local Z-axis (forward).
"""

from typing import Union, Optional
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation as R
import py_trees
from py_trees.common import Status
from geometry_msgs.msg import Pose, PoseStamped, Point

from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class ComputeForwardCartesianGoal(BlackboardBehavior):
    """
    Calculates a new goal pose by moving a specified distance along the
    input pose's local Z-axis (forward).
    """

    def blackboard_inputs(
        self,
        current_ee_pose: Union[BlackboardKey, Pose, PoseStamped],
        forward_distance_m: Union[BlackboardKey, float],
    ) -> None:
        """Define blackboard inputs."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        cartesian_goal_pose: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> Status:
        """Perform the forward pose calculation."""
        try:
            pose_in = self.blackboard_get("current_ee_pose")
            distance = self.blackboard_get("forward_distance_m")

            current_pose = pose_in.pose if isinstance(pose_in, PoseStamped) else pose_in

            # Extract current position and orientation
            p_current = np.array(
                [
                    current_pose.position.x,
                    current_pose.position.y,
                    current_pose.position.z,
                ]
            )
            q_current = current_pose.orientation

            # Convert orientation to a rotation object
            R_current = R.from_quat(
                [q_current.x, q_current.y, q_current.z, q_current.w]
            )

            # Get the local Z-axis (forward vector) in the world frame
            v_forward = R_current.apply([0.0, 0.0, 1.0])

            # Calculate the new goal position
            p_goal = p_current + v_forward * distance

            # The goal orientation is the same as the original orientation
            goal_pose_msg = Pose(
                position=Point(x=p_goal[0], y=p_goal[1], z=p_goal[2]),
                orientation=q_current,
            )

            self.blackboard_set("cartesian_goal_pose", goal_pose_msg)
            self.feedback_message = "Successfully computed forward Cartesian goal."
            return Status.SUCCESS

        except Exception as e:
            self.feedback_message = f"Failed to compute forward goal: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
