# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the OffsetPositionFromPose behavior, which calculates a new 3D point
by applying a vector offset to the position component of an input pose.
"""

# Standard imports
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Pose, PoseStamped, Point, Vector3
import numpy as np
from overrides import override
import py_trees
from py_trees.common import Status
import ros2_numpy

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class OffsetPositionFromPose(BlackboardBehavior):
    """
    Calculates a new position by applying a Vector3 offset to the position
    of an input Pose or PoseStamped. This is useful for creating a target
    point that is relative to another object's frame.
    """

    def blackboard_inputs(
        self,
        input_pose: Union[BlackboardKey, Pose, PoseStamped],
        offset: Union[BlackboardKey, Vector3],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        input_pose: The source pose from which to take the initial position.
        offset: The Vector3 offset to add to the pose's position.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_position: Optional[BlackboardKey],  # -> Optional[Point]
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_position: The resulting position (as a geometry_msgs/Point)
                         after applying the offset.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> Status:
        """
        Performs the position extraction, offset addition, and writes
        the result to the blackboard.
        """
        try:
            pose_in = self.blackboard_get("input_pose")
            offset_in = self.blackboard_get("offset")

            # Extract the position part of the input pose
            if isinstance(pose_in, PoseStamped):
                position_np = ros2_numpy.numpify(pose_in.pose.position)
            elif isinstance(pose_in, Pose):
                position_np = ros2_numpy.numpify(pose_in.position)
            else:
                self.logger.error(
                    f"[{self.name}] Input 'input_pose' is not a Pose or PoseStamped, but {type(pose_in)}."
                )
                return Status.FAILURE

            # Convert offset to numpy array
            offset_np = ros2_numpy.numpify(offset_in)

            # Calculate the new position
            new_position_np = position_np + offset_np

            # Convert back to a ROS message and write to blackboard
            new_position_msg = ros2_numpy.msgify(Point, new_position_np)
            self.blackboard_set("output_position", new_position_msg)

            self.logger.info(f"[{self.name}] Successfully calculated offset position.")
            return Status.SUCCESS

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            return Status.FAILURE
        except Exception as e:
            self.logger.error(f"[{self.name}] Unexpected error: {e}")
            return Status.FAILURE
