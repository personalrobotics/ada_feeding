# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the ExtractOrientationFromPose behavior.
"""

# Standard imports
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Pose, Quaternion
from overrides import override
import py_trees
from py_trees.common import Status

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ExtractOrientationFromPose(BlackboardBehavior):
    """
    Extracts the geometry_msgs/Quaternion from a geometry_msgs/Pose message.

    This is a utility behavior for when a full pose is available, but only the
    orientation is needed for a subsequent component, such as creating an
    orientation constraint.
    """

    def blackboard_inputs(
        self,
        input_pose: Union[BlackboardKey, Pose],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        input_pose: The source Pose message.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_orientation: Optional[BlackboardKey],  # -> Optional[Quaternion]
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_orientation: The extracted Quaternion message.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> Status:
        """
        Performs the extraction and writes the result to the blackboard.
        """
        try:
            pose_in = self.blackboard_get("input_pose")

            if not isinstance(pose_in, Pose):
                self.logger.error(
                    f"[{self.name}] Input 'input_pose' is not a "
                    f"Pose, but {type(pose_in)}."
                )
                return Status.FAILURE

            # The orientation is a direct attribute of the Pose message.
            extracted_orientation = pose_in.orientation

            self.blackboard_set("output_orientation", extracted_orientation)

            self.logger.info(
                f"[{self.name}] Successfully extracted Quaternion from Pose."
            )
            return Status.SUCCESS

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            return Status.FAILURE
        except Exception as e:
            self.logger.error(f"[{self.name}] Unexpected error: {e}")
            return Status.FAILURE
