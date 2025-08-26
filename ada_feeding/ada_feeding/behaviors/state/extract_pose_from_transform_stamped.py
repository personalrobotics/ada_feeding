# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the ExtractPoseFromTransformStamped behavior.
"""

# Standard imports
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Pose, TransformStamped
from overrides import override
import py_trees
from py_trees.common import Status

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ExtractPoseFromTransformStamped(BlackboardBehavior):
    """
    Extracts the geometry_msgs/Pose from a geometry_msgs/TransformStamped message.

    This is a utility behavior to bridge components that output TransformStamped
    (like a TF lookup) with components that expect a Pose as input.
    """

    def blackboard_inputs(
        self,
        input_transform_stamped: Union[BlackboardKey, TransformStamped],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        input_transform_stamped: The source TransformStamped message.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_pose: Optional[BlackboardKey],  # -> Optional[Pose]
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_pose: The extracted Pose message.
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
            transform_stamped_in = self.blackboard_get("input_transform_stamped")

            if not isinstance(transform_stamped_in, TransformStamped):
                self.logger.error(
                    f"[{self.name}] Input 'input_transform_stamped' is not a "
                    f"TransformStamped, but {type(transform_stamped_in)}."
                )
                return Status.FAILURE

            # The transform contains a translation (Vector3) and rotation (Quaternion)
            # which together form a Pose.
            extracted_pose = Pose()
            extracted_pose.position.x = transform_stamped_in.transform.translation.x
            extracted_pose.position.y = transform_stamped_in.transform.translation.y
            extracted_pose.position.z = transform_stamped_in.transform.translation.z
            extracted_pose.orientation = transform_stamped_in.transform.rotation

            self.blackboard_set("output_pose", extracted_pose)

            self.logger.info(
                f"[{self.name}] Successfully extracted Pose from TransformStamped."
            )
            return Status.SUCCESS

        except KeyError as e:
            self.logger.error(f"[{self.name}] Blackboard key error: {e}")
            return Status.FAILURE
        except Exception as e:
            self.logger.error(f"[{self.name}] Unexpected error: {e}")
            return Status.FAILURE
