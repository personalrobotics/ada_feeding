#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines behaviors that manipulate the planning scene during bite
transfer.
"""

# Standard imports
from typing import Optional, Union

# Third-party imports
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    Quaternion,
    QuaternionStamped,
)
from overrides import override
import py_trees
from std_msgs.msg import Header

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class ComputeWheelchairCollisionTransform(BlackboardBehavior):
    """
    A behavior that takes in the detected head pose, the original head pose,
    and the original wheelchair_collision pose, and computes the new
    wheelchair_collision pose and scale.

    NOTE: Although this class is in theory rich enough to compute arbitrary
    updates to the wheelchair collision object, e.g., moving its x and y to be
    centered on the face, currently we don't translate it at all and only scale
    z in (0.0, inf). This is because incorporating translations may bring the
    wheelchair collision object into collision with the robot base, which is not desirable.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    def blackboard_inputs(
        self,
        detected_head_pose: Union[BlackboardKey, PoseStamped],
        # NOTE: The below parameters must match the ones in `ada_planning_scene.yaml`
        original_head_pose: Union[BlackboardKey, PoseStamped] = PoseStamped(
            header=Header(frame_id="j2n6s200_link_base"),
            pose=Pose(
                position=Point(x=0.29, y=0.35, z=0.85),
                orientation=Quaternion(
                    x=-0.0616284, y=-0.0616284, z=-0.704416, w=0.704416
                ),
            ),
        ),
        original_wheelchair_collision_pose: Union[
            BlackboardKey, PoseStamped
        ] = PoseStamped(
            header=Header(frame_id="j2n6s200_link_base"),
            pose=Pose(
                position=Point(x=0.02, y=-0.02, z=-0.05),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        detected_head_pose: The detected head pose in the planning scene.
        original_head_pose: The initial head pose in the planning scene.
        original_wheelchair_collision_pose: The initial wheelchair_collision
            pose in the planning scene.
        """
        # pylint: disable=unused-argument, duplicate-code, too-many-arguments
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        wheelchair_position: Optional[BlackboardKey] = None,  # PointStamped
        wheelchair_orientation: Optional[BlackboardKey] = None,  # QuaternionStamped
        wheelchair_scale: Optional[BlackboardKey] = None,  # Tuple[float, float, float]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        wheelchair_position: The new position of the wheelchair collision object.
        wheelchair_orientation: The new orientation of the wheelchair collision object.
        wheelchair_scale: The new scale of the wheelchair collision object.
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        self.logger.debug(
            f"{self.name} [ComputeWheelchairCollisionTransform::update()]"
        )

        # Validate inputs
        if not self.blackboard_exists(
            [
                "detected_head_pose",
                "original_head_pose",
                "original_wheelchair_collision_pose",
            ]
        ):
            self.logger.error("Missing input arguments")
            return py_trees.common.Status.FAILURE

        # Get the inputs
        detected_head_pose = self.blackboard_get("detected_head_pose")
        original_head_pose = self.blackboard_get("original_head_pose")
        original_wheelchair_collision_pose = self.blackboard_get(
            "original_wheelchair_collision_pose"
        )

        # Verify that they all have the same frame_id
        if not (
            detected_head_pose.header.frame_id
            == original_head_pose.header.frame_id
            == original_wheelchair_collision_pose.header.frame_id
        ):
            self.logger.error("All input poses must have the same frame_id")
            return py_trees.common.Status.FAILURE

        # Compute the new wheelchair collision pose
        wheelchair_position = PointStamped(
            header=detected_head_pose.header,
            point=original_wheelchair_collision_pose.pose.position,
        )
        wheelchair_orientation = QuaternionStamped(
            header=detected_head_pose.header,
            quaternion=original_wheelchair_collision_pose.pose.orientation,
        )
        wheelchair_scale = (
            1.0,
            1.0,
            (
                detected_head_pose.pose.position.z
                - original_wheelchair_collision_pose.pose.position.z
            )
            / (
                original_head_pose.pose.position.z
                - original_wheelchair_collision_pose.pose.position.z
            ),
        )

        # Set the outputs
        self.blackboard_set("wheelchair_position", wheelchair_position)
        self.blackboard_set("wheelchair_orientation", wheelchair_orientation)
        self.blackboard_set("wheelchair_scale", wheelchair_scale)

        # Return success
        return py_trees.common.Status.SUCCESS
