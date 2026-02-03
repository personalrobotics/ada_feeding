# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from typing import Union
from overrides import override

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
import py_trees
from py_trees.common import Status

from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey, set_static_tf


class PublishPoseAsTf(BlackboardBehavior):
    """
    A simple debugging behavior that reads a Pose or PoseStamped from the
    blackboard and publishes it as a static TF transform.
    """

    def blackboard_inputs(
        self,
        pose_to_publish: Union[BlackboardKey, Pose, PoseStamped],
        child_frame_id: Union[BlackboardKey, str],
        parent_frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
    ) -> None:
        """Define blackboard inputs for this behavior."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        self.node = kwargs["node"]

    @override
    def update(self) -> Status:
        """
        Constructs a TransformStamped message and publishes it.
        """
        try:
            pose_in = self.blackboard_get("pose_to_publish")
            child_frame = self.blackboard_get("child_frame_id")
            parent_frame = self.blackboard_get("parent_frame_id")

            # Handle both Pose and PoseStamped inputs
            if isinstance(pose_in, PoseStamped):
                pose = pose_in.pose
                # Optional: Could also use pose_in.header.frame_id as parent
            else:
                pose = pose_in

            # Construct the TransformStamped message
            tfs = TransformStamped()
            tfs.header.stamp = self.node.get_clock().now().to_msg()
            tfs.header.frame_id = parent_frame
            tfs.child_frame_id = child_frame
            tfs.transform.translation.x = pose.position.x
            tfs.transform.translation.y = pose.position.y
            tfs.transform.translation.z = pose.position.z
            tfs.transform.rotation = pose.orientation

            # Use the existing helper to publish the static transform
            set_static_tf(tfs, self.blackboard, self.node)

            self.feedback_message = f"Published TF for frame '{child_frame}'"
            return Status.SUCCESS

        except Exception as e:
            self.feedback_message = f"Failed to publish TF: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
