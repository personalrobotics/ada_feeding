# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import py_trees
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from typing import Any, Dict, List, Optional, Union
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class CreatePoseStampedFromOrientation(BlackboardBehavior):
    """
    Converts an orientation (as a quaternion list) into a full PoseStamped
    message with a default (0,0,0) position. This is a utility for
    visualizing orientation constraints with behaviors like PublishPoseAsTf.
    """

    def __init__(
        self,
        name: str,
        ns: str = "/",
        inputs: Optional[Dict[str, Union[BlackboardKey, Any]]] = None,
        outputs: Optional[Dict[str, Optional[BlackboardKey]]] = None,
    ):
        super().__init__(name, ns=ns, inputs=inputs, outputs=outputs)
        self.node = None
        self.feedback_message = "CreatePoseStampedFromOrientation: "

    def blackboard_inputs(
        self,
        orientation_quat: Union[BlackboardKey, List[float]] = None,
        frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        orientation_quat: The [x, y, z, w] quaternion list.
        frame_id: The TF frame this pose is relative to.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        output_pose_stamped: Union[BlackboardKey, PoseStamped] = None,
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        output_pose_stamped: The resulting PoseStamped message.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        try:
            self.node = kwargs["node"]
        except KeyError:
            self.node = None
            self.feedback_message = "no 'node' in setup's kwargs"

    def update(self):
        if self.node is None:
            py_trees.logging.Logger(self.name).error(self.feedback_message)
            return py_trees.common.Status.FAILURE

        try:
            quat_list = self.blackboard_get("orientation_quat")
            frame_id = self.blackboard_get("frame_id")
        except KeyError as e:
            self.node.get_logger().error(
                f"[{self.name}] Failed to get blackboard key: {e}"
            )
            return py_trees.common.Status.FAILURE

        if not isinstance(quat_list, list) or len(quat_list) != 4:
            self.node.get_logger().error(
                f"[{self.name}] Input 'orientation_quat' is not a list of 4 floats."
            )
            return py_trees.common.Status.FAILURE

        # Create the PoseStamped message
        pose_stamped_msg = PoseStamped()

        # Set Header
        pose_stamped_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped_msg.header.frame_id = frame_id

        # Set Position (default to origin)
        pose_stamped_msg.pose.position = Point(x=0.0, y=0.0, z=0.0)

        # Set Orientation
        pose_stamped_msg.pose.orientation = Quaternion(
            x=quat_list[0], y=quat_list[1], z=quat_list[2], w=quat_list[3]
        )

        # Write to blackboard
        self.blackboard_set("output_pose_stamped", pose_stamped_msg)

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        pass
