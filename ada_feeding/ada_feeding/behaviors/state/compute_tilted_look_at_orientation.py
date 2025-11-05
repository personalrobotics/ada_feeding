# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
import py_trees
from scipy.spatial.transform import Rotation

from typing import Any, Dict, List, Optional, Union
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


def _create_look_at_rotation(z_axis, up_vector) -> Rotation:
    """Helper to create a 'look_at' rotation matrix."""
    # 1. Normalize Z-axis (the "look at" vector)
    z_vec = z_axis / np.linalg.norm(z_axis)

    # 2. Compute X-axis (Right vector)
    # X = Up x Z
    x_vec = np.cross(up_vector, z_vec)
    x_vec = x_vec / np.linalg.norm(x_vec)

    # 3. Compute Y-axis (Orthonormal Up vector)
    # Y = Z x X
    y_vec = np.cross(z_vec, x_vec)

    # 4. Create rotation matrix from basis vectors
    # [X, Y, Z] (column vectors)
    matrix = np.stack([x_vec, y_vec, z_vec], axis=-1)

    return Rotation.from_matrix(matrix)


class ComputeTiltedLookAtOrientation(BlackboardBehavior):
    """
    Computes a nominal orientation quaternion for a constraint.

    The orientation is defined by:
    1.  A "level" pose where the Z-axis points at the target's (x,y)
        position and the Y-axis points to world +Z (up).
    2.  A downward tilt (positive rotation) about that new X-axis.
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
        self.feedback_message = "ComputeTiltedLookAtOrientation: "

    def blackboard_inputs(
        self,
        target_pose: Union[BlackboardKey, PoseStamped, TransformStamped] = None,
        pitch_down_rad: Union[BlackboardKey, float] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        target_pose: The target pose (e.g., of the food) in the arm's
                     base frame. Accepts PoseStamped or TransformStamped.
        pitch_down_rad: The angle in radians to pitch down (e.g., 0.785
                        for 45 degrees).
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        goal_orientation_quat: Union[BlackboardKey, List[float]] = None,
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        goal_orientation_quat: The resulting [x, y, z, w] quaternion as a
                               list[float].
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """
        Get the node from the tree manager for logging.
        """
        try:
            self.node = kwargs["node"]
        except KeyError:
            self.node = None
            self.feedback_message = "no 'node' in setup's kwargs"

    def update(self):
        """
        Reads inputs, computes the orientation, and writes to the blackboard.
        """
        if self.node is None:
            py_trees.logging.Logger(self.name).error(self.feedback_message)
            return py_trees.common.Status.FAILURE

        # --- 1. Read Inputs ---
        try:
            target_pose = self.blackboard_get("target_pose")
            pitch_angle = self.blackboard_get("pitch_down_rad")
        except KeyError as e:
            self.node.get_logger().error(
                f"[{self.name}] Failed to get blackboard key: {e}"
            )
            return py_trees.common.Status.FAILURE

        # --- 2. Extract Position (Handles both PoseStamped and TransformStamped) ---
        target_pos = None
        if isinstance(target_pose, PoseStamped):
            target_pos = target_pose.pose.position
        elif isinstance(target_pose, TransformStamped):
            target_pos = target_pose.transform.translation
        else:
            self.node.get_logger().error(
                f"[{self.name}] Input 'target_pose' must be PoseStamped or "
                f"TransformStamped, but got {type(target_pose)}."
            )
            return py_trees.common.Status.FAILURE

        # --- 3. Compute "Level" Orientation ---
        try:
            # Define the desired Z-axis ("look at" vector)
            # This points from origin to food in the XY plane
            z_axis_vec = np.array([target_pos.x, target_pos.y, 0.0])

            # Handle case where target is at (0,0)
            if np.linalg.norm(z_axis_vec) < 1e-6:
                z_axis_vec = np.array([1.0, 0.0, 0.0])  # Default to +X

            # Define the desired "Up" vector (World +Z)
            up_vec = np.array([0.0, 0.0, 1.0])

            # Compute the "level" rotation matrix
            R_level = _create_look_at_rotation(z_axis_vec, up_vec)

            # --- 4. Compute "Tilted" Orientation ---

            # Define the tilt rotation (positive 45 deg about the new X-axis)
            R_tilt = Rotation.from_euler("x", pitch_angle, degrees=False)

            # Combine rotations: R_final = R_level * R_tilt
            # This applies the tilt in the new "level" frame
            R_final = R_level * R_tilt

            # Get the quaternion in [x, y, z, w] format
            final_quat = R_final.as_quat()

        except Exception as e:
            self.node.get_logger().error(
                f"[{self.name}] Error during quaternion math: {e}"
            )
            return py_trees.common.Status.FAILURE

        # --- 5. Write Output ---
        self.blackboard_set("goal_orientation_quat", final_quat.tolist())

        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status):
        """
        No cleanup needed.
        """
        pass
