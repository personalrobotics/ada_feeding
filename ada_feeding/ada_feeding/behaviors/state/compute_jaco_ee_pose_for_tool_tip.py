# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import math
from typing import Union, Optional, List, Any, Dict
from overrides import override

# Third-party Imports
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from scipy.spatial.transform import Rotation as R
import py_trees
from py_trees.common import Status
import pinocchio as pin

# Local BT Imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


# This helper is copied directly from calculate_skewer_pose_for_tilt.py
#
def _get_relative_transform(
    model: pin.Model,
    data: pin.Data,
    parent_frame: str,
    child_frame: str,
    atool_joints: list,
    atool_joint_names: list,
    logger,
) -> Optional[pin.SE3]:
    """
    Replicates the logic from the benchmark's PinocchioModel wrapper to
    compute a relative transform using raw pinocchio objects.
    """
    try:
        q = pin.neutral(model)
        # Populate only the articutool joints
        for i, joint_name in enumerate(atool_joint_names):
            joint_id = model.getJointId(joint_name)
            joint_obj = model.joints[joint_id]
            angle = atool_joints[i]
            idx_q = joint_obj.idx_q
            if joint_obj.nq == 2:  # Revolute joint with cos/sin representation
                q[idx_q : idx_q + 2] = [math.cos(angle), math.sin(angle)]
            else:
                q[idx_q] = angle

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        parent_id = model.getFrameId(parent_frame)
        T_world_parent = data.oMf[parent_id]
        child_id = model.getFrameId(child_frame)
        T_world_child = data.oMf[child_id]

        return T_world_parent.inverse() * T_world_child
    except Exception as e:
        logger.error(f"[_get_relative_transform] Internal calculation failed: {e}")
        return None


class ComputeJacoEEPoseForToolTip(BlackboardBehavior):
    """
    Calculates a target Jaco EE pose for the 'MoveInto' Cartesian plan.

    It calculates the Jaco EE pose that would place the tool tip at the
    'tool_tip_target_pose' (the 'into' pose).

    Crucially, it then *overwrites* the orientation of this new pose
    with the orientation of the 'jaco_ee_above_pose'. This ensures
    the subsequent Cartesian plan is a pure translation.
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
        self.feedback_message = "ComputeJacoEEPoseForToolTip: "

    def blackboard_inputs(
        self,
        tool_tip_target_pose: Union[BlackboardKey, PoseStamped] = None,
        # --- THIS IS THE FIX ---
        # Changed type hint from Pose to PoseStamped
        jaco_ee_above_pose: Union[BlackboardKey, PoseStamped] = None,
        # --- END FIX ---
        articutool_joint_positions: Union[BlackboardKey, List[float]] = None,
        pinocchio_model: Union[BlackboardKey, pin.Model] = None,
        pinocchio_data: Union[BlackboardKey, pin.Data] = None,
        articutool_joint_names: Union[BlackboardKey, list] = None,
    ) -> None:
        """
        Blackboard Inputs
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        jaco_ee_into_pose: Union[BlackboardKey, Pose] = None,
    ) -> None:
        """
        Blackboard Outputs
        """
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
            # --- 1. Read Inputs ---
            tip_target_pose_stamped = self.blackboard_get("tool_tip_target_pose")
            ee_above_pose_stamped = self.blackboard_get(
                "jaco_ee_above_pose"
            )  # This is a PoseStamped
            atool_joints = self.blackboard_get("articutool_joint_positions")
            model = self.blackboard_get("pinocchio_model")
            data = self.blackboard_get("pinocchio_data")
            atool_names = self.blackboard_get("articutool_joint_names")

            tip_target_pose = tip_target_pose_stamped.pose

            # --- 2. Get T_wrist_tip ---
            T_wrist_tip = _get_relative_transform(
                model,
                data,
                "j2n6s200_end_effector",
                "tool_tip",
                atool_joints,
                atool_names,
                self.logger,
            )
            if T_wrist_tip is None:
                self.feedback_message = "Failed to calculate T_wrist_tip"
                return py_trees.common.Status.FAILURE

            # --- 3. Get T_world_tip (Target 'Into' Pose) ---
            p_tip = tip_target_pose.position
            q_tip = tip_target_pose.orientation
            R_world_tip = R.from_quat([q_tip.x, q_tip.y, q_tip.z, q_tip.w])
            T_world_tip = pin.SE3(
                R_world_tip.as_matrix(), np.array([p_tip.x, p_tip.y, p_tip.z])
            )

            # --- 4. Calculate T_world_wrist (Goal 'Into' Pose) ---
            T_world_wrist_goal = T_world_tip * T_wrist_tip.inverse()

            # --- 5. Construct Final Pose ---
            final_pose = Pose()

            # Position from our IK calculation
            final_pose.position = Point(
                x=T_world_wrist_goal.translation[0],
                y=T_world_wrist_goal.translation[1],
                z=T_world_wrist_goal.translation[2],
            )

            # --- THIS IS THE FIX ---
            # Orientation from the 'jaco_ee_above_pose_stamped' (PoseStamped)
            final_pose.orientation = ee_above_pose_stamped.pose.orientation
            # --- END FIX ---

            self.blackboard_set("jaco_ee_into_pose", final_pose)
            self.feedback_message = "Calculated Jaco EE 'into' pose"
            return py_trees.common.Status.SUCCESS

        except Exception as e:
            self.feedback_message = f"Failed during calculation: {e}"
            self.node.get_logger().error(f"[{self.name}] {self.feedback_message}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass
