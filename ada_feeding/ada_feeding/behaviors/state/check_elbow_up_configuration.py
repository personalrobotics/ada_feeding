import math
import numpy as np
from typing import List, Optional
from overrides import override
import pinocchio as pin
from py_trees.common import Status
import py_trees
import rclpy.node
from sensor_msgs.msg import JointState

from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class CheckElbowUpConfiguration(BlackboardBehavior):
    """
    Checks if a Jaco arm joint configuration is "elbow-up".
    Succeeds if it is, fails otherwise.
    Reads the Pinocchio model from the global blackboard.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._pinocchio_ready = False
        self.node: Optional[rclpy.node.Node] = None

    @override
    def setup(self, **kwargs):
        """Get node and Pinocchio model from global blackboard."""
        try:
            self.node = kwargs["node"]  # Get node for logging
            bb_client = py_trees.blackboard.Client(name=f"{self.name}_BBClient")
            bb_client.register_key(
                key="/pinocchio_model", access=py_trees.common.Access.READ
            )
            bb_client.register_key(
                key="/pinocchio_data", access=py_trees.common.Access.READ
            )

            self._pin_model = bb_client.get("/pinocchio_model")
            self._pin_data = bb_client.get("/pinocchio_data")

            if self._pin_model is None or self._pin_data is None:
                raise ValueError("Pinocchio model/data not found on global blackboard.")

            self._pinocchio_ready = True
        except Exception as e:
            # self.logger might not be set yet if node is missing
            print(
                f"[{self.name}] ERROR: Failed to get Pinocchio model during setup: {e}"
            )
            self._pinocchio_ready = False

    def blackboard_inputs(
        self,
        ik_solution_joint_state: BlackboardKey,
        jaco_joint_names: List[str],
    ) -> None:
        """Define blackboard inputs for this behavior."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def update(self) -> Status:
        """Perform the elbow-up check."""
        if not self._pinocchio_ready:
            self.feedback_message = "Pinocchio model not initialized."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        try:
            # Read model and data from self, not blackboard
            model = self._pin_model
            data = self._pin_data

            joint_state_msg = self.blackboard_get("ik_solution_joint_state")
            jaco_joint_names = self.blackboard_get("jaco_joint_names")

            # Extract the Jaco joint positions in the correct order
            solution_map = dict(zip(joint_state_msg.name, joint_state_msg.position))
            if not all(name in solution_map for name in jaco_joint_names):
                self.feedback_message = "IK solution is incomplete."
                return Status.FAILURE

            jaco_joint_config = [solution_map[name] for name in jaco_joint_names]

            # 1. Populate the Pinocchio configuration vector 'q'.
            q = pin.neutral(model)
            for i, joint_name in enumerate(jaco_joint_names):
                joint_id = model.getJointId(joint_name)
                joint_obj = model.joints[joint_id]
                angle = jaco_joint_config[i]
                idx_q = joint_obj.idx_q
                if joint_obj.nq == 2:
                    q[idx_q : idx_q + 2] = [math.cos(angle), math.sin(angle)]
                else:
                    q[idx_q] = angle

            # 2. Run FK.
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            # 3. Get key frame positions.
            shoulder_id = model.getFrameId("j2n6s200_link_2")
            elbow_id = model.getFrameId("j2n6s200_link_3")
            wrist_id = model.getFrameId("j2n6s200_link_4")

            p_shoulder = data.oMf[shoulder_id].translation
            p_elbow = data.oMf[elbow_id].translation
            p_wrist = data.oMf[wrist_id].translation

            # 4. Create arm vectors.
            v_upper_arm = p_elbow - p_shoulder
            v_forearm = p_wrist - p_elbow

            # 5. Compute the normal and check its dot product with the world's up-vector.
            elbow_normal = np.cross(v_upper_arm, v_forearm)
            is_elbow_up = np.dot(elbow_normal, np.array([0.0, 0.0, 1.0])) > 0

            if is_elbow_up:
                self.feedback_message = "Configuration is elbow-up."
                return Status.SUCCESS
            else:
                self.feedback_message = "Configuration is elbow-down."
                return Status.FAILURE

        except Exception as e:
            self.logger.error(f"[{self.name}] Exception: {e}")
            return Status.FAILURE
