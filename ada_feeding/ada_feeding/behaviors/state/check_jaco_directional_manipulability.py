# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the CheckJacoDirectionalManipulability behavior.
This behavior evaluates if the Jaco arm, at a given configuration,
has sufficient translational manipulability for its end-effector to move
along a specified Cartesian direction. It uses a pre-loaded Pinocchio model
from the blackboard.
"""

# Standard imports
import math
from typing import Union, Optional, List, Dict, Any

# Third-party imports
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import numpy as np
from overrides import override
import pinocchio as pin
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import rclpy.node

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CheckJacoDirectionalManipulability(BlackboardBehavior):
    """
    Checks Jaco arm's directional translational manipulability using a pre-loaded
    Pinocchio model and pre-calculated joint/frame IDs from the blackboard.
    """

    def blackboard_inputs(
        self,
        pinocchio_model: Union[BlackboardKey, pin.Model],
        pinocchio_data: Union[BlackboardKey, pin.Data],
        jaco_vel_indices_pin: Union[
            BlackboardKey, List[int]
        ],  # Pre-calculated velocity indices for Jaco joints
        jaco_ee_frame_id_pin: Union[
            BlackboardKey, int
        ],  # Pre-calculated frame ID for Jaco EE
        current_full_robot_joint_state_MA: Union[BlackboardKey, JointState],
        tool_tip_move_above_pose_world: Union[BlackboardKey, PoseStamped],
        tool_tip_move_into_pose_world: Union[BlackboardKey, PoseStamped],
        directional_manipulability_threshold: Union[BlackboardKey, float],
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        pinocchio_model: Pinocchio model object (from blackboard).
        pinocchio_data: Pinocchio data object (from blackboard).
        jaco_vel_indices_pin: List of Pinocchio velocity indices for Jaco joints (from blackboard).
        jaco_ee_frame_id_pin: Pinocchio frame ID for the Jaco end-effector (from blackboard).
        current_full_robot_joint_state_MA: JointState of the full robot at "Move Above".
        tool_tip_move_above_pose_world: PoseStamped of tool tip at "Move Above" in world.
        tool_tip_move_into_pose_world: PoseStamped of tool tip at "Move Into" in world.
        directional_manipulability_threshold: Minimum acceptable score.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        jaco_directional_manipulability_score: Optional[BlackboardKey],
        jaco_is_manipulable_for_direction: Optional[BlackboardKey],
    ) -> None:
        """
        Blackboard Outputs
        (Same as before)
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[rclpy.node.Node] = None
        # Pinocchio objects will be read from blackboard in update/initialise
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._jaco_vel_indices: Optional[List[int]] = None
        self._jaco_ee_frame_id: Optional[int] = None

    @override
    def setup(self, **kwargs):
        """Gets the ROS2 node from arguments."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            # Node is critical, update will fail if it's None

    @override
    def initialise(self) -> None:
        """
        Reads Pinocchio model, data, and pre-calculated IDs from the blackboard.
        Clears previous outputs.
        """
        self.logger.debug(
            f"[{self.name}] Initializing and attempting to read Pinocchio data from blackboard."
        )
        self.blackboard_set("jaco_directional_manipulability_score", None)
        self.blackboard_set("jaco_is_manipulable_for_direction", None)

        try:
            self._pin_model = self.blackboard_get("pinocchio_model")
            self._pin_data = self.blackboard_get("pinocchio_data")
            self._jaco_vel_indices = self.blackboard_get("jaco_vel_indices_pin")
            self._jaco_ee_frame_id = self.blackboard_get("jaco_ee_frame_id_pin")

            if not isinstance(self._pin_model, pin.Model):
                self.logger.error(
                    f"[{self.name}] 'pinocchio_model' from blackboard is not a valid Pinocchio Model."
                )
                self._pin_model = None  # Invalidate
            if not isinstance(self._pin_data, pin.Data):
                self.logger.error(
                    f"[{self.name}] 'pinocchio_data' from blackboard is not a valid Pinocchio Data."
                )
                self._pin_data = None  # Invalidate
            if not isinstance(self._jaco_vel_indices, list) or not all(
                isinstance(i, int) for i in self._jaco_vel_indices
            ):
                self.logger.error(
                    f"[{self.name}] 'jaco_vel_indices_pin' from blackboard is not a valid List[int]."
                )
                self._jaco_vel_indices = None  # Invalidate
            if not isinstance(self._jaco_ee_frame_id, int):
                self.logger.error(
                    f"[{self.name}] 'jaco_ee_frame_id_pin' from blackboard is not a valid int."
                )
                self._jaco_ee_frame_id = None  # Invalidate

        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Failed to get required Pinocchio info from blackboard: {e}. Ensure LoadPinocchioModel ran successfully."
            )
            self._pin_model = None  # Ensure invalidated on error
            self._pin_data = None
            self._jaco_vel_indices = None
            self._jaco_ee_frame_id = None

        if (
            self._pin_model
            and self._pin_data
            and self._jaco_vel_indices
            and self._jaco_ee_frame_id is not None
        ):
            self.logger.debug(
                f"[{self.name}] Successfully retrieved Pinocchio model, data, and Jaco IDs from blackboard."
            )
        else:
            self.logger.error(
                f"[{self.name}] Failed to properly initialize with Pinocchio data from blackboard."
            )

    @override
    def update(self) -> Status:
        if not self.node:
            self.feedback_message = "Node not initialized in setup."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        # Check if Pinocchio objects were successfully retrieved in initialise
        if (
            not self._pin_model
            or not self._pin_data
            or self._jaco_ee_frame_id is None
            or not self._jaco_vel_indices
        ):
            self.feedback_message = (
                "Pinocchio model/data/IDs not available from blackboard."
            )
            self.logger.error(
                f"[{self.name}] {self.feedback_message} Ensure LoadPinocchioModel ran successfully and populated the blackboard."
            )
            return Status.FAILURE

        try:
            current_q_robot_state: JointState = self.blackboard_get(
                "current_full_robot_joint_state_MA"
            )
            tip_pose_MA_world: PoseStamped = self.blackboard_get(
                "tool_tip_move_above_pose_world"
            )
            tip_pose_MI_world: PoseStamped = self.blackboard_get(
                "tool_tip_move_into_pose_world"
            )
            threshold: float = self.blackboard_get(
                "directional_manipulability_threshold"
            )

            if not all(
                isinstance(p, PoseStamped)
                for p in [tip_pose_MA_world, tip_pose_MI_world]
            ):
                self.feedback_message = "Input tool tip poses are not PoseStamped."
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            # --- Populate Pinocchio's q vector (q_MA) ---
            q_MA = pin.neutral(self._pin_model)  # Use the model from blackboard

            num_updated_joints_in_q = 0
            for i, name_from_js in enumerate(current_q_robot_state.name):
                if self._pin_model.existJointName(name_from_js):
                    joint_id_pin = self._pin_model.getJointId(name_from_js)
                    joint_obj_pin = self._pin_model.joints[joint_id_pin]
                    pin_joint_name = self._pin_model.names[joint_id_pin]

                    theta = current_q_robot_state.position[i]

                    if joint_obj_pin.nq == 2 and joint_obj_pin.nv == 1:
                        q_MA[joint_obj_pin.idx_q] = math.cos(theta)
                        q_MA[joint_obj_pin.idx_q + 1] = math.sin(theta)
                        num_updated_joints_in_q += 1
                    elif joint_obj_pin.nq == 1 and joint_obj_pin.nv == 1:
                        q_MA[joint_obj_pin.idx_q] = theta
                        num_updated_joints_in_q += 1
                    elif joint_obj_pin.nq > 1:
                        if (
                            pin_joint_name == self._pin_model.names[1]
                            and joint_obj_pin.shortname() == "JointModelFreeFlyer"
                        ):
                            if (i + 7) <= len(current_q_robot_state.position):
                                q_MA[joint_obj_pin.idx_q : joint_obj_pin.idx_q + 7] = (
                                    current_q_robot_state.position[i : i + 7]
                                )
                                num_updated_joints_in_q += 1
                            else:
                                self.logger.warn(
                                    f"[{self.name}] FreeFlyer base joint '{pin_joint_name}' found, but not enough values in JointState (from index {i}) to populate q (expected 7). Using neutral for base."
                                )
                        else:
                            self.logger.warn(
                                f"[{self.name}] Joint '{name_from_js}' (Pinocchio name: '{pin_joint_name}') has nq={joint_obj_pin.nq}, nv={joint_obj_pin.nv}. Unhandled complex joint type."
                            )
                else:
                    self.logger.warn(
                        f"[{self.name}] Joint '{name_from_js}' from input JointState not found in Pinocchio model. Skipping."
                    )

            if num_updated_joints_in_q == 0:
                self.logger.error(
                    f"[{self.name}] No joints in Pinocchio's q were updated from input JointState."
                )
                return Status.FAILURE

            jaco_joint_names_bb: List[str] = self.blackboard_get(
                "jaco_joint_names"
            )  # For logging
            log_jaco_q_values = []
            for (
                name_j
            ) in (
                jaco_joint_names_bb
            ):  # Use 'name_j' to avoid conflict with outer 'name'
                if self._pin_model.existJointName(name_j):
                    jid = self._pin_model.getJointId(name_j)
                    j_obj = self._pin_model.joints[jid]
                    if j_obj.nq == 1:
                        log_jaco_q_values.append(f"{name_j}: {q_MA[j_obj.idx_q]:.3f}")
                    elif j_obj.nq == 2:
                        log_jaco_q_values.append(
                            f"{name_j}: [{q_MA[j_obj.idx_q]:.3f}, {q_MA[j_obj.idx_q + 1]:.3f}]"
                        )
            self.logger.info(
                f"[{self.name}] Populated q_MA for Jaco joints: {{{', '.join(log_jaco_q_values)}}}"
            )

            pin.forwardKinematics(self._pin_model, self._pin_data, q_MA)
            pin.updateFramePlacements(self._pin_model, self._pin_data)

            J_jacoEE_world_full = pin.computeFrameJacobian(
                self._pin_model,
                self._pin_data,
                q_MA,
                self._jaco_ee_frame_id,
                pin.ReferenceFrame.WORLD,
            )

            J_jacoEE_world_jaco = J_jacoEE_world_full[
                :, self._jaco_vel_indices
            ]  # Use pre-calculated indices
            J_jacoEE_world_jaco_trans = J_jacoEE_world_jaco[:3, :]
            self.logger.info(
                f"[{self.name}] Jaco EE Trans Jacobian (J_jacoEE_world_jaco_trans):\n{J_jacoEE_world_jaco_trans}"
            )

            pos_MA_tip = np.array(
                [
                    tip_pose_MA_world.pose.position.x,
                    tip_pose_MA_world.pose.position.y,
                    tip_pose_MA_world.pose.position.z,
                ]
            )
            pos_MI_tip = np.array(
                [
                    tip_pose_MI_world.pose.position.x,
                    tip_pose_MI_world.pose.position.y,
                    tip_pose_MI_world.pose.position.z,
                ]
            )

            task_direction_vec = pos_MI_tip - pos_MA_tip
            norm_task_direction = np.linalg.norm(task_direction_vec)

            if norm_task_direction < 1e-6:
                self.feedback_message = "Task direction vector (tool tip) is near zero."
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                self.blackboard_set("jaco_directional_manipulability_score", 0.0)
                self.blackboard_set("jaco_is_manipulable_for_direction", False)
                return Status.FAILURE

            unit_task_direction_jacoEE = task_direction_vec / norm_task_direction

            JJT_trans_jacoEE = J_jacoEE_world_jaco_trans @ J_jacoEE_world_jaco_trans.T
            det_JJT = np.linalg.det(JJT_trans_jacoEE)
            self.logger.debug(
                f"[{self.name}] Determinant of JJT_trans_jacoEE: {det_JJT:.6e}"
            )

            m_d_squared = (
                unit_task_direction_jacoEE.T
                @ JJT_trans_jacoEE
                @ unit_task_direction_jacoEE
            )
            manipulability_score = np.sqrt(max(0.0, m_d_squared))

            self.logger.info(
                f"[{self.name}] Jaco EE directional manipulability (tool tip direction): {manipulability_score:.4f}"
            )
            self.blackboard_set(
                "jaco_directional_manipulability_score", manipulability_score
            )

            is_manipulable = manipulability_score >= threshold
            self.blackboard_set("jaco_is_manipulable_for_direction", is_manipulable)

            if is_manipulable:
                self.feedback_message = f"Jaco EE is manipulable for task direction (Score: {manipulability_score:.4f} >= Threshold: {threshold:.4f})"
                self.logger.info(f"[{self.name}] {self.feedback_message}")
                return Status.SUCCESS
            else:
                self.feedback_message = f"Jaco EE NOT manipulable for task direction (Score: {manipulability_score:.4f} < Threshold: {threshold:.4f})"
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
