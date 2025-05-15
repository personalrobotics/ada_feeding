# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the CheckJacoDirectionalManipulability behavior.
This behavior evaluates if the Jaco arm, at a given configuration,
has sufficient translational manipulability for its end-effector to move
along a specified Cartesian direction (derived from desired tool tip motion).
"""

# Standard imports
import os
import subprocess
import tempfile
import math  # Added for cos/sin
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
import rclpy.time

# Local imports
from ada_feeding.helpers import BlackboardKey  # Assuming this is your helper
from ada_feeding.behaviors import BlackboardBehavior  # Assuming this is your base class


class CheckJacoDirectionalManipulability(BlackboardBehavior):
    """
    Checks the Jaco arm's directional translational manipulability.
    The manipulability is assessed for the Jaco end-effector at the current
    "Move Above" configuration, for moving in a direction derived from the
    desired tool tip motion (from tool_tip_move_above_pose to tool_tip_move_into_pose).
    """

    def blackboard_inputs(
        self,
        current_full_robot_joint_state_MA: Union[
            BlackboardKey, JointState
        ],  # Full state for Pinocchio FK/Jacobian
        tool_tip_move_above_pose_world: Union[BlackboardKey, PoseStamped],
        tool_tip_move_into_pose_world: Union[BlackboardKey, PoseStamped],
        jaco_joint_names: Union[BlackboardKey, List[str]],
        jaco_end_effector_link_name: Union[
            BlackboardKey, str
        ],  # Link whose manipulability is checked
        directional_manipulability_threshold: Union[BlackboardKey, float],
        urdf_file_path: Union[BlackboardKey, str],
    ) -> None:
        """
        Blackboard Inputs
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        jaco_directional_manipulability_score: Optional[
            BlackboardKey
        ],  # -> Optional[float]
        jaco_is_manipulable_for_direction: Optional[BlackboardKey],  # -> Optional[bool]
    ) -> None:
        """
        Blackboard Outputs
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pin_model: Optional[pin.Model] = None
        self.pin_data: Optional[pin.Data] = None
        self.jaco_pin_vel_indices: List[int] = []
        self.jaco_ee_frame_id: Optional[int] = None
        self.node: Optional[rclpy.node.Node] = None
        self._urdf_loaded_successfully = False

    @override
    def setup(self, **kwargs):
        """Loads the Pinocchio model and gets joint/frame IDs."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            return

        if self._urdf_loaded_successfully:  # Avoid reloading if already done
            self.logger.debug(f"[{self.name}] Pinocchio model already loaded.")
            return

        urdf_path_str: str = self.blackboard_get("urdf_file_path")
        if not urdf_path_str:
            self.logger.error(
                f"[{self.name}] 'urdf_file_path' not provided or is empty."
            )
            return

        resolved_urdf_path = self._resolve_package_path(urdf_path_str)
        if not resolved_urdf_path:
            return  # Error already logged by _resolve_package_path

        processed_urdf_path = resolved_urdf_path
        temp_urdf_file_name = None  # Store name for cleanup

        try:
            if resolved_urdf_path.endswith(".xacro"):
                self.logger.info(
                    f"[{self.name}] Processing XACRO file: {resolved_urdf_path}"
                )
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".urdf", delete=False
                ) as temp_file:
                    temp_urdf_file_name = temp_file.name
                    process = subprocess.run(
                        ["ros2", "run", "xacro", "xacro", resolved_urdf_path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    temp_file.write(process.stdout)
                processed_urdf_path = temp_urdf_file_name

            self.logger.info(f"[{self.name}] Loading URDF from: {processed_urdf_path}")
            self.pin_model = pin.buildModelFromUrdf(processed_urdf_path)
            self.pin_data = self.pin_model.createData()
            self.logger.info(
                f"[{self.name}] Pinocchio model loaded: {self.pin_model.name}, Nq={self.pin_model.nq}, Nv={self.pin_model.nv}"
            )

            jaco_joint_names_list: List[str] = self.blackboard_get("jaco_joint_names")
            if not jaco_joint_names_list:
                self.logger.error(
                    f"[{self.name}] 'jaco_joint_names' list is empty or not provided."
                )
                self.pin_model = None  # Invalidate model
                return

            self.jaco_pin_vel_indices = (
                []
            )  # Reset for safety if setup is called multiple times
            for joint_name in jaco_joint_names_list:
                if self.pin_model.existJointName(joint_name):
                    joint_id = self.pin_model.getJointId(joint_name)
                    if (
                        self.pin_model.joints[joint_id].nv == 1
                    ):  # Check if this joint is 1 DoF in velocity space
                        self.jaco_pin_vel_indices.append(
                            self.pin_model.joints[joint_id].idx_v
                        )
                    else:
                        # This error should ideally not be hit for standard Jaco joints if Pinocchio parses them as revolute.
                        self.logger.error(
                            f"[{self.name}] Jaco joint '{joint_name}' (Pinocchio type: {self.pin_model.joints[joint_id].shortname()}) has nv={self.pin_model.joints[joint_id].nv} != 1. Not supported for simple vel_indices list."
                        )
                        self.pin_model = None
                        return  # Invalidate model and return
                else:
                    self.logger.error(
                        f"[{self.name}] Jaco joint '{joint_name}' not found in Pinocchio model."
                    )
                    self.pin_model = None  # Invalidate model
                    return

            if not self.jaco_pin_vel_indices:
                self.logger.error(
                    f"[{self.name}] No Jaco joint velocity indices could be determined."
                )
                self.pin_model = None
                return
            self.logger.info(
                f"[{self.name}] Jaco velocity indices in Pinocchio model: {self.jaco_pin_vel_indices}"
            )

            jaco_ee_link: str = self.blackboard_get("jaco_end_effector_link_name")
            if not jaco_ee_link:
                self.logger.error(
                    f"[{self.name}] 'jaco_end_effector_link_name' not provided."
                )
                self.pin_model = None  # Invalidate model
                return
            if self.pin_model.existFrame(jaco_ee_link):
                self.jaco_ee_frame_id = self.pin_model.getFrameId(jaco_ee_link)
            else:
                self.logger.error(
                    f"[{self.name}] Jaco EE link '{jaco_ee_link}' not found in Pinocchio model."
                )
                self.pin_model = None  # Invalidate model
                return

            self._urdf_loaded_successfully = True

        except FileNotFoundError:
            self.logger.error(
                f"[{self.name}] URDF/XACRO file not found at: {resolved_urdf_path}"
            )
            self.pin_model = None
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"[{self.name}] XACRO processing failed: {e}\nOutput:\n{e.stderr}"
            )
            self.pin_model = None
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to load Pinocchio model: {e}")
            self.pin_model = None
        finally:
            if (
                temp_urdf_file_name
                and os.path.exists(temp_urdf_file_name)
                and processed_urdf_path == temp_urdf_file_name
            ):
                try:
                    os.unlink(processed_urdf_path)
                except OSError as e_unlink:
                    self.logger.error(
                        f"[{self.name}] Failed to delete temp URDF file {processed_urdf_path}: {e_unlink}"
                    )

    def _resolve_package_path(self, path_str: str) -> Optional[str]:
        """Resolves package:// paths to absolute paths."""
        if path_str.startswith("package://"):
            try:
                parts = path_str.split("package://", 1)[1].split("/", 1)
                package_name = parts[0]
                relative_path = parts[1] if len(parts) > 1 else ""

                from ament_index_python.packages import get_package_share_directory

                package_share_directory = get_package_share_directory(package_name)
                return os.path.join(package_share_directory, relative_path)
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] Could not resolve package path '{path_str}': {e}"
                )
                return None
        return path_str

    @override
    def initialise(self) -> None:
        """Clear previous outputs."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("jaco_directional_manipulability_score", None)
        self.blackboard_set("jaco_is_manipulable_for_direction", None)

    @override
    def update(self) -> Status:
        if not self.node:
            self.feedback_message = "Node not initialized"
            return Status.FAILURE
        if (
            not self._urdf_loaded_successfully
            or not self.pin_model
            or not self.pin_data
            or self.jaco_ee_frame_id is None
            or not self.jaco_pin_vel_indices
        ):
            self.feedback_message = (
                "Pinocchio model/data not loaded or required IDs not found"
            )
            self.logger.error(
                f"[{self.name}] {self.feedback_message} during update. Setup likely failed."
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
            q_MA = pin.neutral(self.pin_model)

            num_updated_joints_in_q = 0
            for i, name_from_js in enumerate(current_q_robot_state.name):
                if self.pin_model.existJointName(name_from_js):
                    joint_id_pin = self.pin_model.getJointId(name_from_js)
                    joint_obj_pin = self.pin_model.joints[joint_id_pin]
                    pin_joint_name = self.pin_model.names[joint_id_pin]

                    theta = current_q_robot_state.position[i]

                    if (
                        joint_obj_pin.nq == 2 and joint_obj_pin.nv == 1
                    ):  # Typical for revolute joints in Pinocchio
                        # Expects [cos(theta), sin(theta)]
                        q_MA[joint_obj_pin.idx_q] = math.cos(theta)
                        q_MA[joint_obj_pin.idx_q + 1] = math.sin(theta)
                        num_updated_joints_in_q += (
                            1  # Counts as one "joint" from JointState
                        )
                        self.logger.debug(
                            f"[{self.name}] Updated revolute joint '{pin_joint_name}' (nq=2) in q_MA with [cos({theta:.3f}), sin({theta:.3f})]"
                        )
                    elif (
                        joint_obj_pin.nq == 1 and joint_obj_pin.nv == 1
                    ):  # E.g. Prismatic joint
                        q_MA[joint_obj_pin.idx_q] = theta
                        num_updated_joints_in_q += 1
                        self.logger.debug(
                            f"[{self.name}] Updated prismatic/nq=1 joint '{pin_joint_name}' in q_MA with {theta:.3f}"
                        )
                    elif joint_obj_pin.nq > 1:  # Other complex joints, e.g. FreeFlyer
                        if (
                            pin_joint_name == self.pin_model.names[1]
                            and joint_obj_pin.shortname() == "JointModelFreeFlyer"
                        ):
                            if (i + 7) <= len(
                                current_q_robot_state.position
                            ):  # Check if enough values are present
                                q_MA[joint_obj_pin.idx_q : joint_obj_pin.idx_q + 7] = (
                                    current_q_robot_state.position[i : i + 7]
                                )
                                num_updated_joints_in_q += 1  # Counts as one "entry" from JointState perspective
                                self.logger.debug(
                                    f"[{self.name}] Updated FreeFlyer base joint '{pin_joint_name}' in q_MA."
                                )
                            else:
                                self.logger.warn(
                                    f"[{self.name}] FreeFlyer base joint '{pin_joint_name}' found, but not enough values in JointState (from index {i}) to populate q (expected 7). Using neutral for base."
                                )
                        else:
                            self.logger.warn(
                                f"[{self.name}] Joint '{name_from_js}' (Pinocchio name: '{pin_joint_name}') has nq={joint_obj_pin.nq}, nv={joint_obj_pin.nv}. "
                                f"Unhandled complex joint type in q_MA population. It will remain neutral if not FreeFlyer."
                            )
                else:
                    self.logger.warn(
                        f"[{self.name}] Joint '{name_from_js}' from input JointState not found in Pinocchio model. Skipping."
                    )

            if num_updated_joints_in_q == 0:
                self.logger.error(
                    f"[{self.name}] No joints in Pinocchio's q were updated from input JointState. Check joint names and URDF consistency."
                )
                return Status.FAILURE

            jaco_joint_names_bb: List[str] = self.blackboard_get("jaco_joint_names")
            log_jaco_q_values = []
            for name in jaco_joint_names_bb:
                if self.pin_model.existJointName(name):
                    jid = self.pin_model.getJointId(name)
                    j_obj = self.pin_model.joints[jid]
                    if j_obj.nq == 1:
                        log_jaco_q_values.append(f"{name}: {q_MA[j_obj.idx_q]:.3f}")
                    elif j_obj.nq == 2:
                        log_jaco_q_values.append(
                            f"{name}: [{q_MA[j_obj.idx_q]:.3f}, {q_MA[j_obj.idx_q + 1]:.3f}]"
                        )
            self.logger.info(
                f"[{self.name}] Populated q_MA for Jaco joints: {{{', '.join(log_jaco_q_values)}}}"
            )
            # --- End q_MA Population ---

            pin.forwardKinematics(self.pin_model, self.pin_data, q_MA)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            J_jacoEE_world_full = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q_MA,
                self.jaco_ee_frame_id,
                pin.ReferenceFrame.WORLD,
            )

            J_jacoEE_world_jaco = J_jacoEE_world_full[:, self.jaco_pin_vel_indices]
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
            self.logger.info(
                f"[{self.name}] Tool Tip MA Pose: {pos_MA_tip}, MI Pose: {pos_MI_tip}"
            )
            self.logger.info(
                f"[{self.name}] Task Direction Vector (tool tip): {task_direction_vec}, Norm: {norm_task_direction}"
            )

            if norm_task_direction < 1e-6:
                self.feedback_message = "Task direction vector (tool tip) is near zero."
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                self.blackboard_set("jaco_directional_manipulability_score", 0.0)
                self.blackboard_set("jaco_is_manipulable_for_direction", False)
                return Status.FAILURE

            unit_task_direction_tip = task_direction_vec / norm_task_direction
            unit_task_direction_jacoEE = unit_task_direction_tip
            self.logger.info(
                f"[{self.name}] Unit Task Direction (Jaco EE): {unit_task_direction_jacoEE}"
            )

            JJT_trans_jacoEE = J_jacoEE_world_jaco_trans @ J_jacoEE_world_jaco_trans.T
            self.logger.info(
                f"[{self.name}] JJT_trans_jacoEE (3x3 matrix):\n{JJT_trans_jacoEE}"
            )
            det_JJT = np.linalg.det(JJT_trans_jacoEE)
            self.logger.info(
                f"[{self.name}] Determinant of JJT_trans_jacoEE: {det_JJT:.6e}"
            )

            m_d_squared = (
                unit_task_direction_jacoEE.T
                @ JJT_trans_jacoEE
                @ unit_task_direction_jacoEE
            )
            self.logger.info(
                f"[{self.name}] m_d_squared (before sqrt): {m_d_squared:.6e}"
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
