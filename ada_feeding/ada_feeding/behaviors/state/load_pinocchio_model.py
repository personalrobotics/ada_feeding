# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the LoadPinocchioModel behavior, which loads the robot model
using Pinocchio and places the model, data, and key kinematic information
onto the blackboard for use by other behaviors.
"""

# Standard imports
import os
import subprocess
import tempfile
from typing import Union, Optional, List, Dict, Any
from overrides import override

# Third-party imports
import pinocchio as pin
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import rclpy.node

# Local imports
from ada_feeding.helpers import BlackboardKey  # Assuming this is your helper
from ada_feeding.behaviors import BlackboardBehavior  # Assuming this is your base class


class LoadPinocchioModel(BlackboardBehavior):
    """
    Loads the robot's kinematic model using Pinocchio from a URDF/XACRO file.
    It writes the Pinocchio model, data, and optionally pre-identified
    joint velocity indices and frame IDs to the blackboard.
    This behavior should typically run once at the start of a sequence
    requiring Pinocchio-based calculations. It will only attempt to load
    the model once per instantiation unless explicitly reset or re-set up.
    """

    def blackboard_inputs(
        self,
        urdf_file_path: Union[BlackboardKey, str],
        jaco_joint_names: Optional[Union[BlackboardKey, List[str]]] = None,
        articutool_joint_names: Optional[Union[BlackboardKey, List[str]]] = None,
        jaco_end_effector_link_name: Optional[Union[BlackboardKey, str]] = None,
        tool_tip_link_name: Optional[Union[BlackboardKey, str]] = None,
        # Add other relevant link names if needed, e.g., articutool_base_link
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        urdf_file_path: Path to the robot's URDF or XACRO file.
        jaco_joint_names: Optional. List of Jaco arm joint names. If provided, their
                          velocity indices in the Pinocchio model will be stored.
        articutool_joint_names: Optional. List of Articutool joint names. If provided,
                                their velocity indices will be stored.
        jaco_end_effector_link_name: Optional. Name of the Jaco end-effector link.
                                     If provided, its frame ID will be stored.
        tool_tip_link_name: Optional. Name of the tool tip link. If provided,
                            its frame ID will be stored.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        pinocchio_model: Optional[BlackboardKey],  # -> Optional[pin.Model]
        pinocchio_data: Optional[BlackboardKey],  # -> Optional[pin.Data]
        jaco_vel_indices_pin: Optional[BlackboardKey] = None,  # -> Optional[List[int]]
        articutool_vel_indices_pin: Optional[
            BlackboardKey
        ] = None,  # -> Optional[List[int]]
        jaco_ee_frame_id_pin: Optional[BlackboardKey] = None,  # -> Optional[int]
        tool_tip_frame_id_pin: Optional[BlackboardKey] = None,  # -> Optional[int]
        # Add other relevant frame IDs or indices as outputs if needed
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        pinocchio_model: The loaded Pinocchio model object.
        pinocchio_data: The Pinocchio data object associated with the model.
        jaco_vel_indices_pin: List of velocity indices for Jaco joints in the model.
        articutool_vel_indices_pin: List of velocity indices for Articutool joints.
        jaco_ee_frame_id_pin: Frame ID for the Jaco end-effector link.
        tool_tip_frame_id_pin: Frame ID for the tool tip link.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[rclpy.node.Node] = None
        self._model_loaded_successfully = False  # Flag to prevent re-loading

    @override
    def setup(self, **kwargs):
        """Gets the ROS2 node from arguments."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            # No return here, update will handle node being None

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

    def _get_vel_indices_for_joints(
        self, model: pin.Model, joint_names: List[str], group_name_log: str
    ) -> Optional[List[int]]:
        """Helper to get velocity indices for a list of joint names."""
        vel_indices = []
        for joint_name in joint_names:
            if model.existJointName(joint_name):
                joint_id = model.getJointId(joint_name)
                if model.joints[joint_id].nv == 1:  # Assuming 1 DoF in velocity space
                    vel_indices.append(model.joints[joint_id].idx_v)
                else:
                    self.logger.error(
                        f"[{self.name}] {group_name_log} joint '{joint_name}' has nv={model.joints[joint_id].nv} != 1. Cannot directly get single velocity index."
                    )
                    return None
            else:
                self.logger.error(
                    f"[{self.name}] {group_name_log} joint '{joint_name}' not found in Pinocchio model."
                )
                return None
        if (
            not vel_indices and joint_names
        ):  # If names were provided but no indices found
            self.logger.error(
                f"[{self.name}] No velocity indices determined for {group_name_log} joints: {joint_names}"
            )
            return None
        return vel_indices

    @override
    def update(self) -> Status:
        """
        Loads the Pinocchio model if not already loaded and writes outputs to blackboard.
        Returns SUCCESS if model is loaded (or was already loaded), FAILURE otherwise.
        """
        if not self.node:
            self.feedback_message = "Node not initialized in setup."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        if self._model_loaded_successfully:
            self.feedback_message = "Pinocchio model already loaded and on blackboard."
            self.logger.debug(f"[{self.name}] {self.feedback_message}")
            return Status.SUCCESS  # Already done, no need to reload

        try:
            urdf_path_str: str = self.blackboard_get("urdf_file_path")
            if not urdf_path_str:
                self.feedback_message = "'urdf_file_path' not provided or is empty."
                self.logger.error(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE

            resolved_urdf_path = self._resolve_package_path(urdf_path_str)
            if not resolved_urdf_path:
                self.feedback_message = f"Failed to resolve URDF path: {urdf_path_str}"
                # _resolve_package_path already logged the error
                return Status.FAILURE

            processed_urdf_path = resolved_urdf_path
            temp_urdf_file_name = None

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
            # Important: If your URDF has a floating base (e.g. for a mobile manipulator),
            # you might need to load it with pin.buildModelFromUrdf(..., pin.JointModelFreeFlyer())
            # or ensure the URDF correctly specifies the root joint.
            # For a fixed-base manipulator like Jaco on a table, this is usually fine.
            model = pin.buildModelFromUrdf(processed_urdf_path)
            data = model.createData()
            self.logger.info(
                f"[{self.name}] Pinocchio model loaded: {model.name}, Nq={model.nq}, Nv={model.nv}"
            )

            self.blackboard_set("pinocchio_model", model)
            self.blackboard_set("pinocchio_data", data)

            # Optionally get and store Jaco joint velocity indices
            jaco_joint_names_list = self.blackboard_get("jaco_joint_names")
            if jaco_joint_names_list:
                jaco_vel_indices = self._get_vel_indices_for_joints(
                    model, jaco_joint_names_list, "Jaco"
                )
                if jaco_vel_indices is not None:
                    self.blackboard_set("jaco_vel_indices_pin", jaco_vel_indices)
                    self.logger.info(
                        f"[{self.name}] Stored Jaco velocity indices: {jaco_vel_indices}"
                    )
                # else: return Status.FAILURE # Fail if requested but not found

            # Optionally get and store Articutool joint velocity indices
            articutool_joint_names_list = self.blackboard_get("articutool_joint_names")
            if articutool_joint_names_list:
                articutool_vel_indices = self._get_vel_indices_for_joints(
                    model, articutool_joint_names_list, "Articutool"
                )
                if articutool_vel_indices is not None:
                    self.blackboard_set(
                        "articutool_vel_indices_pin", articutool_vel_indices
                    )
                    self.logger.info(
                        f"[{self.name}] Stored Articutool velocity indices: {articutool_vel_indices}"
                    )
                # else: return Status.FAILURE

            # Optionally get and store Jaco EE frame ID
            jaco_ee_link_name_str = self.blackboard_get("jaco_end_effector_link_name")
            if jaco_ee_link_name_str:
                if model.existFrame(jaco_ee_link_name_str):
                    jaco_ee_id = model.getFrameId(jaco_ee_link_name_str)
                    self.blackboard_set("jaco_ee_frame_id_pin", jaco_ee_id)
                    self.logger.info(
                        f"[{self.name}] Stored Jaco EE frame ID for '{jaco_ee_link_name_str}': {jaco_ee_id}"
                    )
                else:
                    self.logger.warn(
                        f"[{self.name}] Jaco EE link '{jaco_ee_link_name_str}' not found in Pinocchio model."
                    )
                    # Not failing here, as it's optional

            # Optionally get and store tool tip frame ID
            tool_tip_link_name_str = self.blackboard_get("tool_tip_link_name")
            if tool_tip_link_name_str:
                if model.existFrame(tool_tip_link_name_str):
                    tool_tip_id = model.getFrameId(tool_tip_link_name_str)
                    self.blackboard_set("tool_tip_frame_id_pin", tool_tip_id)
                    self.logger.info(
                        f"[{self.name}] Stored tool tip frame ID for '{tool_tip_link_name_str}': {tool_tip_id}"
                    )
                else:
                    self.logger.warn(
                        f"[{self.name}] Tool tip link '{tool_tip_link_name_str}' not found in Pinocchio model."
                    )

            self._model_loaded_successfully = True
            self.feedback_message = "Pinocchio model loaded successfully."
            self.logger.info(f"[{self.name}] {self.feedback_message}")
            return Status.SUCCESS

        except FileNotFoundError as e:
            self.feedback_message = f"URDF/XACRO file not found: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
        except subprocess.CalledProcessError as e:
            self.feedback_message = f"XACRO processing failed: {e}\nOutput:\n{e.stderr}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
        except KeyError as e:  # For blackboard_get if a required key is missing
            self.feedback_message = (
                f"Blackboard key error during setup/input reading: {e}"
            )
            self.logger.error(f"[{self.name}] {self.feedback_message}")
        except Exception as e:
            self.feedback_message = f"Failed to load Pinocchio model: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
        finally:
            if temp_urdf_file_name and os.path.exists(temp_urdf_file_name):
                try:
                    os.unlink(temp_urdf_file_name)
                except OSError as e_unlink:
                    self.logger.error(
                        f"[{self.name}] Failed to delete temp URDF file: {e_unlink}"
                    )

        return Status.FAILURE  # Reached if any exception occurred

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
