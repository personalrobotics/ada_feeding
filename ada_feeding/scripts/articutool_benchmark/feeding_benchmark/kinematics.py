# Standard imports
import subprocess
import math
from typing import Optional, List

# Third-party imports
import numpy as np
import pinocchio as pin

# Local application imports
from .constants import (
    LOGGER,
    JOINT_NAMES_JACO,
    JOINT_NAMES_ATOOL,
    PLANNING_GROUP_JACO,
    PLANNING_GROUP_ATOOL,
)


class PinocchioModel:
    """
    A wrapper for Pinocchio to provide a seamless interface for kinematic queries.
    """

    def __init__(self, xacro_file_path: str):
        """Loads the robot model from a XACRO file."""
        self.model: Optional[pin.Model] = None
        self.data: Optional[pin.Data] = None
        self._is_ready = False
        self.jaco_joint_ids = []
        self.atool_joint_ids = []
        self.jaco_vel_indices = []
        self.atool_vel_indices = []
        self.full_vel_indices = []
        try:
            process = subprocess.run(
                ["ros2", "run", "xacro", "xacro", xacro_file_path],
                check=True,
                capture_output=True,
                text=True,
            )
            urdf_xml_string = process.stdout
            self.model = pin.buildModelFromXML(urdf_xml_string)
            self.data = self.model.createData()
            self._map_joint_indices()
            self._is_ready = True
            LOGGER.info("Pinocchio model loaded successfully.")
        except Exception as e:
            LOGGER.error(f"Failed to initialize Pinocchio model: {e}", exc_info=True)

    def _map_joint_indices(self):
        for name in JOINT_NAMES_JACO:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.jaco_joint_ids.append(joint_id)
                self.jaco_vel_indices.append(self.model.joints[joint_id].idx_v)
        for name in JOINT_NAMES_ATOOL:
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                self.atool_joint_ids.append(joint_id)
                self.atool_vel_indices.append(self.model.joints[joint_id].idx_v)
        self.full_vel_indices = self.jaco_vel_indices + self.atool_vel_indices
        self.jaco_vel_indices.sort()
        self.atool_vel_indices.sort()
        self.full_vel_indices.sort()

    def is_ready(self) -> bool:
        return self._is_ready

    def _update_configuration(
        self, jaco_joints: List[float], atool_joints: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Populates Pinocchio's configuration vector `q` from joint lists.
        """
        q = pin.neutral(self.model)
        all_joints = jaco_joints + (atool_joints if atool_joints is not None else [])
        joint_ids = self.jaco_joint_ids + (
            self.atool_joint_ids if atool_joints is not None else []
        )
        for i, joint_id in enumerate(joint_ids):
            joint_obj = self.model.joints[joint_id]
            angle = all_joints[i]
            idx_q = joint_obj.idx_q
            if joint_obj.nq == 2:
                q[idx_q : idx_q + 2] = [math.cos(angle), math.sin(angle)]
            else:
                q[idx_q] = angle
        return q

    def get_frame_transform(
        self,
        frame_name: str,
        jaco_joints: List[float],
        atool_joints: Optional[List[float]] = None,
    ) -> Optional[pin.SE3]:
        """
        Performs Forward Kinematics to get the transform of a specific frame.
        """
        if not self.is_ready():
            return None
        try:
            q = self._update_configuration(jaco_joints, atool_joints)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            frame_id = self.model.getFrameId(frame_name)
            return self.data.oMf[frame_id]
        except Exception as e:
            LOGGER.error(f"Pinocchio FK failed for frame '{frame_name}': {e}")
            return None

    def get_relative_transform(
        self,
        parent_frame: str,
        child_frame: str,
        jaco_joints: List[float],
        atool_joints: Optional[List[float]] = None,
    ) -> Optional[pin.SE3]:
        """
        Computes the relative SE(3) transform from a parent frame to a child frame.
        """
        if not self.is_ready():
            return None
        try:
            q = self._update_configuration(jaco_joints, atool_joints)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            parent_id = self.model.getFrameId(parent_frame)
            T_world_parent = self.data.oMf[parent_id]
            child_id = self.model.getFrameId(child_frame)
            T_world_child = self.data.oMf[child_id]
            return T_world_parent.inverse() * T_world_child
        except Exception as e:
            LOGGER.error(
                f"Pinocchio relative transform for '{parent_frame}' -> '{child_frame}' failed: {e}"
            )
            return None

    def get_frame_jacobian(
        self,
        frame_name: str,
        jaco_joints: List[float],
        atool_joints: Optional[List[float]] = None,
        group: str = "jaco_arm_with_articutool",
        reference_frame: pin.ReferenceFrame = pin.ReferenceFrame.WORLD,
    ) -> Optional[np.ndarray]:
        """
        Computes the full Jacobian for a specific frame in the specified reference frame.
        """
        if not self.is_ready():
            return None
        try:
            q = self._update_configuration(jaco_joints, atool_joints)
            frame_id = self.model.getFrameId(frame_name)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            J_full = pin.getFrameJacobian(
                self.model, self.data, frame_id, reference_frame
            )
            if group == PLANNING_GROUP_JACO:
                vel_indices = self.jaco_vel_indices
            elif group == PLANNING_GROUP_ATOOL:
                vel_indices = self.atool_vel_indices
            else:
                vel_indices = self.full_vel_indices
            return J_full[:, vel_indices]
        except Exception as e:
            LOGGER.error(
                f"Pinocchio Jacobian calculation for frame '{frame_name}' failed: {e}"
            )
            return None
