# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the MoveIt2ComputeIK behavior, which uses the
ada_feeding MoveIt2 wrapper to compute Inverse Kinematics for a target pose.
"""

# Standard imports
from typing import Any, Union, Optional, Dict, List, Tuple
from threading import Lock

# Third-party imports
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.msg import Constraints, MoveItErrorCodes # Added MoveItErrorCodes potentially used by wrapper
from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access
import rclpy.node

# Local imports from ada_feeding (adjust paths if necessary)
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
    # get_tf_object, # Likely not needed here if pose is already in desired frame for IK
)
from ada_feeding.behaviors import BlackboardBehavior

from pymoveit2 import MoveIt2


class MoveIt2ComputeIK(BlackboardBehavior):
    """
    Computes Inverse Kinematics (IK) using the ada_feeding MoveIt2 interface wrapper.

    Takes a target pose (PoseStamped) and computes a joint state configuration
    that achieves that pose for the planning group associated with the MoveIt2 object.
    """

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments

    @override
    def blackboard_inputs(
        self,
        target_pose: Union[BlackboardKey, PoseStamped],
        group_name: Union[BlackboardKey, str],
        start_joint_state: Optional[Union[BlackboardKey, JointState]] = None,
        constraints: Optional[Union[BlackboardKey, Constraints]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        target_pose: The target pose (PoseStamped) for the end-effector of the group.
                     The frame_id within PoseStamped is important for the IK solver.
        group_name: The MoveIt2 planning group name. Used during setup to retrieve
                    the correct MoveIt2 object instance.
        start_joint_state: Optional seed state (JointState or List[float]) for the IK solver.
        constraints: Optional MoveIt constraints message to respect during IK solving.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def blackboard_outputs(
        self,
        ik_solution_joint_state: Optional[BlackboardKey], # -> Optional[JointState]
        success: Optional[BlackboardKey], # -> bool
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        ik_solution_joint_state: The resulting JointState if IK is successful, otherwise None.
        success: Boolean flag indicating whether a valid IK solution was found (True) or not (False).
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Get the ROS2 node and acquire the MoveIt2 object for the specified group."""
        # pylint: disable=attribute-defined-outside-init
        self.node: rclpy.node.Node = kwargs["node"]
        self.moveit2_obj: Optional[MoveIt2] = None
        self.moveit2_lock: Optional[Lock] = None
        self.ik_group_name: Optional[str] = None

        # Get group name from blackboard - needed to select the correct MoveIt2 instance
        try:
            self.ik_group_name = self.blackboard_get("group_name")
            if not isinstance(self.ik_group_name, str) or not self.ik_group_name:
                 raise ValueError("group_name must be a non-empty string")
        except (KeyError, ValueError) as e:
            self.logger.error(f"[{self.name}] Invalid or missing input 'group_name': {e}")
            return

        # Get the MoveIt2 object using the specific group name
        try:
            self.moveit2_obj, self.moveit2_lock = get_moveit2_object(
                blackboard=self.blackboard,
                group_name=self.ik_group_name,
                node=self.node,
            )
            if self.moveit2_obj is None or self.moveit2_lock is None:
                 # Ensure setup fails completely if objects aren't retrieved
                 raise RuntimeError("get_moveit2_object returned None for MoveIt2 object or lock")
            self.logger.info(f"[{self.name}] Successfully obtained MoveIt2 object for IK group '{self.ik_group_name}'")
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to get MoveIt2 object for IK group '{self.ik_group_name}': {e}")
            self.moveit2_obj = None
            self.moveit2_lock = None

    @override
    def initialise(self) -> None:
        """Reset status variables if needed."""
        self.logger.debug(f"[{self.name}] Initialising IK computation.")
        self.blackboard_set("success", False)
        self.blackboard_set("ik_solution_joint_state", None)


    @override
    def update(self) -> py_trees.common.Status:
        """Perform the IK computation using the synchronous compute_ik method."""
        # Check if setup was successful
        if self.moveit2_obj is None or self.moveit2_lock is None:
            self.logger.error(f"[{self.name}] MoveIt2 object not initialized. Setup likely failed.")
            return py_trees.common.Status.FAILURE

        # Check if MoveIt2 is currently locked by another behavior
        if self.moveit2_lock.locked():
            self.logger.debug(f"[{self.name}] MoveIt2 is locked, waiting.")
            return py_trees.common.Status.RUNNING

        # Acquire lock and perform IK computation
        with self.moveit2_lock:
            try:
                # Read inputs from blackboard at runtime
                target_pose_stamped: PoseStamped = self.blackboard_get("target_pose")
                start_state_seed = self.blackboard_get("start_joint_state", None) # Optional
                ik_constraints = self.blackboard_get("constraints", None) # Optional

                # Validate target_pose_stamped type
                if not isinstance(target_pose_stamped, PoseStamped):
                     self.logger.error(f"[{self.name}] Input 'target_pose' is not a PoseStamped message.")
                     return py_trees.common.Status.FAILURE

                # --- Call the synchronous compute_ik method provided in the API ---
                # It takes position and orientation separately.
                self.logger.info(f"[{self.name}] Computing IK for group '{self.ik_group_name}' targeting pose in frame '{target_pose_stamped.header.frame_id}'...")

                # The MoveIt service internally uses the frame_id from the PoseStamped in its request,
                # even though the wrapper function takes position/orientation separately.
                # We pass the pose components directly.
                ik_result_state: Optional[JointState] = self.moveit2_obj.compute_ik(
                    position=target_pose_stamped.pose.position,
                    quat_xyzw=target_pose_stamped.pose.orientation,
                    start_joint_state=start_state_seed,
                    constraints=ik_constraints
                )

                # Determine success based on whether a result was returned
                ik_success = ik_result_state is not None

                # Write results to blackboard
                self.blackboard_set("success", ik_success)
                self.blackboard_set("ik_solution_joint_state", ik_result_state)

                if ik_success:
                    self.logger.info(f"[{self.name}] IK computation successful.")
                    return py_trees.common.Status.SUCCESS
                else:
                    self.logger.warning(f"[{self.name}] IK computation failed (see MoveIt2 wrapper logs for error code).")
                    return py_trees.common.Status.FAILURE

            except KeyError as e:
                self.logger.error(f"[{self.name}] Blackboard key error during IK update: {e}")
                return py_trees.common.Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Unexpected error during IK computation: {e}", exc_info=True)
                return py_trees.common.Status.FAILURE

        # This line should not be reachable if the lock logic is correct
        return py_trees.common.Status.FAILURE


    @override
    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
