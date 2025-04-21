# -*- coding: utf-8 -*-
# (Add appropriate Copyright/License if desired)

"""
This module defines the MoveIt2ComputeFK behavior, which uses the
ada_feeding MoveIt2 wrapper to compute Forward Kinematics.
"""

# Standard imports
from typing import Union, Optional, Dict, Any, List, Tuple
from threading import Lock

# Third-party imports
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveItErrorCodes
from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import rclpy.node

# Local imports
from ada_feeding.helpers import BlackboardKey, get_moveit2_object
from ada_feeding.behaviors import BlackboardBehavior

from pymoveit2 import MoveIt2


class MoveIt2ComputeFK(BlackboardBehavior):
    """
    Computes Forward Kinematics (FK) using the MoveIt2 interface wrapper
    for a given joint state and specified links.

    If joint_state is not provided, uses the current state known by the MoveIt2 object.
    If fk_link_names is not provided, computes FK for the default end-effector link
    associated with the MoveIt2 object's group.
    """

    def blackboard_inputs(
        self,
        group_name: Union[BlackboardKey, str],
        joint_state: Optional[Union[BlackboardKey, JointState, List[float]]] = None,
        fk_link_names: Optional[Union[BlackboardKey, List[str]]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        group_name: The MoveIt2 planning group name. Used during setup to retrieve
                    the correct MoveIt2 object instance.
        joint_state: Optional state (JointState or List[float]) for FK calculation.
                     If None or key not present, uses current state from MoveIt2 object.
        fk_link_names: Optional list of link names. If None or key not present, uses
                       the default end-effector link from the MoveIt2 object.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        fk_poses: Optional[BlackboardKey], # -> Optional[Union[PoseStamped, List[PoseStamped]]]
        success: Optional[BlackboardKey], # -> bool
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        fk_poses: The resulting PoseStamped (if single link/default) or List[PoseStamped]
                  (if multiple fk_link_names requested). None on failure.
        success: Boolean flag indicating if FK computation was successful.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Get the ROS2 node and acquire the MoveIt2 object for the specified group."""
        # pylint: disable=attribute-defined-outside-init
        self.node: rclpy.node.Node = kwargs['node']
        self.moveit2_obj: Optional[MoveIt2] = None
        self.moveit2_lock: Optional[Lock] = None
        self.fk_group_name: Optional[str] = None

        try:
            self.fk_group_name = self.blackboard_get("group_name")
            if not isinstance(self.fk_group_name, str) or not self.fk_group_name:
                 raise ValueError("group_name must be a non-empty string")
        except (KeyError, ValueError) as e:
            self.logger.error(f"[{self.name}] Invalid or missing input 'group_name': {e}")
            return

        try:
            self.moveit2_obj, self.moveit2_lock = get_moveit2_object(
                blackboard=self.blackboard,
                group_name=self.fk_group_name,
                node=self.node,
            )
            if self.moveit2_obj is None or self.moveit2_lock is None:
                 raise RuntimeError("get_moveit2_object returned None")
            self.logger.info(f"[{self.name}] Successfully obtained MoveIt2 object for FK group '{self.fk_group_name}'")
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to get MoveIt2 object for FK group '{self.fk_group_name}': {e}")
            self.moveit2_obj = None
            self.moveit2_lock = None


    @override
    def initialise(self) -> None:
        """Reset status variables if needed."""
        self.logger.debug(f"[{self.name}] Initialising FK computation.")
        # Clear previous results from blackboard
        self.blackboard_set("fk_poses", None)
        self.blackboard_set("success", False)


    @override
    def update(self) -> Status:
        """Perform the FK computation using the synchronous compute_fk method."""
        # Check if setup was successful
        if self.moveit2_obj is None or self.moveit2_lock is None:
            self.logger.error(f"[{self.name}] MoveIt2 object not initialized. Setup likely failed.")
            return Status.FAILURE

        # Check if MoveIt2 is currently locked by another behavior
        # Note: FK is usually read-only, lock might be optional depending on wrapper implementation
        # but using it ensures consistency if wrapper reads internal state like current joints.
        if self.moveit2_lock.locked():
            self.logger.debug(f"[{self.name}] MoveIt2 is locked, waiting.")
            return Status.RUNNING

        # Acquire lock and perform FK computation
        with self.moveit2_lock:
            try:
                # Read optional inputs using try-except to handle missing keys gracefully
                joint_state_input = None
                try:
                    joint_state_input = self.blackboard_get("joint_state")
                    # Optional validation
                    if joint_state_input is not None and not isinstance(joint_state_input, (JointState, list)):
                         self.logger.warning(f"[{self.name}] Input 'joint_state' is not JointState or List, ignoring.")
                         joint_state_input = None
                except KeyError:
                    self.logger.debug(f"[{self.name}] Optional input 'joint_state' not found, using MoveIt2 object's current state.")
                    pass # Keep default None -> wrapper uses current state

                fk_link_names_input = None
                try:
                    fk_link_names_input = self.blackboard_get("fk_link_names")
                    if fk_link_names_input is not None and not (isinstance(fk_link_names_input, list) and all(isinstance(n, str) for n in fk_link_names_input)):
                         self.logger.warning(f"[{self.name}] Input 'fk_link_names' is not List[str], ignoring.")
                         fk_link_names_input = None
                except KeyError:
                    self.logger.debug(f"[{self.name}] Optional input 'fk_link_names' not found, using MoveIt2 object's default end-effector.")
                    pass # Keep default None -> wrapper uses default EE link

                # Call the synchronous compute_fk method
                self.logger.info(f"[{self.name}] Computing FK for group '{self.fk_group_name}' "
                                 f"with links: {fk_link_names_input or 'default EE'}.")

                result_poses: Optional[Union[PoseStamped, List[PoseStamped]]] = self.moveit2_obj.compute_fk(
                    joint_state=joint_state_input,
                    fk_link_names=fk_link_names_input,
                )

                # Determine success based on whether a result was returned
                fk_success = result_poses is not None

                # Write results to blackboard
                self.blackboard_set("success", fk_success)
                self.blackboard_set("fk_poses", result_poses)

                if fk_success:
                    num_poses = len(result_poses) if isinstance(result_poses, list) else 1
                    self.logger.info(f"[{self.name}] FK computation successful ({num_poses} pose(s) found).")
                    return Status.SUCCESS
                else:
                    # The compute_fk wrapper likely logs the specific error code/reason
                    self.logger.warning(f"[{self.name}] FK computation failed (see MoveIt2 wrapper logs).")
                    return Status.FAILURE

            except KeyError as e:
                self.logger.error(f"[{self.name}] Blackboard key error during FK update: {e}")
                self.blackboard_set("success", False)
                self.blackboard_set("fk_poses", None)
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Unexpected error during FK computation: {e}", exc_info=True)
                self.blackboard_set("success", False)
                self.blackboard_set("fk_poses", None)
                return Status.FAILURE

        # Should not be reached if lock logic is correct
        return Status.FAILURE


    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
