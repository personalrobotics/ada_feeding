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
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
)  # Added MoveItErrorCodes potentially used by wrapper
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
from .moveit2_plan import MoveIt2ConstraintType


class MoveIt2ComputeIK(BlackboardBehavior):
    """
    Computes Inverse Kinematics (IK) using the ada_feeding MoveIt2 interface wrapper.

    Takes a target pose (PoseStamped) and computes a joint state configuration
    that achieves that pose for the planning group associated with the MoveIt2 object.
    """

    # pylint: disable=arguments-differ
    # pylint: disable=too-many-arguments

    def blackboard_inputs(
        self,
        target_pose: Union[BlackboardKey, PoseStamped],
        group_name: Union[BlackboardKey, str],
        lock_joints: Union[BlackboardKey, bool],
        start_joint_state: Optional[Union[BlackboardKey, JointState]] = None,
        constraints: Optional[
            Union[BlackboardKey, List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
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
        ik_constraints: Optional MoveIt constraints message to respect during IK solving.
        """
        # pylint: disable=unused-argument, duplicate-code
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        ik_solution_joint_state: Optional[BlackboardKey],  # -> Optional[JointState]
        success: Optional[BlackboardKey],  # -> bool
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
        self.lock_joints: Optional[bool] = self.blackboard_get("lock_joints")

        # Get group name from blackboard - needed to select the correct MoveIt2 instance
        try:
            self.ik_group_name = self.blackboard_get("group_name")
            if not isinstance(self.ik_group_name, str) or not self.ik_group_name:
                raise ValueError("group_name must be a non-empty string")
        except (KeyError, ValueError) as e:
            self.logger.error(
                f"[{self.name}] Invalid or missing input 'group_name': {e}"
            )
            return

        # Get the MoveIt2 object using the specific group name
        try:
            self.moveit2_obj, self.moveit2_lock = get_moveit2_object(
                blackboard=self.blackboard,
                group_name=self.ik_group_name,
                node=self.node,
                lock_joints=self.lock_joints,
            )
            if self.moveit2_obj is None or self.moveit2_lock is None:
                # Ensure setup fails completely if objects aren't retrieved
                raise RuntimeError(
                    "get_moveit2_object returned None for MoveIt2 object or lock"
                )
            self.logger.info(
                f"[{self.name}] Successfully obtained MoveIt2 object for IK group '{self.ik_group_name}'"
            )
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Failed to get MoveIt2 object for IK group '{self.ik_group_name}': {e}"
            )
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
            self.logger.error(
                f"[{self.name}] MoveIt2 object not initialized. Setup likely failed."
            )
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
                start_state_seed = None
                try:
                    start_state_seed = self.blackboard_get("start_joint_state")
                except KeyError:
                    self.logger.debug(
                        f"[{self.name}] Optional input 'start_joint_state' not found, using None."
                    )
                    pass
                constraints = None
                try:
                    constraints = self.blackboard_get("constraints")
                except KeyError:
                    self.logger.debug(
                        f"[{self.name}] Optional input 'constraints' not found, using None."
                    )
                    pass

                # Validate target_pose_stamped type
                if not isinstance(target_pose_stamped, PoseStamped):
                    self.logger.error(
                        f"[{self.name}] Input 'target_pose' is not a PoseStamped message."
                    )
                    return py_trees.common.Status.FAILURE

                ik_constraints_msg: Optional[Constraints] = None
                if isinstance(constraints, list) and constraints:
                    ik_constraints_msg = Constraints()
                    ik_constraints_msg.name = "ik_constraints_from_list"
                    self.logger.debug(
                        f"[{self.name}] Processing {len(constraints)} constraint specifications for IK."
                    )
                    for constraint_type, constraint_kwargs in constraints:
                        try:
                            if constraint_type == MoveIt2ConstraintType.JOINT:
                                # Assumes wrapper has create_joint_constraint(**kwargs) -> JointConstraint
                                constraint_obj = (
                                    self.moveit2_obj.create_joint_constraints(
                                        **constraint_kwargs
                                    )
                                )
                                if constraint_obj:
                                    ik_constraints_msg.joint_constraints.extend(
                                        constraint_obj
                                    )
                            elif constraint_type == MoveIt2ConstraintType.POSITION:
                                # Assumes wrapper has create_position_constraint(**kwargs) -> PositionConstraint
                                constraint_obj = (
                                    self.moveit2_obj.create_position_constraint(
                                        **constraint_kwargs
                                    )
                                )
                                if constraint_obj:
                                    ik_constraints_msg.position_constraints.append(
                                        constraint_obj
                                    )
                            elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                                # Assumes wrapper has create_orientation_constraint(**kwargs) -> OrientationConstraint
                                constraint_obj = (
                                    self.moveit2_obj.create_orientation_constraint(
                                        **constraint_kwargs
                                    )
                                )
                                if constraint_obj:
                                    ik_constraints_msg.orientation_constraints.append(
                                        constraint_obj
                                    )
                            else:
                                self.logger.warning(
                                    f"[{self.name}] Unknown constraint type '{constraint_type}' in list."
                                )

                        except AttributeError as ae:
                            self.logger.error(
                                f"[{self.name}] MoveIt2 wrapper missing 'create_X_constraint' method for type {constraint_type}? Error: {ae}"
                            )
                            return py_trees.common.Status.FAILURE
                        except Exception as e_constr:
                            self.logger.error(
                                f"[{self.name}] Error processing constraint {constraint_type} with args {constraint_kwargs}: {e_constr}"
                            )
                            return py_trees.common.Status.FAILURE
                    # Optional: Write generated message to blackboard for debugging
                    # self.blackboard_set("generated_ik_constraints_msg", ik_constraints_msg)
                elif constraints is not None:
                    self.logger.warning(
                        f"[{self.name}] Input 'constraints' is not a list, ignoring. Type: {type(constraints)}"
                    )

                # --- Call the synchronous compute_ik method provided in the API ---
                # It takes position and orientation separately.
                self.logger.info(
                    f"[{self.name}] Computing IK for group '{self.ik_group_name}' targeting pose in frame '{target_pose_stamped.header.frame_id}'..."
                )

                # The MoveIt service internally uses the frame_id from the PoseStamped in its request,
                # even though the wrapper function takes position/orientation separately.
                # We pass the pose components directly.
                ik_result_state: Optional[JointState] = self.moveit2_obj.compute_ik(
                    position=target_pose_stamped.pose.position,
                    quat_xyzw=target_pose_stamped.pose.orientation,
                    start_joint_state=start_state_seed,
                    constraints=ik_constraints_msg,
                )

                # Determine success based on whether a result was returned
                ik_success = ik_result_state is not None

                # Write results to blackboard
                self.blackboard_set("success", ik_success)
                self.blackboard_set("ik_solution_joint_state", ik_result_state)

                if ik_success:
                    self.logger.info(f"[{self.name}] IK computation successful.")
                    self.logger.info(f"IK solution: {ik_result_state}")
                    return py_trees.common.Status.SUCCESS
                else:
                    self.logger.warning(
                        f"[{self.name}] IK computation failed (see MoveIt2 wrapper logs for error code)."
                    )
                    return py_trees.common.Status.FAILURE

            except KeyError as e:
                self.logger.error(
                    f"[{self.name}] Blackboard key error during IK update: {e}"
                )
                return py_trees.common.Status.FAILURE
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] Unexpected error during IK computation: {e}",
                )
                return py_trees.common.Status.FAILURE

        # This line should not be reachable if the lock logic is correct
        return py_trees.common.Status.FAILURE

    @override
    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
