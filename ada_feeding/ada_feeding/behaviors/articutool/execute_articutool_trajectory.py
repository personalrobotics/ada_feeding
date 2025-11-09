# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the ExecuteArticutoolTrajectory behavior, which sends a planned
trajectory to the Articutool's FollowJointTrajectory action server.
It assumes controllers are already appropriately switched by another behavior.
"""

# Standard imports
from typing import Union, Optional, Dict, Any, List
from enum import Enum

# Third-party imports
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, GoalStatus
from rclpy.node import Node as RclpyNode  # Alias for clarity
from rclpy.executors import Future
from rclpy.qos import (
    qos_profile_services_default,
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory  # Input type

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


# Define constants for action status reporting
class ActionExecutionStatus(Enum):
    IDLE = "IDLE"
    CHECKING_DEPENDENCIES = "CHECKING_DEPENDENCIES"
    SENDING_GOAL = "SENDING_GOAL"
    WAITING_FOR_ACCEPTANCE = "WAITING_FOR_ACCEPTANCE"
    EXECUTING = "EXECUTING"
    WAITING_FOR_RESULT = "WAITING_FOR_RESULT"
    GOAL_REJECTED = "GOAL_REJECTED"
    GOAL_CANCELLED = "GOAL_CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"  # From server
    FAILED = "FAILED"  # General failure


class ExecuteArticutoolTrajectory(BlackboardBehavior):
    """
    Executes a JointTrajectory on the Articutool using the
    FollowJointTrajectory action server. This behavior assumes that the
    necessary controllers (e.g., 'joint_trajectory_controller') have already
    been activated by a separate behavior (like SwitchArticutoolControllers)
    prior to this behavior being ticked.

    Returns RUNNING while executing, SUCCESS on completion,
    FAILURE on error, rejection, or abortion.
    """

    # --- Constants ---
    DEFAULT_ACTION_SERVER = (
        "/articutool/joint_trajectory_controller/follow_joint_trajectory"
    )

    def __init__(self, name: str, ns: str = "/", **kwargs):
        super().__init__(name=name, ns=ns, **kwargs)
        # Internal state variables
        self._action_client: Optional[ActionClient] = None
        self._send_goal_future: Optional[Future] = None
        self._get_result_future: Optional[Future] = None
        self._goal_handle: Optional[ClientGoalHandle] = None
        self._current_status: ActionExecutionStatus = ActionExecutionStatus.IDLE
        self._result_code: Optional[int] = None  # Store the final result code

        self._action_client_initialized: bool = False
        self.node: Optional[RclpyNode] = None  # Store the node instance
        self._action_server_name_str: str = (
            ""  # To store the resolved action server name for logging
        )

    def blackboard_inputs(
        self,
        trajectory: Union[BlackboardKey, JointTrajectory],
        action_server_name: Union[BlackboardKey, str] = DEFAULT_ACTION_SERVER,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        trajectory: The trajectory_msgs/JointTrajectory message to execute.
        action_server_name: Name of the FollowJointTrajectory action server.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        action_goal_accepted: Optional[BlackboardKey] = None,  # -> bool
        action_result_code: Optional[
            BlackboardKey
        ] = None,  # -> int (FollowJointTrajectory.Result.error_code)
        action_status: Optional[
            BlackboardKey
        ] = None,  # -> str (from ActionExecutionStatus enum)
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        action_goal_accepted: True if the action server accepted the goal request.
        action_result_code: The final error_code from the FollowJointTrajectory result.
        action_status: The current status of the action call (e.g., "IDLE", "EXECUTING", "SUCCEEDED").
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Get node and create action client."""
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            return  # Critical failure

        try:
            action_server_name_bb = self.blackboard_get("action_server_name")
            if not isinstance(action_server_name_bb, str) or not action_server_name_bb:
                self.logger.warning(
                    f"[{self.name}] Blackboard 'action_server_name' invalid or not found, using default: {self.DEFAULT_ACTION_SERVER}"
                )
                self._action_server_name_str = self.DEFAULT_ACTION_SERVER
            else:
                self._action_server_name_str = action_server_name_bb
        except KeyError:
            self.logger.info(
                f"[{self.name}] Input 'action_server_name' not found on blackboard, using default: {self.DEFAULT_ACTION_SERVER}"
            )
            self._action_server_name_str = self.DEFAULT_ACTION_SERVER

        # Define QoS profiles for action client components
        service_qos_profile = qos_profile_services_default
        feedback_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        status_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            self._action_server_name_str,  # Use stored name
            goal_service_qos_profile=service_qos_profile,
            result_service_qos_profile=service_qos_profile,
            cancel_service_qos_profile=service_qos_profile,
            feedback_sub_qos_profile=feedback_qos_profile,
            status_sub_qos_profile=status_qos_profile,
        )
        self._action_client_initialized = True
        self.logger.info(
            f"[{self.name}] Action client created for '{self._action_server_name_str}' (readiness check deferred to update)."
        )
        self.logger.info(f"[{self.name}] Setup method complete.")

    @override
    def initialise(self) -> None:
        """Reset action client state."""
        self.logger.debug(f"[{self.name}] Initialising.")
        self._send_goal_future = None
        self._get_result_future = None
        self._goal_handle = None
        self._result_code = None

        if self._action_client_initialized:
            self._current_status = ActionExecutionStatus.CHECKING_DEPENDENCIES
        else:
            self.logger.error(
                f"[{self.name}] Action client was not initialized during setup. Failing."
            )
            self._current_status = ActionExecutionStatus.FAILED

        # Clear blackboard outputs
        self.blackboard_set("action_goal_accepted", False)
        self.blackboard_set("action_result_code", self._result_code)
        self.blackboard_set("action_status", self._current_status.value)

    def _check_dependencies(self) -> Status:
        """Helper to check and wait for the action server."""
        if not self._action_client_initialized:
            self.logger.error(
                f"[{self.name}] Action client instance not created. Critical setup failure."
            )
            return self._handle_failure(
                "Action client not initialized", ActionExecutionStatus.FAILED
            )

        if not self._action_client.wait_for_server(
            timeout_sec=0.01
        ):  # Non-blocking check
            self.logger.info(
                f"[{self.name}] Waiting for action server '{self._action_server_name_str}'..."  # Use stored name
            )
            return Status.RUNNING

        self.logger.info(
            f"[{self.name}] Action server '{self._action_server_name_str}' is ready."
        )  # Use stored name
        self._current_status = ActionExecutionStatus.IDLE
        self.blackboard_set("action_status", self._current_status.value)
        return Status.RUNNING

    @override
    def update(self) -> Status:
        """Manage the action client state machine."""
        self.blackboard_set("action_status", self._current_status.value)

        if self._current_status == ActionExecutionStatus.FAILED:
            return Status.FAILURE

        if self._current_status == ActionExecutionStatus.CHECKING_DEPENDENCIES:
            return self._check_dependencies()

        if self._current_status == ActionExecutionStatus.IDLE:
            try:
                trajectory: JointTrajectory = self.blackboard_get("trajectory")
                if not isinstance(trajectory, JointTrajectory):
                    self.logger.error(
                        f"[{self.name}] Input 'trajectory' is not a JointTrajectory message."
                    )
                    return self._handle_failure("Invalid trajectory input type")
                if not trajectory.points:
                    self.logger.warning(
                        f"[{self.name}] Input 'trajectory' has no points. Succeeding trivially."
                    )
                    return self._handle_success(
                        result_code=FollowJointTrajectory.Result.SUCCESSFUL
                    )

                goal_msg = FollowJointTrajectory.Goal()
                goal_msg.trajectory = trajectory
                self.logger.info(
                    f"[{self.name}] Sending trajectory goal to action server '{self._action_server_name_str}'..."  # Use stored name
                )
                self._send_goal_future = self._action_client.send_goal_async(goal_msg)
                self._current_status = ActionExecutionStatus.SENDING_GOAL
                self.blackboard_set("action_status", self._current_status.value)
                return Status.RUNNING
            except KeyError as e:
                return self._handle_failure(f"Blackboard key error: {e}")
            except Exception as e:
                return self._handle_failure(f"Error preparing to send goal: {e}")

        if self._current_status == ActionExecutionStatus.SENDING_GOAL:
            if self._send_goal_future is None:
                return self._handle_invalid_state("SENDING_GOAL future is None")
            if self._send_goal_future.done():
                try:
                    goal_handle: ClientGoalHandle = self._send_goal_future.result()
                except Exception as e:
                    self._send_goal_future = None
                    self.logger.error(
                        f"[{self.name}] Exception waiting for goal handle: {e}. Server might have shut down."
                    )
                    self._current_status = ActionExecutionStatus.CHECKING_DEPENDENCIES
                    self.logger.warning(
                        f"[{self.name}] Re-checking dependencies due to goal send error."
                    )
                    return Status.RUNNING
                self._send_goal_future = None
                if not goal_handle.accepted:
                    self.logger.warning(
                        f"[{self.name}] Goal rejected by action server."
                    )
                    self.blackboard_set("action_goal_accepted", False)
                    return self._handle_failure(
                        "Goal rejected", ActionExecutionStatus.GOAL_REJECTED
                    )
                else:
                    try:
                        uuid_hex = bytes(goal_handle.goal_id.uuid).hex()
                    except Exception:
                        uuid_hex = "[Error Formatting UUID]"
                    self.logger.info(
                        f"[{self.name}] Goal accepted by action server (ID: {uuid_hex})."
                    )
                    self._goal_handle = goal_handle
                    self.blackboard_set("action_goal_accepted", True)
                    self._get_result_future = self._goal_handle.get_result_async()
                    self._current_status = ActionExecutionStatus.EXECUTING
                    self.blackboard_set("action_status", self._current_status.value)
                    return Status.RUNNING
            else:
                return Status.RUNNING

        if (
            self._current_status == ActionExecutionStatus.EXECUTING
            or self._current_status == ActionExecutionStatus.WAITING_FOR_RESULT
        ):
            if self._get_result_future is None or self._goal_handle is None:
                return self._handle_invalid_state(
                    "EXECUTING/WAITING future or handle None"
                )
            status = self._goal_handle.status
            if status == GoalStatus.STATUS_EXECUTING:
                self._current_status = ActionExecutionStatus.EXECUTING
            elif status == GoalStatus.STATUS_ABORTED:
                self.logger.warning(
                    f"[{self.name}] Goal aborted by server (status update)."
                )
                self._current_status = ActionExecutionStatus.WAITING_FOR_RESULT
            elif status == GoalStatus.STATUS_CANCELED:
                self.logger.warning(f"[{self.name}] Goal canceled (status update).")
                self._current_status = ActionExecutionStatus.WAITING_FOR_RESULT
            elif status == GoalStatus.STATUS_SUCCEEDED:
                self.logger.info(
                    f"[{self.name}] Goal status reported SUCCEEDED (status update)."
                )
                self._current_status = ActionExecutionStatus.WAITING_FOR_RESULT
            if self._get_result_future.done():
                try:
                    result_wrapper = self._get_result_future.result()
                    action_result = result_wrapper.result
                    self._result_code = action_result.error_code
                    result_string = action_result.error_string or "None"
                    self.logger.info(
                        f"[{self.name}] Action result received: Code={self._result_code}, Msg='{result_string}'"
                    )
                    self.blackboard_set("action_result_code", self._result_code)
                    if self._result_code == FollowJointTrajectory.Result.SUCCESSFUL:
                        return self._handle_success(result_code=self._result_code)
                    else:
                        final_status_map = {
                            GoalStatus.STATUS_ABORTED: ActionExecutionStatus.ABORTED,
                            GoalStatus.STATUS_CANCELED: ActionExecutionStatus.GOAL_CANCELLED,
                        }
                        final_exec_status = final_status_map.get(
                            status, ActionExecutionStatus.FAILED
                        )
                        return self._handle_failure(
                            f"Action Failed by server: Code={self._result_code}, Msg='{result_string}'",
                            final_exec_status,
                            result_code=self._result_code,
                        )
                except Exception as e:
                    self.logger.error(
                        f"[{self.name}] Exception getting result from future: {e}"
                    )
                    self._current_status = ActionExecutionStatus.CHECKING_DEPENDENCIES
                    self.logger.warning(
                        f"[{self.name}] Re-checking dependencies due to result error."
                    )
                    return Status.RUNNING
            else:
                self.blackboard_set("action_status", self._current_status.value)
                return Status.RUNNING

        if self._current_status == ActionExecutionStatus.SUCCEEDED:
            return Status.SUCCESS
        if self._current_status in [
            ActionExecutionStatus.FAILED,
            ActionExecutionStatus.GOAL_REJECTED,
            ActionExecutionStatus.GOAL_CANCELLED,
            ActionExecutionStatus.ABORTED,
        ]:
            return Status.FAILURE
        return self._handle_invalid_state(f"Unhandled status {self._current_status}")

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        if new_status == Status.INVALID and self._goal_handle is not None:
            status = self._goal_handle.status
            if (
                status == GoalStatus.STATUS_ACCEPTED
                or status == GoalStatus.STATUS_EXECUTING
            ):
                self.logger.warning(
                    f"[{self.name}] Behavior terminated externally while action goal was active (Status: {GoalStatus(status).name}). Requesting cancellation."
                )
                self._goal_handle.cancel_goal_async()
        self._send_goal_future = None
        self._get_result_future = None
        self._goal_handle = None
        if self._current_status not in [
            ActionExecutionStatus.SUCCEEDED,
            ActionExecutionStatus.FAILED,
            ActionExecutionStatus.GOAL_REJECTED,
            ActionExecutionStatus.GOAL_CANCELLED,
            ActionExecutionStatus.ABORTED,
        ]:
            self._current_status = ActionExecutionStatus.IDLE

    def _handle_success(
        self, result_code: int = FollowJointTrajectory.Result.SUCCESSFUL
    ) -> Status:
        self.logger.info(f"[{self.name}] Action Succeeded. Code: {result_code}")
        self._current_status = ActionExecutionStatus.SUCCEEDED
        self.blackboard_set("action_status", self._current_status.value)
        self.blackboard_set("action_result_code", result_code)
        self._goal_handle = None
        self._get_result_future = None
        return Status.SUCCESS

    def _handle_failure(
        self,
        reason: str,
        final_status: ActionExecutionStatus = ActionExecutionStatus.FAILED,
        result_code: Optional[int] = None,
    ) -> Status:
        self.logger.error(f"[{self.name}] Failure: {reason}")
        self._current_status = final_status
        self.blackboard_set("action_status", self._current_status.value)
        if result_code is not None:
            self.blackboard_set("action_result_code", result_code)
        elif (
            self._result_code is not None
            and self._result_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self.blackboard_set("action_result_code", self._result_code)
        elif final_status == ActionExecutionStatus.GOAL_REJECTED:
            self.blackboard_set("action_result_code", -998)
        else:
            self.blackboard_set("action_result_code", -999)
        self._goal_handle = None
        self._get_result_future = None
        self._send_goal_future = None
        return Status.FAILURE

    def _handle_invalid_state(self, message: str) -> Status:
        self.logger.error(
            f"[{self.name}] Reached invalid state: {message}. This is a bug."
        )
        return self._handle_failure(
            f"Invalid internal state: {message}", ActionExecutionStatus.FAILED
        )
