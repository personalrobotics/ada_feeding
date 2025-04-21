# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

# (Add appropriate Copyright/License if desired)

"""
Defines the ExecuteArticutoolTrajectory behavior, which sends a planned
trajectory to the Articutool's FollowJointTrajectory action server,
handling controller switching.
"""

# Standard imports
from typing import Union, Optional, Dict, Any, List
from enum import Enum

# Third-party imports
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, GoalStatus
from rclpy.node import Node
from rclpy.executors import Future  # For type hints

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory  # Input type

# from builtin_interfaces.msg import Duration # Used internally by action goal
# from sensor_msgs.msg import JointState # Not directly needed

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior

# Assuming ControllerSwitcher is importable and works with a passed node
from articutool_control.controller_switcher import ControllerSwitcher


# Define constants for action status reporting (optional but clearer)
class ActionExecutionStatus(Enum):
    IDLE = "IDLE"
    SWITCHING_CONTROLLERS = "SWITCHING_CONTROLLERS"
    SENDING_GOAL = "SENDING_GOAL"
    WAITING_FOR_ACCEPTANCE = "WAITING_FOR_ACCEPTANCE"
    EXECUTING = "EXECUTING"
    WAITING_FOR_RESULT = (
        "WAITING_FOR_RESULT"  # If execution status isn't monitored closely
    )
    GOAL_REJECTED = "GOAL_REJECTED"
    GOAL_CANCELLED = "GOAL_CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"  # From server
    FAILED = "FAILED"  # General failure


class ExecuteArticutoolTrajectory(BlackboardBehavior):
    """
    Executes a JointTrajectory on the Articutool using the
    FollowJointTrajectory action server. Handles controller switching before
    sending the goal. Returns RUNNING while executing, SUCCESS on completion,
    FAILURE on error, rejection, or abortion.
    """

    # --- Constants ---
    DEFAULT_ACTION_SERVER = (
        "/articutool/joint_trajectory_controller/follow_joint_trajectory"
    )
    DEFAULT_CONTROLLER_TO_ACTIVATE = ["joint_trajectory_controller"]  # Expects list
    DEFAULT_CONTROLLERS_TO_DEACTIVATE = ["velocity_controller"]  # Expects list

    def __init__(self, name: str, ns: str = "/", **kwargs):
        # Pass kwargs to parent if BlackboardBehavior supports it
        super().__init__(name=name, ns=ns, **kwargs)
        # Internal state variables
        self._action_client: Optional[ActionClient] = None
        self.controller_switcher: Optional[ControllerSwitcher] = None
        self._send_goal_future: Optional[Future] = None
        self._get_result_future: Optional[Future] = None
        self._goal_handle: Optional[ClientGoalHandle] = None
        self._current_status: ActionExecutionStatus = ActionExecutionStatus.IDLE
        self._result_code: Optional[int] = None  # Store the final result code

    def blackboard_inputs(
        self,
        trajectory: Union[BlackboardKey, JointTrajectory],
        controllers_to_activate: Union[
            BlackboardKey, List[str]
        ] = DEFAULT_CONTROLLER_TO_ACTIVATE,
        controllers_to_deactivate: Union[
            BlackboardKey, List[str]
        ] = DEFAULT_CONTROLLERS_TO_DEACTIVATE,
        action_server_name: Union[BlackboardKey, str] = DEFAULT_ACTION_SERVER,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        trajectory: The trajectory_msgs/JointTrajectory message to execute.
        controllers_to_activate: List of controller names to activate before sending.
        controllers_to_deactivate: List of controller names to deactivate before sending.
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
        """Get node, create action client and controller switcher."""
        # pylint: disable=attribute-defined-outside-init
        try:
            self.node: Node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            return  # Cannot function without node

        try:
            try:
                action_server_name = self.blackboard_get("action_server_name")
                if not isinstance(action_server_name, str) or not action_server_name:
                    self.logger.warning(
                        f"[{self.name}] Blackboard 'action_server_name' invalid, using default."
                    )
                    action_server_name = self.DEFAULT_ACTION_SERVER
            except KeyError:
                self.logger.info(
                    f"[{self.name}] Input 'action_server_name' not found, using default: {self.DEFAULT_ACTION_SERVER}"
                )
                action_server_name = self.DEFAULT_ACTION_SERVER

            self._action_client = ActionClient(
                self.node, FollowJointTrajectory, action_server_name
            )
            # Check server availability briefly during setup
            timeout_sec = 1.0
            if not self._action_client.wait_for_server(timeout_sec=timeout_sec):
                self.logger.error(
                    f"[{self.name}] Action server '{action_server_name}' not available after {timeout_sec}s during setup."
                )
                self._action_client = None  # Mark as unavailable
                return  # Setup fails

            # Instantiate Controller Switcher (pass node if its __init__ requires it)
            try:
                # Assuming ControllerSwitcher might take node as arg
                self.controller_switcher = ControllerSwitcher(node=self.node)
            except TypeError:  # If ControllerSwitcher() doesn't take node
                self.controller_switcher = ControllerSwitcher()

            self.logger.info(
                f"[{self.name}] Setup complete. Action client created for '{action_server_name}'."
            )

        except Exception as e:
            self.logger.error(f"[{self.name}] Failed during setup: {e}")
            self._action_client = None
            self.controller_switcher = None

    @override
    def initialise(self) -> None:
        """Reset action client state."""
        self.logger.debug(f"[{self.name}] Initialising.")
        self._send_goal_future = None
        self._get_result_future = None
        self._goal_handle = None
        self._current_status = ActionExecutionStatus.IDLE
        self._result_code = None  # Reset result code
        # Clear blackboard outputs
        self.blackboard_set("action_goal_accepted", False)
        self.blackboard_set(
            "action_result_code", self._result_code
        )  # Set to None or initial value
        self.blackboard_set("action_status", self._current_status.value)

    @override
    def update(self) -> Status:
        """Manage the action client state machine."""
        # Update blackboard status at start of tick
        self.blackboard_set("action_status", self._current_status.value)

        if self._action_client is None or self.controller_switcher is None:
            self.logger.error(
                f"[{self.name}] Action client or controller switcher not available. Setup failed?"
            )
            return Status.FAILURE  # Setup must have failed

        # --- State 1: IDLE -> Try sending goal ---
        if self._current_status == ActionExecutionStatus.IDLE:
            try:
                trajectory: JointTrajectory = self.blackboard_get("trajectory")
                controllers_to_activate: List[str] = self.blackboard_get(
                    "controllers_to_activate"
                )
                controllers_to_deactivate: List[str] = self.blackboard_get(
                    "controllers_to_deactivate"
                )

                # Validate inputs
                if not isinstance(trajectory, JointTrajectory):
                    self.logger.error(
                        f"[{self.name}] Input 'trajectory' is not a JointTrajectory message."
                    )
                    return Status.FAILURE
                if not trajectory.points:
                    self.logger.warning(
                        f"[{self.name}] Input 'trajectory' has no points. Succeeding trivially."
                    )
                    self._current_status = ActionExecutionStatus.SUCCEEDED
                    self.blackboard_set("action_status", self._current_status.value)
                    self.blackboard_set(
                        "action_result_code", FollowJointTrajectory.Result.SUCCESSFUL
                    )
                    return Status.SUCCESS
                if not isinstance(controllers_to_activate, list) or not isinstance(
                    controllers_to_deactivate, list
                ):
                    self.logger.error(f"[{self.name}] Controller inputs must be lists.")
                    return Status.FAILURE

                # 1a. Switch Controllers
                self._current_status = ActionExecutionStatus.SWITCHING_CONTROLLERS
                self.blackboard_set("action_status", self._current_status.value)
                self.logger.info(
                    f"[{self.name}] Requesting controller switch: start={controllers_to_activate}, stop={controllers_to_deactivate}"
                )
                try:
                    # Assuming switch_controllers blocks or returns success/failure quickly
                    self.controller_switcher.switch_controllers(
                        activate_controllers=controllers_to_activate,
                        deactivate_controllers=controllers_to_deactivate,
                    )
                except Exception as sw_e:
                    self.logger.error(
                        f"[{self.name}] Error during controller switch: {sw_e}"
                    )
                    self._current_status = ActionExecutionStatus.FAILED
                    self.blackboard_set("action_status", self._current_status.value)
                    return Status.FAILURE

                # 1b. Construct Goal Message
                goal_msg = FollowJointTrajectory.Goal()
                # Important: Make sure the trajectory has correct joint names for the action server
                goal_msg.trajectory = trajectory
                # Optional: Add goal time tolerance, path tolerance if needed
                # goal_msg.goal_time_tolerance = Duration(sec=1, nanosec=0).to_msg()

                # 1c. Send Goal Asynchronously
                self.logger.info(
                    f"[{self.name}] Sending trajectory goal to action server..."
                )
                self._send_goal_future = self._action_client.send_goal_async(goal_msg)
                self._current_status = ActionExecutionStatus.SENDING_GOAL
                self.blackboard_set("action_status", self._current_status.value)
                return Status.RUNNING

            except KeyError as e:
                self.logger.error(f"[{self.name}] Blackboard key error: {e}")
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Error preparing to send goal: {e}")
                return Status.FAILURE

        # --- State 2: Waiting for Goal Acceptance ---
        if self._current_status == ActionExecutionStatus.SENDING_GOAL:
            if self._send_goal_future is None:
                return self._handle_invalid_state(
                    "SENDING_GOAL future None"
                )  # Should not happen

            if self._send_goal_future.done():
                try:
                    goal_handle: ClientGoalHandle = self._send_goal_future.result()
                except Exception as e:
                    self.logger.error(
                        f"[{self.name}] Exception getting goal handle from future: {e}"
                    )
                    return self._handle_failure("Exception getting goal handle")

                self._send_goal_future = None  # Clear future

                if not goal_handle.accepted:
                    self.logger.warning(
                        f"[{self.name}] Goal rejected by action server."
                    )
                    self.blackboard_set("action_goal_accepted", False)
                    return self._handle_failure(
                        "Goal rejected", ActionExecutionStatus.GOAL_REJECTED
                    )
                else:
                    # Convert UUID numpy array to hex string for logging
                    try:
                        # Access the uuid field (numpy array of uint8)
                        uuid_array = goal_handle.goal_id.uuid
                        # Convert the numpy array to standard Python bytes
                        uuid_bytes = bytes(uuid_array)
                        # Convert bytes to a hex string
                        uuid_hex = uuid_bytes.hex()
                    except Exception as log_e:
                        # Fallback in case something goes wrong with conversion
                        self.logger.warning(
                            f"[{self.name}] Could not format goal ID UUID for logging: {log_e}"
                        )
                        uuid_hex = "[Error Formatting UUID]"

                    self.logger.info(
                        f"[{self.name}] Goal accepted by action server (ID: {uuid_hex})."
                    )
                    self._goal_handle = goal_handle
                    self.blackboard_set("action_goal_accepted", True)
                    self._get_result_future = self._goal_handle.get_result_async()
                    self._current_status = (
                        ActionExecutionStatus.EXECUTING
                    )  # Or WAITING_FOR_RESULT if preferred
                    self.blackboard_set("action_status", self._current_status.value)
                    return Status.RUNNING
            else:
                # Future not done yet
                return Status.RUNNING

        # --- State 3: Waiting for Result / Monitoring Execution ---
        if (
            self._current_status == ActionExecutionStatus.EXECUTING
            or self._current_status == ActionExecutionStatus.WAITING_FOR_RESULT
        ):
            if self._get_result_future is None or self._goal_handle is None:
                return self._handle_invalid_state(
                    "EXECUTING/WAITING future or handle None"
                )

            # Check goal handle status
            status = self._goal_handle.status
            if status == GoalStatus.STATUS_EXECUTING:
                self._current_status = (
                    ActionExecutionStatus.EXECUTING
                )  # Ensure state is correct
                # Optional: Process feedback if needed
                # self.logger.debug(f"[{self.name}] Goal executing...")
            elif status == GoalStatus.STATUS_ABORTED:
                self.logger.warning(f"[{self.name}] Goal aborted by server.")
                # Result future should complete soon, wait for it
                self._current_status = (
                    ActionExecutionStatus.WAITING_FOR_RESULT
                )  # Move to check result
            elif status == GoalStatus.STATUS_CANCELED:
                self.logger.warning(f"[{self.name}] Goal canceled.")
                self._current_status = ActionExecutionStatus.WAITING_FOR_RESULT
            elif status == GoalStatus.STATUS_SUCCEEDED:
                self.logger.info(f"[{self.name}] Goal status reported SUCCEEDED.")
                self._current_status = ActionExecutionStatus.WAITING_FOR_RESULT

            # Now check if the result future is done
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

                    # Final state determination
                    if self._result_code == FollowJointTrajectory.Result.SUCCESSFUL:
                        return self._handle_success()
                    elif (
                        status == GoalStatus.STATUS_CANCELED
                    ):  # Check status again if needed
                        return self._handle_failure(
                            "Goal Canceled", ActionExecutionStatus.GOAL_CANCELLED
                        )
                    else:  # Any other error code implies failure
                        return self._handle_failure(
                            f"Action Failed: Code={self._result_code}, Msg='{result_string}'",
                            ActionExecutionStatus.ABORTED
                            if status == GoalStatus.STATUS_ABORTED
                            else ActionExecutionStatus.FAILED,
                        )

                except Exception as e:
                    self.logger.error(
                        f"[{self.name}] Exception getting result from future: {e}"
                    )
                    return self._handle_failure("Exception getting result")
            else:
                # Result future not done yet, goal still active (or transitioning)
                self.blackboard_set("action_status", self._current_status.value)
                return Status.RUNNING

        # --- Handle Terminal States (Should be reached via returns above) ---
        if self._current_status == ActionExecutionStatus.SUCCEEDED:
            return Status.SUCCESS
        if self._current_status in [
            ActionExecutionStatus.FAILED,
            ActionExecutionStatus.GOAL_REJECTED,
            ActionExecutionStatus.GOAL_CANCELLED,
            ActionExecutionStatus.ABORTED,
        ]:
            return Status.FAILURE

        # Fallback if state is somehow invalid
        return self._handle_invalid_state(f"Unhandled status {self._current_status}")

    @override
    def terminate(self, new_status: Status) -> None:
        """Cancel active goal if behavior is terminated unexpectedly."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        # If terminated INTERNALLY (SUCCESS/FAILURE), goal is already finished or failed.
        # If terminated EXTERNALLY (INVALID), cancel the goal if it's active.
        if new_status == Status.INVALID and self._goal_handle is not None:
            status = self._goal_handle.status
            if (
                status == GoalStatus.STATUS_ACCEPTED
                or status == GoalStatus.STATUS_EXECUTING
            ):
                self.logger.warning(
                    f"[{self.name}] Behavior terminated externally while action goal was active (Status: {GoalStatus(status).name}). Requesting cancellation."
                )
                cancel_future = self._goal_handle.cancel_goal_async()
                # Optional: Spin briefly/add callback to ensure cancel sent? Usually not needed.
            # else:
            #     self.logger.debug(f"[{self.name}] Goal handle exists but not in active state ({GoalStatus(status).name}) on termination.")
        # else:
        # self.logger.debug(f"[{self.name}] No active goal handle or clean termination, no cancellation needed.")

        # Reset internal state variables regardless
        self._send_goal_future = None
        self._get_result_future = None
        self._goal_handle = None
        # Don't reset self._current_status here, it might be SUCCEEDED/FAILED from last update
        # self._current_status = ActionExecutionStatus.IDLE # Resetting might hide final state

    # --- Helper methods for state transitions ---
    def _handle_success(self) -> Status:
        self._current_status = ActionExecutionStatus.SUCCEEDED
        self.blackboard_set("action_status", self._current_status.value)
        self._goal_handle = None
        self._get_result_future = None
        return Status.SUCCESS

    def _handle_failure(
        self,
        reason: str,
        final_status: ActionExecutionStatus = ActionExecutionStatus.FAILED,
    ) -> Status:
        self.logger.error(f"[{self.name}] Failure: {reason}")
        self._current_status = final_status
        self.blackboard_set("action_status", self._current_status.value)
        # Set result code if available and not already set failure code
        if (
            self._result_code is not None
            and self._result_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self.blackboard_set("action_result_code", self._result_code)
        elif self._current_status == ActionExecutionStatus.GOAL_REJECTED:
            # Use a convention for rejected?
            self.blackboard_set("action_result_code", -998)  # Example arbitrary code
        else:  # General failure
            self.blackboard_set("action_result_code", -999)  # Example arbitrary code

        self._goal_handle = None
        self._get_result_future = None
        self._send_goal_future = None  # Ensure this is cleared too
        return Status.FAILURE

    def _handle_invalid_state(self, message: str) -> Status:
        self.logger.error(f"[{self.name}] Reached invalid state: {message}")
        return self._handle_failure(f"Invalid internal state: {message}")
