#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the ExecuteNamedPrimitive behavior, which executes a specified
Articutool primitive action by reading its name and parameters from the
blackboard.
"""

from typing import Union, Optional, Dict, Any, List
from enum import Enum
import traceback

import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from action_msgs.msg import GoalStatus
from rclpy.node import Node
from rclpy.executors import Future
from rclpy.duration import Duration as RCLPYDuration

from articutool_interfaces.action import ExecuteArticutoolPrimitive

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Status

# Assuming these are your actual base classes and helpers
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class ActionExecutionStatus(Enum):
    """Enum for tracking the state of the action client behavior."""

    IDLE = "IDLE"
    CHECKING_SERVER = "CHECKING_SERVER"
    SENDING_GOAL = "SENDING_GOAL"
    EXECUTING = "EXECUTING"
    GOAL_REJECTED = "GOAL_REJECTED"
    GOAL_CANCELLED = "GOAL_CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


class ExecuteNamedPrimitive(BlackboardBehavior):
    """
    Executes a named Articutool primitive by calling the ExecuteArticutoolPrimitive
    action server. The primitive name and its parameters are read from the blackboard.
    If the primitive name is "NONE" or empty, it succeeds immediately.
    """

    DEFAULT_ACTION_SERVER_NAME = "/orientation_control/execute_primitive"
    DEFAULT_WAIT_TIMEOUT_SEC = 2.0

    def __init__(self, name: str, ns: Optional[str] = None, **kwargs):
        super().__init__(name=name, ns=ns if ns else name, **kwargs)
        self.node: Optional[Node] = None
        self._action_client: Optional[ActionClient] = None
        self._send_goal_future: Optional[Future] = None
        self._get_result_future: Optional[Future] = None
        self._goal_handle: Optional[ClientGoalHandle] = None
        self._current_status: ActionExecutionStatus = ActionExecutionStatus.IDLE
        self._result: Optional[ExecuteArticutoolPrimitive.Result] = None

        self._action_server_name_str: str = self.DEFAULT_ACTION_SERVER_NAME
        self._wait_timeout_float: float = self.DEFAULT_WAIT_TIMEOUT_SEC

    def blackboard_inputs(
        self,
        primitive_name: Union[BlackboardKey, str],
        primitive_params: Union[BlackboardKey, List[float]] = [],
        action_server_name: Union[BlackboardKey, str] = DEFAULT_ACTION_SERVER_NAME,
        wait_for_server_timeout_sec: Union[
            BlackboardKey, float
        ] = DEFAULT_WAIT_TIMEOUT_SEC,
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        primitive_result: Optional[BlackboardKey] = None,
        primitive_status: Optional[BlackboardKey] = None,
    ) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs: Any) -> None:
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            return  # Critical failure

        try:
            self._action_server_name_str = self.blackboard_get("action_server_name")
            self._wait_timeout_float = self.blackboard_get(
                "wait_for_server_timeout_sec"
            )

            self._action_client = ActionClient(
                self.node, ExecuteArticutoolPrimitive, self._action_server_name_str
            )
            self.logger.info(
                f"[{self.name}] Action client created for '{self._action_server_name_str}'."
            )
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Failed to create action client during setup: {e} {traceback.format_exc()}"
            )
            self._action_client = None

    @override
    def initialise(self) -> None:
        self.logger.debug(f"[{self.name}] Initialising.")
        self._send_goal_future = None
        self._get_result_future = None
        self._goal_handle = None
        self._current_status = ActionExecutionStatus.IDLE
        self._result = None
        self.blackboard_set("primitive_result", None)
        self.blackboard_set("primitive_status", self._current_status.value)

    def _handle_failure(
        self,
        reason: str,
        final_status: ActionExecutionStatus = ActionExecutionStatus.FAILED,
    ) -> Status:
        self.logger.error(f"[{self.name}] Failure: {reason}")
        self._current_status = final_status
        self.blackboard_set("primitive_status", self._current_status.value)
        return Status.FAILURE

    def _handle_success(self, message: str) -> Status:
        self.logger.info(f"[{self.name}] Success: {message}")
        self._current_status = ActionExecutionStatus.SUCCEEDED
        self.blackboard_set("primitive_status", self._current_status.value)
        return Status.SUCCESS

    @override
    def update(self) -> Status:
        self.blackboard_set("primitive_status", self._current_status.value)

        if self.node is None or self._action_client is None:
            return self._handle_failure(
                "Behavior not initialized properly (node or action client is None)."
            )

        # --- State: IDLE -> Check inputs and transition ---
        if self._current_status == ActionExecutionStatus.IDLE:
            try:
                primitive_name = self.blackboard_get("primitive_name")
                # Handle the "do nothing" case
                if not primitive_name or primitive_name.upper() == "NONE":
                    return self._handle_success("No primitive specified.")

                # If a primitive is specified, transition to check for server
                self._current_status = ActionExecutionStatus.CHECKING_SERVER
                self.blackboard_set("primitive_status", self._current_status.value)
                return Status.RUNNING  # Return RUNNING to re-tick in the new state

            except KeyError as e:
                return self._handle_failure(f"Blackboard key error: {e}")

        # --- State: CHECKING_SERVER -> Wait for server then send goal ---
        if self._current_status == ActionExecutionStatus.CHECKING_SERVER:
            if not self._action_client.wait_for_server(
                timeout_sec=self._wait_timeout_float
            ):
                return self._handle_failure(
                    f"Action server '{self._action_server_name_str}' not available after {self._wait_timeout_float}s timeout."
                )

            try:
                # Re-read inputs in case they changed while waiting
                primitive_name = self.blackboard_get("primitive_name")
                primitive_params = self.blackboard_get("primitive_params") or []

                goal_msg = ExecuteArticutoolPrimitive.Goal()
                goal_msg.primitive_name = primitive_name
                goal_msg.parameters = [float(p) for p in primitive_params]

                self.logger.info(
                    f"[{self.name}] Sending goal '{primitive_name}' with params {goal_msg.parameters}"
                )
                self._send_goal_future = self._action_client.send_goal_async(goal_msg)
                self._current_status = ActionExecutionStatus.SENDING_GOAL
                return Status.RUNNING
            except Exception as e:
                return self._handle_failure(
                    f"Error sending goal: {e} {traceback.format_exc()}"
                )

        # --- State: SENDING_GOAL -> Waiting for goal acceptance ---
        if self._current_status == ActionExecutionStatus.SENDING_GOAL:
            if not self._send_goal_future.done():
                return Status.RUNNING
            try:
                self._goal_handle = self._send_goal_future.result()
                if not self._goal_handle.accepted:
                    return self._handle_failure(
                        "Goal rejected by action server.",
                        ActionExecutionStatus.GOAL_REJECTED,
                    )
                self._get_result_future = self._goal_handle.get_result_async()
                self._current_status = ActionExecutionStatus.EXECUTING
                return Status.RUNNING
            except Exception as e:
                return self._handle_failure(f"Exception getting goal handle: {e}")

        # --- State: EXECUTING -> Waiting for the result ---
        if self._current_status == ActionExecutionStatus.EXECUTING:
            if not self._get_result_future.done():
                return Status.RUNNING
            try:
                result_wrapper = self._get_result_future.result()
                self._result = result_wrapper.result
                self.blackboard_set("primitive_result", self._result)
                final_status = result_wrapper.status
                if final_status == GoalStatus.STATUS_SUCCEEDED and self._result.success:
                    return self._handle_success(
                        f"Primitive completed successfully: {self._result.message}"
                    )
                else:
                    final_enum = (
                        ActionExecutionStatus.ABORTED
                        if final_status == GoalStatus.STATUS_ABORTED
                        else (
                            ActionExecutionStatus.GOAL_CANCELLED
                            if final_status == GoalStatus.STATUS_CANCELED
                            else ActionExecutionStatus.FAILED
                        )
                    )
                    return self._handle_failure(
                        f"Primitive did not succeed. Final status: {final_status}, Result msg: '{self._result.message}'",
                        final_enum,
                    )
            except Exception as e:
                return self._handle_failure(
                    f"Exception while getting action result: {e}"
                )

        # Fallback
        return self._handle_failure(
            f"Reached unhandled state: {self._current_status.value}"
        )

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        if (
            new_status == Status.INVALID
            and self._goal_handle
            and self._goal_handle.is_active
        ):
            self.logger.warning(
                f"[{self.name}] Terminated externally. Attempting to cancel active goal."
            )
            self._goal_handle.cancel_goal_async()
        self.initialise()
