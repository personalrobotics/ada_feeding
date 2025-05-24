# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the TriggerArticutoolCalibration behavior, which calls the service
to trigger the Articutool's IMU orientation calibration routine.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import Future  # For type hints
from rclpy.duration import Duration as RCLPYDuration  # For service call timeout

from typing import Union, Optional, Dict, Any
import time  # For service call timeout logic

from geometry_msgs.msg import Quaternion as QuaternionMsg
from articutool_interfaces.srv import TriggerCalibration  # Service type

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status

# Local imports
from ada_feeding.helpers import BlackboardKey  # Assuming this is your helper
from ada_feeding.behaviors import BlackboardBehavior  # Assuming this is your base class


class TriggerArticutoolCalibration(BlackboardBehavior):
    """
    Calls the TriggerCalibration service to initiate Articutool's
    IMU orientation calibration.

    Returns SUCCESS if the service call completes and the service response indicates success.
    Returns FAILURE if the service is unavailable, the call fails, or the service response indicates failure.
    Returns RUNNING while waiting for the service response.
    """

    DEFAULT_SERVICE_NAME = (
        "/orientation_calibration_service/trigger_calibration"  # Default name
    )
    DEFAULT_WAIT_FOR_SERVER_TIMEOUT_SEC = 2.0
    DEFAULT_SERVICE_CALL_TIMEOUT_SEC = 10.0  # Calibration might take some time

    def blackboard_inputs(
        self,
        service_name: Union[BlackboardKey, str] = DEFAULT_SERVICE_NAME,
        wait_for_server_timeout_sec: Union[
            BlackboardKey, float
        ] = DEFAULT_WAIT_FOR_SERVER_TIMEOUT_SEC,
        service_call_timeout_sec: Union[
            BlackboardKey, float
        ] = DEFAULT_SERVICE_CALL_TIMEOUT_SEC,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        service_name: Name of the TriggerCalibration service.
        wait_for_server_timeout_sec: Max time (sec) to wait for server availability.
        service_call_timeout_sec: Max time (sec) to wait for the service call to complete.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        calibration_call_succeeded: Optional[
            BlackboardKey
        ] = None,  # bool: Did the ROS service call itself succeed?
        calibration_routine_success: Optional[
            BlackboardKey
        ] = None,  # bool: From service response .success
        calibration_message: Optional[
            BlackboardKey
        ] = None,  # string: From service response .message
        computed_calibration_offset_quat: Optional[
            BlackboardKey
        ] = None,  # QuaternionMsg: From service response
    ) -> None:
        """
        Blackboard Outputs
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """Get the node and create the service client."""
        self.node: Optional[Node] = kwargs.get("node")
        self.client = None
        self._service_name_str: str = self.DEFAULT_SERVICE_NAME
        self._wait_timeout_float: float = self.DEFAULT_WAIT_FOR_SERVER_TIMEOUT_SEC
        self._call_timeout_float: float = self.DEFAULT_SERVICE_CALL_TIMEOUT_SEC
        self._initialized_properly = False

        if not self.node:
            self.logger.error(
                f"[{self.name}] Node object not provided in setup kwargs."
            )
            return

        try:
            # It's better to get these once in setup if they don't change per tick,
            # or get them in update if they can change dynamically from the blackboard.
            # For service clients, usually setup is fine.
            self._service_name_str = self.blackboard_get("service_name")
            self._wait_timeout_float = self.blackboard_get(
                "wait_for_server_timeout_sec"
            )
            self._call_timeout_float = self.blackboard_get("service_call_timeout_sec")

            self.client = self.node.create_client(
                TriggerCalibration, self._service_name_str
            )
            self.logger.info(
                f"[{self.name}] Service client created for '{self._service_name_str}'"
            )
            self._initialized_properly = True
        except KeyError as e:
            self.logger.error(f"[{self.name}] Missing blackboard key during setup: {e}")
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Failed to create service client during setup: {e}"
            )

    def update(self) -> Status:
        """Manage the asynchronous service call state machine."""
        if not self._initialized_properly or self.client is None:
            self.logger.error(
                f"[{self.name}] Service client not available. Setup likely failed."
            )
            self.blackboard_set(
                "calibration_call_succeeded", False
            )  # Use internal port name
            self.blackboard_set("calibration_routine_success", False)
            return Status.FAILURE

        # --- State 1: Ready to Call ---
        if self.service_future is None:
            if self._wait_timeout_float >= 0.0:
                if not self.client.wait_for_service(
                    timeout_sec=self._wait_timeout_float
                ):
                    self.logger.error(
                        f"[{self.name}] Service '{self._service_name_str}' not available after {self._wait_timeout_float}s timeout."
                    )
                    self.blackboard_set("calibration_call_succeeded", False)
                    return Status.FAILURE

            # If wait_for_service timeout is < 0, we attempt call without waiting (not recommended for critical services)
            # elif not self.client.service_is_ready():
            #     self.logger.error(f"[{self.name}] Service '{self._service_name_str}' not ready (no wait).")
            #     self.blackboard_set("calibration_call_succeeded", False)
            #     return Status.FAILURE

            req = TriggerCalibration.Request()  # Empty request
            self.logger.info(
                f"[{self.name}] Calling service '{self._service_name_str}' to trigger calibration."
            )
            try:
                self.service_future = self.client.call_async(req)
                self.call_sent_time = time.monotonic()  # Record time when call was sent
                return Status.RUNNING
            except Exception as e:
                self.logger.error(f"[{self.name}] Error sending service request: {e}")
                self.blackboard_set("calibration_call_succeeded", False)
                return Status.FAILURE

        # --- State 2: Waiting for Response ---
        elif self.service_future and not self.service_future.done():
            if self.call_sent_time is not None and (
                time.monotonic() - self.call_sent_time > self._call_timeout_float
            ):
                self.logger.error(
                    f"[{self.name}] Service call to '{self._service_name_str}' timed out after {self._call_timeout_float}s."
                )
                # Attempt to cancel, though ROS 2 service call cancellation is not straightforward
                if hasattr(self.service_future, "cancel") and callable(
                    self.service_future.cancel
                ):
                    if (
                        not self.service_future.cancelled()
                    ):  # Check if it can be cancelled
                        if self.service_future.cancel():  # Try to cancel
                            self.logger.info(
                                f"[{self.name}] Service call future cancelled."
                            )
                        else:
                            self.logger.warn(
                                f"[{self.name}] Failed to cancel service call future (may have already completed or cannot be cancelled)."
                            )
                self.service_future = None  # Abandon the future
                self.call_sent_time = None
                self.blackboard_set("calibration_call_succeeded", False)
                self.blackboard_set("calibration_message", "Service call timed out")
                return Status.FAILURE

            self.logger.debug(
                f"[{self.name}] Waiting for calibration service response..."
            )
            return Status.RUNNING

        # --- State 3: Processing Response ---
        elif self.service_future:  # Future exists and is done
            self.logger.debug(f"[{self.name}] Calibration service call future done.")
            response_payload = None
            final_status = Status.FAILURE  # Default to failure
            try:
                response_payload = self.service_future.result()
                self.blackboard_set("calibration_call_succeeded", True)

                if response_payload is not None:
                    self.blackboard_set(
                        "calibration_routine_success", response_payload.success
                    )
                    self.blackboard_set("calibration_message", response_payload.message)
                    self.blackboard_set(
                        "computed_calibration_offset_quat",
                        response_payload.computed_offset_jacobase_to_filterworld,
                    )

                    self.logger.info(
                        f"[{self.name}] Calibration service response: success={response_payload.success}, msg='{response_payload.message}'"
                    )
                    if response_payload.success:
                        self.logger.info(
                            f"  Computed Offset (xyzw): ["
                            f"{response_payload.computed_offset_jacobase_to_filterworld.x:.4f}, "
                            f"{response_payload.computed_offset_jacobase_to_filterworld.y:.4f}, "
                            f"{response_payload.computed_offset_jacobase_to_filterworld.z:.4f}, "
                            f"{response_payload.computed_offset_jacobase_to_filterworld.w:.4f}]"
                        )
                    final_status = (
                        Status.SUCCESS if response_payload.success else Status.FAILURE
                    )
                else:
                    self.logger.error(
                        f"[{self.name}] Service call future done but result is None."
                    )
                    self.blackboard_set("calibration_routine_success", False)
                    self.blackboard_set(
                        "calibration_message", "Service call returned None result"
                    )
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] Service call failed with exception: {e}"
                )
                self.blackboard_set("calibration_call_succeeded", False)
                self.blackboard_set("calibration_routine_success", False)
                self.blackboard_set("calibration_message", f"Exception: {e}")

            self.service_future = None  # Reset for next call
            self.call_sent_time = None
            return final_status

        # Should not be reached
        self.logger.warn(f"[{self.name}] Reached unexpected state in update method.")
        return Status.FAILURE

    def terminate(self, new_status: Status) -> None:
        """Log termination status and clear future if needed."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        if self.service_future is not None and not self.service_future.done():
            self.logger.warning(
                f"[{self.name}] Service call terminated while waiting for response."
            )
            # Note: ROS 2 futures are not easily cancellable once sent if the service doesn't support it.
            # We just abandon it here.
        self.service_future = None
        self.call_sent_time = None
