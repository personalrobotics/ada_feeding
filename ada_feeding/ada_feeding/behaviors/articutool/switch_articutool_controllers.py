#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the SwitchArticutoolControllers behavior, which calls the standard
controller_manager service to switch controllers for the Articutool.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import Future  # For type hints
from rclpy.duration import Duration

# Import the standard controller manager service definition
from controller_manager_msgs.srv import SwitchController

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status

# Import base class and helpers (Adjust path as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior

# Typing imports
from typing import Union, Optional, Dict, Any, List


class SwitchArticutoolControllers(BlackboardBehavior):
    """
    Calls the /articutool/controller_manager/switch_controller service
    to activate and deactivate specified controllers.

    Returns SUCCESS if the service call completes and the response indicates success (ok=True).
    Returns FAILURE if the service is unavailable, the call fails, or the response indicates failure.
    Returns RUNNING while waiting for the service response.
    """

    # Define default values used if not provided via blackboard inputs during instantiation
    DEFAULT_SERVICE_NAME = "/articutool/controller_manager/switch_controller"
    DEFAULT_WAIT_TIMEOUT_SEC = 2.0  # Timeout for wait_for_service
    DEFAULT_STRICTNESS = SwitchController.Request.BEST_EFFORT  # Default: 1 (STRICT=2)

    def blackboard_inputs(
        self,
        controllers_to_activate: Union[BlackboardKey, List[str]],
        controllers_to_deactivate: Union[BlackboardKey, List[str]],
        strictness: Union[BlackboardKey, int] = DEFAULT_STRICTNESS,
        service_name: Union[BlackboardKey, str] = DEFAULT_SERVICE_NAME,
        wait_for_server_timeout_sec: Union[
            BlackboardKey, float
        ] = DEFAULT_WAIT_TIMEOUT_SEC,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        controllers_to_activate: List of controller names to activate (start).
        controllers_to_deactivate: List of controller names to deactivate (stop).
        strictness: Strictness level for the switch operation. Options are
                    SwitchController.Request.BEST_EFFORT (1) or
                    SwitchController.Request.STRICT (2).
        service_name: Name of the controller manager's switch_controller service.
        wait_for_server_timeout_sec: Max time (sec) to wait for server in initial check.
                                     Use 0.0 or negative to skip check.
        """
        # Use the base class method to register keys/defaults
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        switch_call_succeeded: Optional[
            BlackboardKey
        ] = None,  # -> bool (Did call complete?)
        switch_response_ok: Optional[
            BlackboardKey
        ] = None,  # -> bool (Payload 'ok' field)
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        switch_call_succeeded: True if the service call finished without client-side errors.
        switch_response_ok: The boolean 'ok' field from the SwitchController response payload,
                            indicating if the controller manager performed the switch successfully.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        """Get node and create the service client."""
        # pylint: disable=attribute-defined-outside-init
        self.node: Node = kwargs["node"]
        self.client: Optional[rclpy.client.Client] = None
        try:
            # Read configuration parameters using the blackboard helper method
            self.service_name = self.blackboard_get("service_name")
            self.wait_timeout = self.blackboard_get("wait_for_server_timeout_sec")

            # Create client for the standard controller manager service
            self.client = self.node.create_client(SwitchController, self.service_name)
            self.logger.info(
                f"[{self.name}] Service client created for '{self.service_name}'"
            )

        except KeyError as e:
            self.logger.error(f"[{self.name}] Missing blackboard key during setup: {e}")
            self.client = None  # Ensure client is None if setup fails
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to create service client: {e}")
            self.client = None  # Ensure client is None if setup fails

    @override
    def initialise(self) -> None:
        """Reset the future before potentially calling the service."""
        # pylint: disable=attribute-defined-outside-init
        self.service_future: Optional[Future] = None
        self.logger.debug(f"[{self.name}] Initializing controller switch call state.")
        # Clear outputs to default non-success state
        self.blackboard_set("switch_call_succeeded", False)
        self.blackboard_set("switch_response_ok", False)

    @override
    def update(self) -> Status:
        """Manage the asynchronous service call state machine."""
        if self.client is None:
            self.logger.error(
                f"[{self.name}] Service client not available. Setup failed?"
            )
            return Status.FAILURE

        # --- State 1: Ready to Call ---
        if self.service_future is None:
            # Check server availability (only if timeout >= 0)
            wait_timeout = self.wait_timeout
            if wait_timeout >= 0.0:
                if not self.client.wait_for_service(timeout_sec=wait_timeout):
                    self.logger.error(
                        f"[{self.name}] Service '{self.service_name}' not available after {wait_timeout}s timeout."
                    )
                    return Status.FAILURE  # Service unavailable is a failure

            try:
                # Read inputs from blackboard
                activate_list = self.blackboard_get("controllers_to_activate")
                deactivate_list = self.blackboard_get("controllers_to_deactivate")
                strict_level = self.blackboard_get("strictness")

                # Basic Type Validation
                if not isinstance(activate_list, list) or not all(
                    isinstance(s, str) for s in activate_list
                ):
                    self.logger.error(
                        f"[{self.name}] Input 'controllers_to_activate' must be a List[str]."
                    )
                    return Status.FAILURE
                if not isinstance(deactivate_list, list) or not all(
                    isinstance(s, str) for s in deactivate_list
                ):
                    self.logger.error(
                        f"[{self.name}] Input 'controllers_to_deactivate' must be a List[str]."
                    )
                    return Status.FAILURE
                if not isinstance(strict_level, int):
                    self.logger.error(
                        f"[{self.name}] Input 'strictness' must be an integer."
                    )
                    return Status.FAILURE
                if strict_level not in [
                    SwitchController.Request.BEST_EFFORT,
                    SwitchController.Request.STRICT,
                ]:
                    self.logger.warning(
                        f"[{self.name}] Invalid strictness value {strict_level}, using BEST_EFFORT."
                    )
                    strict_level = SwitchController.Request.BEST_EFFORT

                # Create request
                req = SwitchController.Request()
                req.activate_controllers = activate_list
                req.deactivate_controllers = deactivate_list
                req.strictness = strict_level
                req.activate_asap = False  # Usually False for planned switches
                req.timeout = Duration(
                    seconds=5.0
                ).to_msg()  # Server-side timeout within CM

                # Call service async
                self.logger.info(
                    f"[{self.name}] Calling switch_controller: "
                    f"Activate={req.activate_controllers}, Deactivate={req.deactivate_controllers}, Strictness={req.strictness}"
                )
                self.service_future = self.client.call_async(req)
                return Status.RUNNING  # Waiting for response

            except KeyError as e:
                self.logger.error(f"[{self.name}] Blackboard key error: {e}")
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Error preparing service call: {e}")
                return Status.FAILURE

        # --- State 2: Waiting for Response ---
        elif not self.service_future.done():
            self.logger.debug(
                f"[{self.name}] Waiting for switch_controller response..."
            )
            return Status.RUNNING

        # --- State 3: Processing Response ---
        else:
            self.logger.info(f"[{self.name}] switch_controller call finished.")
            response = None
            call_succeeded = (
                False  # Did the call itself complete without client-side error?
            )
            response_ok = False  # Did the service response payload indicate success?
            try:
                response = self.service_future.result()  # Get the response payload
                call_succeeded = True  # If we get here, the call itself completed

                if response is not None:
                    response_ok = response.ok  # Check the 'ok' field in the response
                    self.logger.info(
                        f"[{self.name}] switch_controller response: ok={response_ok}"
                    )
                else:
                    self.logger.error(
                        f"[{self.name}] Service call future done but result is None."
                    )
                    response_ok = False

            except Exception as e:
                # Error during call execution on server or getting result
                self.logger.error(
                    f"[{self.name}] Service call failed with exception: {e}"
                )
                call_succeeded = False
                response_ok = False

            # Write outputs
            self.blackboard_set("switch_call_succeeded", call_succeeded)
            self.blackboard_set("switch_response_ok", response_ok)

            # Reset future
            self.service_future = None

            # Return status based on response 'ok' field
            return Status.SUCCESS if response_ok else Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination status and clear future if needed."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        # If terminated while waiting for response, clear the future
        if self.service_future is not None and not self.service_future.done():
            self.logger.warning(
                f"[{self.name}] Terminated while waiting for switch_controller response."
            )
            # We can't easily cancel a service call future, just abandon it
            self.service_future = None
