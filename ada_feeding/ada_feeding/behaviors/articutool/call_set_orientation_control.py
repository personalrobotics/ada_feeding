# -*- coding: utf-8 -*-
# (Add appropriate Copyright/License if desired)

"""
Defines the CallSetOrientationControl behavior, which calls the service
to enable/disable the Articutool orientation controller.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import Future # For type hints
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from typing import Union, Optional, Dict, Any, List, Tuple # Added Tuple just in case

from geometry_msgs.msg import PoseStamped, Quaternion # Need Quaternion if creating default PoseStamped
# Import your service definition (adjust path as needed)
from articutool_interfaces.srv import SetOrientationControl

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import numpy as np

# Local imports (adjust paths as needed)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CallSetOrientationControl(BlackboardBehavior):
    """
    Calls the SetOrientationControl service to enable/disable the
    Articutool's orientation controller and optionally set its target pose/orientation.

    Returns SUCCESS if the service call completes and the service response indicates success.
    Returns FAILURE if the service is unavailable, the call fails, or the service response indicates failure.
    Returns RUNNING while waiting for the service response.
    """

    DEFAULT_SERVICE_NAME = "/articutool/set_orientation_control"
    DEFAULT_WAIT_TIMEOUT_SEC = 1.0

    def blackboard_inputs(
        self,
        enable: Union[BlackboardKey, bool],
        quat_xyzw: Union[BlackboardKey, PoseStamped],
        service_name: Union[BlackboardKey, str] = DEFAULT_SERVICE_NAME,
        wait_for_server_timeout_sec: Union[BlackboardKey, float] = DEFAULT_WAIT_TIMEOUT_SEC,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        enable: True to enable orientation control, False to disable.
        quat_xyzw: The target orientation, provided as a geometry_msgs/Quaternion
                      message or an [x, y, z, w] list/tuple. Must be relative
                      to the frame expected by the orientation controller node.
        service_name: Name of the SetOrientationControl service.
        wait_for_server_timeout_sec: Max time (sec) to wait for server in initial check.
                                     Use 0.0 or negative to skip check (not recommended).
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        service_call_succeeded: Optional[BlackboardKey] = None, # -> bool (Did call complete?)
        service_response_success: Optional[BlackboardKey] = None, # -> bool (Payload 'success' field)
        service_response_message: Optional[BlackboardKey] = None, # -> str (Payload 'message' field)
    ) -> None:
        """
        Blackboard Outputs

        Parameters
        ----------
        service_call_succeeded: True if the service call finished without client-side errors/timeouts.
        service_response_success: The boolean 'success' field from the service response payload.
        service_response_message: The string 'message' field from the service response payload.
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """Get the node and create the service client."""
        # pylint: disable=attribute-defined-outside-init
        self.node: Node = kwargs['node']
        self.client = None # Initialize client to None
        try:
            self.service_name = self.blackboard_get("service_name")
            self.wait_timeout = self.blackboard_get("wait_for_server_timeout_sec")
            self.client = self.node.create_client(SetOrientationControl, self.service_name)
            self.logger.info(f"[{self.name}] Service client created for '{self.service_name}'")
        except KeyError as e:
             self.logger.error(f"[{self.name}] Missing blackboard key during setup: {e}")
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to create service client: {e}")

    def initialise(self) -> None:
        """Reset the future."""
        # pylint: disable=attribute-defined-outside-init
        self.service_future: Optional[Future] = None
        self.logger.debug(f"[{self.name}] Initializing service call state.")
        # Clear blackboard outputs
        self.blackboard_set("service_call_succeeded", False)
        self.blackboard_set("service_response_success", False)
        self.blackboard_set("service_response_message", "")

    def update(self) -> Status:
        """Manage the asynchronous service call state machine."""
        if self.client is None:
            self.logger.error(f"[{self.name}] Service client not available. Setup failed?")
            return Status.FAILURE

        # --- State 1: Ready to Call ---
        if self.service_future is None:
            # Check server availability (only if timeout > 0)
            wait_timeout = self.wait_timeout # Use timeout from setup
            if wait_timeout >= 0.0:
                 if not self.client.wait_for_service(timeout_sec=wait_timeout):
                     self.logger.error(f"[{self.name}] Service '{self.service_name}' not available after {wait_timeout}s timeout.")
                     return Status.FAILURE # Service unavailable is a failure

            try:
                # Read inputs from blackboard
                enable_flag = self.blackboard_get("enable")
                quat_xyzw = self.blackboard_get("quat_xyzw")

                # Input validation
                if not isinstance(enable_flag, bool):
                     self.logger.error(f"[{self.name}] Input 'enable' type mismatch: expected bool, got {type(enable_flag)}.")
                     return Status.FAILURE

                target_quat_msg = Quaternion()
                if isinstance(quat_xyzw, Quaternion):
                    target_quat_msg = quat_xyzw
                    self.logger.debug(f"[{self.name}] Using Quaternion message input.")
                elif isinstance(quat_xyzw, (list, tuple)) and len(quat_xyzw) == 4:
                    self.logger.debug(f"[{self.name}] Converting list/tuple input to Quaternion.")
                    try:
                        target_quat_msg.x = float(quat_xyzw[0])
                        target_quat_msg.y = float(quat_xyzw[1])
                        target_quat_msg.z = float(quat_xyzw[2])
                        target_quat_msg.w = float(quat_xyzw[3])
                        norm = np.linalg.norm([target_quat_msg.x, target_quat_msg.y, target_quat_msg.z, target_quat_msg.w])
                        if not np.isclose(norm, 1.0, atol=0.01): self.logger.warning(f"Input quat norm is {norm:.3f}")
                    except (ValueError, TypeError, IndexError) as e:
                        self.logger.error(f"Could not convert list/tuple {quat_xyzw} to Quaternion: {e}"); return Status.FAILURE
                else:
                    if enable_flag: # Target is mandatory if enabling
                        self.logger.error(f"[{self.name}] Input 'target_orientation_input' invalid type {type(quat_xyzw)} when enable=True.")
                        return Status.FAILURE
                    else: # Use default identity if disabling
                        self.logger.debug(f"[{self.name}] Using default identity quaternion because enable=False.")
                        target_quat_msg = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                # Create request
                req = SetOrientationControl.Request()
                req.enable = enable_flag
                req.target_orientation = target_quat_msg

                # Call service asynchronously
                self.logger.info(f"[{self.name}] Calling service '{self.service_name}' (enable={req.enable})...")
                self.service_future = self.client.call_async(req)
                return Status.RUNNING # Waiting for response

            except KeyError as e:
                self.logger.error(f"[{self.name}] Blackboard key error preparing service call: {e}")
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Error preparing service call: {e}")
                return Status.FAILURE

        # --- State 2: Waiting for Response ---
        elif not self.service_future.done():
            self.logger.debug(f"[{self.name}] Waiting for service response...")
            return Status.RUNNING

        # --- State 3: Processing Response ---
        else:
            self.logger.debug(f"[{self.name}] Service call future done.")
            response = None
            try:
                response = self.service_future.result() # Get the response payload
                self.blackboard_set("service_call_succeeded", True) # Call itself completed

                if response is not None:
                    self.blackboard_set("service_response_success", response.success)
                    self.blackboard_set("service_response_message", response.message)
                    self.logger.info(f"[{self.name}] Service response: success={response.success}, msg='{response.message}'")
                    final_status = Status.SUCCESS if response.success else Status.FAILURE
                else:
                    # Should not happen if future.result() didn't raise exception
                    self.logger.error(f"[{self.name}] Service call future done but result is None.")
                    self.blackboard_set("service_response_success", False)
                    final_status = Status.FAILURE

            except Exception as e:
                # Error during call execution on server or getting result
                self.logger.error(f"[{self.name}] Service call failed with exception: {e}")
                self.blackboard_set("service_call_succeeded", False) # Call failed
                self.blackboard_set("service_response_success", False)
                self.blackboard_set("service_response_message", f"Exception: {e}")
                final_status = Status.FAILURE

            # Reset future regardless of outcome for next tick
            self.service_future = None
            return final_status

    def terminate(self, new_status: Status) -> None:
        """Log termination status and clear future if needed."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        # If terminated while waiting for response, clear the future
        if self.service_future is not None and not self.service_future.done():
             self.logger.warning(f"[{self.name}] Service call terminated while waiting for response.")
             # We can't easily cancel a service call future, just abandon it
             self.service_future = None
