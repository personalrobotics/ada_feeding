# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
Defines the CallSetOrientationControl behavior, which calls the service
to set the Articutool orientation controller's mode and targets.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import Future  # For type hints

from typing import Union, Optional, Dict, Any, List, Tuple
import math

from geometry_msgs.msg import Quaternion
from articutool_interfaces.srv import SetOrientationControl

from overrides import override
import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status
import numpy as np

# Local imports
from ada_feeding.helpers import BlackboardKey
from ada_feeding.behaviors import BlackboardBehavior


class CallSetOrientationControl(BlackboardBehavior):
    """
    Calls the SetOrientationControl service to set the Articutool's
    orientation control mode and corresponding targets.

    Returns SUCCESS if the service call completes and the service response indicates success.
    Returns FAILURE if the service is unavailable, the call fails, or the service response indicates failure.
    Returns RUNNING while waiting for the service response.
    """

    DEFAULT_SERVICE_NAME = "/articutool/set_orientation_control"
    DEFAULT_WAIT_TIMEOUT_SEC = 1.0

    def blackboard_inputs(
        self,
        control_mode: Union[
            BlackboardKey, int
        ],  # MODE_DISABLED, MODE_LEVELING, MODE_FULL_ORIENTATION
        pitch_offset_deg: Optional[
            Union[BlackboardKey, float]
        ] = 0.0,  # For MODE_LEVELING
        roll_offset_deg: Optional[
            Union[BlackboardKey, float]
        ] = 0.0,  # For MODE_LEVELING
        target_orientation_robot_base_quat: Optional[
            Union[BlackboardKey, Quaternion, List[float], Tuple[float, ...]]
        ] = None,  # For MODE_FULL_ORIENTATION
        service_name: Union[BlackboardKey, str] = DEFAULT_SERVICE_NAME,
        wait_for_server_timeout_sec: Union[
            BlackboardKey, float
        ] = DEFAULT_WAIT_TIMEOUT_SEC,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        control_mode: The desired control mode (e.g., SetOrientationControl.Request.MODE_LEVELING).
        pitch_offset_deg: Target pitch offset in degrees (for MODE_LEVELING).
        roll_offset_deg: Target roll offset in degrees (for MODE_LEVELING).
        target_orientation_robot_base_quat: Target orientation as Quaternion msg or [x,y,z,w] list/tuple
                                            (for MODE_FULL_ORIENTATION), relative to robot_base_frame.
        service_name: Name of the SetOrientationControl service.
        wait_for_server_timeout_sec: Max time (sec) to wait for server.
        """
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        service_call_succeeded: Optional[BlackboardKey] = None,
        service_response_success: Optional[BlackboardKey] = None,
        service_response_message: Optional[BlackboardKey] = None,
    ) -> None:
        """
        Blackboard Outputs
        (Same as before)
        """
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def setup(self, **kwargs):
        """Get the node and create the service client."""
        self.node: Optional[Node] = kwargs.get("node")  # Use .get() for safety
        self.client = None
        if not self.node:
            self.logger.error(
                f"[{self.name}] Node object not provided in setup kwargs."
            )
            return

        try:
            # It's better to get these once in setup if they don't change per tick
            # Or get them in update if they can change dynamically from the blackboard
            self._service_name_str = self.blackboard_get("service_name")
            self._wait_timeout_float = self.blackboard_get(
                "wait_for_server_timeout_sec"
            )

            self.client = self.node.create_client(
                SetOrientationControl, self._service_name_str
            )
            self.logger.info(
                f"[{self.name}] Service client created for '{self._service_name_str}'"
            )
        except KeyError as e:
            self.logger.error(f"[{self.name}] Missing blackboard key during setup: {e}")
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to create service client: {e}")

    def initialise(self) -> None:
        """Reset the future."""
        self.service_future: Optional[Future] = None
        self.logger.debug(f"[{self.name}] Initializing service call state.")
        self.blackboard_set("service_call_succeeded", False)
        self.blackboard_set("service_response_success", False)
        self.blackboard_set("service_response_message", "")

    def update(self) -> Status:
        """Manage the asynchronous service call state machine."""
        if self.client is None:
            self.logger.error(
                f"[{self.name}] Service client not available. Setup likely failed or did not complete."
            )
            return Status.FAILURE

        # --- State 1: Ready to Call ---
        if self.service_future is None:
            if (
                self._wait_timeout_float >= 0.0
            ):  # Check server availability only if timeout is non-negative
                if not self.client.wait_for_service(
                    timeout_sec=self._wait_timeout_float
                ):
                    self.logger.error(
                        f"[{self.name}] Service '{self._service_name_str}' not available after {self._wait_timeout_float}s timeout."
                    )
                    return Status.FAILURE

            try:
                # Read all necessary inputs from blackboard for this call
                control_mode_input = self.blackboard_get("control_mode")

                # Validate control_mode
                if control_mode_input not in [
                    SetOrientationControl.Request.MODE_DISABLED,
                    SetOrientationControl.Request.MODE_LEVELING,
                    SetOrientationControl.Request.MODE_FULL_ORIENTATION,
                ]:
                    self.logger.error(
                        f"[{self.name}] Invalid 'control_mode' input: {control_mode_input}."
                    )
                    return Status.FAILURE

                req = SetOrientationControl.Request()
                req.control_mode = int(control_mode_input)  # Ensure it's int

                # Populate fields based on mode
                if req.control_mode == SetOrientationControl.Request.MODE_LEVELING:
                    pitch_deg = self.blackboard_try_get(
                        "pitch_offset_deg", 0.0
                    )  # Default to 0 if not found
                    roll_deg = self.blackboard_try_get(
                        "roll_offset_deg", 0.0
                    )  # Default to 0 if not found
                    req.pitch_offset = float(math.radians(pitch_deg))
                    req.roll_offset = float(math.radians(roll_deg))
                    # Set a default for target_orientation_robot_base when not in MODE_FULL_ORIENTATION
                    req.target_orientation_robot_base = Quaternion(
                        x=0.0, y=0.0, z=0.0, w=1.0
                    )
                    self.logger.info(
                        f"[{self.name}] Setting MODE_LEVELING with pitch_offset_rad={req.pitch_offset:.3f}, roll_offset_rad={req.roll_offset:.3f}"
                    )

                elif (
                    req.control_mode
                    == SetOrientationControl.Request.MODE_FULL_ORIENTATION
                ):
                    target_orient_input = self.blackboard_try_get(
                        "target_orientation_robot_base_quat"
                    )
                    if target_orient_input is None:
                        self.logger.error(
                            f"[{self.name}] 'target_orientation_robot_base_quat' is required for MODE_FULL_ORIENTATION but not found or None."
                        )
                        return Status.FAILURE

                    temp_quat_msg = Quaternion()
                    if isinstance(target_orient_input, Quaternion):
                        temp_quat_msg = target_orient_input
                    elif (
                        isinstance(target_orient_input, (list, tuple))
                        and len(target_orient_input) == 4
                    ):
                        try:
                            temp_quat_msg.x = float(target_orient_input[0])
                            temp_quat_msg.y = float(target_orient_input[1])
                            temp_quat_msg.z = float(target_orient_input[2])
                            temp_quat_msg.w = float(target_orient_input[3])
                            norm = np.linalg.norm(
                                [
                                    temp_quat_msg.x,
                                    temp_quat_msg.y,
                                    temp_quat_msg.z,
                                    temp_quat_msg.w,
                                ]
                            )
                            if not np.isclose(norm, 1.0, atol=0.01) and not np.isclose(
                                norm, 0.0, atol=0.01
                            ):  # Allow zero quat if it means "don't care"
                                self.logger.warning(
                                    f"Input target_orientation_robot_base_quat norm is {norm:.3f}, not 1.0."
                                )
                        except (ValueError, TypeError, IndexError) as e:
                            self.logger.error(
                                f"Could not convert list/tuple {target_orient_input} to Quaternion: {e}"
                            )
                            return Status.FAILURE
                    else:
                        self.logger.error(
                            f"[{self.name}] Input 'target_orientation_robot_base_quat' invalid type {type(target_orient_input)} for MODE_FULL_ORIENTATION."
                        )
                        return Status.FAILURE
                    req.target_orientation_robot_base = temp_quat_msg
                    # Set defaults for leveling offsets when not in MODE_LEVELING
                    req.pitch_offset = 0.0
                    req.roll_offset = 0.0
                    self.logger.info(
                        f"[{self.name}] Setting MODE_FULL_ORIENTATION with target_quat (xyzw): [{req.target_orientation_robot_base.x:.3f}, {req.target_orientation_robot_base.y:.3f}, {req.target_orientation_robot_base.z:.3f}, {req.target_orientation_robot_base.w:.3f}]"
                    )

                else:  # MODE_DISABLED or other
                    # Set defaults for all orientation fields
                    req.pitch_offset = 0.0
                    req.roll_offset = 0.0
                    req.target_orientation_robot_base = Quaternion(
                        x=0.0, y=0.0, z=0.0, w=1.0
                    )
                    self.logger.info(
                        f"[{self.name}] Setting control_mode to {req.control_mode} (likely DISABLED)."
                    )

                self.logger.info(
                    f"[{self.name}] Calling service '{self._service_name_str}' with mode={req.control_mode}..."
                )
                self.service_future = self.client.call_async(req)
                return Status.RUNNING

            except KeyError as e:
                self.logger.error(
                    f"[{self.name}] Blackboard key error preparing service call: {e}"
                )
                return Status.FAILURE
            except Exception as e:
                self.logger.error(f"[{self.name}] Error preparing service call: {e}")
                return Status.FAILURE

        # --- State 2: Waiting for Response ---
        elif (
            self.service_future and not self.service_future.done()
        ):  # Added check for self.service_future not None
            self.logger.debug(f"[{self.name}] Waiting for service response...")
            return Status.RUNNING

        # --- State 3: Processing Response ---
        elif self.service_future:  # Ensure future exists before trying to get result
            self.logger.debug(f"[{self.name}] Service call future done.")
            response_payload = None
            try:
                response_payload = self.service_future.result()
                self.blackboard_set("service_call_succeeded", True)

                if response_payload is not None:
                    self.blackboard_set(
                        "service_response_success", response_payload.success
                    )
                    self.blackboard_set(
                        "service_response_message", response_payload.message
                    )
                    self.logger.info(
                        f"[{self.name}] Service response: success={response_payload.success}, msg='{response_payload.message}'"
                    )
                    final_status = (
                        Status.SUCCESS if response_payload.success else Status.FAILURE
                    )
                else:
                    self.logger.error(
                        f"[{self.name}] Service call future done but result is None."
                    )
                    self.blackboard_set("service_response_success", False)
                    final_status = Status.FAILURE
            except Exception as e:
                self.logger.error(
                    f"[{self.name}] Service call failed with exception: {e}",
                )
                self.blackboard_set("service_call_succeeded", False)
                self.blackboard_set("service_response_success", False)
                self.blackboard_set("service_response_message", f"Exception: {e}")
                final_status = Status.FAILURE

            self.service_future = None  # Reset for next call
            return final_status

        # Should not be reached if logic is correct
        self.logger.warn(f"[{self.name}] Reached unexpected state in update method.")
        return Status.FAILURE

    def terminate(self, new_status: Status) -> None:
        """Log termination status and clear future if needed."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
        if self.service_future is not None and not self.service_future.done():
            self.logger.warning(
                f"[{self.name}] Service call terminated while waiting for response."
            )
            # Future cannot be truly cancelled easily, just abandoned
            self.service_future = None
