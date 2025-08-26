#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the ConditionallyRotateFoodFrame behavior.
It takes an existing food_frame transform and, based on a boolean flag,
rotates it around its Z-axis to align with the robot's approach vector.
It then re-publishes the updated static TF transform.
"""

import math
from typing import Union, Optional
from copy import deepcopy

from geometry_msgs.msg import TransformStamped
import numpy as np
from overrides import override
import py_trees
from py_trees.common import Status
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import ros2_numpy
import tf2_ros

from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey, get_tf_object, set_static_tf


class ConditionallyRotateFoodFrame(BlackboardBehavior):
    """
    Conditionally rotates the food_frame around its Z-axis to align its
    local X-axis with the vector from the robot base to the food origin.
    This behavior reads the decision from the loaded action schema.
    """

    EPSILON = 1e-6

    def blackboard_inputs(
        self,
        # The initial, un-aligned food_frame transform
        initial_food_frame: Union[BlackboardKey, TransformStamped],
        # The boolean flag read from the action schema
        should_align_to_base: Union[BlackboardKey, bool],
        # The frames needed for the calculation
        food_frame_id: Union[BlackboardKey, str] = "food",
        robot_base_frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        # This behavior UPDATES the food_frame on the blackboard, so it needs
        # to declare it as an output to get write access.
        # It reads the key from its inputs and writes back to the same key.
        food_frame_updated: BlackboardKey,
    ) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[Node] = None
        self.tf_buffer: Optional[tf2_ros.Buffer] = None
        self.tf_listener: Optional[tf2_ros.TransformListener] = None
        # _initialized_properly is handled by the base class setup

    @override
    def setup(self, **kwargs):
        # This setup pattern is from your original code
        try:
            self.node = kwargs["node"]
            self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during setup: {e}")

    def _normalize_angle(self, angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @override
    def update(self) -> Status:
        if not hasattr(self, "node") or self.node is None or self.tf_buffer is None:
            self.feedback_message = "Behavior not properly initialized."
            return Status.FAILURE

        try:
            should_align = self.blackboard_get("should_align_to_base")

            if not should_align:
                self.feedback_message = (
                    "Alignment not required by action schema. Succeeding."
                )
                self.logger.info(f"[{self.name}] {self.feedback_message}")
                # We still need to write the original frame to the output key
                # so the rest of the tree has a consistent key to read from.
                original_transform: TransformStamped = self.blackboard_get(
                    "initial_food_frame"
                )
                self.blackboard_set("food_frame_updated", original_transform)
                return Status.SUCCESS

            initial_transform: TransformStamped = self.blackboard_get(
                "initial_food_frame"
            )
            food_fid: str = self.blackboard_get("food_frame_id")
            robot_base_fid: str = self.blackboard_get("robot_base_frame_id")

            if initial_transform is None:
                self.feedback_message = (
                    "Initial food_frame transform not found on blackboard."
                )
                return Status.FAILURE

            # This logic assumes world_frame and robot_base_frame are the same.
            P_food_in_rb = ros2_numpy.numpify(initial_transform.transform.translation)
            R_rb_food_initial = R.from_quat(
                ros2_numpy.numpify(initial_transform.transform.rotation)
            )

            # --- Rotation logic from RotateLocalApproachPoses ---
            L_xy_rb_numpy = np.array([P_food_in_rb[0], P_food_in_rb[1]])
            if np.linalg.norm(L_xy_rb_numpy) < self.EPSILON:
                self.logger.info(
                    f"[{self.name}] Food origin too close to robot base. No alignment rotation applied."
                )
                # Pass through the original transform
                self.blackboard_set("food_frame_updated", initial_transform)
                return Status.SUCCESS

            desired_global_approach_yaw_rb = math.atan2(
                L_xy_rb_numpy[1], L_xy_rb_numpy[0]
            )

            X_axis_in_food_frame = np.array([1.0, 0.0, 0.0])
            X_axis_in_rb_frame = R_rb_food_initial.apply(X_axis_in_food_frame)
            X_axis_xy_in_rb_frame = np.array(
                [X_axis_in_rb_frame[0], X_axis_in_rb_frame[1]]
            )

            if np.linalg.norm(X_axis_xy_in_rb_frame) < self.EPSILON:
                self.logger.warn(
                    f"[{self.name}] Food frame's X-axis has no projection. Cannot align."
                )
                self.blackboard_set(
                    "food_frame_updated", initial_transform
                )  # Pass through
                return Status.SUCCESS

            current_food_x_yaw_rb = math.atan2(
                X_axis_xy_in_rb_frame[1], X_axis_xy_in_rb_frame[0]
            )
            delta_psi_food_z = self._normalize_angle(
                desired_global_approach_yaw_rb - current_food_x_yaw_rb
            )

            self.logger.info(
                f"[{self.name}] Applying alignment rotation of {math.degrees(delta_psi_food_z):.1f} deg to food frame."
            )
            R_correction_z = R.from_euler("z", delta_psi_food_z)

            # --- Apply the rotation and prepare the final transform ---
            R_world_food_final = R_rb_food_initial * R_correction_z

            final_transform = deepcopy(initial_transform)
            final_quat_xyzw = R_world_food_final.as_quat()
            final_transform.transform.rotation.x = final_quat_xyzw[0]
            final_transform.transform.rotation.y = final_quat_xyzw[1]
            final_transform.transform.rotation.z = final_quat_xyzw[2]
            final_transform.transform.rotation.w = final_quat_xyzw[3]

            # Re-publish the static TF with the new, rotated orientation
            set_static_tf(final_transform, self.blackboard, self.node)

            # Write the final transform to the output blackboard key
            self.blackboard_set("food_frame_updated", final_transform)

            self.feedback_message = (
                f"Food frame realigned by {math.degrees(delta_psi_food_z):.1f} deg."
            )
            return Status.SUCCESS

        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
