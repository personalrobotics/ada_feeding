# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the RotateLocalApproachPoses behavior.
It takes original MoveAbove and MoveInto poses defined in a food_frame,
and rotates them around the food_frame's Z-axis. The rotation angle
is calculated to align the food_frame's Y-axis (when projected onto the
robot_base_frame's XY plane) with the vector from the robot_base_frame's
origin to the food_frame's origin (also projected onto the robot_base_frame's XY plane).
This is intended to align the tool's functional direction (assumed to be
oriented along the food_frame's Y-axis by a pre_quat) with the
robot-to-food line.
"""

# Standard imports
import math
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Pose, Point as PointMsg, Quaternion as QuaternionMsg
import numpy as np
from overrides import override
import py_trees
from py_trees.common import Status
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
import ros2_numpy

# TF2 imports
import tf2_ros

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class RotateLocalApproachPoses(BlackboardBehavior):
    """
    Rotates local MoveAbove and MoveInto poses around the Z-axis of their
    defining food_frame to align the food_frame's Y-axis (and thus a
    tool Z-axis aligned with it) with the robot-base-to-food-origin direction.
    """

    EPSILON = 1e-6

    def blackboard_inputs(
        self,
        move_above_pose_in_food_frame_orig: Union[BlackboardKey, Pose],
        move_into_pose_in_food_frame_orig: Union[BlackboardKey, Pose],
        food_frame_id: Union[BlackboardKey, str] = "food",
        robot_base_frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        move_above_pose_in_food_frame_rotated: Optional[BlackboardKey],
        move_into_pose_in_food_frame_rotated: Optional[BlackboardKey],
    ) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[Node] = None
        self.tf_buffer: Optional[tf2_ros.Buffer] = None
        self.tf_listener: Optional[tf2_ros.TransformListener] = None
        self._initialized_properly = False

    @override
    def setup(self, **kwargs):
        try:
            self.node = kwargs["node"]
            self.tf_buffer = tf2_ros.Buffer(node=self.node)
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node, spin_thread=False
            )
            self.logger.debug(f"[{self.name}] TF Buffer and Listener initialized.")
            self._initialized_properly = True
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
            self._initialized_properly = False
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during setup: {e}")
            self._initialized_properly = False

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @override
    def update(self) -> Status:
        if not self._initialized_properly or not self.node or not self.tf_buffer:
            self.feedback_message = "Behavior not properly initialized in setup."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        try:
            ma_pose_orig_food: Pose = self.blackboard_get(
                "move_above_pose_in_food_frame_orig"
            )
            mi_pose_orig_food: Pose = self.blackboard_get(
                "move_into_pose_in_food_frame_orig"
            )
            food_fid: str = self.blackboard_get("food_frame_id")
            robot_base_fid: str = self.blackboard_get("robot_base_frame_id")

            # 1. Get current transform T_RobotBase_FoodOriginal
            try:
                t_rb_food_stamped: TransformStamped = self.tf_buffer.lookup_transform(
                    robot_base_fid,
                    food_fid,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=1.0),
                )
            except Exception as e:
                self.feedback_message = (
                    f"TF lookup {robot_base_fid} -> {food_fid} failed: {e}"
                )
                self.logger.warn(
                    f"[{self.name}] {self.feedback_message}. Retrying if BT ticks again."
                )
                return Status.RUNNING

            R_rb_food = R.from_quat(
                [
                    t_rb_food_stamped.transform.rotation.x,
                    t_rb_food_stamped.transform.rotation.y,
                    t_rb_food_stamped.transform.rotation.z,
                    t_rb_food_stamped.transform.rotation.w,
                ]
            )
            P_food_in_rb = ros2_numpy.numpify(t_rb_food_stamped.transform.translation)

            # 2. Calculate Desired Approach Line in Robot Base XY plane (L_xy_rb)
            L_xy_rb_numpy = np.array(
                [P_food_in_rb[0], P_food_in_rb[1]]
            )  # Only XY components
            desired_global_approach_yaw_rb: float
            if np.linalg.norm(L_xy_rb_numpy) < self.EPSILON:
                self.logger.info(
                    f"[{self.name}] Food origin too close to robot base XY origin. Desired global yaw is ill-defined, using 0."
                )
                desired_global_approach_yaw_rb = 0.0
                # If desired direction is undefined, delta_psi should ideally make current X-axis align with robot_base X-axis,
                # or simply make delta_psi = 0 if current_food_x_yaw_rb is also ill-defined.
                # For now, if desired is ill-defined, we'll effectively try to rotate food X to robot X.
            else:
                L_xy_rb_norm_vec = L_xy_rb_numpy / np.linalg.norm(L_xy_rb_numpy)
                desired_global_approach_yaw_rb = math.atan2(
                    L_xy_rb_norm_vec[1], L_xy_rb_norm_vec[0]
                )

            # 3. Calculate Current Direction of Food Frame's X-axis in Robot Base XY plane
            X_axis_in_food_frame = np.array([1.0, 0.0, 0.0])  # Food Frame's X-axis
            X_axis_in_rb_frame = R_rb_food.apply(
                X_axis_in_food_frame
            )  # Transform to robot base
            X_axis_xy_in_rb_frame = np.array(
                [X_axis_in_rb_frame[0], X_axis_in_rb_frame[1]]
            )

            current_food_x_yaw_rb: float
            if np.linalg.norm(X_axis_xy_in_rb_frame) < self.EPSILON:
                self.logger.warn(
                    f"[{self.name}] Food frame's X-axis has near-zero XY projection in robot base. "
                    f"This implies food_frame Z is nearly aligned with robot_base XY plane (e.g. food frame is 'on its side'). "
                    f"Cannot reliably determine current food_X_yaw. Assuming 0 adjustment needed relative to this."
                )
                # If current X-axis has no defined yaw, any desired_global_approach_yaw_rb becomes the delta directly,
                # but this might not be what we want. It's better to make no change if current state is ambiguous.
                delta_psi_food_z = 0.0
            else:
                X_axis_xy_in_rb_norm_vec = X_axis_xy_in_rb_frame / np.linalg.norm(
                    X_axis_xy_in_rb_frame
                )
                current_food_x_yaw_rb = math.atan2(
                    X_axis_xy_in_rb_norm_vec[1], X_axis_xy_in_rb_norm_vec[0]
                )

                # 4. Calculate Yaw Correction delta_psi_food_z
                # This is the rotation around the food_frame's Z-axis needed to align its X-axis
                # (projected to robot_base XY) with the desired_global_approach_yaw_rb.
                delta_psi_food_z = self._normalize_angle(
                    desired_global_approach_yaw_rb - current_food_x_yaw_rb
                )
                self.logger.info(
                    f"[{self.name}] Aligning Food_X to base-food line. "
                    f"Current Food_X Yaw (in RB): {math.degrees(current_food_x_yaw_rb):.1f} deg. "
                    f"Desired Global Yaw (for Food_X in RB): {math.degrees(desired_global_approach_yaw_rb):.1f} deg. "
                    f"Required Food_Z Rotation: {math.degrees(delta_psi_food_z):.1f} deg."
                )

            # 5. Apply rotation to local poses (around Z-axis of food_frame)
            R_correction_in_food_z = R.from_euler("z", delta_psi_food_z)

            # Rotate MoveAbove pose
            P_ma_orig_food_numpy = ros2_numpy.numpify(ma_pose_orig_food.position)
            R_ma_orig_food_scipy = R.from_quat(
                [
                    ma_pose_orig_food.orientation.x,
                    ma_pose_orig_food.orientation.y,
                    ma_pose_orig_food.orientation.z,
                    ma_pose_orig_food.orientation.w,
                ]
            )
            P_ma_rotated_food_numpy = R_correction_in_food_z.apply(P_ma_orig_food_numpy)
            R_ma_rotated_food_scipy = R_correction_in_food_z * R_ma_orig_food_scipy

            ma_pose_rotated_food = Pose()
            ma_pose_rotated_food.position = ros2_numpy.msgify(
                PointMsg, P_ma_rotated_food_numpy
            )
            ma_quat_xyzw_rotated = R_ma_rotated_food_scipy.as_quat()
            ma_pose_rotated_food.orientation.x = ma_quat_xyzw_rotated[0]
            ma_pose_rotated_food.orientation.y = ma_quat_xyzw_rotated[1]
            ma_pose_rotated_food.orientation.z = ma_quat_xyzw_rotated[2]
            ma_pose_rotated_food.orientation.w = ma_quat_xyzw_rotated[3]
            self.blackboard_set(
                "move_above_pose_in_food_frame_rotated", ma_pose_rotated_food
            )

            # Rotate MoveInto pose
            P_mi_orig_food_numpy = ros2_numpy.numpify(mi_pose_orig_food.position)
            R_mi_orig_food_scipy = R.from_quat(
                [
                    mi_pose_orig_food.orientation.x,
                    mi_pose_orig_food.orientation.y,
                    mi_pose_orig_food.orientation.z,
                    mi_pose_orig_food.orientation.w,
                ]
            )  # Should be same as R_ma_orig_food_scipy
            P_mi_rotated_food_numpy = R_correction_in_food_z.apply(P_mi_orig_food_numpy)
            R_mi_rotated_food_scipy = R_correction_in_food_z * R_mi_orig_food_scipy

            mi_pose_rotated_food = Pose()
            mi_pose_rotated_food.position = ros2_numpy.msgify(
                PointMsg, P_mi_rotated_food_numpy
            )
            mi_quat_xyzw_rotated = R_mi_rotated_food_scipy.as_quat()
            mi_pose_rotated_food.orientation.x = mi_quat_xyzw_rotated[0]
            mi_pose_rotated_food.orientation.y = mi_quat_xyzw_rotated[1]
            mi_pose_rotated_food.orientation.z = mi_quat_xyzw_rotated[2]
            mi_pose_rotated_food.orientation.w = mi_quat_xyzw_rotated[3]
            self.blackboard_set(
                "move_into_pose_in_food_frame_rotated", mi_pose_rotated_food
            )

            self.feedback_message = f"Local poses rotated by {math.degrees(delta_psi_food_z):.1f} deg around food Z."
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error in {self.name}: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
