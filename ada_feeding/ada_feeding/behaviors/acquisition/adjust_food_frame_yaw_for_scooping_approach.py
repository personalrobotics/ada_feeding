# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the AdjustFoodFrameYawForScoopingApproach behavior.
It modifies the TF of the 'food_frame' to align the approach vector
(MoveAbove to MoveInto) with the vector from the robot base to the food,
affecting only the yaw.
"""

# Standard imports
import math
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import (
    Pose,
    TransformStamped,
    PointStamped,
    Vector3 as Vector3Msg,
)
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
from tf2_geometry_msgs import do_transform_point  # For transforming PointStamped

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    set_static_tf,
)  # Assuming set_static_tf is available


class AdjustFoodFrameYawForScoopingApproach(BlackboardBehavior):
    """
    Adjusts the yaw of the 'food_frame' so that the local approach vector
    (from MoveAbove to MoveInto poses) aligns its XY projection in the
    robot_base_frame with the XY projection of the vector from the
    robot_base_frame's origin to the food_frame's origin.
    The pitch and roll of the original food_frame relative to the world,
    and the pitch/roll of the tool approach, are preserved.
    """

    EPSILON = 1e-6

    def blackboard_inputs(
        self,
        move_above_pose_food_frame: Union[
            BlackboardKey, Pose
        ],  # In current (old) food_frame
        move_into_pose_food_frame: Union[
            BlackboardKey, Pose
        ],  # In current (old) food_frame
        food_frame_id: Union[BlackboardKey, str] = "food",
        robot_base_frame_id: Union[BlackboardKey, str] = "j2n6s200_link_base",
    ) -> None:
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    # No direct blackboard outputs other than status, effect is on TF
    def blackboard_outputs(self) -> None:
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.node: Optional[Node] = None
        self.tf_buffer: Optional[tf2_ros.Buffer] = None
        self.tf_listener: Optional[tf2_ros.TransformListener] = None
        # self.static_tf_broadcaster: Optional[tf2_ros.StaticTransformBroadcaster] = None # If set_static_tf needs it

    @override
    def setup(self, **kwargs):
        try:
            self.node = kwargs["node"]
            self.tf_buffer = tf2_ros.Buffer(node=self.node)
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node, spin_thread=False
            )
            # If set_static_tf requires a broadcaster passed in:
            # self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self.node)
            self.logger.debug(f"[{self.name}] TF Buffer and Listener initialized.")
        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs. {e}"
            )
        except Exception as e:
            self.logger.error(f"[{self.name}] Error during setup: {e}")

    @override
    def update(self) -> Status:
        if not self.node or not self.tf_buffer:
            self.feedback_message = "Node or TF Buffer not initialized."
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

        try:
            ma_pose_food: Pose = self.blackboard_get("move_above_pose_food_frame")
            mi_pose_food: Pose = self.blackboard_get("move_into_pose_food_frame")
            food_fid: str = self.blackboard_get("food_frame_id")
            robot_base_fid: str = self.blackboard_get("robot_base_frame_id")

            # 1. Get current transform T_RobotBase_FoodOld
            try:
                t_rb_foodold_stamped: TransformStamped = (
                    self.tf_buffer.lookup_transform(
                        robot_base_fid,
                        food_fid,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=1.0),
                    )
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                self.feedback_message = (
                    f"TF lookup {robot_base_fid} -> {food_fid} failed: {e}"
                )
                self.logger.warn(f"[{self.name}] {self.feedback_message}")
                return Status.FAILURE  # Or RUNNING if you want to retry

            R_rb_foodold_scipy = R.from_quat(
                [
                    t_rb_foodold_stamped.transform.rotation.x,
                    t_rb_foodold_stamped.transform.rotation.y,
                    t_rb_foodold_stamped.transform.rotation.z,
                    t_rb_foodold_stamped.transform.rotation.w,
                ]
            )
            P_foodcentroid_rb_numpy = ros2_numpy.numpify(
                t_rb_foodold_stamped.transform.translation
            )

            # 2. Calculate approach vector D_FoodOld (in old food frame)
            P_ma_foodold_numpy = ros2_numpy.numpify(ma_pose_food.position)
            P_mi_foodold_numpy = ros2_numpy.numpify(mi_pose_food.position)
            D_foodold_numpy = P_mi_foodold_numpy - P_ma_foodold_numpy

            # 3. Transform D_FoodOld to D_RobotBase (in robot base frame)
            D_rb_numpy = R_rb_foodold_scipy.apply(D_foodold_numpy)

            # 4. Determine desired approach direction L_RobotBase (from robot base origin to food centroid)
            # This is simply P_foodcentroid_rb_numpy if robot_base_fid is the reference
            L_rb_numpy = P_foodcentroid_rb_numpy

            # 5. Project D_RobotBase and L_RobotBase onto XY plane of RobotBase frame
            D_xy_rb = np.array([D_rb_numpy[0], D_rb_numpy[1]])
            L_xy_rb = np.array([L_rb_numpy[0], L_rb_numpy[1]])

            if (
                np.linalg.norm(D_xy_rb) < self.EPSILON
                or np.linalg.norm(L_xy_rb) < self.EPSILON
            ):
                self.logger.info(
                    f"[{self.name}] Approach vector or base-to-food vector has zero XY projection. Skipping yaw adjustment."
                )
                # No adjustment needed, or situation is ambiguous for pure yaw.
                # The original food_frame TF remains.
                self.feedback_message = (
                    "Skipped yaw adjustment due to zero XY projection."
                )
                return Status.SUCCESS  # Or FAILURE if this state is problematic

            # 6. Calculate yaw correction angle delta_psi
            current_approach_yaw = math.atan2(D_xy_rb[1], D_xy_rb[0])
            desired_approach_yaw = math.atan2(L_xy_rb[1], L_xy_rb[0])
            delta_psi = self._normalize_angle(
                desired_approach_yaw - current_approach_yaw
            )

            self.logger.info(
                f"[{self.name}] Current approach yaw (in robot_base XY): {math.degrees(current_approach_yaw):.1f} deg. "
                f"Desired: {math.degrees(desired_approach_yaw):.1f} deg. Delta: {math.degrees(delta_psi):.1f} deg."
            )

            # 7. Compute new food frame orientation R_RobotBase_FoodNew
            R_deltapsi_z = R.from_euler("z", delta_psi)
            R_rb_foodnew_scipy = (
                R_deltapsi_z * R_rb_foodold_scipy
            )  # Apply yaw correction in RobotBase frame

            # 8. Create and publish new TransformStamped for T_RobotBase_FoodNew
            t_rb_foodnew_stamped = TransformStamped()
            t_rb_foodnew_stamped.header.stamp = self.node.get_clock().now().to_msg()
            t_rb_foodnew_stamped.header.frame_id = robot_base_fid
            t_rb_foodnew_stamped.child_frame_id = food_fid

            t_rb_foodnew_stamped.transform.translation = ros2_numpy.msgify(
                Vector3Msg, P_foodcentroid_rb_numpy
            )

            new_quat_xyzw = R_rb_foodnew_scipy.as_quat()
            t_rb_foodnew_stamped.transform.rotation.x = new_quat_xyzw[0]
            t_rb_foodnew_stamped.transform.rotation.y = new_quat_xyzw[1]
            t_rb_foodnew_stamped.transform.rotation.z = new_quat_xyzw[2]
            t_rb_foodnew_stamped.transform.rotation.w = new_quat_xyzw[3]

            # Use the set_static_tf helper.
            # Assuming set_static_tf is: set_static_tf(transform_stamped, blackboard_client, node_instance)
            # Or if it manages its own broadcaster: set_static_tf(transform_stamped, node_instance)
            # For this example, I'll assume the latter simplified version.
            # If it needs the blackboard, you'd pass self.blackboard
            if (
                hasattr(self, "static_tf_broadcaster")
                and self.static_tf_broadcaster is not None
            ):
                # If you were managing it directly (less likely if set_static_tf is a true helper)
                self.static_tf_broadcaster.sendTransform(t_rb_foodnew_stamped)
            elif "set_static_tf" in globals() or hasattr(
                ada_feeding.helpers, "set_static_tf"
            ):
                # Call the helper, assuming it's imported or available
                set_static_tf(t_rb_foodnew_stamped, self.blackboard, self.node)
                self.logger.info(
                    f"[{self.name}] Updated static TF for '{food_fid}' relative to '{robot_base_fid}'."
                )
            else:
                self.logger.error(
                    f"[{self.name}] 'set_static_tf' helper not available. Cannot update TF."
                )
                self.feedback_message = "set_static_tf helper missing."
                return Status.FAILURE

            self.feedback_message = (
                f"Food frame yaw adjusted by {math.degrees(delta_psi):.1f} deg."
            )
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            return Status.FAILURE

    def _normalize_angle(self, angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @override
    def terminate(self, new_status: Status) -> None:
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")
