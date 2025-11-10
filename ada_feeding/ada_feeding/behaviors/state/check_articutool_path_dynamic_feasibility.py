# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

"""
This module defines the CheckArticutoolPathDynamicFeasibility behavior.

This behavior checks if a Jaco trajectory is dynamically feasible for the
Articutool to maintain a level orientation. It uses the Articutool's
analytic kinematic model and applies a fixed coordinate frame transformation
to account for the difference between the JacoEE frame and the
Articutool's kinematic base frame.

It reads the Pinocchio model from the global blackboard.
"""

# Standard imports
import math
from typing import Union, Optional, List, Tuple
import traceback

# Third-party imports
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory
import numpy as np
from overrides import override
from scipy.spatial.transform import Rotation
import py_trees
from py_trees.common import Status
import rclpy.node
import pinocchio as pin

# ROS / System Imports
from articutool_control.articutool_kinematics import ArticutoolAnalyticalKinematics

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import BlackboardKey


class CheckArticutoolPathDynamicFeasibility(BlackboardBehavior):
    """
    Checks if a Jaco trajectory is dynamically feasible for the Articutool
    to maintain a level orientation.
    """

    EPSILON = 1e-6
    WORLD_Z_UP_VECTOR = np.array([0.0, 0.0, 1.0])

    # Fixed rotation to convert a vector from the Jaco End-Effector frame
    # to the Articutool's base frame (the frame used by the analytic IK solver).
    # v_ArticutoolBase = R_z(-90) * v_JacoEE
    R_JACOEE_TO_ATOOL_BASE = Rotation.from_euler("z", -math.pi / 2, degrees=False)

    def blackboard_inputs(
        self,
        jaco_trajectory: Union[BlackboardKey, RobotTrajectory, JointTrajectory],
        articutool_pitch_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_roll_limits_rad: Union[BlackboardKey, Tuple[float, float]],
        articutool_max_joint_velocity: Union[BlackboardKey, float],
        num_trajectory_points_to_check: Union[BlackboardKey, int] = 10,
    ) -> None:
        """Define blackboard inputs."""
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        articutool_is_dynamic_feasible: Optional[BlackboardKey],
    ) -> None:
        """Define blackboard outputs."""
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def __init__(self, name: str, **kwargs):
        """Initialize the behavior."""
        super().__init__(name=name, **kwargs)
        self.kin_model = ArticutoolAnalyticalKinematics(epsilon=self.EPSILON)

        self.node: Optional[rclpy.node.Node] = None
        self._pin_model: Optional[pin.Model] = None
        self._pin_data: Optional[pin.Data] = None
        self._jaco_joint_names_pin: List[str] = (
            [  # Get from param or blackboard if not fixed
                "j2n6s200_joint_1",
                "j2n6s200_joint_2",
                "j2n6s200_joint_3",
                "j2n6s200_joint_4",
                "j2n6s200_joint_5",
                "j2n6s200_joint_6",
            ]
        )
        self._jaco_vel_indices_pin: Optional[List[int]] = None
        self._jaco_ee_frame_id_pin: Optional[int] = None
        self._pitch_limits_rad: Optional[Tuple[float, float]] = None
        self._roll_limits_rad: Optional[Tuple[float, float]] = None
        self._max_atool_vel: Optional[float] = None
        self._pinocchio_ready = False

    @override
    def setup(self, **kwargs):
        """Get the ROS2 node and Pinocchio essentials from the global blackboard."""
        try:
            self.node = kwargs["node"]

            # Read static parameters from local blackboard
            self._pitch_limits_rad = self.blackboard_get("articutool_pitch_limits_rad")
            self._roll_limits_rad = self.blackboard_get("articutool_roll_limits_rad")
            self._max_atool_vel = self.blackboard_get("articutool_max_joint_velocity")

            # Create a client to read from the global blackboard
            bb_client = py_trees.blackboard.Client(name=f"{self.name}_BBClient")
            bb_client.register_key(
                key="/pinocchio_model", access=py_trees.common.Access.READ
            )
            bb_client.register_key(
                key="/pinocchio_data", access=py_trees.common.Access.READ
            )
            bb_client.register_key(
                key="/jaco_vel_indices_pin", access=py_trees.common.Access.READ
            )
            bb_client.register_key(
                key="/jaco_ee_frame_id_pin", access=py_trees.common.Access.READ
            )

            # Read the values
            self._pin_model = bb_client.get("/pinocchio_model")
            self._pin_data = bb_client.get("/pinocchio_data")
            self._jaco_vel_indices_pin = bb_client.get("/jaco_vel_indices_pin")
            self._jaco_ee_frame_id_pin = bb_client.get("/jaco_ee_frame_id_pin")

            if any(
                v is None
                for v in [
                    self._pin_model,
                    self._pin_data,
                    self._jaco_vel_indices_pin,
                    self._jaco_ee_frame_id_pin,
                ]
            ):
                raise ValueError(
                    "One or more required Pinocchio objects not found on global blackboard."
                )

            self._pinocchio_ready = True
            self.logger.info(
                f"[{self.name}] Successfully read Pinocchio model from global blackboard."
            )

        except KeyError as e:
            self.logger.error(
                f"[{self.name}] Behaviour expects 'node' in setup kwargs or local blackboard key missing: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"[{self.name}] Failed to get Pinocchio model during setup: {e}"
            )

    def _get_jaco_trajectory_points(
        self, trajectory_input: Union[RobotTrajectory, JointTrajectory]
    ) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        """Extracts joint positions and velocities from a trajectory message."""
        if isinstance(trajectory_input, RobotTrajectory):
            jt = trajectory_input.joint_trajectory
        elif isinstance(trajectory_input, JointTrajectory):
            jt = trajectory_input
        else:
            return None

        # TODO: Need to respect joint_names order from trajectory msg
        # This implementation assumes the points are already in the
        # order matching self._jaco_joint_names_pin.
        # A robust way is to get indices from jt.joint_names
        # and re-order the points.

        points = []
        for point in jt.points:
            positions = np.array(point.positions)
            velocities = np.array(point.velocities)
            points.append((positions, velocities))
        return points

    @override
    def update(self) -> Status:
        """Execute the behavior's logic."""
        if not self.node or not self._pinocchio_ready:
            self.feedback_message = (
                "Behavior not properly initialized or Pinocchio model not loaded."
            )
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE

        try:
            trajectory_input = self.blackboard_get("jaco_trajectory")
            num_points = self.blackboard_get("num_trajectory_points_to_check")

            jaco_points = self._get_jaco_trajectory_points(trajectory_input)

            if not jaco_points:
                self.logger.warn(
                    f"[{self.name}] Trajectory is empty. Assuming feasible."
                )
                self.blackboard_set("articutool_is_dynamic_feasible", True)
                return Status.SUCCESS

            indices = np.linspace(0, len(jaco_points) - 1, num_points, dtype=int)

            last_valid_atool_q = None

            for idx in indices:
                q_jaco_partial, v_jaco_partial = jaco_points[idx]

                # This assumes q_jaco_partial maps 1:1 to self._jaco_joint_names_pin
                if len(q_jaco_partial) != len(self._jaco_joint_names_pin):
                    self.feedback_message = (
                        f"Trajectory point {idx} has wrong number of joints."
                    )
                    self.logger.error(f"[{self.name}] {self.feedback_message}")
                    return Status.FAILURE

                # Construct the full configuration vector for Pinocchio
                q_full_robot = pin.neutral(self._pin_model)
                for i, joint_name in enumerate(self._jaco_joint_names_pin):
                    joint_id = self._pin_model.getJointId(joint_name)
                    joint_obj = self._pin_model.joints[joint_id]
                    theta = q_jaco_partial[i]
                    if joint_obj.nq == 2 and joint_obj.nv == 1:
                        q_full_robot[joint_obj.idx_q] = math.cos(theta)
                        q_full_robot[joint_obj.idx_q + 1] = math.sin(theta)
                    elif joint_obj.nq == 1 and joint_obj.nv == 1:
                        q_full_robot[joint_obj.idx_q] = theta

                pin.forwardKinematics(self._pin_model, self._pin_data, q_full_robot)
                pin.updateFramePlacements(self._pin_model, self._pin_data)

                # This pose is of the JacoEE frame
                T_world_jacoee = self._pin_data.oMf[self._jaco_ee_frame_id_pin]
                R_world_jacoee = Rotation.from_matrix(T_world_jacoee.rotation)

                # 2. Transform world "up" vector to the JacoEE frame
                target_y_in_JacoEE = R_world_jacoee.inv().apply(self.WORLD_Z_UP_VECTOR)

                # 3. Convert from JacoEE frame to Articutool Base frame
                target_y_in_ArticutoolBase = self.R_JACOEE_TO_ATOOL_BASE.apply(
                    target_y_in_JacoEE
                )

                # 4. Solve Articutool IK for leveling
                ik_solutions = self.kin_model.solve_ik_for_leveling(
                    target_y_in_ArticutoolBase
                )

                valid_solutions = [
                    sol
                    for sol in ik_solutions
                    if self._pitch_limits_rad[0] - self.EPSILON
                    <= sol[0]
                    <= self._pitch_limits_rad[1] + self.EPSILON
                    and self._roll_limits_rad[0] - self.EPSILON
                    <= sol[1]
                    <= self._roll_limits_rad[1] + self.EPSILON
                ]

                if not valid_solutions:
                    self.feedback_message = (
                        f"Path is kinematically infeasible at point {idx}."
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

                # Choose best solution (e.g., closest to previous)
                if last_valid_atool_q is None:
                    q_atool = valid_solutions[0]
                else:
                    distances = [
                        np.linalg.norm(np.array(sol) - last_valid_atool_q)
                        for sol in valid_solutions
                    ]
                    q_atool = valid_solutions[np.argmin(distances)]
                last_valid_atool_q = np.array(q_atool)

                # --- DYNAMIC FEASIBILITY CHECK ---

                # 1. Calculate disturbance velocity from Jaco arm
                J_jaco_full = pin.computeFrameJacobian(
                    self._pin_model,
                    self._pin_data,
                    q_full_robot,
                    self._jaco_ee_frame_id_pin,
                    pin.ReferenceFrame.WORLD,
                )

                # Construct full velocity vector
                v_full_robot = np.zeros(self._pin_model.nv)
                v_full_robot[self._jaco_vel_indices_pin] = v_jaco_partial

                v_jaco_full = J_jaco_full @ v_full_robot
                omega_disturbance_world = v_jaco_full[3:6]

                # Transform disturbance omega into the local JacoEE frame
                omega_disturbance_local_JacoEE = R_world_jacoee.inv().apply(
                    omega_disturbance_world
                )

                # 3. Compute Articutool's Jacobian at the required configuration
                J_atool_ArticutoolBase = self.kin_model.compute_jacobian(
                    q_atool[0], q_atool[1]
                )

                # 4. Transform disturbance omega into the Articutool Base frame
                omega_disturbance_local_ArticutoolBase = (
                    self.R_JACOEE_TO_ATOOL_BASE.apply(omega_disturbance_local_JacoEE)
                )

                # We want to find joint velocities `q_dot` that *cancel* this disturbance.
                # J_ArticuloolBase * q_dot = -omega_disturbance_ArticutoolBase
                omega_correction_local_ArticutoolBase = (
                    -omega_disturbance_local_ArticutoolBase
                )

                # 5. Solve for required Articutool joint velocities
                try:
                    J_atool_pinv = np.linalg.pinv(J_atool_ArticutoolBase, rcond=1e-4)
                    q_dot_atool_required = (
                        J_atool_pinv @ omega_correction_local_ArticutoolBase
                    )
                except np.linalg.LinAlgError:
                    self.feedback_message = (
                        f"Articutool Jacobian is singular at point {idx}."
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

                # 6. The final check
                if (
                    abs(q_dot_atool_required[0]) > self._max_atool_vel
                    or abs(q_dot_atool_required[1]) > self._max_atool_vel
                ):
                    self.feedback_message = (
                        f"Path is dynamically infeasible at point {idx}. "
                        f"Required vel [p,r]: {np.round(q_dot_atool_required, 2)} rad/s > "
                        f"Max vel: {self._max_atool_vel:.2f} rad/s"
                    )
                    self.logger.warning(f"[{self.name}] {self.feedback_message}")
                    self.blackboard_set("articutool_is_dynamic_feasible", False)
                    return Status.FAILURE

            # If all checks pass
            self.feedback_message = "Trajectory is dynamically feasible."
            self.blackboard_set("articutool_is_dynamic_feasible", True)
            return Status.SUCCESS

        except KeyError as e:
            self.feedback_message = f"Blackboard key error: {e}"
            self.logger.error(f"[{self.name}] {self.feedback_message}")
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE
        except Exception as e:
            self.feedback_message = f"Unexpected error: {e}"
            self.logger.error(
                f"[{self.name}] {self.feedback_message}\n{traceback.format_exc()}"
            )
            self.blackboard_set("articutool_is_dynamic_feasible", False)
            return Status.FAILURE

    @override
    def terminate(self, new_status: Status) -> None:
        """Log termination status."""
        self.logger.debug(f"[{self.name}] Terminating with status {new_status}.")

    @override
    def initialise(self) -> None:
        """Reset blackboard output on initialization."""
        self.logger.debug(f"[{self.name}] Initializing.")
        self.blackboard_set("articutool_is_dynamic_feasible", None)
