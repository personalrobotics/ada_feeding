#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveIt2Plan behavior, which uses pymoveit2
to plan a path using the provided path and goal constraints.
"""

# Standard imports
from enum import Enum
import math
import os
import time
from typing import Any, Union, Optional, Dict, List, Tuple

# Third-party imports
from geometry_msgs.msg import Point, PoseStamped, Quaternion
import numpy as np
from overrides import override
import py_trees
import rclpy
import ros2_numpy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
import tf2_py as tf2
from trajectory_msgs.msg import JointTrajectory

# Local imports
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
    get_tf_object,
)
from ada_feeding.behaviors import BlackboardBehavior


class MoveIt2ConstraintType(Enum):
    """
    Used to specify the type of constraint_kwargs
    Passed to MoveIt2Plan
    """

    JOINT = 0
    POSITION = 1
    ORIENTATION = 2


class MoveIt2Plan(BlackboardBehavior):
    """
    Runs moveit2.py plan_async with the provided
    path and goal constraints.

    Returns SUCCESS with the found trajectory.
    Or SUCCESS + None if the goal constraints are all satisfied.

    Or FAILURE + None if no trajectory can be found.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    # pylint: disable=too-many-arguments
    # These are effectively config definitions
    # They require a lot of arguments.

    def blackboard_inputs(
        self,
        goal_constraints: Union[
            BlackboardKey, List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]
        ],
        path_constraints: Optional[
            Union[BlackboardKey, List[Tuple[MoveIt2ConstraintType, Dict[str, Any]]]]
        ] = None,
        ignore_violated_path_constraints: Union[BlackboardKey, bool] = False,
        pipeline_id: Union[BlackboardKey, str] = "ompl",
        planner_id: Union[BlackboardKey, str] = "RRTstarkConfigDefault",
        allowed_planning_time: Union[BlackboardKey, float] = 0.5,
        max_velocity_scale: Union[BlackboardKey, float] = 0.1,
        max_acceleration_scale: Union[BlackboardKey, float] = 0.1,
        cartesian: Union[BlackboardKey, bool] = False,
        cartesian_max_step: Union[BlackboardKey, float] = 0.0025,
        cartesian_jump_threshold: Union[BlackboardKey, float] = 0.0,
        cartesian_fraction_threshold: Union[BlackboardKey, float] = 0.0,
        start_joint_state: Union[
            BlackboardKey, Optional[Union[JointState, List[float]]]
        ] = None,
        debug_trajectory_viz: Union[BlackboardKey, bool] = False,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        goal_constraints: list of constraints, each one is a pair mapping the type to the kwargs
                        passed to the corresponding pymoveit2 function. See moveit2.py:set_X_goal()
                        Note: defines a goal region (i.e. "and" between constraints)
        path_constraints: list of constraints, each one is a pair mapping the type to the kwargs
                        passed to the corresponding pymoveit2 function.
                        See moveit2.py:set_path_X_constraint()
        ignore_violated_path_constraints: if True, if a path constraint starts violated,
                                          it's okay to plan without it.
        pipeline_id: MoveIt2 Pipeline, usually "ompl"
        planner_id: MoveIt2 planner to use
        allowed_planning_time: Anytime planner timeout, passed to MoveIt2 Directly.
                                NOTE: Not guaranteed, use a Timeout decorator for that,
                                but make sure Timeout > allowed_planning_time, otherwise
                                the behavior will very likely fail
        max_velocity_scale: [0,1] determines max joint velocity of the trajectory
        max_acceleration_scale: [0,1] determines max joint acceleration of the trajectory
        cartesian: True to use MoveIt2's cartesian planner, uses the below parameters
        cartesian_max_step: m to step for each IK of the cartesian planner, note 1cm == 2deg
                            for cartesian rotations
        cartesian_jump_threshold: If >0, large relative joint jumps between cartesian planner IKs
                                  will cause planning to stop
        cartesian_fraction_threshold: [0,1], % of the geodesic that must be planned collision-free
                                      to return success.
        start_joint_state: JointState from which to start planning, useful for chaining
                           planning calls
        debug_trajectory_viz: Whether to generate and save a visualization of the
            trajectory in joint space. This is useful for debugging, but should
            be disabled in production.
        """
        # TODO: consider cartesian parameter struct
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        trajectory: Optional[BlackboardKey],  # Optional[JointTrajectory]
        end_joint_state: Optional[BlackboardKey] = None,  # Optional[JointState]
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        trajectory (JointTrajectory): planned, timed trajectory, or None if planning failed
                                      or goal is already satisfied
        end_joint_state (JointState): get the last joint state of the planned trajectoy,
                                      useful for chaining planning calls
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

        # Get TF Listener from blackboard
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

    @override
    def initialise(self) -> None:
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        self.planning_future = None

        # Copy config to MoveIt2 object.
        # Note: this *could* block, but it is unlikely
        with self.moveit2_lock:
            self.moveit2.planner_id = self.blackboard_get("planner_id")
            self.moveit2.pipeline_id = self.blackboard_get("pipeline_id")
            self.moveit2.max_velocity = self.blackboard_get("max_velocity_scale")
            self.moveit2.max_acceleration = self.blackboard_get(
                "max_acceleration_scale"
            )
            self.moveit2.allowed_planning_time = self.blackboard_get(
                "allowed_planning_time"
            )
            self.moveit2.cartesian_jump_threshold = self.blackboard_get(
                "cartesian_jump_threshold"
            )
            # If the plan is cartesian, it should always avoid collisions
            self.moveit2.cartesian_avoid_collisions = True

            # Clear constraints
            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # pylint: disable=too-many-return-statements, too-many-statements, too-many-branches
        # All raised exceptions map to a return statement.
        # This is the main function, it is okay for it to be long.

        # Lock MoveIt2 Object
        if self.moveit2_lock.locked():
            return py_trees.common.Status.RUNNING
        if self.tf_lock.locked():
            return py_trees.common.Status.RUNNING
        with self.moveit2_lock, self.tf_lock:
            ### Check if plan done
            if self.planning_future is not None:
                if self.planning_future.done():
                    traj = self.moveit2.get_trajectory(
                        self.planning_future,
                        cartesian=self.blackboard_get("cartesian"),
                        cartesian_fraction_threshold=self.blackboard_get(
                            "cartesian_fraction_threshold"
                        ),
                    )

                    if traj is None:
                        self.logger.error(
                            f"{self.name} [MoveIt2Plan::update()] Planning Failure!"
                        )
                        return py_trees.common.Status.FAILURE

                    # MoveIt's default cartesian interpolator doesn't respect velocity
                    # scaling, so we need to manually add that.
                    if (
                        self.blackboard_get("cartesian")
                        and self.blackboard_get("max_velocity_scale") > 0.0
                    ):
                        MoveIt2Plan.scale_velocity(traj, self.moveit2.max_velocity)

                    # Save the trajectory visualization
                    if self.blackboard_get("debug_trajectory_viz"):
                        MoveIt2Plan.visualize_trajectory(self.name, traj)

                    # Write blackboard outputs and SUCCEED
                    self.blackboard_set("trajectory", traj)
                    end_joint_state = JointState()
                    end_joint_state.name = traj.joint_names[:]
                    end_joint_state.position = traj.points[-1].positions[:]
                    end_joint_state.velocity = traj.points[-1].velocities[:]
                    end_joint_state.effort = traj.points[-1].effort[:]
                    self.blackboard_set("end_joint_state", end_joint_state)
                    return py_trees.common.Status.SUCCESS
                # Planning goal still running
                return py_trees.common.Status.RUNNING

            ### New Plan
            ### Add Constraints to MoveIt Object

            # Check all goal constraints
            goals_satisfied = True

            for constraint_type, constraint_kwargs in self.blackboard_get(
                "goal_constraints"
            ):
                try:
                    if constraint_type == MoveIt2ConstraintType.JOINT:
                        goals_satisfied = (
                            goals_satisfied
                            and self.joint_constraint_satisfied(constraint_kwargs)
                        )
                        self.moveit2.set_joint_goal(**constraint_kwargs)
                    elif constraint_type == MoveIt2ConstraintType.POSITION:
                        goals_satisfied = (
                            goals_satisfied
                            and self.position_constraint_satisfied(constraint_kwargs)
                        )
                        ## If Cartesian, transform to base link frame
                        if self.blackboard_get("cartesian"):
                            self.transform_goal_to_base_link(constraint_kwargs)
                        self.moveit2.set_position_goal(**constraint_kwargs)
                    elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                        goals_satisfied = (
                            goals_satisfied
                            and self.orientation_constraint_satisfied(constraint_kwargs)
                        )
                        ## If Cartesian, transform to base link frame
                        if self.blackboard_get("cartesian"):
                            self.transform_goal_to_base_link(constraint_kwargs)
                        self.moveit2.set_orientation_goal(**constraint_kwargs)
                except tf2.LookupException:
                    # Wait on Frame Transform
                    # Use Timeout decorator to fail
                    # pylint: disable=unexpected-keyword-arg
                    # We overwrite self.logger w/ node.get_logger()
                    self.logger.warning(
                        "Waiting on TF to test constraint satisfaction", once=True
                    )
                    return py_trees.common.Status.RUNNING
                except (IndexError, AttributeError, ValueError) as error:
                    self.logger.error(
                        f"Malformed Goal Constraint: {constraint_kwargs}; Error: {error}"
                    )
                    return py_trees.common.Status.FAILURE

            # If all goals are satisfied, return SUCCESS, no planning necessary
            # Skip if a start position is defined
            if (
                not self.blackboard_exists("start_joint_state")
                or self.blackboard_get("start_joint_state") is None
            ):
                if goals_satisfied:
                    self.blackboard_set("trajectory", None)
                    self.blackboard_set("end_joint_state", self.moveit2.joint_state)
                    return py_trees.common.Status.SUCCESS

            # Check all path constraints
            paths_satisfied = True
            ignore_violated_path_constraints = self.blackboard_get(
                "ignore_violated_path_constraints"
            )
            if (
                self.blackboard_exists("path_constraints")
                and self.blackboard_get("path_constraints") is not None
            ):
                for constraint_type, constraint_kwargs in self.blackboard_get(
                    "path_constraints"
                ):
                    try:
                        if constraint_type == MoveIt2ConstraintType.JOINT:
                            if self.joint_constraint_satisfied(constraint_kwargs):
                                self.moveit2.set_path_joint_constraint(
                                    **constraint_kwargs
                                )
                            elif ignore_violated_path_constraints:
                                continue
                            else:
                                paths_satisfied = False
                                break
                        elif constraint_type == MoveIt2ConstraintType.POSITION:
                            if self.position_constraint_satisfied(constraint_kwargs):
                                self.moveit2.set_path_position_constraint(
                                    **constraint_kwargs
                                )
                            elif ignore_violated_path_constraints:
                                continue
                            else:
                                paths_satisfied = False
                                break
                        elif constraint_type == MoveIt2ConstraintType.ORIENTATION:
                            if self.orientation_constraint_satisfied(constraint_kwargs):
                                self.moveit2.set_path_orientation_constraint(
                                    **constraint_kwargs
                                )
                            elif ignore_violated_path_constraints:
                                continue
                            else:
                                paths_satisfied = False
                                break
                    except tf2.LookupException:
                        # Wait on Frame Transform
                        # Use Timeout decorator to fail
                        # pylint: disable=unexpected-keyword-arg
                        # We overwrite self.logger w/ node.get_logger()
                        self.logger.warning(
                            "Waiting on TF to test constraint satisfaction", once=True
                        )
                        return py_trees.common.Status.RUNNING
                    except (IndexError, AttributeError, ValueError):
                        self.logger.error(
                            f"Malformed Path Constraint: {constraint_kwargs}"
                        )
                        return py_trees.common.Status.FAILURE

                # If any path constraints are unsatisfied, return FAILURE
                # Skip if a start position is defined
                if (
                    self.blackboard_exists("start_joint_state")
                    and self.blackboard_get("start_joint_state") is not None
                ):
                    if not paths_satisfied:
                        self.blackboard_set("end_joint_state", None)
                        self.blackboard_set("trajectory", None)
                        return py_trees.common.Status.FAILURE

            ### Begin Planning
            # pylint: disable=attribute-defined-outside-init
            self.planning_future = self.moveit2.plan_async(
                cartesian=self.blackboard_get("cartesian"),
                max_step=self.blackboard_get("cartesian_max_step"),
                start_joint_state=self.blackboard_get("start_joint_state"),
            )
            return py_trees.common.Status.RUNNING

    @override
    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Docstring copied from @override

        # Clear constraints just in case
        # Note plan_async should already do this
        with self.moveit2_lock:
            self.moveit2.clear_goal_constraints()
            self.moveit2.clear_path_constraints()

    def transform_goal_to_base_link(self, constraint_kwargs: Dict[str, Any]) -> None:
        """
        Transforms the given constraint into the base link frame.

        Necessary because MoveIt2 Cartesian Planner doesn't take in frame_id.

        Parameters
        ----------
        constraint_kwargs: MODIFIED with transformed constraint
        """
        if constraint_kwargs["frame_id"] is None:
            return

        cons_pose = PoseStamped()
        cons_pose.header.frame_id = constraint_kwargs["frame_id"]
        if "quat_xyzw" in constraint_kwargs:
            cons_pose.pose.orientation = (
                constraint_kwargs["quat_xyzw"]
                if isinstance(constraint_kwargs["quat_xyzw"], Quaternion)
                else ros2_numpy.msgify(
                    Quaternion, np.array(constraint_kwargs["quat_xyzw"])
                )
            )
            constraint_kwargs["quat_xyzw"] = self.tf_buffer.transform(
                cons_pose, self.moveit2.base_link_name
            ).pose.orientation

        if "position" in constraint_kwargs:
            cons_pose.pose.position = (
                constraint_kwargs["position"]
                if isinstance(constraint_kwargs["position"], Point)
                else ros2_numpy.msgify(Point, np.array(constraint_kwargs["position"]))
            )
            constraint_kwargs["position"] = self.tf_buffer.transform(
                cons_pose, self.moveit2.base_link_name
            ).pose.position

        constraint_kwargs["frame_id"] = self.moveit2.base_link_name

    def joint_constraint_satisfied(self, constraint_kwargs: Dict[str, Any]) -> bool:
        """
        Check if current joint state satisfies provided joint constraint.

        Assume self.moveit2 is locked.

        Parameters
        ----------
        constraint_kwargs: See moveit2.py; dictionary with keys: "joint_positions"
                           "joint_names", "tolerance"
        Returns
        -------
        True if the constraint is satisfied


        Raises
        ------
        ValueError: if the joint_names kwarg is non-None and doesn't match the
        length of the joint_positions kwarg
        """

        curr_joint_state = self.moveit2.joint_state
        tol = np.fabs(constraint_kwargs["tolerance"])

        des_positions = list(constraint_kwargs["joint_positions"])
        curr_positions = list(curr_joint_state.position)

        # Remap current joints based on joint names
        joint_names = constraint_kwargs["joint_names"]
        if joint_names is None:
            joint_names = []
        if len(joint_names) == len(des_positions):
            for i in range(len(des_positions)):
                index = curr_joint_state.name.index(joint_names[i])
                curr_positions[i] = curr_joint_state.position[index]
        elif len(joint_names) > 0:
            raise ValueError("Joint names array should match joint positions array.")

        # Compare with desired joints
        curr_positions = curr_positions[: len(des_positions)]
        diff = np.fabs(np.array(curr_positions) - np.array(des_positions))

        return np.all(diff < tol)

    def position_constraint_satisfied(self, constraint_kwargs: Dict[str, Any]) -> bool:
        """
        Check if current position state satisfies provided position constraint.

        Assume self.moveit2_lock and self.tf_lock are locked.

        Assume TF from frame_id to target_link exists.

        Parameters
        ----------
        constraint_kwargs: See moveit2.py; dictionary with keys: "position"
                           "frame_id", "target_link", "tolerance"
        Returns
        -------
        True if the constraint is satisfied
        """

        tol = np.fabs(constraint_kwargs["tolerance"])

        t_stamped = self.tf_buffer.lookup_transform(
            constraint_kwargs["frame_id"]
            if constraint_kwargs["frame_id"] is not None
            else self.moveit2.base_link_name,
            constraint_kwargs["target_link"]
            if constraint_kwargs["target_link"] is not None
            else self.moveit2.end_effector_name,
            rclpy.time.Time(),
        )
        current = ros2_numpy.numpify(t_stamped.transform.translation)
        desired = None
        if isinstance(constraint_kwargs["position"], Point):
            desired = ros2_numpy.numpify(constraint_kwargs["position"])
        else:
            desired = np.array(constraint_kwargs["position"])

        return np.linalg.norm(desired - current) <= tol

    def orientation_constraint_satisfied(
        self, constraint_kwargs: Dict[str, Any]
    ) -> bool:
        """
        Check if current orientation satisfies provided orientation constraint.

        Assume self.moveit2_lock and self.tf_lock are locked.

        Assume TF from frame_id to target_link exists.

        Parameters
        ----------
        constraint_kwargs: See moveit2.py; dictionary with keys: "quat_xyzw"
                           "frame_id", "target_link", "tolerance", and "parameterization"
        Returns
        -------
        True if the constraint is satisfied
        """

        tol = np.fabs(np.array(constraint_kwargs["tolerance"]) * np.ones(3))

        t_stamped = self.tf_buffer.lookup_transform(
            constraint_kwargs["frame_id"]
            if constraint_kwargs["frame_id"] is not None
            else self.moveit2.base_link_name,
            constraint_kwargs["target_link"]
            if constraint_kwargs["target_link"] is not None
            else self.moveit2.end_effector_name,
            rclpy.time.Time(),
        )
        current = R.from_quat(list(ros2_numpy.numpify(t_stamped.transform.rotation)))
        desired = None
        if isinstance(constraint_kwargs["quat_xyzw"], Quaternion):
            desired = R.from_quat(
                list(ros2_numpy.numpify(constraint_kwargs["quat_xyzw"]))
            )
        else:
            desired = R.from_quat(list(constraint_kwargs["quat_xyzw"]))

        diff = R.from_matrix(np.dot(current.as_matrix().T, desired.as_matrix()))

        # For specifics, see:
        # https://github.com/ros-planning/moveit2/blob/f9989626491ca63e9f7f612964b03afcf0749bea/moveit_core/kinematic_constraints/src/kinematic_constraint.cpp
        if constraint_kwargs["parameterization"] == 0:  # Euler Angles
            diff_xyz = diff.as_euler("xyz", degrees=False)
            if MoveIt2Plan.normalize_absolute_angle(diff_xyz[0]) > tol[2]:
                diff_xyz[2] = diff_xyz[0]
                diff_xyz[0] = 0.0
            # Account for angle wrapping
            diff_xyz = np.array(
                [MoveIt2Plan.normalize_absolute_angle(angle) for angle in diff_xyz],
                dtype=diff_xyz.dtype,
            )
        else:  # Rotation Vector
            diff_xyz = np.fabs(diff.as_rotvec())

        return np.all(diff_xyz <= tol)

    @staticmethod
    def scale_velocity(traj: JointTrajectory, scale_factor: float) -> None:
        """
        Scale the velocity of the trajectory by the given factor. The resulting
        trajectory should execute the same trajectory with the same continuity,
        but just take 1/scale_factor as long to execute.

        This function keeps positions the same and scales time, velocities, and
        accelerations. It does not modify effort.

        Parameters
        ----------
        traj: The trajectory to scale.
        scale_factor: The factor to scale the velocity by, in [0, 1].
        """
        for point in traj.points:
            # Scale time_from_start
            nsec = point.time_from_start.sec * 10.0**9
            nsec += point.time_from_start.nanosec
            nsec /= scale_factor  # scale time
            sec = int(math.floor(nsec / 10.0**9))
            point.time_from_start.sec = sec
            point.time_from_start.nanosec = int(nsec - sec * 10.0**9)

            # Scale the velocities
            # pylint: disable=consider-using-enumerate
            # Necessary because we want to destructively modify the trajectory
            for i in range(len(point.velocities)):
                point.velocities[i] *= scale_factor

            # Scale the accelerations
            for i in range(len(point.accelerations)):
                point.accelerations[i] *= scale_factor**2

    @staticmethod
    def normalize_absolute_angle(angle: float) -> float:
        """
        Normalizes an angle to the interval [-pi, +pi] and then take the absolute value
        The returned values will be in the following range [0, +pi]
        """
        normalized_angle = np.fmod(np.fabs(angle), 2 * np.pi)
        return min(2 * np.pi - normalized_angle, normalized_angle)

    @staticmethod
    def visualize_trajectory(action_name: str, traj: JointTrajectory) -> None:
        """
        Generates a visualization of the positions, velocities, and accelerations
        of each joint in the trajectory. Saves the visualization in the share
        directory for `ada_feeding`, as `trajectories/{timestamp}_{action}.png`.
        Also saves a CSV of the trajectory.

        Parameters
        ----------
        action_name: The name of the action that generated this trajectory.
        traj: The trajectory to visualize.
        """

        # pylint: disable=too-many-locals
        # Necessary because this function saves both the image and CSV

        # pylint: disable=import-outside-toplevel
        # No need to import graphing libraries if we aren't saving the trajectory
        import csv
        from ament_index_python.packages import get_package_share_directory
        import matplotlib.pyplot as plt

        # Get the filepath, excluding the extension
        file_dir = os.path.join(
            get_package_share_directory("ada_feeding"),
            "trajectories",
        )
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        filepath = os.path.join(
            file_dir,
            f"{int(time.time()*10**9)}_{action_name}",
        )

        # Generate the CSV header
        csv_header = ["time_from_start"]
        for descr in ["Position", "Velocity", "Acceleration"]:
            for joint_name in traj.joint_names:
                csv_header.append(f"{joint_name} {descr}")
        csv_data = [csv_header]

        # Generate the axes for the graph
        nrows = 2
        ncols = int(math.ceil(len(traj.joint_names) / float(nrows)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
        positions = [[] for _ in traj.joint_names]
        velocities = [[] for _ in traj.joint_names]
        accelerations = [[] for _ in traj.joint_names]
        time_from_start = []

        # Loop over the trajectory
        for point in traj.points:
            timestamp = (
                point.time_from_start.sec + point.time_from_start.nanosec * 10.0**-9
            )
            row = [timestamp]
            time_from_start.append(timestamp)
            for descr in ["positions", "velocities", "accelerations"]:
                for i, joint_name in enumerate(traj.joint_names):
                    row.append(getattr(point, descr)[i])
                    if descr == "positions":
                        positions[i].append(getattr(point, descr)[i])
                    elif descr == "velocities":
                        velocities[i].append(getattr(point, descr)[i])
                    elif descr == "accelerations":
                        accelerations[i].append(getattr(point, descr)[i])
            csv_data.append(row)

        # Generate and save the figure
        for i, joint_name in enumerate(traj.joint_names):
            ax = axes[i // ncols, i % ncols]
            ax.plot(time_from_start, positions[i], label="Position (rad)")
            ax.plot(time_from_start, velocities[i], label="Velocity (rad/s)")
            ax.plot(time_from_start, accelerations[i], label="Acceleration (rad/s^2)")
            ax.set_xlabel("Time (s)")
            ax.set_title(joint_name)
            ax.legend()
        fig.tight_layout()
        fig.savefig(filepath + ".png")
        plt.clf()

        # Save the CSV
        with open(filepath + ".csv", "w", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(csv_data)
