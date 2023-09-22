#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveTo behavior, which contains the core
logic to move the robot arm using pymoveit2.
"""
# Standard imports
import csv
import math
import os
import time

# Third-party imports
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.msg import MoveItErrorCodes
import py_trees
from pymoveit2 import MoveIt2State
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

# Local imports
from ada_feeding.helpers import get_from_blackboard_with_default, get_moveit2_object


class MoveTo(py_trees.behaviour.Behaviour):
    """
    MoveTo is a generic behavior to move the robot arm. It creates the MoveIt
    interface, plans to the goal, handles the tree tick, generates feedback,
    and handles termination.

    Note that this class does not specify any goal; goals must be specified by
    adding decorators on top of this class.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # This behavior needs to keep track of lots of information across ticks,
    # so the number of attributes is reasonable. It is also intended to be generic,
    # so the number of arguments is reasonable.
    # pylint: disable=duplicate-code
    # The MoveIt2 object will have similar code in any file it is created.
    def __init__(
        self,
        name: str,
        tree_name: str,
        node: Node,
        terminate_timeout_s: float = 10.0,
        terminate_rate_hz: float = 30.0,
        planning_service_timeout_s: float = 10.0,
        save_trajectory_viz: bool = False,
    ):
        """
        Initialize the MoveTo class.

        Parameters
        ----------
        name: The name of the behavior.
        tree_name: The name of the behavior tree. This is necessary because the
            blackboard elements related to feedback, e.g., is_planning, are
            defined in the behavior tree's namespace.
        node: The ROS2 node that this behavior is associated with. Necessary to
            connect to the MoveIt action server.
        terminate_timeout_s: How long after a terminate is requested to wait for a
            response from the MoveIt2 action server.
        terminate_rate_hz: How often to check whether a terminate request has been
            processed.
        planning_service_timeout_s: How long to wait for the planning service to be
            ready before failing.
        save_trajectory_viz: Whether to generate and save a visualization of the
            trajectory in joint space. This is useful for debugging, but should
            be disabled in production.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.tree_name = tree_name
        self.terminate_timeout_s = terminate_timeout_s
        self.terminate_rate_hz = terminate_rate_hz
        self.planning_service_timeout_s = planning_service_timeout_s
        self.save_trajectory_viz = save_trajectory_viz

        # Initialization the blackboard for this behavior
        self.move_to_blackboard = self.attach_blackboard_client(
            name=name + " MoveTo", namespace=name
        )
        # All planning calls have the option to either be cartesian or kinematic
        self.move_to_blackboard.register_key(
            key="cartesian", access=py_trees.common.Access.READ
        )
        # Add the ability to set a pipeline_id
        self.move_to_blackboard.register_key(
            key="pipeline_id", access=py_trees.common.Access.READ
        )
        # Add the ability to set a planner_id
        self.move_to_blackboard.register_key(
            key="planner_id", access=py_trees.common.Access.READ
        )
        # Add the ability to set an allowed planning time
        self.move_to_blackboard.register_key(
            key="allowed_planning_time", access=py_trees.common.Access.READ
        )
        # Add the ability to set velocity scaling
        self.move_to_blackboard.register_key(
            key="max_velocity_scaling_factor", access=py_trees.common.Access.READ
        )
        # Add the ability to set acceleration scaling
        self.move_to_blackboard.register_key(
            key="max_acceleration_scaling_factor", access=py_trees.common.Access.READ
        )
        # Add the ability to set the cartesian jump threshold
        self.move_to_blackboard.register_key(
            key="cartesian_jump_threshold", access=py_trees.common.Access.READ
        )
        # Add the ability to set the cartesian max step
        self.move_to_blackboard.register_key(
            key="cartesian_max_step", access=py_trees.common.Access.READ
        )
        # Add the ability to set a cartesian fraction threshold (e.g., only
        # accept plans that completed at least this fraction of the path)
        self.move_to_blackboard.register_key(
            key="cartesian_fraction_threshold", access=py_trees.common.Access.READ
        )

        # Initialize the blackboard to read from the parent behavior tree
        self.tree_blackboard = self.attach_blackboard_client(
            name=name + " MoveTo", namespace=self.tree_name
        )
        # Feedback from MoveTo for the ROS2 Action Server
        self.tree_blackboard.register_key(
            key="is_planning", access=py_trees.common.Access.WRITE
        )
        self.tree_blackboard.register_key(
            key="planning_time", access=py_trees.common.Access.WRITE
        )
        self.tree_blackboard.register_key(
            key="motion_time", access=py_trees.common.Access.WRITE
        )
        self.tree_blackboard.register_key(
            key="motion_initial_distance", access=py_trees.common.Access.WRITE
        )
        self.tree_blackboard.register_key(
            key="motion_curr_distance", access=py_trees.common.Access.WRITE
        )

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.move_to_blackboard,
            self.node,
        )

    # pylint: disable=attribute-defined-outside-init
    # For attributes that are only used during the execution of the tree
    # and get reset before the next execution, it is reasonable to define
    # them in `initialise`.
    def initialise(self) -> None:
        """
        Reset the blackboard and configure all parameters for motion.
        """
        self.logger.info(f"{self.name} [MoveTo::initialise()]")

        with self.moveit2_lock:
            # Set the planner_id
            self.moveit2.planner_id = get_from_blackboard_with_default(
                self.move_to_blackboard, "planner_id", "RRTstarkConfigDefault"
            )
            # Set the pipeline_id
            self.moveit2.pipeline_id = get_from_blackboard_with_default(
                self.move_to_blackboard, "pipeline_id", "ompl"
            )

            # Set the planner_id
            self.moveit2.planner_id = get_from_blackboard_with_default(
                self.move_to_blackboard, "planner_id", "RRTstarkConfigDefault"
            )

            # Set the max velocity
            self.moveit2.max_velocity = get_from_blackboard_with_default(
                self.move_to_blackboard, "max_velocity_scaling_factor", 0.1
            )

            # Set the allowed planning time
            self.moveit2.allowed_planning_time = get_from_blackboard_with_default(
                self.move_to_blackboard, "allowed_planning_time", 0.5
            )

            # Set the max acceleration
            self.moveit2.max_acceleration = get_from_blackboard_with_default(
                self.move_to_blackboard, "max_acceleration_scaling_factor", 0.1
            )

            # Set the allowed planning time
            self.moveit2.allowed_planning_time = get_from_blackboard_with_default(
                self.move_to_blackboard, "allowed_planning_time", 0.5
            )

        # Set the cartesian jump threshold
        self.moveit2.cartesian_jump_threshold = get_from_blackboard_with_default(
            self.move_to_blackboard, "cartesian_jump_threshold", 0.0
        )

        # Get whether we should use the cartesian interpolator
        self.cartesian = get_from_blackboard_with_default(
            self.move_to_blackboard, "cartesian", False
        )

        # Get the cartesian max step
        self.cartesian_max_step = get_from_blackboard_with_default(
            self.move_to_blackboard, "cartesian_max_step", 0.0025
        )

        # Get the cartesian fraction threshold
        self.cartesian_fraction_threshold = get_from_blackboard_with_default(
            self.move_to_blackboard, "cartesian_fraction_threshold", 0.0
        )

        # If the plan is cartesian, it should always avoid collisions
        self.moveit2.cartesian_avoid_collisions = True

        # Reset local state variables
        self.prev_query_state = None
        self.planning_start_time = time.time()
        self.planning_future = None
        self.motion_start_time = None
        self.motion_future = None

        # Reset the feedback. The robot starts in planning.
        self.tree_blackboard.is_planning = True
        self.tree_blackboard.planning_time = 0.0
        self.tree_blackboard.motion_time = 0.0
        self.tree_blackboard.motion_initial_distance = 0.0
        self.tree_blackboard.motion_curr_distance = 0.0

        # Store the trajectory to be able to monitor the robot's progress
        self.traj = None
        self.traj_i = 0
        self.aligned_joint_indices = None

    def update(self) -> py_trees.common.Status:
        """
        Monitor the progress of moving the robot. This includes:
            - Checking if planning is complete
            - Transitioning to motion
            - Checking if motion is complete
            - Updating feedback in the blackboard
        """

        # pylint: disable=too-many-branches, too-many-return-statements
        # This is the heart of the MoveTo behavior, so the number of branches
        # and return statements is reasonable.

        self.logger.info(f"{self.name} [MoveTo::update()]")

        # Check the state of MoveIt
        if self.tree_blackboard.is_planning:  # Is planning
            # Update the feedback
            self.tree_blackboard.planning_time = time.time() - self.planning_start_time

            # Check if we have succesfully initiated planning
            if self.planning_future is None:
                # Check if we have timed out waiting for the planning service
                if self.tree_blackboard.planning_time > self.planning_service_timeout_s:
                    self.logger.error(
                        f"{self.name} [MoveTo::update()] Planning timed out!"
                    )
                    return py_trees.common.Status.FAILURE
                # Initiate an asynchronous planning call
                with self.moveit2_lock:
                    planning_future = self.moveit2.plan_async(
                        cartesian=self.cartesian, max_step=self.cartesian_max_step
                    )
                if planning_future is None:
                    self.logger.error(
                        f"{self.name} [MoveTo::update()] Failed to initiate planning!"
                    )
                    return py_trees.common.Status.FAILURE
                self.planning_future = planning_future
                return py_trees.common.Status.RUNNING

            # Check if planning is complete
            if self.planning_future.done():
                # Transition from planning to motion
                self.tree_blackboard.is_planning = False
                self.motion_start_time = time.time()

                # Get the trajectory
                with self.moveit2_lock:
                    self.traj = self.moveit2.get_trajectory(
                        self.planning_future,
                        cartesian=self.cartesian,
                        cartesian_fraction_threshold=self.cartesian_fraction_threshold,
                    )
                self.logger.info(f"Trajectory: {self.traj}")
                if self.traj is None:
                    self.logger.error(
                        f"{self.name} [MoveTo::update()] Failed to get trajectory from MoveIt!"
                    )
                    return py_trees.common.Status.FAILURE

                # MoveIt's default cartesian interpolator doesn't respect velocity
                # scaling, so we need to manually add that.
                if self.cartesian and self.moveit2.max_velocity > 0.0:
                    MoveTo.scale_velocity(self.traj, self.moveit2.max_velocity)

                # Save the trajectory visualization
                if self.save_trajectory_viz:
                    MoveTo.visualize_trajectory(self.tree_name, self.traj)

                # Set the trajectory's initial distance to goal
                self.tree_blackboard.motion_initial_distance = float(
                    len(self.traj.points)
                )
                self.tree_blackboard.motion_curr_distance = (
                    self.tree_blackboard.motion_initial_distance
                )

                # Send the trajectory to MoveIt
                with self.moveit2_lock:
                    self.moveit2.execute(self.traj)
                return py_trees.common.Status.RUNNING

            # Still planning
            return py_trees.common.Status.RUNNING

        # Is moving
        self.tree_blackboard.motion_time = time.time() - self.motion_start_time
        if self.motion_future is None:
            with self.moveit2_lock:
                query_state = self.moveit2.query_state()
            if query_state == MoveIt2State.REQUESTING:
                # The goal has been sent to the action server, but not yet accepted
                return py_trees.common.Status.RUNNING
            if query_state == MoveIt2State.EXECUTING:
                # The goal has been accepted and is executing. In this case
                # don't return a status since we drop down into the below
                # for when the robot is in motion.
                with self.moveit2_lock:
                    self.motion_future = self.moveit2.get_execution_future()
            elif query_state == MoveIt2State.IDLE:
                with self.moveit2_lock:
                    last_error_code = self.moveit2.get_last_execution_error_code()
                if last_error_code is None or last_error_code.val != 1:
                    # If we get here then something went wrong (e.g., controller
                    # is already executing a trajectory, action server not
                    # available, goal was rejected, etc.)
                    self.logger.error(
                        f"{self.name} [MoveTo::update()] Failed to execute trajectory before goal "
                        "was accepted!"
                    )
                    return py_trees.common.Status.FAILURE
                # If we get here, the goal finished executing within the
                # last tick.
                self.tree_blackboard.motion_curr_distance = 0.0
                return py_trees.common.Status.SUCCESS
        if self.motion_future is not None:
            self.tree_blackboard.motion_curr_distance = self.get_distance_to_goal()
            if self.motion_future.done():
                # The goal has finished executing
                if self.motion_future.result().status == GoalStatus.STATUS_SUCCEEDED:
                    error_code = self.motion_future.result().result.error_code
                    if error_code.val == MoveItErrorCodes.SUCCESS:
                        # The goal succeeded
                        self.tree_blackboard.motion_curr_distance = 0.0
                        return py_trees.common.Status.SUCCESS

                    # The goal failed
                    return py_trees.common.Status.FAILURE

                # The goal failed
                return py_trees.common.Status.FAILURE

        # The goal is still executing
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate this behavior.

        This will cancel any active goal and wait for the MoveIt2 action
        server to complete the termination.
        """
        self.logger.info(
            f"{self.name} [MoveTo::terminate()][{self.status}->{new_status}]"
        )

        # Cancel execution of any active goals
        #   - If we have requested a goal but it has not yet been accepted/rejected,
        #     (i.e., MoveIt2State.REQUESTING) then wait until it is accepted/rejected.
        #   - If a goal has been accepted and is therefore executing (i.e.,
        #     MoveIt2State.EXECUTING), then cancel the goal and wait until it has canceled.
        #   - If the goal has finished executing (i.e., MoveIt2State.IDLE), then do nothing.
        terminate_requested_time = time.time()
        rate = self.node.create_rate(self.terminate_rate_hz)
        # A termination request has not succeeded until the MoveIt2 action server is IDLE
        with self.moveit2_lock:
            while self.moveit2.query_state() != MoveIt2State.IDLE:
                self.logger.info(
                    f"MoveTo Update MoveIt2State not Idle {time.time()} {terminate_requested_time} "
                    f"{self.terminate_timeout_s}"
                )
                # If the goal is executing, cancel it
                if self.moveit2.query_state() == MoveIt2State.EXECUTING:
                    self.moveit2.cancel_execution()

                # Check for terminate timeout
                if time.time() - terminate_requested_time > self.terminate_timeout_s:
                    self.logger.error(
                        f"{self.name} [MoveTo::terminate()] Terminate timed out!"
                    )
                    break

                rate.sleep()

    def shutdown(self) -> None:
        """
        Shutdown infrastructure created in setup().
        """
        self.logger.info(f"{self.name} [MoveTo::shutdown()]")

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
    def joint_position_dist(point1: float, point2: float) -> float:
        """
        Given two joint positions in radians, this function computes the
        distance between then, accounting for rotational symmetry.

        Parameters
        ----------
        point1: The first joint position, in radians.
        point2: The second joint position, in radians.
        """
        abs_dist = abs(point1 - point2) % (2 * math.pi)
        return min(abs_dist, 2 * math.pi - abs_dist)

    def get_distance_to_goal(self) -> float:
        """
        Returns the remaining distance to the goal.

        In practice, it keeps track of what joint state along the trajectory the
        robot is currently in, and returns the number of remaining joint states. As
        a result, this is not technically a measure of either distance or time, but
        should give some intuition of how much of the trajectory is left to execute.
        """
        # If the trajectory has not been set, something is wrong.
        if self.traj is None:
            self.logger.error(
                f"{self.name} [MoveTo::get_distance_to_goal()] Trajectory is None!"
            )
            return 0.0

        # Get the latest joint state of the robot
        with self.moveit2_lock:
            curr_joint_state = self.moveit2.joint_state

        # Lazily align the joints between the trajectory and the joint states message
        if self.aligned_joint_indices is None:
            self.aligned_joint_indices = []
            for joint_traj_i, joint_name in enumerate(self.traj.joint_names):
                if joint_name in curr_joint_state.name:
                    joint_state_i = curr_joint_state.name.index(joint_name)
                    self.aligned_joint_indices.append(
                        (joint_name, joint_state_i, joint_traj_i)
                    )
                else:
                    self.logger.error(
                        f"{self.name} [MoveTo::get_distance_to_goal()] Joint {joint_name} not in "
                        "current joint state, despite being in the trajectory. Skipping this joint "
                        "in distance to goal calculation."
                    )

        # Get the point along the trajectory closest to the robot's current joint state
        min_dist = None
        for i in range(self.traj_i, len(self.traj.points)):
            # Compute the distance between the current joint state and the
            # ujoint state at index i.
            traj_joint_state = self.traj.points[i]
            dist = sum(
                MoveTo.joint_position_dist(
                    curr_joint_state.position[joint_state_i],
                    traj_joint_state.positions[joint_traj_i],
                )
                for (_, joint_state_i, joint_traj_i) in self.aligned_joint_indices
            )

            # If the distance is increasing, we've found the local min.
            if min_dist is not None:
                if dist >= min_dist:
                    self.traj_i = i - 1
                    return float(len(self.traj.points) - self.traj_i)

            min_dist = dist

        # If the distance never increased, we are nearing the final waypoint.
        # Because the robot may still have slight motion even after this point,
        # we conservatively return 1.0.
        return 1.0

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
