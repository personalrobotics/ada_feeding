#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveTo behavior, which contains the core
logic to move the robot arm using pymoveit2.
"""
# Standard imports
import math
import time
from typing import List, Optional

# Third-party imports
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes
import py_trees
from pymoveit2 import MoveIt2, MoveIt2State
from pymoveit2.robots import kinova
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

# Local imports
from ada_feeding.helpers import get_from_blackboard_with_default


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
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.terminate_timeout_s = terminate_timeout_s
        self.terminate_rate_hz = terminate_rate_hz
        self.planning_service_timeout_s = planning_service_timeout_s

        # Initialization the blackboard for this behavior
        self.move_to_blackboard = self.attach_blackboard_client(
            name=name + " MoveTo", namespace=name
        )
        # All planning calls have the option to either be cartesian or kinematic
        self.move_to_blackboard.register_key(
            key="cartesian", access=py_trees.common.Access.READ
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
        # Initialize the blackboard to read from the parent behavior tree
        self.tree_blackboard = self.attach_blackboard_client(
            name=name + " MoveTo", namespace=tree_name
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

        # Create MoveIt 2 interface for moving the Jaco arm. This must be done
        # in __init__ and not setup since the MoveIt2 interface must be
        # initialized before the ROS2 node starts spinning.
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name=kinova.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        # Subscribe to the joint state and track the distance to goal while the
        # robot is executing the trajectory.
        self.distance_to_goal = DistanceToGoal()
        node.create_subscription(
            msg_type=JointState,
            topic="joint_states",
            callback=self.distance_to_goal.joint_state_callback,
            qos_profile=QoSPresetProfiles.SENSOR_DATA.value,
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

        # Get all parameters for motion, resorting to default values if unset.
        self.joint_names = kinova.joint_names()
        self.cartesian = get_from_blackboard_with_default(
            self.move_to_blackboard, "cartesian", False
        )

        # Set the joint names
        self.distance_to_goal.set_joint_names(self.joint_names)

    # pylint: disable=too-many-branches, too-many-return-statements
    # This is the heart of the MoveTo behavior, so the number of branches
    # and return statements is reasonable.
    def update(self) -> py_trees.common.Status:
        """
        Monitor the progress of moving the robot. This includes:
            - Checking if planning is complete
            - Transitioning to motion
            - Checking if motion is complete
            - Updating feedback in the blackboard
        """
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
                self.planning_future = self.moveit2.plan_async(cartesian=self.cartesian)
                return py_trees.common.Status.RUNNING

            # Check if planning is complete
            if self.planning_future.done():
                # Transition from planning to motion
                self.tree_blackboard.is_planning = False
                self.motion_start_time = time.time()

                # Get the trajectory
                traj = self.moveit2.get_trajectory(
                    self.planning_future, cartesian=self.cartesian
                )
                self.logger.info(f"Trajectory: {traj} | type {type(traj)}")
                if traj is None:
                    self.logger.error(
                        f"{self.name} [MoveTo::update()] Failed to get trajectory from MoveIt!"
                    )
                    return py_trees.common.Status.FAILURE

                # Set the trajectory's initial distance to goal
                self.tree_blackboard.motion_initial_distance = (
                    self.distance_to_goal.set_trajectory(traj)
                )
                self.tree_blackboard.motion_curr_distance = (
                    self.tree_blackboard.motion_initial_distance
                )

                # Send the trajectory to MoveIt
                self.moveit2.execute(traj)
                return py_trees.common.Status.RUNNING

            # Still planning
            return py_trees.common.Status.RUNNING

        # Is moving
        self.tree_blackboard.motion_time = time.time() - self.motion_start_time
        if self.motion_future is None:
            if self.moveit2.query_state() == MoveIt2State.REQUESTING:
                # The goal has been sent to the action server, but not yet accepted
                return py_trees.common.Status.RUNNING
            if self.moveit2.query_state() == MoveIt2State.EXECUTING:
                # The goal has been accepted and is executing. In this case
                # don't return a status since we drop down into the below
                # for when the robot is in motion.
                self.motion_future = self.moveit2.get_execution_future()
            elif self.moveit2.query_state() == MoveIt2State.IDLE:
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
            self.tree_blackboard.motion_curr_distance = (
                self.distance_to_goal.get_distance()
            )
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
        while self.moveit2.query_state() != MoveIt2State.IDLE:
            self.node.get_logger().info(
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


class DistanceToGoal:
    """
    The DistanceToGoal class is used to determine how much of the trajectory
    the robot arm has yet to execute.

    In practice, it keeps track of what joint state along the trajectory the
    robot is currently in, and returns the number of remaining joint states. As
    a result, this is not technically a measure of either distance or time, but
    should give some intuition of how much of the trajectory is left to execute.
    """

    def __init__(self):
        """
        Initializes the DistanceToGoal class.
        """
        self.joint_names = None
        self.aligned_joint_indices = None

        self.trajectory = None
        self.curr_joint_state_i = 0
        self.curr_joint_state = None

    def set_joint_names(self, joint_names: List[str]) -> None:
        """
        This function stores the robot's joint names.

        Parameters
        ----------
        joint_names: The names of the joints that the robot arm is moving.
        """
        self.joint_names = joint_names

    def set_trajectory(self, trajectory: JointTrajectory) -> float:
        """
        This function takes in the robot's trajectory and returns the initial
        distance to goal e.g., the distance between the starting and ending
        joint state. In practice, this returns the length of the trajectory.
        """
        self.trajectory = trajectory
        self.curr_joint_state_i = 0
        return float(len(self.trajectory.points))

    def joint_state_callback(self, msg: JointState) -> None:
        """
        This function stores the robot's current joint state, and aligns those
        messages with the order of joints in the robot's computed trajectory.

        It assumes:
            (a) that there are joint state messages that contain all of the robot's
                joints that are in the trajectory.
            (b) that the order of joins in the trajectory and in the joint state
                messages never changes.
        """
        # Only store messages that have all of the robot's joints. This is
        # because sometimes nodes other than the robot will be publishing to
        # the joint state publisher.
        contains_all_robot_joints = True
        if self.joint_names is not None:
            for joint_name in self.joint_names:
                if joint_name not in msg.name:
                    contains_all_robot_joints = False
                    break
        if contains_all_robot_joints:
            self.curr_joint_state = msg

            # Align the joint names between the JointState and JointTrajectory
            # messages.
            if self.aligned_joint_indices is None and self.trajectory is not None:
                self.aligned_joint_indices = []
                for joint_name in self.joint_names:
                    if (
                        joint_name in msg.name
                        and joint_name in self.trajectory.joint_names
                    ):
                        joint_state_i = msg.name.index(joint_name)
                        joint_traj_i = self.trajectory.joint_names.index(joint_name)
                        self.aligned_joint_indices.append(
                            (joint_name, joint_state_i, joint_traj_i)
                        )

    @staticmethod
    def joint_position_dist(point1: float, point2: float) -> float:
        """
        Given two joint positions in radians, this function computes the
        distance between then, accounting for rotational symmetry.
        """
        abs_dist = abs(point1 - point2) % (2 * math.pi)
        return min(abs_dist, 2 * math.pi - abs_dist)

    def get_distance(self) -> Optional[float]:
        """
        This function determines where in the trajectory the robot is. It does
        this by computing the distance (L1 distance across the joint positions)
        between the current joint state and the upcoming joint states in the
        trajectory, and selecting the nearest local min.

        This function assumes the joint names are aligned between the JointState
        and JointTrajectory messages.
        """
        # If we haven't yet received a joint state message to the trajectory,
        # immediately return
        if self.aligned_joint_indices is None:
            if self.trajectory is None:
                return None
            return float(len(self.trajectory.points) - self.curr_joint_state_i)

        # Else, determine how much remaining the robot has of the trajectory
        prev_dist = None
        for i in range(self.curr_joint_state_i, len(self.trajectory.points)):
            # Compute the distance between the current joint state and the
            # ujoint state at index i.
            traj_joint_state = self.trajectory.points[i]
            dist = sum(
                DistanceToGoal.joint_position_dist(
                    self.curr_joint_state.position[joint_state_i],
                    traj_joint_state.positions[joint_traj_i],
                )
                for (_, joint_state_i, joint_traj_i) in self.aligned_joint_indices
            )

            # If the distance is increasing, we've found the local min.
            if prev_dist is not None:
                if dist >= prev_dist:
                    self.curr_joint_state_i = i - 1
                    return float(len(self.trajectory.points) - self.curr_joint_state_i)

            prev_dist = dist

        # If the distance never increased, we are nearing the final waypoint.
        # Because the robot may still have slight motion even after this point,
        # we conservatively return 1.0.
        return 1.0
