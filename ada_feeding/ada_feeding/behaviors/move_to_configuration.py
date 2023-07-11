#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfiguration behavior tree, which moves the Jaco
arm to a specified joint configuration.
"""
# Standard imports
import time

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

# Local imports
from ada_feeding.helpers import DistanceToGoal


class MoveToConfiguration(py_trees.behaviour.Behaviour):
    """
    A generic behavior for moving the Jaco arm a specified joint configuration.
    """

    def __init__(
        self,
        name: str,
        node: Node,
        terminate_timeout_s: float = 10.0,
        terminate_rate_hz: float = 30.0,
    ):
        """
        A generic behavior for moving the Jaco arm a specified joint configuration.

        Parameters
        ----------
        node: The ROS2 node that this behavior is associated with. Necessary to
            connect to the MoveIt action server.
        terminate_timeout_s: How long after a terminate is requested to wait for a
            response from the MoveIt2 action server.
        terminate_rate_hz: How often to check whether a terminate request has been
            processed.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.node = node
        self.terminate_timeout_s = terminate_timeout_s
        self.terminate_rate_hz = terminate_rate_hz

        # Initialization the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " MoveToConfigurationBehavior", namespace=name
        )
        # Inputs for MoveToConfiguration
        self.blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="joint_names", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="cartesian", access=py_trees.common.Access.READ
        )
        # Feedback from MoveToConfiguration for the ROS2 Action Server
        self.blackboard.register_key(
            key="is_planning", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.register_key(
            key="planning_time", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.register_key(
            key="motion_time", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.register_key(
            key="motion_initial_distance", access=py_trees.common.Access.EXCLUSIVE_WRITE
        )
        self.blackboard.register_key(
            key="motion_curr_distance", access=py_trees.common.Access.EXCLUSIVE_WRITE
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

    def setup(self, **kwargs) -> None:
        """
        Create the MoveIt interface.
        """
        self.logger.info("%s [MoveToConfiguration::setup()]" % self.name)

    def initialise(self) -> None:
        """
        Send the goal to MoveIt and reset the blackboard.
        """
        self.logger.info("%s [MoveToConfiguration::initialise()]" % self.name)

        # Reset local state variables
        self.prev_query_state = None
        self.planning_start_time = time.time()
        self.motion_start_time = None
        self.motion_future = None

        # Reset the blackboard. The robot starts in planning.
        self.blackboard.is_planning = True
        self.blackboard.planning_time = 0.0
        self.blackboard.motion_time = 0.0
        self.blackboard.motion_initial_distance = 0.0
        self.blackboard.motion_curr_distance = 0.0

        # Get all parameters for planning, resorting to default values if unset.
        joint_positions = self.blackboard.joint_positions  # required
        try:
            joint_names = self.blackboard.joint_names
        except KeyError:
            joint_names = None  # default value
        try:
            tolerance = self.blackboard.tolerance
        except KeyError:
            tolerance = 0.001  # default value
        try:
            weight = self.blackboard.weight
        except KeyError:
            weight = 1.0  # default value
        try:
            self.cartesian = self.blackboard.cartesian
        except KeyError:
            self.cartesian = False  # default value

        # Set the joint names
        self.distance_to_goal.set_joint_names(joint_names)

        # Send a new goal to MoveIt
        self.planning_future = self.moveit2.plan_async(
            joint_positions=joint_positions,
            joint_names=joint_names,
            tolerance_joint_position=tolerance,
            weight_joint_position=weight,
            cartesian=self.cartesian,
        )

    def update(self) -> py_trees.common.Status:
        """
        Check if the MoveGroup dummy action server has finished.
        """
        self.logger.info("%s [MoveToConfiguration::update()]" % self.name)

        # Check the state of MoveIt
        if self.blackboard.is_planning:  # Is planning
            # Update the feedback
            self.blackboard.planning_time = time.time() - self.planning_start_time

            if self.planning_future.done():  # Finished planning
                # Transition from planning to motion
                self.blackboard.is_planning = False
                self.motion_start_time = time.time()

                # Get the trajectory
                traj = self.moveit2.get_trajectory(
                    self.planning_future, cartesian=self.cartesian
                )
                self.logger.info("Trajectory: %s | type %s" % (traj, type(traj)))
                if traj is None:
                    self.logger.error(
                        "%s [MoveToConfiguration::update()] Failed to get trajectory from MoveIt!"
                        % self.name
                    )
                    return py_trees.common.Status.FAILURE

                # Set the trajectory's initial distance to goal
                self.blackboard.motion_initial_distance = (
                    self.distance_to_goal.set_trajectory(traj)
                )
                self.blackboard.motion_curr_distance = (
                    self.blackboard.motion_initial_distance
                )

                # Send the trajectory to MoveIt
                self.moveit2.execute(traj)
                return py_trees.common.Status.RUNNING

            else:  # Still planning
                return py_trees.common.Status.RUNNING
        else:  # Is moving
            self.blackboard.motion_time = time.time() - self.motion_start_time
            if self.motion_future is None:
                if self.moveit2.query_state() == MoveIt2State.REQUESTING:
                    # The goal has been sent to the action server, but not yet accepted
                    return py_trees.common.Status.RUNNING
                elif self.moveit2.query_state() == MoveIt2State.EXECUTING:
                    # The goal has been accepted and is executing. In this case
                    # don't return a status since we drop down into the below
                    # for when the robot is in motion.
                    self.motion_future = self.moveit2.get_execution_future()
                elif self.moveit2.query_state() == MoveIt2State.IDLE:
                    # If we get here (i.e., self.moveit2 returned to IDLE without executing)
                    # then something went wrong (e.g., controller is already executing a
                    # trajectory, action server not available, goal was rejected, etc.)
                    return py_trees.common.Status.FAILURE
            if self.motion_future is not None:
                self.blackboard.motion_curr_distance = (
                    self.distance_to_goal.get_distance()
                )
                if self.motion_future.done():
                    # The goal has finished executing
                    if (
                        self.motion_future.result().status
                        == GoalStatus.STATUS_SUCCEEDED
                    ):
                        error_code = self.motion_future.result().result.error_code
                        if error_code.val == MoveItErrorCodes.SUCCESS:
                            # The goal succeeded
                            self.blackboard.motion_curr_distance = 0.0
                            return py_trees.common.Status.SUCCESS
                        else:
                            # The goal failed
                            return py_trees.common.Status.FAILURE
                    else:
                        # The goal failed
                        return py_trees.common.Status.FAILURE
                else:
                    # The goal is still executing
                    return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate this behavior.

        This will cancel any active goal and wait for the MoveIt2 action
        server to complete the termination.
        """
        self.logger.info(
            "%s [MoveToConfiguration::terminate()][%s->%s]"
            % (self.name, self.status, new_status)
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
            # If the goal is executing, cancel it
            if self.moveit2.query_state() == MoveIt2State.EXECUTING:
                self.moveit2.cancel_execution()

            # Check for terminate timeout
            if time.time() - terminate_requested_time > self.terminate_timeout_s:
                self.logger.error(
                    "%s [MoveToConfiguration::terminate()] Terminate timed out!"
                    % self.name
                )
                break

            rate.sleep()

    def shutdown(self) -> None:
        """
        Shutdown infrastructure created in setup().
        """
        self.logger.info("%s [MoveToConfiguration::shutdown()]" % self.name)
