#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a dummy behavior for moving to a location. It spins up a
dummy process that very loosely mimics MoveIt2's MoveGroup action, and then
sends and receives commands from that. That dummy process sleeps for a
configurable number of secs for "planning" and another configurable number of
secs for "motion" before returning success.
"""
# Standard imports
import multiprocessing
import multiprocessing.connection
import time

# Third-party imports
import py_trees

# Constants used to communicate between the main behavior process and the process
# that loosely mimic's MoveIt2's MoveGroup action.
MOVEGROUP_STATE_PLANNING = "PLAN"
MOVEGROUP_STATE_MOTION = "MONITOR"
MOVEGROUP_STATE_IDLE = "IDLE"
ROSACTION_SHUTDOWN = "shutdown"
ROSACTION_NEW_GOAL = "new_goal"
ROSACTION_PREEMPT_GOAL = "preempt_goal"
ROSACTION_GOAL_SUCCEEDED = "goal_succeeded"
ROSACTION_GOAL_ABORTED = "goal_aborted"
ROSACTION_GOAL_PREEMPTED = "goal_preempted"


def _move_group_dummy(
    dummy_plan_time_s: float,
    dummy_motion_time_s: float,
    pipe_connection: multiprocessing.connection.Connection,
    feedback_s: float = 0.2,
) -> None:
    """
    This function is loosely designed to mimic a call to the MoveGroup action:
    https://github.com/ros-planning/moveit_msgs/blob/ros2/action/MoveGroup.action

    Specifically, it sends a string with the state MOVEGROUP_STATE_PLANNING while
    it is planning, and MOVEGROUP_STATE_MOTION while it is moving. It plans for
    `dummy_plan_time_s` seconds, and moves for `dummy_motion_time_s` seconds.

    Parameters
    ----------
    dummy_plan_time_s: How many seconds this dummy node should spend in planning.
    dummy_motion_time_s: How many seconds this dummy node should spend in motion.
    pipe_connection: The pipe connection to the main behavior process.
    feedback_s: How often to send feedback to the main behavior process. NOTE
        that because Pipe has no option to only take the latest message,
        `feedback_s` must be >= the period at which this tree is ticked, else
        `update` will be reading old messages from the Pipe.
    """
    idle = True
    planning_start_time_s = None
    motion_start_time_s = None
    try:
        while True:
            # See if the main behavior process has sent a command
            if pipe_connection.poll():
                command = pipe_connection.recv().pop()
                if command == ROSACTION_SHUTDOWN:
                    break
                if command == ROSACTION_NEW_GOAL:
                    idle = False
                    planning_start_time_s = time.time()
                elif command == ROSACTION_PREEMPT_GOAL:
                    idle = True
                    planning_start_time_s = None
                    motion_start_time_s = None
                    pipe_connection.send([ROSACTION_GOAL_PREEMPTED])
            # If we're not idle, continue with the action
            if not idle:
                if motion_start_time_s is None:  # If we're planning
                    planning_elapsed_time_s = time.time() - planning_start_time_s
                    if planning_elapsed_time_s < dummy_plan_time_s:
                        pipe_connection.send([MOVEGROUP_STATE_PLANNING])
                    else:
                        motion_start_time_s = time.time()
                else:  # If we're moving
                    motion_elapsed_time_s = time.time() - motion_start_time_s
                    if motion_elapsed_time_s < dummy_motion_time_s:
                        pipe_connection.send([MOVEGROUP_STATE_MOTION])
                    else:
                        idle = True
                        planning_start_time_s = None
                        motion_start_time_s = None
                        pipe_connection.send([ROSACTION_GOAL_SUCCEEDED])
            # Sleep for a bit before checking again
            time.sleep(feedback_s)
    except KeyboardInterrupt:
        pass


class MoveToDummy(py_trees.behaviour.Behaviour):
    """
    A dummy behavior for moving to a target position.

    This behavior is designed to loosely mimic calling MoveIt2's `MoveGroup`
    action.
    """

    def __init__(
        self,
        name: str,
        dummy_plan_time_s: float = 2.5,
        dummy_motion_time_s: float = 7.5,
        preempt_timeout_s: float = 10.0,
    ):
        """
        A dummy behavior for moving to a target position.

        This behavior will sleep for `dummy_plan_time_s` sec, then sleep for
        `dummy_motion_time_s` sec, and then succeed.

        Parameters
        ----------
        name: The name of the behavior.
        dummy_plan_time_s: How many seconds this dummy node should spend in planning.
        dummy_motion_time_s: How many seconds this dummy node should spend in motion.
        preempt_timeout_s: How long after a preempt is requested to wait for a
            response from the dummy MoveGroup action.
        """
        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.dummy_plan_time_s = dummy_plan_time_s
        self.dummy_motion_time_s = dummy_motion_time_s
        self.preempt_timeout_s = preempt_timeout_s
        self.prev_response = None
        self.planning_start_time = None
        self.motion_start_time = None

        # Initialization the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " MoveToDummyBehavior", namespace=name
        )
        # Note that the dummy node doesn't actually access the goal, but this is
        # included for completeness
        self.blackboard.register_key(key="goal", access=py_trees.common.Access.READ)
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

    def setup(self, **kwargs) -> None:
        """
        Start the MoveGroup dummy action server.

        Note that in the actual scenario, this action would already be running.
        """
        self.logger.info("%s [MoveToDummy::setup()]" % self.name)
        # Create the pipe to communicate between processes
        self.parent_connection, self.child_connection = multiprocessing.Pipe()
        # Start the move group process
        self.move_group = multiprocessing.Process(
            target=_move_group_dummy,
            args=(
                self.dummy_plan_time_s,
                self.dummy_motion_time_s,
                self.child_connection,
            ),
        )
        self.move_group.start()

    def initialise(self) -> None:
        """
        Send a new goal to the MoveGroup dummy action server and reset the
        blackboard.
        """
        self.logger.info("%s [MoveToDummy::initialise()]" % self.name)

        # Reset local state variables
        self.prev_response = None
        self.planning_start_time = None
        self.motion_start_time = None

        # Reset the blackboard
        self.blackboard.is_planning = False
        self.blackboard.planning_time = 0.0
        self.blackboard.motion_time = 0.0
        self.blackboard.motion_initial_distance = 0.0
        self.blackboard.motion_curr_distance = 0.0

        # Send a new goal to the MoveGroup dummy action server
        self.parent_connection.send([ROSACTION_NEW_GOAL])

    def update(self) -> py_trees.common.Status:
        """
        Check if the MoveGroup dummy action server has finished.
        """
        self.logger.info("%s [MoveToDummy::update()]" % self.name)

        # Check if the process has died
        if not self.move_group.is_alive():
            self.logger.error(
                "%s [MoveToDummy::update()] MoveGroup dummy process died!" % self.name
            )
            return py_trees.common.Status.FAILURE

        # Check the status of the MoveGroup dummy action server
        if self.parent_connection.poll():
            # Get the response and update the previous response
            prev_response = self.prev_response
            response = self.parent_connection.recv().pop()
            self.prev_response = response

            # Process the response
            if response == ROSACTION_GOAL_SUCCEEDED:
                return py_trees.common.Status.SUCCESS
            if response == ROSACTION_GOAL_ABORTED:
                return py_trees.common.Status.FAILURE
            if response == ROSACTION_GOAL_PREEMPTED:
                return py_trees.common.Status.INVALID
            # Write to blackboard the information for feedback messages
            if response == MOVEGROUP_STATE_PLANNING:
                if prev_response != response:  # If the robot just started planning
                    self.blackboard.is_planning = True
                    self.planning_start_time = time.time()
                self.blackboard.planning_time = time.time() - self.planning_start_time
            elif response == MOVEGROUP_STATE_MOTION:
                if prev_response != response:  # If the robot just started moving
                    self.blackboard.is_planning = False
                    self.blackboard.planning_time = (
                        time.time() - self.planning_start_time
                    )
                    self.motion_start_time = time.time()
                    # TODO: On the actual robot, `motion_initial_distance` and
                    #       `motion_curr_distance` should determine the distance
                    #       to the goal.
                    self.blackboard.motion_initial_distance = self.dummy_motion_time_s
                self.blackboard.motion_time = time.time() - self.motion_start_time
                self.blackboard.motion_curr_distance = (
                    self.dummy_motion_time_s - self.blackboard.motion_time
                )

        # If it hasn't finished, return running
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate this behavior.

        This will cancel any active goal and wait for the MoveGroup dummy action
        server to complete the preemption.
        """
        self.logger.info(
            "%s [MoveToDummy::terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )

        # Cancel any active goal
        if self.move_group.is_alive():
            # Send the preempt request
            self.parent_connection.send([ROSACTION_PREEMPT_GOAL])
            preempt_requested_time = time.time()
            # Wait for the response
            while time.time() - preempt_requested_time < self.preempt_timeout_s:
                if self.parent_connection.poll():
                    response = self.parent_connection.recv().pop()
                    if response == ROSACTION_GOAL_PREEMPTED:
                        break
        self.logger.info("%s [MoveToDummy::terminate()] completed" % self.name)

    def shutdown(self) -> None:
        """
        Shutdown infrastructure created in setup().

        In this case, terminate the MoveGroup dummy action server.
        """
        self.logger.info("%s [MoveToDummy::shutdown()]" % self.name)
        # Terminate the MoveGroup dummy action server
        self.parent_connection.send([ROSACTION_SHUTDOWN])
        self.move_group.join()
        self.move_group.close()
        # Close the pipe
        self.parent_connection.close()
        self.child_connection.close()
