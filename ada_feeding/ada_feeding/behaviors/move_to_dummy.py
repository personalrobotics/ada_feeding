#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import atexit
import multiprocessing
import multiprocessing.connection
import py_trees
import time

# Constants used to communicate between the main behavior process and the process
# that loosely mimic's MoveIt2's MoveGroup action.
MOVEGROUP_STATE_PLANNING = "PLAN"
MOVEGROUP_STATE_MOTION = "MONITOR"
MOVEGROUP_STATE_IDLE = "IDLE"
ROSACTION_NEW_GOAL = "new_goal"
ROSACTION_PREEMPT_GOAL = "preempt_goal"
ROSACTION_GOAL_SUCCEEDED = "goal_succeeded"
ROSACTION_GOAL_ABORTED = "goal_aborted"
ROSACTION_GOAL_PREEMPTED = "goal_preempted"


def move_group_dummy(
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
                if command == ROSACTION_NEW_GOAL:
                    idle = False
                    planning_start_time_s = time.time()
                elif command == ROSACTION_PREEMPT_GOAL:
                    idle = True
                    planning_start_time_s = None
                    motion_start_time = None
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
        """
        # Initiatilize the behavior
        super(MoveToDummy, self).__init__(name=name)

        # Store parameters
        self.dummy_plan_time_s = dummy_plan_time_s
        self.dummy_motion_time_s = dummy_motion_time_s
        self.prev_response = None
        self.planning_start_time = None
        self.motion_start_time = None

        # Initialization the blackboard
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(
            key="preempt_requested", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="was_preempted", access=py_trees.common.Access.WRITE
        )
        self.blackboard.preempt_requested = False
        self.blackboard.register_key(
            key="is_planning", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="planning_time", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="motion_initial_distance", access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="motion_curr_distance", access=py_trees.common.Access.WRITE
        )

    def setup(self, **kwargs: int) -> None:
        """
        Start the MoveGroup dummy action server.

        Note that in the actual scenario, this action would already be running.
        """
        self.logger.info("%s [MoveToDummy::setup()]" % self.name)
        # Create the pipe to communicate between processes
        self.parent_connection, self.child_connection = multiprocessing.Pipe()
        # Start the move group process
        self.move_group = multiprocessing.Process(
            target=move_group_dummy,
            args=(
                self.dummy_plan_time_s,
                self.dummy_motion_time_s,
                self.child_connection,
            ),
        )
        self.move_group.start()
        # Close it on exit
        atexit.register(self.move_group.terminate)

    def initialise(self) -> None:
        """
        Send a new goal to the MoveGroup dummy action server and reset the
        blackboard.
        """
        self.logger.info("%s [MoveToDummy::initialise()]" % self.name)

        # Reset the blackboard
        self.blackboard.preempt_requested = False
        self.blackboard.was_preempted = False
        self.blackboard.is_planning = False
        self.blackboard.planning_time = 0.0
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

        # Check if a preempt has been requested
        if self.blackboard.preempt_requested:
            self.parent_connection.send([ROSACTION_PREEMPT_GOAL])
            self.blackboard.was_preempted = True
            self.blackboard.preempt_requested = False

        # Check the status of the MoveGroup dummy action server
        if self.parent_connection.poll():
            # Get the response and update the previous response
            prev_response = self.prev_response
            response = self.parent_connection.recv().pop()
            self.prev_response = response

            # Process the response
            if response == ROSACTION_GOAL_SUCCEEDED:
                return py_trees.common.Status.SUCCESS
            elif response == ROSACTION_GOAL_ABORTED:
                return py_trees.common.Status.FAILURE
            elif response == ROSACTION_GOAL_PREEMPTED:
                return py_trees.common.Status.INVALID
            # Write to blackboard the information for feedback messages
            elif response == MOVEGROUP_STATE_PLANNING:
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
                self.blackboard.motion_curr_distance = self.dummy_motion_time_s - (
                    time.time() - self.motion_start_time
                )

        # If it hasn't finished, return running
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """
        Terminate this behavior.

        There is nothing to terminate in this case, since we already terminate
        the move group process in `atexit`
        """
        self.logger.info(
            "%s [MoveToDummy::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )