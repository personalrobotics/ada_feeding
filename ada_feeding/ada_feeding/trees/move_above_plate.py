#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ada_feeding_msgs.action import MoveTo
import py_trees
import random
import threading
import time

class MoveToDummy(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        """
        """
        super(Foo, self).__init__(name=name)
        self.blackboard = self.attach_blackboard_client(name="Move Above Plate Blackboard")

    def setup(self):
        """
        """
        # TODO (amaln): Make these parameters of the class or setup function, not hardcoded.
        self.dummy_plan_time = 2.5 # secs
        self.dummy_motion_time = 7.5 # secs

    def plan(self, plan, success):
        """
        A dummy thread for planning to the target position. This thread
        will sleep for `self.dummy_plan_time` sec and then set the plan to None.

        Parameters
        ----------
        plan: A mutable object, which will contain the plan once the thread has
              finished. For this dummy thread, it contains None.
        success: A mutable object, which will contain the success status once
                 the thread has finished.
        """
        time.sleep(self.dummy_plan_time)
        plan.append(None)
        success[0] = True

    def move(self, plan, success):
        """
        A dummy thread for moving the robot arm along the plan. This thread
        will sleep for `self.dummy_motion_time` sec and then return success.

        Parameters
        ----------
        plan: Contains the plan.
        success: A mutable object, which will contain the success status once
                 the thread has finished.
        """
        time.sleep(self.dummy_motion_time)
        success[0] = True

    def initialise(self):
        """
        """
        # Start the planning thread
        self.plan = []
        self.plan_success = [False]
        self.planning_thread = threading.Thread(
            target=self.plan, args=(self.plan, self.plan_success), daemon=True
        )
        self.planning_thread.start()
        self.planning_start_time = self.get_clock().now()
        self.is_planning = True

        # Create (but don't yet start) the motion thread
        self.is_moving = False
        self.motion_success = [False]
        self.motion_thread = threading.Thread(
            target=self.move, args=(self.plan, self.motion_success), daemon=True
        )

    def update(self):
        """
        Checks the progress of the planning and motion threads, writes out 
        feedback to the blackboard, as well as the response. Feedback and 
        response must have the same structure as the expected ROS action messages.
        """

        # Check if the planning thread has finished
        if self.is_planning:
            if not self.planning_thread.is_alive():
                self.is_planning = False
                if self.plan_success[0]: # Plan succeeded
                    # Start the motion thread
                    self.motion_thread.start()
                    self.motion_start_time = self.get_clock().now()
                    self.is_moving = True
                    return py_trees.common.Status.RUNNING
                else: # Plan failed
                    # Abort the goal
                    result = MoveTo.Result()
                    result.status = result.STATUS_PLANNING_FAILED
                    return result

        # Check if the motion thread has finished
        if is_moving:
            if not motion_thread.is_alive():
                is_moving = False
                if motion_success[0]:
                    self.get_logger().info("Motion succeeded, returning")
                    # Succeed the goal
                    goal_handle.succeed()
                    result = self.action_class.Result()
                    result.status = result.STATUS_SUCCESS
                    self.active_goal_request = None  # Clear the active goal
                    return result
                else:
                    self.get_logger().info("Motion failed, aborting")
                    # Abort the goal
                    goal_handle.abort()
                    result = self.action_class.Result()
                    result.status = result.STATUS_MOTION_FAILED
                    self.active_goal_request = None  # Clear the active goal
                    return result

        # Send feedback
        feedback_msg.is_planning = is_planning
        if is_planning:
            feedback_msg.planning_time = (
                self.get_clock().now() - planning_start_time
            ).to_msg()
        elif is_moving:
            # TODO: In the actual (not dummy) implementation, this should
            # return the distance (not time) the robot has yet to move.
            feedback_msg.motion_initial_distance = self.dummy_motion_time
            elapsed_time = self.get_clock().now() - motion_start_time
            elapsed_time_float = elapsed_time.nanoseconds / 1.0e9
            feedback_msg.motion_curr_distance = (
                self.dummy_motion_time - elapsed_time_float
            )
        self.get_logger().info("Feedback: %s" % feedback_msg)
        goal_handle.publish_feedback(feedback_msg)


        self.logger.debug("  %s [Foo::update()]" % self.name)
        ready_to_make_a_decision = random.choice([True, False])
        decision = random.choice([True, False])
        if not ready_to_make_a_decision:
            return py_trees.common.Status.RUNNING
        elif decision:
            self.feedback_message = "We are not bar!"
            return py_trees.common.Status.SUCCESS
        else:
            self.feedback_message = "Uh oh"
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """
        When is this called?
           Whenever your behaviour switches to a non-running state.
            - SUCCESS || FAILURE : your behaviour's work cycle has finished
            - INVALID : a higher priority branch has interrupted, or shutting down
        """
        self.logger.debug("  %s [Foo::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))
