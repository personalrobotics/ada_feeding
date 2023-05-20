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
        super(MoveToDummy, self).__init__(name=name)
        # self.blackboard = self.attach_blackboard_client(name=name)

    def setup(self):
        """
        """
        self.logger.debug("  %s [MoveToDummy::setup()]" % self.name)

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
        self.logger.debug("  %s [MoveToDummy::initialise()]" % self.name)

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
        self.logger.debug("  %s [MoveToDummy::update()]" % self.name)

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
                    return py_trees.common.Status.FAILURE

        # Check if the motion thread has finished
        if self.is_moving:
            if not self.motion_thread.is_alive():
                self.is_moving = False
                if self.motion_success[0]: # Motion succeeded
                    return py_trees.common.Status.SUCCESS
                else: # Motion failed
                    return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """
        When is this called?
           Whenever your behaviour switches to a non-running state.
            - SUCCESS || FAILURE : your behaviour's work cycle has finished
            - INVALID : a higher priority branch has interrupted, or shutting down
        """
        self.logger.debug("  %s [MoveToDummy::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))
