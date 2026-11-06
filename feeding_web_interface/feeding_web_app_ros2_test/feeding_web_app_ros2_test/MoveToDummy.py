# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
import threading
import time


class MoveToDummy(Node):
    def __init__(
        self,
        name,
        action_class,
        send_feedback_hz=10,
        dummy_plan_time=2.5,
        dummy_motion_time=7.5,
    ):
        """
        Initialize the MoveToDummy action node.

        Parameters
        ----------
        name: The name of the action server.
        action_class: The action class to use for the action server.
        send_feedback_hz: The target frequency at which to send feedback.
        dummy_plan_time: How many seconds this dummy node should spend in planning.
        dummy_motion_time: How many seconds this dummy node should spend in motion.
        """
        super().__init__(name)

        self.send_feedback_hz = send_feedback_hz
        self.dummy_plan_time = dummy_plan_time
        self.dummy_motion_time = dummy_motion_time
        self.action_class = action_class

        self.active_goal_request = None

        self._action_server = ActionServer(
            self,
            action_class,
            name,
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def goal_callback(self, goal_request):
        """
        Accept a goal if this action does not already have an active goal,
        else reject.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info("Received goal request")
        if self.active_goal_request is None:
            self.get_logger().info("Accepting goal request")
            self.active_goal_request = goal_request
            return GoalResponse.ACCEPT
        self.get_logger().info("Rejecting goal request")
        return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """
        Always accept client requests to cancel the active goal. Note that this
        function should not actually implement the cancel; that is handled in
        `execute_callback`

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

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

    async def execute_callback(self, goal_handle):
        """
        First, plan to the target position. Then, move to that position.
        As a "dummy node," this specific node will spend `self.dummy_plan_time`
        sec in planning and `self.dummy_motion_time` sec in motion.

        NOTE: In the actual (not dummy) implementation, we will be calling a
        ROS action defined by MoveIt to do both planning and execution. Because
        actions are non-blocking, we won't need to break off separate threads.
        Using separate threads is an artiface of using time.sleep() in this
        dummy implementation.
        """
        self.get_logger().info("Executing goal...%s" % (goal_handle.request,))

        # Load the feedback parameters
        feedback_rate = self.create_rate(self.send_feedback_hz)
        feedback_msg = self.action_class.Feedback()

        # Start the planning thread
        plan = []
        plan_success = [False]
        planning_thread = threading.Thread(
            target=self.plan, args=(plan, plan_success), daemon=True
        )
        planning_thread.start()
        planning_start_time = self.get_clock().now()
        is_planning = True

        # Create (but don't yet start) the motion thread
        is_moving = False
        motion_success = [False]
        motion_thread = threading.Thread(
            target=self.move, args=(plan, motion_success), daemon=True
        )

        # Monitor the planning and motion threads, and send feedback
        while rclpy.ok() and (is_planning or is_moving):
            # Check if there is a cancel request
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                result = self.action_class.Result()
                result.status = result.STATUS_CANCELED
                self.active_goal_request = None  # Clear the active goal
                return result

            # Check if the planning thread has finished
            if is_planning:
                if not planning_thread.is_alive():
                    is_planning = False
                    if plan_success[0]:  # Plan succeeded
                        self.get_logger().info(
                            "Planning succeeded, proceeding to motion"
                        )
                        # Start the motion thread
                        motion_thread.start()
                        motion_start_time = self.get_clock().now()
                        is_moving = True
                        continue
                    else:  # Plan failed
                        self.get_logger().info("Planning failed, aborting")
                        # Abort the goal
                        goal_handle.abort()
                        result = self.action_class.Result()
                        result.status = result.STATUS_PLANNING_FAILED
                        self.active_goal_request = None  # Clear the active goal
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
                feedback_msg.motion_time = elapsed_time.to_msg()
                elapsed_time_float = elapsed_time.nanoseconds / 1.0e9
                feedback_msg.motion_curr_distance = (
                    self.dummy_motion_time - elapsed_time_float
                )
            self.get_logger().info("Feedback: %s" % feedback_msg)
            goal_handle.publish_feedback(feedback_msg)

            # Sleep for the specified feedback rate
            feedback_rate.sleep()

        # If we get here, something went wrong
        self.get_logger().info("Unknown error, aborting")
        goal_handle.abort()
        result = self.action_class.Result()
        result.status = result.STATUS_UNKNOWN
        self.active_goal_request = None  # Clear the active goal
        return result
