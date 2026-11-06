#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

import inspect
from feeding_web_app_ros2_msgs.action import SortByCharacterFrequency


class MinimalAction(Node):
    def __init__(self, sleep_time=0.1):
        super().__init__("sort_by_character_frequency")
        self.sleep_time = sleep_time
        self._action_server = ActionServer(
            self,
            SortByCharacterFrequency,
            "sort_by_character_frequency",
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        # This server allows multiple goals in parallel
        self.get_logger().info("Received goal request")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info(
            "Executing goal...%s | %s" % (goal_handle, repr(goal_handle.request.input))
        )
        input = goal_handle.request.input

        feedback_msg = SortByCharacterFrequency.Feedback()

        character_frequency = {}
        for i in range(len(input)):
            # Check if there is a cancel request
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal canceled")
                goal_handle.canceled()
                return SortByCharacterFrequency.Result()

            # Update the character frequency
            character = input[i]
            if character not in character_frequency:
                character_frequency[character] = 0
            character_frequency[character] += 1

            # Update our progress
            feedback_msg.progress = float(i) / float(len(input))
            self.get_logger().info("Feedback: %f" % feedback_msg.progress)
            goal_handle.publish_feedback(feedback_msg)

            # Sleep (to simulate the delay that a robot might have during a
            # ROS action)
            time.sleep(self.sleep_time)

        goal_handle.succeed()

        result = SortByCharacterFrequency.Result()
        uniqueLetters = "".join(character_frequency.keys())
        result.result = "".join(
            sorted(uniqueLetters, key=lambda x: character_frequency[x], reverse=True)
        )
        return result


def main(args=None):
    rclpy.init(args=args)

    minimal_action = MinimalAction()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(minimal_action, executor=executor)


if __name__ == "__main__":
    main()
