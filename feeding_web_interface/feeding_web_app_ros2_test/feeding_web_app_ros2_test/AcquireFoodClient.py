#!/usr/bin/env python3

# Standard imports
import pickle

# Third-party imports
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# Local imports
from ada_feeding_msgs.action import AcquireFood, MoveTo


class AcquireFoodClient(Node):
    def __init__(self):
        super().__init__("acquire_food_client")
        self._above_client = ActionClient(self, MoveTo, "/MoveAbovePlate")
        self._action_client = ActionClient(self, AcquireFood, "/AcquireFood")

        self.declare_parameter("request_path", rclpy.Parameter.Type.STRING)
        request_path = self.get_parameter("request_path")
        with open(request_path.value, "rb") as file:
            self.goal_msg = pickle.load(file)

    def move_above_plate(self):
        self._above_client.wait_for_server()
        return self._above_client.send_goal_async(
            MoveTo.Goal(), feedback_callback=self.feedback_callback
        )

    def acquire_food(self):
        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(
            self.goal_msg, feedback_callback=self.feedback_callback
        )

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: {feedback}")


def main(args=None):
    rclpy.init(args=args)

    action_client = AcquireFoodClient()

    # First Move Above Plate
    # Send Goal
    future = action_client.move_above_plate()
    rclpy.spin_until_future_complete(action_client, future)

    # Check Accept
    goal_handle = future.result()
    if not goal_handle.accepted:
        action_client.get_logger().error("MoveAbovePlate Rejected")
        return
    action_client.get_logger().info("MoveAbovePlate Accepted")

    # Get Result
    future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(action_client, future)
    result = future.result().result
    if result.status != result.STATUS_SUCCESS:
        action_client.get_logger().error("MoveAbovePlate Failed")
        return

    # Second, send the acquire food action
    # Send Goal
    future = action_client.acquire_food()
    rclpy.spin_until_future_complete(action_client, future)

    # Check Accept
    goal_handle = future.result()
    if not goal_handle.accepted:
        action_client.get_logger().error("AcquireFood Goal Rejected")
        return
    action_client.get_logger().info("AcquireFood Goal Accepted")

    # Get Result
    future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(action_client, future)

    # Print Result + Exit
    result = future.result().result
    action_client.get_logger().info(f"Result: {result}")


if __name__ == "__main__":
    main()
