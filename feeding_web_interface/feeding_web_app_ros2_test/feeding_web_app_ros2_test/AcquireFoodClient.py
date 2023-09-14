#!/usr/bin/env python3

# Standard imports
import pickle

# Third-party imports
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# Local imports
from ada_feeding_msgs.action import AcquireFood


class AcquireFoodClient(Node):
    def __init__(self):
        super().__init__("acquire_food_client")
        self._action_client = ActionClient(self, AcquireFood, "/AcquireFood")

        self.declare_parameter("request_path", rclpy.Parameter.Type.STRING)
        request_path = self.get_parameter("request_path")
        with open(request_path.value, "rb") as file:
            self.goal_msg = pickle.load(file)

    def send_goal(self):
        self._action_client.wait_for_server()

        return self._action_client.send_goal_async(self.goal_msg)


def main(args=None):
    rclpy.init(args=args)

    action_client = AcquireFoodClient()

    # Send Goal
    future = action_client.send_goal()
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
