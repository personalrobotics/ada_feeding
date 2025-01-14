#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from ada_feeding_msgs.action import AcquireFood
from feeding_web_app_ros2_test.MoveToDummy import MoveToDummy
import rclpy
from rclpy.executors import MultiThreadedExecutor


def main(args=None):
    rclpy.init(args=args)

    acquire_food = MoveToDummy("AcquireFood", AcquireFood)

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    rclpy.spin(acquire_food, executor=executor)


if __name__ == "__main__":
    main()
