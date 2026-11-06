#!/usr/bin/env python3
# Copyright (c) 2024-2025, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from feeding_web_app_ros2_msgs.srv import ReverseString

import rclpy
from rclpy.node import Node


class MinimalService(Node):
    def __init__(self):
        super().__init__("reverse_string")
        self.srv = self.create_service(
            ReverseString, "reverse_string", self.reverse_string_callback
        )

    def reverse_string_callback(self, request, response):
        response.reversed = request.input[::-1]
        self.get_logger().info("Incoming request\ninput: %s" % (request.input))
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
