#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a node that launches a 2 ROS2 services.
This service implement AcquisitionSelect and AcquisitionReport.
"""

from ada_feeding_msgs.srv import AcquisitionSelect, AcquisitionReport

import rclpy
from rclpy.node import Node


class PolicyServices(Node):

    def __init__(self):
    	"""
    	Declare ROS2 Parameters
    	"""
        super().__init__('policy_service')

    def init_services() -> bool:
    	"""
    	Verify all parameters, load the specific requested policy
    	"""
        self.select_srv = self.create_service(AcquisitionSelect, '~/action_select', self.select_callback)
        self.report_srv = self.create_service(AcquisitionReport, '~/action_report', self.report_callback)

    def select_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response


def main():
    rclpy.init()

    node = PolicyServices()

	if node.init_services():
    	rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()