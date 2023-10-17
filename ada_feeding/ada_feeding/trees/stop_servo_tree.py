#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the StopServoTree behavior tree, which stops MoveIt Servo.
"""

# Standard imports
import operator

# Third-party imports
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node
from std_srvs.srv import Trigger

# Local imports
from ada_feeding.idioms import retry_call_ros_service
from ada_feeding.trees import TriggerTree


class StopServoTree(TriggerTree):
    """
    This behavior tree calls one ROS2 Services:
      1. `~/stop_servo` to stop MoveIt Servo
    """

    @override
    def create_tree(
        self,
        name: str,
        tree_root_name: str,  # DEPRECATED
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the behavior to stop servo
        stop_servo_key_response = Blackboard.separator.join(
            [name, "stop_servo", "response"]
        )
        stop_servo = retry_call_ros_service(
            name=name + "Stop Servo",
            service_type=Trigger,
            service_name="~/stop_servo",
            key_request=None,
            request=Trigger.Request(),
            key_response=stop_servo_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=stop_servo_key_response + ".success",
                    value=True,
                    operator=operator.eq,
                )
            ],
        )

        # Wrap the behavior in a behavior tree
        return py_trees.trees.BehaviourTree(root=stop_servo)
