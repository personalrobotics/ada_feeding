#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the StartServoTree behavior tree, which turns the servo
controller on and starts MoveIt Servo.
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
from .activate_controller import ActivateControllerTree
from .trigger_tree import TriggerTree


class StartServoTree(TriggerTree):
    """
    This behavior tree calls two ROS2 Services:
      1. `~/switch_controller` to turn the servo controller on
      2. `~/start_servo` to start MoveIt Servo
    """

    def __init__(
        self,
        node: Node,
        servo_controller_name: str = "jaco_arm_servo_controller",
        start_moveit_servo: bool = True,
    ) -> None:
        """
        Initializes the behavior tree.

        Parameters
        ----------
        node: The ROS node.
        servo_controller_name: The name of the servo controller.
        start_moveit_servo: Whether to start MoveIt Servo.
        """
        # Initialize the TriggerTree class
        super().__init__(node=node)
        self.servo_controller_name = servo_controller_name
        self.start_moveit_servo = start_moveit_servo

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the behavior to switch controllers
        switch_controllers = (
            ActivateControllerTree(
                self._node,
                controller_to_activate=self.servo_controller_name,
            )
            .create_tree(name=name + "Activate Servo Controller")
            .root
        )
        children = [switch_controllers]

        # Create the behavior to start servo
        if self.start_moveit_servo:
            start_servo_key_response = Blackboard.separator.join(
                [name, "start_servo", "response"]
            )
            start_servo = retry_call_ros_service(
                name=name + "Start Servo",
                service_type=Trigger,
                service_name="~/start_servo",
                key_request=None,
                request=Trigger.Request(),
                key_response=start_servo_key_response,
                response_checks=[
                    py_trees.common.ComparisonExpression(
                        variable=start_servo_key_response + ".success",
                        value=True,
                        operator=operator.eq,
                    )
                ],
            )
            children.append(start_servo)

        # Put them together in a sequence
        # pylint: disable=duplicate-code
        return py_trees.trees.BehaviourTree(
            root=py_trees.composites.Sequence(
                name=name,
                memory=True,
                children=children,
            )
        )
