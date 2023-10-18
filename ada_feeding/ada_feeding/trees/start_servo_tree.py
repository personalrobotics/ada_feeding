#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the StartServoTree behavior tree, which turns the servo
controller on and starts MoveIt Servo.
"""

# Standard imports
import operator

# Third-party imports
from controller_manager_msgs.srv import SwitchController
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node
from std_srvs.srv import Trigger

# Local imports
from ada_feeding.idioms import retry_call_ros_service
from ada_feeding.trees import TriggerTree


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
        move_group_controller_name: str = "jaco_arm_controller",
    ) -> None:
        """
        Initializes the behavior tree.

        Parameters
        ----------
        node: The ROS node.
        servo_controller_name: The name of the servo controller.
        move_group_controller_name: The name of the move group controller.
        """
        # Initialize the TriggerTree class
        super().__init__(node=node)
        self.servo_controller_name = servo_controller_name
        self.move_group_controller_name = move_group_controller_name

    @override
    def create_tree(
        self,
        name: str,
        tree_root_name: str,  # DEPRECATED
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the behavior to switch controllers
        switch_controller_req = SwitchController.Request(
            activate_controllers=[self.servo_controller_name],
            deactivate_controllers=[self.move_group_controller_name],
            activate_asap=True,
        )
        switch_controllers_key_response = Blackboard.separator.join(
            [name, "switch_controllers", "response"]
        )
        switch_controllers = retry_call_ros_service(
            name=name + "Activate Servo Controller",
            service_type=SwitchController,
            service_name="~/switch_controller",
            key_request=None,
            request=switch_controller_req,
            key_response=switch_controllers_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=switch_controllers_key_response + ".ok",
                    value=True,
                    operator=operator.eq,
                )
            ],
        )

        # Create the behavior to start servo
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

        # Put them together in a sequence
        # pylint: disable=duplicate-code
        return py_trees.trees.BehaviourTree(
            root=py_trees.composites.Sequence(
                name=name,
                memory=True,
                children=[
                    switch_controllers,
                    start_servo,
                ],
            )
        )
