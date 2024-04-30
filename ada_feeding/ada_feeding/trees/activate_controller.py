#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ActivateController behavior tree, which activates a
user-specified controller and deactivates all others.
"""

# Standard imports
import operator
from typing import List, Optional

# Third-party imports
from controller_manager_msgs.srv import SwitchController
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import retry_call_ros_service
from .trigger_tree import TriggerTree


class ActivateController(TriggerTree):
    """
    Thsi behavior tree calls the `~/switch_controller` service to activate a
    user-specified controller and deactivate all others.
    """

    def __init__(
        self,
        node: Node,
        controller_to_activate: str = "jaco_arm_cartesian_controller",
        all_controller_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the behavior tree.

        Parameters
        ----------
        node: The ROS node.
        controller_to_activate: The name of the controller to activate.
        all_controller_names: The names of all controllers. If None, the default
            controllers are "jaco_arm_cartesian_controller", "jaco_arm_controller",
            and "jaco_arm_servo_controller"
        """
        # Initialize the TriggerTree class
        super().__init__(node=node)
        self.controller_to_activate = controller_to_activate
        self.all_controller_names = all_controller_names
        if all_controller_names is None:
            self.all_controller_names = [
                "jaco_arm_cartesian_controller",
                "jaco_arm_controller",
                "jaco_arm_servo_controller",
            ]

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the behavior to switch controllers
        switch_controller_req = SwitchController.Request(
            activate_controllers=[self.controller_to_activate],
            deactivate_controllers=[
                controller
                for controller in self.all_controller_names
                if controller != self.controller_to_activate
            ],
            activate_asap=True,
            strictness=SwitchController.Request.BEST_EFFORT,
        )
        switch_controllers_key_response = Blackboard.separator.join(
            [name, "switch_controllers", "response"]
        )
        switch_controllers = retry_call_ros_service(
            name=name + "Activate Controller",
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

        # Put them together in a sequence
        # pylint: disable=duplicate-code
        return py_trees.trees.BehaviourTree(
            root=switch_controllers,
        )
