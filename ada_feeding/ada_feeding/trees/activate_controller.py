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
from std_srvs.srv import SetBool

# Local imports
from ada_feeding_msgs.action import ActivateController
from ada_feeding.idioms import retry_call_ros_service
from .trigger_tree import TriggerTree


class ActivateControllerTree(TriggerTree):
    """
    This behavior tree calls the `~/switch_controller` service to activate a
    user-specified controller and deactivate all others.
    """

    def __init__(
        self,
        node: Node,
        controller_to_activate: Optional[str] = "jaco_arm_cartesian_controller",
        all_controller_names: Optional[List[str]] = None,
        re_tare: bool = False,
    ) -> None:
        """
        Initializes the behavior tree.

        Parameters
        ----------
        node: The ROS node.
        controller_to_activate: The name of the controller to activate. If None,
            deactive all controllers without activating any.
        all_controller_names: The names of all controllers. If None, the default
            controllers are "jaco_arm_cartesian_controller", "jaco_arm_controller",
            and "jaco_arm_servo_controller"
        re_tare: If True, re-tare the force-torque sensor before activating the controller.
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
        self.re_tare = re_tare

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Write tree inputs to blackboard. Note that write access also gives
        # read access.
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(
            key="controller_to_activate", access=py_trees.common.Access.WRITE
        )
        blackboard.controller_to_activate = self.controller_to_activate
        blackboard.register_key(key="re_tare", access=py_trees.common.Access.WRITE)
        blackboard.re_tare = self.re_tare

        # Create the behavior to re-tare the force-torque sensor
        re_tare_ft_sensor_key_response = Blackboard.separator.join(
            [name, "re_tare_ft_sensor", "response"]
        )
        re_tare = py_trees.composites.Selector(
            name=name + "Re-tare Selector",
            memory=True,
            children=[
                # Check if re-tare is **not** requested
                py_trees.behaviours.CheckBlackboardVariableValue(
                    name=name + "Check Re-tare",
                    check=py_trees.common.ComparisonExpression(
                        variable=Blackboard.separator.join([name, "re_tare"]),
                        value=False,
                        operator=operator.eq,
                    ),
                ),
                # If the above is false (e.g., re-tare is requested), re-tare the sensor
                retry_call_ros_service(
                    name=name + "Re-tare F/T Sensor",
                    service_type=SetBool,
                    service_name="~/re_tare_ft",
                    key_request=None,
                    request=SetBool.Request(data=True),
                    key_response=re_tare_ft_sensor_key_response,
                    response_checks=[
                        py_trees.common.ComparisonExpression(
                            variable=re_tare_ft_sensor_key_response + ".success",
                            value=True,
                            operator=operator.eq,
                        )
                    ],
                ),
            ],
        )

        # Create the behavior to switch controllers
        def switch_controller_request() -> SwitchController.Request:
            return SwitchController.Request(
                activate_controllers=(
                    []  # Only possible via the __init__ parameter, not from the Action interface
                    if blackboard.controller_to_activate is None
                    else [blackboard.controller_to_activate]
                ),
                deactivate_controllers=[
                    controller
                    for controller in self.all_controller_names
                    if controller != blackboard.controller_to_activate
                ],
                activate_asap=True,
                strictness=SwitchController.Request.BEST_EFFORT,
            )

        switch_controllers_key_request = Blackboard.separator.join(
            [name, "switch_controllers", "request"]
        )
        switch_controllers_key_response = Blackboard.separator.join(
            [name, "switch_controllers", "response"]
        )
        switch_controllers = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                py_trees.behaviours.SetBlackboardVariable(
                    name=name + "Set SwitchController Request",
                    variable_name=switch_controllers_key_request,
                    variable_value=switch_controller_request,
                    overwrite=True,
                ),
                retry_call_ros_service(
                    name=name + "Activate Controller",
                    service_type=SwitchController,
                    service_name="~/switch_controller",
                    key_request=switch_controllers_key_request,
                    key_response=switch_controllers_key_response,
                    response_checks=[
                        py_trees.common.ComparisonExpression(
                            variable=switch_controllers_key_response + ".ok",
                            value=True,
                            operator=operator.eq,
                        )
                    ],
                ),
            ],
        )

        # Put them together in a sequence
        # pylint: disable=duplicate-code
        return py_trees.trees.BehaviourTree(
            root=py_trees.composites.Sequence(
                name=name,
                memory=True,
                children=[
                    re_tare,
                    switch_controllers,
                ],
            )
        )

    # Override goal to read arguments into local blackboard
    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree

        # If it is a ActivateController.Goal, override the blackboard
        if isinstance(goal, ActivateController.Goal):
            # Write tree inputs to blackboard
            name = tree.root.name
            blackboard = py_trees.blackboard.Client(name=name, namespace=name)
            blackboard.register_key(
                key="controller_to_activate", access=py_trees.common.Access.WRITE
            )
            blackboard.controller_to_activate = goal.controller_to_activate
            blackboard.register_key(key="re_tare", access=py_trees.common.Access.WRITE)
            blackboard.re_tare = goal.re_tare

        # Adds MoveToVisitor for Feedback
        return super().send_goal(tree, goal)
