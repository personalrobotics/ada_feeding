#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the StopServoTree behavior tree, which stops MoveIt Servo.
"""

# Standard imports
import operator

# Third-party imports
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist, TwistStamped
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Header
from std_srvs.srv import Trigger

# Local imports
from ada_feeding.behaviors.ros import UpdateTimestamp
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import retry_call_ros_service, wait_for_secs
from ada_feeding.trees import TriggerTree


class StopServoTree(TriggerTree):
    """
    This behavior tree does the following:
      1. Pubishes one 0-velocity twist message to `~/servo_twist_cmds`.
      2. Waits for `delay` seconds, to allow servo to publish stop commands
         to the controller.
      3. Calls the `~/stop_servo` service to stop MoveIt Servo.
      4. Calls the `~/switch_controller` service to turn off the servo controller.
    """

    def __init__(
        self,
        node: Node,
        base_frame_id: str = "j2n6s200_link_base",
        servo_controller_name: str = "jaco_arm_servo_controller",
        delay: float = 0.5,
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
        self.base_frame_id = base_frame_id
        self.servo_controller_name = servo_controller_name
        self.delay = delay

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Create the blackboard for this tree class
        blackboard = py_trees.blackboard.Client(name=name + " Tree")

        # Write a 0-velocity TwistStamped message to the blackboard
        twist_key = Blackboard.separator.join([name, "twist"])
        blackboard.register_key(
            key=twist_key,
            access=py_trees.common.Access.WRITE,
        )
        blackboard.set(
            twist_key,
            TwistStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=self.base_frame_id,
                ),
                twist=Twist(),
            ),
        )

        # Update the timestamp of the twist message so its not stale
        update_timestamp = UpdateTimestamp(
            name=name + "Update Timestamp",
            inputs={
                "stamped_msg": BlackboardKey(twist_key),
            },
            outputs={
                "stamped_msg": BlackboardKey(twist_key),
            },
        )

        # Create the behavior to publish the twist message
        twist_pub = py_trees_ros.publishers.FromBlackboard(
            name=name + "Publish Twist",
            topic_name="~/servo_twist_cmds",
            topic_type=TwistStamped,
            qos_profile=QoSProfile(depth=1),
            blackboard_variable=twist_key,
        )

        # Add a delay to allow servo to publish stop commands
        delay_behavior = wait_for_secs(name + "Delay", self.delay)

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

        # Create the behavior to turn off the controllers
        stop_controller_req = SwitchController.Request(
            activate_controllers=[],
            deactivate_controllers=[self.servo_controller_name],
            activate_asap=True,
            strictness=SwitchController.Request.BEST_EFFORT,
        )
        stop_controllers_key_response = Blackboard.separator.join(
            [name, "switch_controllers", "response"]
        )
        stop_controllers = retry_call_ros_service(
            name=name + "Activate Servo Controller",
            service_type=SwitchController,
            service_name="~/switch_controller",
            key_request=None,
            request=stop_controller_req,
            key_response=stop_controllers_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=stop_controllers_key_response + ".ok",
                    value=True,
                    operator=operator.eq,
                )
            ],
        )

        # Put them together in a sequence with memory
        # pylint: disable=duplicate-code
        return py_trees.trees.BehaviourTree(
            root=py_trees.composites.Sequence(
                name=name,
                memory=True,
                children=[
                    update_timestamp,
                    twist_pub,
                    delay_behavior,
                    stop_servo,
                    stop_controllers,
                ],
            )
        )
