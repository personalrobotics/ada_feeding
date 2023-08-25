#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the pre_moveto_config idiom, which returns a behavior that
calls the ROS services that should be called before any MoveTo behavior.
Specifically, it does the following:
    1. Re-tare the force-torque sensor.
        a. Toggle the Watchdog Listener off. This is necessary because re-taring
           the force-torque sensor will briefly stop force-torque sensor readings,
           which would cause the watchdog to fail.
        b. Re-tare the force-torque sensor.
        c. Toggle the Watchdog Listener on.
    2. Set force-torque thresholds (so the ForceGateController will trip if
       the force-torque sensor readings exceed the thresholds).
"""

# Standard imports
import logging
import operator
from typing import Optional

# Third-part_y imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding.idioms import retry_call_ros_service


def pre_moveto_config(
    node: Node,
    name: str,
    re_tare: bool = True,
    f_mag: float = 0.0,
    f_x: float = 0.0,
    f_y: float = 0.0,
    f_z: float = 0.0,
    t_mag: float = 0.0,
    t_x: float = 0.0,
    t_y: float = 0.0,
    t_z: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that calls the ROS services that should be called before
    any MoveTo behavior. Specifically, it does the following, in a Sequence:
        1. Re-tare the force-torque sensor.
            a. Toggle the Watchdog Listener off. This is necessary because re-taring
               the force-torque sensor will briefly stop force-torque sensor readings,
               which would cause the watchdog to fail.
            b. Re-tare the force-torque sensor.
            c. Toggle the Watchdog Listener on.
        2. Set force-torque thresholds (so the ForceGateController will trip if
           the force-torque sensor readings exceed the thresholds).

    Parameters
    ----------
    node: The ROS node to associate all service calls with.
    name: The name to associate with this behavior.
    re_tare: Whether to re-tare the force-torque sensor.
    f_mag: The magnitude of the overall force threshold. No threshold if 0.0.
    f_x: The magnitude of the x component of the force threshold. No threshold if 0.0.
    f_y: The magnitude of the y component of the force threshold. No threshold if 0.0.
    f_z: The magnitude of the z component of the force threshold. No threshold if 0.0.
    t_mag: The magnitude of the overall torque threshold. No threshold if 0.0.
    t_x: The magnitude of the x component of the torque threshold. No threshold if 0.0.
    t_y: The magnitude of the y component of the torque threshold. No threshold if 0.0.
    t_z: The magnitude of the z component of the torque threshold. No threshold if 0.0.
    logger: The logger for the tree that this behavior is in.
    """
    # All the children of the sequence
    children = []

    # Separate the namespace of each sub-behavior
    turn_watchdog_listener_off_prefix = "turn_watchdog_listener_off"
    re_tare_ft_sensor_prefix = "re_tare_ft_sensor"
    turn_watchdog_listener_on_prefix = "turn_watchdog_listener_on"
    set_force_torque_thresholds_prefix = "set_force_torque_thresholds"

    if re_tare:
        # Toggle the Watchdog Listener off
        turn_watchdog_listener_off_name = Blackboard.separator.join(
            [name, turn_watchdog_listener_off_prefix]
        )
        turn_watchdog_listener_off_key_response = Blackboard.separator.join(
            [turn_watchdog_listener_off_name, "response"]
        )
        turn_watchdog_listener_off = retry_call_ros_service(
            name=turn_watchdog_listener_off_name,
            service_type=SetBool,
            service_name="~/toggle_watchdog_listener",
            key_request=None,
            request=SetBool.Request(data=False),
            key_response=turn_watchdog_listener_off_key_response,
            response_check=py_trees.common.ComparisonExpression(
                variable=turn_watchdog_listener_off_key_response + ".success",
                value=True,
                operator=operator.eq,
            ),
            logger=logger,
        )
        children.append(turn_watchdog_listener_off)

        # Re-tare the force-torque sensor
        re_tare_ft_sensor_name = Blackboard.separator.join(
            [name, re_tare_ft_sensor_prefix]
        )
        re_tare_ft_sensor_key_response = Blackboard.separator.join(
            [re_tare_ft_sensor_name, "response"]
        )
        re_tare_ft_sensor = retry_call_ros_service(
            name=re_tare_ft_sensor_name,
            service_type=SetBool,
            service_name="~/re_tare_ft",
            key_request=None,
            request=SetBool.Request(data=True),
            key_response=re_tare_ft_sensor_key_response,
            response_check=py_trees.common.ComparisonExpression(
                variable=re_tare_ft_sensor_key_response + ".success",
                value=True,
                operator=operator.eq,
            ),
            logger=logger,
        )
        children.append(re_tare_ft_sensor)

        # Toggle the Watchdog Listener on
        turn_watchdog_listener_on_name = Blackboard.separator.join(
            [name, turn_watchdog_listener_on_prefix]
        )
        turn_watchdog_listener_on_key_response = Blackboard.separator.join(
            [turn_watchdog_listener_on_name, "response"]
        )
        turn_watchdog_listener_on = retry_call_ros_service(
            name=turn_watchdog_listener_on_name,
            service_type=SetBool,
            service_name="~/toggle_watchdog_listener",
            key_request=None,
            request=SetBool.Request(data=True),
            key_response=turn_watchdog_listener_on_key_response,
            response_check=py_trees.common.ComparisonExpression(
                variable=turn_watchdog_listener_on_key_response + ".success",
                value=True,
                operator=operator.eq,
            ),
            logger=logger,
        )
        children.append(turn_watchdog_listener_on)

    # TODO: Set FT Thresholds

    # Link all the behaviours together in a sequence with memory
    root = py_trees.composites.Sequence(name=name, memory=True, children=children)
    root.logger = logger

    tree = py_trees.trees.BehaviourTree(root)
    return tree
