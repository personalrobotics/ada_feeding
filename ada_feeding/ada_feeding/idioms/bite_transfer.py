#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains helper functions to generate the behaviors used in bite
transfer, e.g., the MoveToMouth and MoveFromMouth behaviors.
"""

# Standard imports
import logging
import operator
from typing import List

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding.behaviors import ToggleCollisionObject
from .retry_call_ros_service import retry_call_ros_service


def get_toggle_collision_object_behavior(
    tree_name: str,
    behavior_name_prefix: str,
    node: Node,
    collision_object_ids: List[str],
    allow: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles a collision object.

    Parameters
    ----------
    tree_name: The name of the tree that this behaviour belongs to.
    behavior_name_prefix: The prefix for the name of the behaviour.
    node: The ROS2 node that this behaviour belongs to.
    collision_object_ids: The IDs of the collision object to toggle.
    allow: Whether to allow or disallow the collision object.

    Returns
    -------
    behavior: The behaviour that toggles the collision object.
    """

    # pylint: disable=too-many-arguments
    # This is acceptable, as these are the parameters necessary to create
    # the behaviour.

    toggle_collision_object = ToggleCollisionObject(
        name=Blackboard.separator.join([tree_name, behavior_name_prefix]),
        node=node,
        collision_object_ids=collision_object_ids,
        allow=allow,
    )
    return toggle_collision_object


def get_toggle_face_detection_behavior(
    tree_name: str,
    behavior_name_prefix: str,
    request_data: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles face detection.

    Parameters
    ----------
    tree_name: The name of the tree that this behaviour belongs to.
    behavior_name_prefix: The prefix for the name of the behaviour.
    request_data: The request data to send to the toggle_face_detection service.
        True to turn it on, False otherwise.

    Returns
    -------
    behavior: The behaviour that toggles face detection.
    """
    behavior_name = Blackboard.separator.join([tree_name, behavior_name_prefix])
    key_response = Blackboard.separator.join([behavior_name, "response"])
    toggle_face_detection_behavior = retry_call_ros_service(
        name=behavior_name,
        service_type=SetBool,
        service_name="~/toggle_face_detection",
        key_request=None,
        request=SetBool.Request(data=request_data),
        key_response=key_response,
        response_checks=[
            py_trees.common.ComparisonExpression(
                variable=key_response + ".success",
                value=True,
                operator=operator.eq,
            )
        ],
    )
    return toggle_face_detection_behavior


def get_toggle_watchdog_listener_behavior(
    tree_name: str,
    behavior_name_prefix: str,
    request_data: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles the watchdog listener.

    Parameters
    ----------
    tree_name: The name of the tree that this behaviour belongs to.
    behavior_name_prefix: The prefix for the name of the behaviour.
    request_data: The request data to send to the toggle_watchdog_listener service.
        True to turn it on, False otherwise.

    Returns
    -------
    behavior: The behaviour that toggles the watchdog listener.
    """
    toggle_watchdog_listener_name = Blackboard.separator.join(
        [tree_name, behavior_name_prefix]
    )
    toggle_watchdog_listener_key_response = Blackboard.separator.join(
        [toggle_watchdog_listener_name, "response"]
    )
    toggle_watchdog_listener = retry_call_ros_service(
        name=toggle_watchdog_listener_name,
        service_type=SetBool,
        service_name="~/toggle_watchdog_listener",
        key_request=None,
        request=SetBool.Request(data=request_data),
        key_response=toggle_watchdog_listener_key_response,
        response_checks=[
            py_trees.common.ComparisonExpression(
                variable=toggle_watchdog_listener_key_response + ".success",
                value=True,
                operator=operator.eq,
            )
        ],
    )
    return toggle_watchdog_listener
