#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains helper functions to generate the behaviors used in bite
transfer, e.g., the MoveToMouth and MoveFromMouth behaviors.
"""

# Standard imports
import operator
from typing import List

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from std_srvs.srv import SetBool

# Local imports
from ada_feeding.behaviors.moveit2 import (
    ModifyCollisionObject,
    ModifyCollisionObjectOperation,
    ToggleCollisionObject,
)
from .retry_call_ros_service import retry_call_ros_service


def get_toggle_collision_object_behavior(
    name: str,
    collision_object_ids: List[str],
    allow: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles a collision object.

    Parameters
    ----------
    name: The name of the behaviour.
    collision_object_ids: The IDs of the collision object to toggle.
    allow: Whether to allow or disallow the collision object.
    ns: The namespace of the toggle_collision_object service.

    Returns
    -------
    behavior: The behaviour that toggles the collision object.
    """

    toggle_collision_object = ToggleCollisionObject(
        name=name,
        inputs={
            "collision_object_ids": collision_object_ids,
            "allow": allow,
        },
    )
    return toggle_collision_object


def get_toggle_face_detection_behavior(
    name: str,
    request_data: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles face detection.

    Parameters
    ----------
    name: The name of the behaviour.
    request_data: The request data to send to the toggle_face_detection service.
        True to turn it on, False otherwise.

    Returns
    -------
    behavior: The behaviour that toggles face detection.
    """
    key_response = Blackboard.separator.join([name, "response"])
    toggle_face_detection_behavior = retry_call_ros_service(
        name=name,
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
    name: str,
    request_data: bool,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behaviour that toggles the watchdog listener.

    Parameters
    ----------
    name: The name of the behaviour.
    request_data: The request data to send to the toggle_watchdog_listener service.
        True to turn it on, False otherwise.

    Returns
    -------
    behavior: The behaviour that toggles the watchdog listener.
    """
    toggle_watchdog_listener_key_response = Blackboard.separator.join(
        [name, "response"]
    )
    toggle_watchdog_listener = retry_call_ros_service(
        name=name,
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


def get_add_in_front_of_wheelchair_wall_behavior(
    name: str,
    collision_object_id: str,
):
    """
    Creates a behavior that adds a collision wall between the staging pose and the user,
    to prevent the robot from moving closer to the user.

    Parameters
    ----------
    name: The name of the behaviour.
    collision_object_id: The ID of the collision object to add.

    Returns
    -------
    behavior: The behaviour that adds the collision wall.
    """
    # Create the behavior to add a collision wall between the staging pose and the user,
    # to prevent the robot from moving closer to the user.
    return ModifyCollisionObject(
        name=name,
        inputs={
            "operation": ModifyCollisionObjectOperation.ADD,
            "collision_object_id": collision_object_id,
            "collision_object_position": (0.37, 0.17, 0.85),
            "collision_object_orientation": (0.0, 0.0, 0.0, 1.0),
            "prim_type": 1,  # Box=1. See shape_msgs/SolidPrimitive.msg
            "dims": [
                0.75,
                0.01,
                0.4,
            ],  # Box has 3 dims: [x, y, z]
            "frame_id": "root",
        },
    )


def get_remove_in_front_of_wheelchair_wall_behavior(
    name: str,
    collision_object_id: str,
):
    """
    Creates a behavior that removes the collision wall between the staging pose and the user.

    Parameters
    ----------
    name: The name of the behaviour.

    Returns
    -------
    behavior: The behaviour that removes the collision wall.
    """
    # Create the behavior to remove the collision wall between the staging pose and the user.
    remove_in_front_of_wheelchair_wall = ModifyCollisionObject(
        name=name,
        inputs={
            "operation": ModifyCollisionObjectOperation.REMOVE,
            "collision_object_id": collision_object_id,
        },
    )

    return remove_in_front_of_wheelchair_wall
