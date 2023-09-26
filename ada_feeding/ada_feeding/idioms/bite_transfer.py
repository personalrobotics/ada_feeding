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
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding.behaviors import (
    ModifyCollisionObject,
    ModifyCollisionObjectOperation,
    ToggleCollisionObject,
)
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


def get_add_in_front_of_wheelchair_wall_behavior(
    tree_name: str,
    behavior_name_prefix: str,
    collision_object_id: str,
    node: Node,
    blackboard: Blackboard,
):
    """
    Creates a behavior that adds a collision wall between the staging pose and the user,
    to prevent the robot from moving closer to the user.

    Parameters
    ----------
    tree_name: The name of the tree that this behaviour belongs to.
    behavior_name_prefix: The prefix for the name of the behaviour.
    collision_object_id: The ID of the collision object to add.
    node: The ROS2 node that this behaviour belongs to.

    Returns
    -------
    behavior: The behaviour that adds the collision wall.
    """
    # Create the behavior to add a collision wall between the staging pose and the user,
    # to prevent the robot from moving closer to the user.
    in_front_of_wheelchair_wall_prim_type = (
        1  # Box=1. See shape_msgs/SolidPrimitive.msg
    )
    in_front_of_wheelchair_wall_dims = [
        0.75,
        0.01,
        0.4,
    ]  # Box has 3 dims: [x, y, z]
    add_in_front_of_wheelchair_wall = ModifyCollisionObject(
        name=Blackboard.separator.join([tree_name, behavior_name_prefix]),
        node=node,
        operation=ModifyCollisionObjectOperation.ADD,
        collision_object_id=collision_object_id,
        collision_object_position_input_key="position",
        collision_object_orientation_input_key="quat_xyzw",
        prim_type=in_front_of_wheelchair_wall_prim_type,
        dims=in_front_of_wheelchair_wall_dims,
    )

    # Write the position, orientation, and frame_id to the blackboard
    position_key = Blackboard.separator.join([behavior_name_prefix, "position"])
    blackboard.register_key(
        key=position_key,
        access=py_trees.common.Access.WRITE,
    )
    blackboard.set(position_key, (0.37, 0.17, 0.85))
    quat_xyzw_key = Blackboard.separator.join([behavior_name_prefix, "quat_xyzw"])
    blackboard.register_key(
        key=quat_xyzw_key,
        access=py_trees.common.Access.WRITE,
    )
    blackboard.set(quat_xyzw_key, (0.0, 0.0, 0.0, 1.0))
    frame_id_key = Blackboard.separator.join([behavior_name_prefix, "frame_id"])
    blackboard.register_key(
        key=frame_id_key,
        access=py_trees.common.Access.WRITE,
    )
    blackboard.set(frame_id_key, "root")

    return add_in_front_of_wheelchair_wall


def get_remove_in_front_of_wheelchair_wall_behavior(
    tree_name: str,
    behavior_name_prefix: str,
    collision_object_id: str,
    node: Node,
):
    """
    Creates a behavior that removes the collision wall between the staging pose and the user.

    Parameters
    ----------
    tree_name: The name of the tree that this behaviour belongs to.
    behavior_name_prefix: The prefix for the name of the behaviour.
    node: The ROS2 node that this behaviour belongs to.

    Returns
    -------
    behavior: The behaviour that removes the collision wall.
    """
    # Create the behavior to remove the collision wall between the staging pose and the user.
    remove_in_front_of_wheelchair_wall = ModifyCollisionObject(
        name=Blackboard.separator.join([tree_name, behavior_name_prefix]),
        node=node,
        operation=ModifyCollisionObjectOperation.REMOVE,
        collision_object_id=collision_object_id,
    )

    return remove_in_front_of_wheelchair_wall
