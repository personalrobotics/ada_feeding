#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ft_thresh_satisifed idiom.
Which subscribes to the FT sensor and detects whether
a given FT threshold is satisfied at the point of execution.
"""

# Standard imports
from typing import Union

# Third-part_y imports
from geometry_msgs.msg import WrenchStamped
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
import rclpy

# Local imports
from ada_feeding.helpers import BlackboardKey


def ft_thresh_satisfied(
    name: str,
    ns: str,
    f_mag: Union[BlackboardKey, float] = 4.0,
    f_x: Union[BlackboardKey, float] = 0.0,
    f_y: Union[BlackboardKey, float] = 0.0,
    f_z: Union[BlackboardKey, float] = 0.0,
    t_mag: Union[BlackboardKey, float] = 0.0,
    t_x: Union[BlackboardKey, float] = 0.0,
    t_y: Union[BlackboardKey, float] = 0.0,
    t_z: Union[BlackboardKey, float] = 0.0,
    ft_topic: str = "~/ft_topic",
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that subscribes to the FT sensor and checks whether threshold
    is immediately satisifed. FAILURE if not, SUCCESS if so. RUNNING while waiting
    for force/torque message.

    Parameters
    ----------
    name: The name to associate with this behavior.
    ns: The namespace to use for the blackboard.
    ft_topic: FT topic to subscribe to
    f_mag: The magnitude of the overall force threshold. No threshold if 0.0.
    f_x: The magnitude of the x component of the force threshold. No threshold if 0.0.
    f_y: The magnitude of the y component of the force threshold. No threshold if 0.0.
    f_z: The magnitude of the z component of the force threshold. No threshold if 0.0.
    t_mag: The magnitude of the overall torque threshold. No threshold if 0.0.
    t_x: The magnitude of the x component of the torque threshold. No threshold if 0.0.
    t_y: The magnitude of the y component of the torque threshold. No threshold if 0.0.
    t_z: The magnitude of the z component of the torque threshold. No threshold if 0.0.
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # If any of the thresholds are BlackboardKeys, create a blackboard and give it permissions
    # to read that key.
    blackboard_keys = []
    for var in [f_mag, f_x, f_y, f_z, t_mag, t_x, t_y, t_z]:
        if isinstance(var, BlackboardKey):
            blackboard_keys.append(var)
    blackboard = None
    if len(blackboard_keys) > 0:
        blackboard = py_trees.blackboard.Client(name=name, namespace=ns)
        for key in blackboard_keys:
            key_base = key.split(".")[0]
            blackboard.register_key(key=key_base, access=py_trees.common.Access.READ)

    def is_satisfied(stamped: WrenchStamped, _: WrenchStamped):
        """
        Inner function to test threshold. The many Ifs / Returns
        make it easiest to read.
        """
        # pylint: disable=too-many-arguments, too-many-return-statements, too-many-branches

        local_f_x = f_x
        if isinstance(f_x, BlackboardKey):
            local_f_x = blackboard.get(f_x)
        local_f_y = f_y
        if isinstance(f_y, BlackboardKey):
            local_f_y = blackboard.get(f_y)
        local_f_z = f_z
        if isinstance(f_z, BlackboardKey):
            local_f_z = blackboard.get(f_z)
        local_f_mag = f_mag
        if isinstance(f_mag, BlackboardKey):
            local_f_mag = blackboard.get(f_mag)
        local_t_x = t_x
        if isinstance(t_x, BlackboardKey):
            local_t_x = blackboard.get(t_x)
        local_t_y = t_y
        if isinstance(t_y, BlackboardKey):
            local_t_y = blackboard.get(t_y)
        local_t_z = t_z
        if isinstance(t_z, BlackboardKey):
            local_t_z = blackboard.get(t_z)
        local_t_mag = t_mag
        if isinstance(t_mag, BlackboardKey):
            local_t_mag = blackboard.get(t_mag)

        wrench = stamped.wrench

        if local_f_x > 0.0:
            if abs(wrench.force.x) > local_f_x:
                return False
        if local_f_y > 0.0:
            if abs(wrench.force.y) > local_f_y:
                return False
        if local_f_z > 0.0:
            if abs(wrench.force.z) > local_f_z:
                return False
        if local_f_mag > 0.0:
            if (
                wrench.force.x * wrench.force.x
                + wrench.force.y * wrench.force.y
                + wrench.force.z * wrench.force.z
                > local_f_mag * local_f_mag
            ):
                return False
        if local_t_x > 0.0:
            if abs(wrench.torque.x) > local_t_x:
                return False
        if local_t_y > 0.0:
            if abs(wrench.torque.y) > local_t_y:
                return False
        if local_t_z > 0.0:
            if abs(wrench.torque.z) > local_t_z:
                return False
        if local_t_mag > 0.0:
            if (
                wrench.torque.x * wrench.torque.x
                + wrench.torque.y * wrench.torque.y
                + wrench.torque.z * wrench.torque.z
                > local_t_mag * local_t_mag
            ):
                return False
        return True

    ft_absolute_key = Blackboard.separator.join([name, "ft_wrench"])

    return py_trees.composites.Sequence(
        name=name,
        memory=True,
        children=[
            py_trees_ros.subscribers.ToBlackboard(
                name=name + " FTSubscriber",
                topic_name=ft_topic,
                topic_type=WrenchStamped,
                qos_profile=rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
                blackboard_variables={
                    ft_absolute_key: None,
                },
                initialise_variables={
                    ft_absolute_key: WrenchStamped(),
                },
            ),
            py_trees.behaviours.CheckBlackboardVariableValue(
                name=name + " CheckFTThresholdSatisfied",
                check=py_trees.common.ComparisonExpression(
                    variable=ft_absolute_key,
                    value=WrenchStamped(),
                    operator=is_satisfied,
                ),
            ),
        ],
    )
