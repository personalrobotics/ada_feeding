#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ft_thresh_satisifed idiom.
Which subscribes to the FT sensor and detects whether
a given FT threshold is satisfied at the point of execution.
"""

# Standard imports

# Third-part_y imports
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from geometry_msgs.msg import WrenchStamped

# Local imports


def ft_thresh_satisfied(
    name: str,
    f_mag: float = 4.0,
    f_x: float = 0.0,
    f_y: float = 0.0,
    f_z: float = 0.0,
    t_mag: float = 0.0,
    t_x: float = 0.0,
    t_y: float = 0.0,
    t_z: float = 0.0,
    ft_topic: str = "~/ft_topic",
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that subscribes to the FT sensor and checks whether threshold
    is immediately satisifed. FAILURE if not, SUCCESS if so. RUNNING while waiting
    for force/torque message.

    Parameters
    ----------
    name: The name to associate with this behavior.
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
    # pylint: disable=too-many-arguments

    def is_satisfied(wrench: WrenchStamped, _: WrenchStamped):
        """
        Inner function to test threshold. The many Ifs / Returns
        make it easiest to read.
        """
        # pylint: disable=too-many-arguments, too-many-return-statements, too-many-branches

        if f_x > 0.0:
            if wrench.force.x > f_x:
                return False
        if f_y > 0.0:
            if wrench.force.y > f_y:
                return False
        if f_z > 0.0:
            if wrench.force.z > f_z:
                return False
        if f_mag > 0.0:
            if (
                wrench.force.x * wrench.force.x
                + wrench.force.y * wrench.force.y
                + wrench.force.z * wrench.force.z
                > f_mag * f_mag
            ):
                return False
        if t_x > 0.0:
            if wrench.torque.x > t_x:
                return False
        if t_y > 0.0:
            if wrench.torque.y > t_y:
                return False
        if t_z > 0.0:
            if wrench.torque.z > t_z:
                return False
        if t_mag > 0.0:
            if (
                wrench.torque.x * wrench.torque.x
                + wrench.torque.y * wrench.torque.y
                + wrench.torque.z * wrench.torque.z
                > t_mag * t_mag
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
                qos_profile=(py_trees_ros.utilities.qos_profile_unlatched()),
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
