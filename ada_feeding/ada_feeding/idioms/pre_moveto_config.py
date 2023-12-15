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
import operator

# Third-part_y imports
import py_trees
from py_trees.blackboard import Blackboard
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import SetBool

# Local imports
from .bite_transfer import get_toggle_watchdog_listener_behavior
from .retry_call_ros_service import retry_call_ros_service


def set_parameter_response_all_success(
    blackboard_value: SetParameters.Response,
    _: SetParameters.Response,
) -> bool:
    """
    Checks that all the parameters were successfully set.
    """
    return all(result.successful for result in blackboard_value.results)


def create_ft_thresh_request(
    f_mag: float = 0.0,
    f_x: float = 0.0,
    f_y: float = 0.0,
    f_z: float = 0.0,
    t_mag: float = 0.0,
    t_x: float = 0.0,
    t_y: float = 0.0,
    t_z: float = 0.0,
) -> SetParameters.Request:
    """
    Create a SetParameters request from requested force/torque thresholds

    Parameters
    ----------
    f_mag: Max magnitude of the force on utensil
    t_mag: Max magnitude of torque on utensil

    Returns
    -------
    SetParameters ROS2 service request object
    designed to operate with `forque_sensor_hardware`
    """

    # pylint: disable=too-many-arguments, too-many-function-args

    parameters = []
    for key, val in [
        ("fMag", f_mag),
        ("fx", f_x),
        ("fy", f_y),
        ("fz", f_z),
        ("tMag", t_mag),
        ("tx", t_x),
        ("ty", t_y),
        ("tz", t_z),
    ]:
        parameters.append(
            Parameter(
                name=f"wrench_threshold.{key}",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE, double_value=val
                ),
            )
        )
    return SetParameters.Request(parameters=parameters)


def pre_moveto_config(
    name: str,
    re_tare: bool = True,
    toggle_watchdog_listener: bool = False,
    set_ft_thresholds: bool = True,
    f_mag: float = 4.0,
    f_x: float = 0.0,
    f_y: float = 0.0,
    f_z: float = 0.0,
    t_mag: float = 0.0,
    t_x: float = 0.0,
    t_y: float = 0.0,
    t_z: float = 0.0,
    param_service_name: str = "~/set_force_gate_controller_parameters",
) -> py_trees.behaviour.Behaviour:
    """
    Returns a behavior that calls the ROS services that should be called before
    any MoveTo behavior. Specifically, it does the following, in a Sequence:
        1. Re-tare the force-torque sensor.
            a. Toggle the Watchdog Listener off.
            b. Re-tare the force-torque sensor.
            c. Toggle the Watchdog Listener on.
        2. Set force-torque thresholds (so the ForceGateController will trip if
           the force-torque sensor readings exceed the thresholds).

    Parameters
    ----------
    name: The name to associate with this behavior.
    re_tare: Whether to re-tare the force-torque sensor.
    toggle_watchdog_listener: Whether to toggle the watchdog listener on and off
        during re-taring.
    set_ft_thresholds: Whether to set the force-torque thresholds.
    f_mag: The magnitude of the overall force threshold. No threshold if 0.0.
    f_x: The magnitude of the x component of the force threshold. No threshold if 0.0.
    f_y: The magnitude of the y component of the force threshold. No threshold if 0.0.
    f_z: The magnitude of the z component of the force threshold. No threshold if 0.0.
    t_mag: The magnitude of the overall torque threshold. No threshold if 0.0.
    t_x: The magnitude of the x component of the torque threshold. No threshold if 0.0.
    t_y: The magnitude of the y component of the torque threshold. No threshold if 0.0.
    t_z: The magnitude of the z component of the torque threshold. No threshold if 0.0.
    """

    # pylint: disable=too-many-arguments, too-many-locals, too-many-function-args
    # Idioms tend to be hefty, in order to prevent other functions from being hefty.

    # All the children of the sequence
    children = []

    # Separate the namespace of each sub-behavior
    turn_watchdog_listener_off_prefix = "turn_watchdog_listener_off"
    re_tare_ft_sensor_prefix = "re_tare_ft_sensor"
    turn_watchdog_listener_on_prefix = "turn_watchdog_listener_on"
    set_force_torque_thresholds_prefix = "set_force_torque_thresholds"

    if re_tare:
        # Toggle the Watchdog Listener off
        if toggle_watchdog_listener:
            turn_watchdog_listener_off = get_toggle_watchdog_listener_behavior(
                name,
                turn_watchdog_listener_off_prefix,
                False,
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
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=re_tare_ft_sensor_key_response + ".success",
                    value=True,
                    operator=operator.eq,
                )
            ],
        )
        children.append(re_tare_ft_sensor)

        # Toggle the Watchdog Listener on
        if toggle_watchdog_listener:
            turn_watchdog_listener_on = get_toggle_watchdog_listener_behavior(
                name,
                turn_watchdog_listener_on_prefix,
                True,
            )
            children.append(turn_watchdog_listener_on)

    # Set FT Thresholds
    if set_ft_thresholds:
        ft_threshold_request = create_ft_thresh_request(
            f_mag, f_x, f_y, f_z, t_mag, t_x, t_y, t_z
        )
        set_force_torque_thresholds_name = Blackboard.separator.join(
            [name, set_force_torque_thresholds_prefix]
        )
        set_force_torque_thresholds_key_response = Blackboard.separator.join(
            [set_force_torque_thresholds_name, "response"]
        )
        set_force_torque_thresholds = retry_call_ros_service(
            name=set_force_torque_thresholds_name,
            service_type=SetParameters,
            service_name=param_service_name,
            key_request=None,
            request=ft_threshold_request,
            key_response=set_force_torque_thresholds_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=set_force_torque_thresholds_key_response,
                    value=SetParameters.Response(),  # Unused
                    operator=set_parameter_response_all_success,
                )
            ],
        )
        children.append(set_force_torque_thresholds)

    # Link all the behaviours together in a sequence with memory
    root = py_trees.composites.Sequence(name=name, memory=True, children=children)

    return root
