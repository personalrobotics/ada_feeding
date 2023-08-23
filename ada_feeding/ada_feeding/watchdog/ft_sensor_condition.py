#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains FTSensorCondition, a watchdog condition that accumulates all
force-torque sensor readings and checks that the sensor is still publishing and
its data is not zero-variance.
"""

# Standard imports
from threading import Lock
from typing import List, Tuple

# Third party imports
from geometry_msgs.msg import WrenchStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

# Local imports
from ada_feeding.watchdog import WatchdogCondition


class FTSensorCondition(WatchdogCondition):
    """
    The FTSensorCondition class accumulates all force-torque sensor readings and
    checks that the sensor is still publishing and its data is not zero-variance.
    """

    def __init__(self, node: Node) -> None:
        """
        Initialize the FTSensorCondition class.

        Parameters
        ----------
        node: Node
            The ROS2 node that this watchdog condition belongs to.
        """
        # Store the node
        self._node = node

        # Initialize the parameter for how long the watchdog can go without
        # receiving at least two force-torque sensor data
        ft_timeout_sec = self._node.declare_parameter(
            "ft_timeout_sec",
            0.5,
            ParameterDescriptor(
                name="ft_timeout_sec",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The number of seconds within which the force-torque"
                    "sensor must have: (a) published messages; and (b) had"
                    "them be nonzero-variance"
                ),
                read_only=True,
            ),
        )
        self.ft_timeout = Duration(seconds=ft_timeout_sec.value)

        # For each dimension of a single force-torque datapoint, store the most
        # recent unique value and the time at which that value was received.
        self.last_unique_values = None
        self.last_unique_values_timestamp = None
        self.last_unique_values_lock = Lock()

        # Subscribe to the FT sensor topic
        self.ft_sensor_topic = self._node.resolve_topic_name("~/ft_topic")
        self.ft_sensor_subscription = self._node.create_subscription(
            WrenchStamped,
            self.ft_sensor_topic,
            self.__ft_sensor_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
        )

    def __ft_sensor_callback(self, msg: WrenchStamped) -> None:
        """
        Callback function for the force-torque sensor topic.

        Parameters
        ----------
        msg: WrenchStamped
            The message received from the force-torque sensor topic.
        """
        # Convert the message to a numpy array
        ft_data = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )
        ft_time = Time.from_msg(msg.header.stamp)

        # Get the last unique values and the last unique values timestamp
        with self.last_unique_values_lock:
            last_unique_values = self.last_unique_values
            last_unique_values_timestamp = self.last_unique_values_timestamp

        # If this is the first message received, initialize the last unique values
        # and the last unique values timestamp
        if last_unique_values is None:
            last_unique_values = ft_data
            last_unique_values_timestamp = np.repeat(ft_time, ft_data.shape[0])
        else:
            # Update the last unique values
            dimensions_that_havent_changed = np.isclose(last_unique_values, ft_data)
            last_unique_values = np.where(
                dimensions_that_havent_changed,
                last_unique_values,
                ft_data,
            )
            last_unique_values_timestamp = np.where(
                dimensions_that_havent_changed,
                last_unique_values_timestamp,
                ft_time,
            )

        # Set the last unique values and the last unique values timestamp
        with self.last_unique_values_lock:
            self.last_unique_values = last_unique_values
            self.last_unique_values_timestamp = last_unique_values_timestamp

    def check_startup(self) -> List[Tuple[bool, str]]:
        """
        Check whether at least one FT sensor message has been received.

        Returns
        -------
        startup_status: A list of tuples, where each tuple contains a boolean
            status of a startup condition and a string describing the condition.
            All conditions must be True for the startup condition to be considered
            passed. For example, [(False, "Has received at least one message on
            topic X")] means that the startup condition has not passed because
            the node has not received any messages on topic X yet.
        """

        condition_1 = (
            f"Has received at least one message on topic {self.ft_sensor_topic}"
        )
        with self.last_unique_values_lock:
            status_1 = self.last_unique_values is not None

        return [(status_1, condition_1)]

    def check_status(self) -> List[Tuple[bool, str]]:
        """
        Check if the force-torque sensor is still publishing and its data is not
        zero-variance.

        Specifically, it returns True if over that last `self.timeout` seconds,
        every dimension of the force-torque sensor data has changed. Inversely,
        it returns False if either the force-torque sensor has not published
        data within the last `timeout` seconds, or at least one dimension
        of that data has not changed.

        Returns
        -------
        status: A list of tuples, where each tuple contains a boolean status
            of a condition and a string describing the condition. All conditions
            must be True for the status to be considered True. For example,
            [(True, "Has received a message on topic X within the last Y secs"),
            (False, "Messages on topic X over the last Y secs have non-zero variance")]
            means that the status is False and the watchdog should fail.
        """
        condition_1 = (
            "Every dimension of the F/T sensor data has received a new unique "
            f"value in the last {self.ft_timeout.nanoseconds / 10.0**9} secs"
        )
        now = self._node.get_clock().now()
        with self.last_unique_values_lock:
            status_1 = np.all(
                (now - self.last_unique_values_timestamp) <= self.ft_timeout
            )

        return [(status_1, condition_1)]
