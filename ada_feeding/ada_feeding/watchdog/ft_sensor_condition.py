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

    # pylint: disable=too-many-instance-attributes
    # One over is fine, since each attribute is necessary

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
        self.ft_sensor_callback_group = (
            rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        )
        self.ft_sensor_subscription = self._node.create_subscription(
            WrenchStamped,
            self.ft_sensor_topic,
            self.__ft_sensor_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=self.ft_sensor_callback_group,
        )

        # Get a function to take the abs of a Duration array
        self.duration_array_abs = np.vectorize(FTSensorCondition.duration_abs)

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

    @staticmethod
    def duration_abs(duration: Duration) -> Duration:
        """
        Return the absolute value of a duration.
        """
        if duration.nanoseconds < 0:
            return Duration(nanoseconds=-1.0 * duration.nanoseconds)
        return duration

    def check_startup(self) -> List[Tuple[bool, str, str]]:
        """
        Check whether at least one FT sensor message has been received.

        Returns
        -------
        startup_status: A list of tuples, where each tuple contains a boolean
            status of a startup condition, a string name describing the condition,
            and a string detailing the status of the condition. All conditions
            must be True for the startup condition to be considered passed.
            For example, [(False, "Recieved Topic X Data", "Has not received at
            least one message on topic X")] means that the startup condition has not
            passed because the node has not received any messages on topic X yet.
        """
        name_1 = "Startup: Received Force/Torque Data"
        with self.last_unique_values_lock:
            status_1 = self.last_unique_values is not None
        condition_1 = (
            f"Has {'' if status_1 else 'not '}received at least one message on topic "
            f"{self.ft_sensor_topic}"
        )

        return [(status_1, name_1, condition_1)]

    def check_status(self) -> List[Tuple[bool, str, str]]:
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
            of a condition, a string name describing the condition, and a string
            detailing the status of the condition. All conditions must be True for
            the status to be considered True. For example, [(True, "Received Topic
            X Data", "Has received a message on topic X within the last Y secs"),
            (False, "Non-Corruped Topic X Data", "Messages on topic X over the
            last Y secs have zero variance")] means that the status is False and
            the watchdog should fail.
        """
        name_1 = "Receiving Force-Torque Data"
        now = self._node.get_clock().now()
        with self.last_unique_values_lock:
            status_1 = np.all(
                self.duration_array_abs(now - self.last_unique_values_timestamp)
                <= self.ft_timeout
            )
        if status_1:
            condition_1 = (
                f"Over the last {self.ft_timeout.nanoseconds / 10.0**9} secs, "
                "the F/T sensor has published data and every dimension of that "
                "data has non-zero variance."
            )
        else:
            condition_1 = (
                f"Over the last {self.ft_timeout.nanoseconds / 10.0**9} secs, either "
                "the F/T sensor has not published data, or at least one dimension of "
                "that data has zero variance."
            )

        return [(status_1, name_1, condition_1)]

    def terminate(self) -> None:
        """
        Terminate the FT Sensor condition. In this case, no termination is
        necessary, since the only state this watchdog condition maintains has
        to do with ROS subscribers, which will be terminated when the watchdog
        node terminates anyway.
        """
        return None
