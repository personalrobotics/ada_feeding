#!/usr/bin/env python3
"""
This module contains a node, ADAWatchdog, which does the following:
    1. Monitors the state of the force-torque sensor, to ensure it is still
       publishing and its data is not zero-variance.
    2. (TODO) Monitors the state of the physical e-stop button, to ensure it is
       not pressed.
This node publishes an output to the /ada_watchdog topic. Any node that moves
the robot should subscribe to this topic and immediately stop if any of the
watchdog conditions fail, or if the watchdog stops publishing.
"""

# Standard imports
from threading import Lock

# Third-party imports
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import WrenchStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

# Local imports


class FTSensorCondition:
    """
    The FTSensorCondition class accumulates all force-torque sensor readings and
    checks that the sensor is still publishing and its data is not zero-variance.
    """

    def __init__(self, timeout: Duration) -> None:
        """
        Initialize the force-torque sensor condition.

        Parameters
        ----------
        timeout: the maximum time (s) that the force-torque sensor can go without
            publishing or changing its data before the condition fails
        """
        # Configure parameters
        self.timeout = timeout

        # For each dimension of a single force-torque datapoint, store the most
        # recent unique value and the time at which that value was received.
        self.last_unique_values = None
        self.last_unique_values_timestamp = None

    def update(self, ft_msg: WrenchStamped) -> None:
        """
        Update the accumulators with the latest force-torque sensor reading.

        Parameters
        ----------
        ft_msg: the message from the force-torque sensor
        """
        # Get the data from the message
        ft_array = np.array(
            [
                ft_msg.wrench.force.x,
                ft_msg.wrench.force.y,
                ft_msg.wrench.force.z,
                ft_msg.wrench.torque.x,
                ft_msg.wrench.torque.y,
                ft_msg.wrench.torque.z,
            ]
        )
        ft_time = Time.from_msg(ft_msg.header.stamp)

        # Update the last unique values
        if self.last_unique_values is None:
            self.last_unique_values = ft_array
            self.last_unique_values_timestamp = np.repeat(ft_time, 6)
        else:
            # Update the last unique values
            dimensions_that_havent_changed = np.isclose(
                self.last_unique_values, ft_array
            )
            self.last_unique_values = np.where(
                dimensions_that_havent_changed,
                self.last_unique_values,
                ft_array,
            )
            self.last_unique_values_timestamp = np.where(
                dimensions_that_havent_changed,
                self.last_unique_values_timestamp,
                ft_time,
            )

    def check(self, now: Time) -> bool:
        """
        Check if the force-torque sensor is still publishing and its data is not
        zero-variance.

        Specifically, it returns True if over that last `self.timeout` seconds,
        every dimension of the force-torque sensor data has changed. Inversely,
        it returns False if either the force-torque sensor has not published
        data within the last `timeout` seconds, or at least one dimension
        of that data has not changed.

        Parameters
        ----------
        now: the current time

        Returns
        -------
        True if the force-torque sensor is still publishing and its data is not
        zero-variance, False otherwise.
        """
        return np.all((now - self.last_unique_values_timestamp) <= self.timeout)


class ADAWatchdog(Node):
    """
    A watchdog node for the ADA robot. This node monitors the state of the
    force-torque sensor and the physical e-stop button (TODO), and publishes its
    output to the /ada_watchdog topic.
    """

    def __init__(self) -> None:
        """
        Initialize the watchdog node.
        """
        super().__init__("ada_watchdog")

        # Load parameters
        self.load_parameters()

        # Create a watchdog publisher
        self.watchdog_publisher = self.create_publisher(
            DiagnosticArray,
            self.watchdog_topic.value,
            1,
        )

        # Parameters for the force-torque conditions
        self.ft_sensor_condition = FTSensorCondition(
            Duration(seconds=self.ft_timeout_sec.value)
        )
        self.recv_first_ft_msg = False
        self.ft_sensor_condition_lock = Lock()
        ft_sensor_ok_message = (
            "Over the last %f sec, the force-torque sensor has published data with nonzero variance"
            % self.ft_timeout_sec.value
        )
        self.ft_ok_status = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name=self.ft_topic.value,
            message=ft_sensor_ok_message,
        )
        ft_sensor_error_message = (
            "Over the last %f sec, the force-torque sensor has either not published data or its data is zero-variance"
            % self.ft_timeout_sec.value
        )
        self.ft_error_status = DiagnosticStatus(
            level=DiagnosticStatus.ERROR,
            name=self.ft_topic.value,
            message=ft_sensor_error_message,
        )

        # Create the watchdog output
        self.watchdog_output = DiagnosticArray()

        # Subscribe to the force-torque sensor topic
        self.ft_sensor_subscription = self.create_subscription(
            WrenchStamped,
            self.ft_topic.value,
            self.ft_sensor_callback,
            1,
        )

        # Publish at the specified rate
        timer_period = 1.0 / self.publish_rate_hz.value  # seconds
        self.timer = self.create_timer(
            timer_period, self.check_and_publish_watchdog_output
        )

    def load_parameters(self) -> None:
        """
        Load parameters from the parameter server.
        """
        self.ft_topic = self.declare_parameter(
            "ft_topic",
            "/wireless_ft/ftSensor1",
            ParameterDescriptor(
                name="ft_topic",
                type=ParameterType.PARAMETER_STRING,
                description="The name of topic that the force-torque sensor is publishing on",
                read_only=True,
            ),
        )
        self.ft_timeout_sec = self.declare_parameter(
            "ft_timeout_sec",
            0.5,
            ParameterDescriptor(
                name="ft_timeout_sec",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The number of seconds within which the force-torque sensor must have: (a) published messages; and (b) had them be nonzero-variance",
                read_only=True,
            ),
        )
        self.watchdog_topic = self.declare_parameter(
            "watchdog_topic",
            "/ada_watchdog",
            ParameterDescriptor(
                name="watchdog_topic",
                type=ParameterType.PARAMETER_STRING,
                description="The name of the topic for the watchdog to publish on",
                read_only=True,
            ),
        )
        self.publish_rate_hz = self.declare_parameter(
            "publish_rate_hz",
            30.0,
            ParameterDescriptor(
                name="publish_rate_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The target rate (Hz) for the watchdog to publish its output",
                read_only=True,
            ),
        )

    def ft_sensor_callback(self, ft_msg: WrenchStamped) -> None:
        """
        Callback function for the force-torque sensor topic. This function
        stores the latest force-torque sensor reading, to be checked in
        check_watchdog_conditions().

        Parameters
        ----------
        ft_msg: the message from the force-torque sensor
        """
        with self.ft_sensor_condition_lock:
            self.ft_sensor_condition.update(ft_msg)
            self.recv_first_ft_msg = True

    def check_and_publish_watchdog_output(self) -> None:
        """
        Checks the watchdog conditions and publishes its output.
        """
        # Only publish if we've received the first force-torque sensor message
        recv_first_ft_msg = False
        with self.ft_sensor_condition_lock:
            recv_first_ft_msg = self.recv_first_ft_msg

        # Configure the output message
        now = self.get_clock().now()
        self.watchdog_output.header.stamp = now.to_msg()
        self.watchdog_output.status = []

        # Return the output
        if recv_first_ft_msg:
            # Check the force-torque sensor conditions
            ft_condition = self.ft_sensor_condition.check(now)

            # Generate the watchdog output
            if ft_condition:
                self.watchdog_output.status.append(self.ft_ok_status)
            else:
                self.watchdog_output.status.append(self.ft_error_status)
        else:
            self.watchdog_output.status.append(self.ft_error_status)

        # Publish the watchdog output
        self.watchdog_publisher.publish(self.watchdog_output)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    ada_watchdog = ADAWatchdog()
    rclpy.spin(ada_watchdog)


if __name__ == "__main__":
    main()
