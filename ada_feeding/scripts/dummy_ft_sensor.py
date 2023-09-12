#!/usr/bin/env python3
"""
This module contains a node, DummyForceTorqueSensor, which publishes sample data
to mimic the force-torque sensor on the robot. By setting parameters, users can
toggle the sensor on and off (i.e., to mimic communication with the sensor
dying) and/or start publishing zero-variance values (i.e., to mimic the sensor
getting corrupted). The node is intended to be used to test the ADAWatchdog
node.

Usage:
- Run the node: `ros2 run ada_feeding dummy_ft_sensor`
- Subscribe to the sensor data: `ros2 topic echo /wireless_ft/ftSensor1`
- Turn the sensor off: `ros2 param set /dummy_ft_sensor is_on False`
- Start publishing zero-variance data: 
    `ros2 param set /dummy_ft_sensor std [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Start publishing data where one dimension is zero-variance:
    `ros2 param set /dummy_ft_sensor std [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]`
"""

# Standard imports
import threading

# Third-party imports
from geometry_msgs.msg import WrenchStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import SetBool


class DummyForceTorqueSensor(Node):
    """
    The DummyForceTorqueSensor class publishes sample data to mimic the
    force-torque sensor on the robot. By setting parameters, users can toggle
    the sensor on and off (i.e., to mimic communication with the sensor dying)
    and/or start publishing zero-variance values (i.e., to mimic the sensor
    getting corrupted). The node is intended to be used to test the ADAWatchdog
    node.
    """

    # pylint: disable=too-many-instance-attributes
    # Two above is fine for a dummy node.

    def __init__(self, rate_hz: float = 100.0) -> None:
        """
        Initialize the dummy force-torque sensor node.

        Parameters
        ----------
        mean: the mean of the force-torque sensor data
        std: the standard deviation of the force-torque sensor data
        rate_hz: the rate (Hz) at which to publish the force-torque sensor data
        """
        super().__init__("dummy_ft_sensor")

        # Get the mean and standard deviaion of the distribution
        self.mean = self.declare_parameter(
            "mean",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ParameterDescriptor(
                name="mean",
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="The mean of the force-torque sensor data",
            ),
        )
        self.std = self.declare_parameter(
            "std",
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ParameterDescriptor(
                name="std",
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="The standard deviation of the force-torque sensor data",
            ),
        )

        # Create a parameter to toggle the sensor on and off
        self.is_on = self.declare_parameter(
            "is_on",
            True,
            ParameterDescriptor(
                name="is_on",
                type=ParameterType.PARAMETER_BOOL,
                description="Whether or not the simulated force-torque sensor is on",
            ),
        )

        # Create a service to (dummy) re-tare the sensor
        self.set_bias_request_time = None
        self.messages_since_bias_request = 0
        self.set_bias_request_time_lock = threading.Lock()
        self.set_bias_service = self.create_service(
            SetBool,
            "/wireless_ft/set_bias",
            self.set_bias_callback,
        )

        # Create the publisher
        self.ft_msg = WrenchStamped()
        self.publisher_ = self.create_publisher(
            WrenchStamped, "/wireless_ft/ftSensor1", 1
        )

        # Publish at the specified rate
        timer_period = 1.0 / rate_hz  # seconds
        self.timer = self.create_timer(timer_period, self.publish_msg)

        self.get_logger().info("Initialized!")

    def set_bias_callback(self, request: SetBool.Request, response: SetBool.Response):
        """
        Callback for the set_bias service. In order to mimic the actual service,
        this returns immediately, but then stops publishing data for 0.75 sec
        (handled in `publish_msg`). This is to mimic the time it takes to
        re-tare the sensor.
        """
        response.success = True
        if request.data:
            response.message = "Successfully set the bias"
        else:
            response.message = "Successfully unset the bias"
        with self.set_bias_request_time_lock:
            self.set_bias_request_time = self.get_clock().now()
        return response

    def publish_msg(self) -> None:
        """
        Publish a message to the force-torque sensor topic.
        """
        # Only publish if the sensor is on
        if self.get_parameter("is_on").value:
            # Reduce the publication rate by 10x while retaring
            with self.set_bias_request_time_lock:
                if (
                    self.set_bias_request_time is not None
                    and self.get_clock().now() - self.set_bias_request_time
                    < rclpy.duration.Duration(seconds=0.75)
                ):
                    self.messages_since_bias_request += 1
                    if (self.messages_since_bias_request % 10) != 0:
                        return

            # Get the simulated data
            ft_data = np.random.normal(
                self.get_parameter("mean").value, self.get_parameter("std").value
            )

            # Generate the force-torque sensor message
            self.ft_msg.header.stamp = self.get_clock().now().to_msg()
            self.ft_msg.wrench.force.x = ft_data[0]
            self.ft_msg.wrench.force.y = ft_data[1]
            self.ft_msg.wrench.force.z = ft_data[2]
            self.ft_msg.wrench.torque.x = ft_data[3]
            self.ft_msg.wrench.torque.y = ft_data[4]
            self.ft_msg.wrench.torque.z = ft_data[5]

            # Publish the message
            self.publisher_.publish(self.ft_msg)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    dummy_ft_sensor = DummyForceTorqueSensor()
    executor = MultiThreadedExecutor()
    rclpy.spin(dummy_ft_sensor, executor=executor)


if __name__ == "__main__":
    main()
