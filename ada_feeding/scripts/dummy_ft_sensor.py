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
- Start publishing zero-variance data: `ros2 param set /dummy_ft_sensor std [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Start publishing data where one dimension is zero-variance: `ros2 param set /dummy_ft_sensor std [0.0, 0.1, 0.1, 0.1, 0.1, 0.1]`
"""

# Third-party imports
from geometry_msgs.msg import WrenchStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class DummyForceTorqueSensor(Node):
    """
    The DummyForceTorqueSensor class publishes sample data to mimic the
    force-torque sensor on the robot. By setting parameters, users can toggle
    the sensor on and off (i.e., to mimic communication with the sensor dying)
    and/or start publishing zero-variance values (i.e., to mimic the sensor
    getting corrupted). The node is intended to be used to test the ADAWatchdog
    node.
    """

    def __init__(self, rate_hz: float = 30.0) -> None:
        """
        Initialize the dummy force-torque sensor node.

        Parameters
        ----------
        mean: the mean of the force-torque sensor data
        std: the standard deviation of the force-torque sensor data
        rate_hz: the rate (Hz) at which to publish the force-torque sensor data
        """
        super().__init__("dummy_ft_sensor")

        # get the mean and standard deviaion of the distribution
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

        # Create the publisher
        self.ft_msg = WrenchStamped()
        self.publisher_ = self.create_publisher(
            WrenchStamped, "/wireless_ft/ftSensor1", 1
        )

        # Publish at the specified rate
        timer_period = 1.0 / rate_hz  # seconds
        self.timer = self.create_timer(timer_period, self.publish_msg)

    def publish_msg(self) -> None:
        """
        Publish a message to the force-torque sensor topic.
        """
        # Only publish if the sensor is on
        if self.get_parameter("is_on").value:
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