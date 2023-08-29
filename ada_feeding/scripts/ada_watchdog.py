#!/usr/bin/env python3
"""
This module contains a node, ADAWatchdog, which does the following:
    1. Monitors the state of the force-torque sensor, to ensure it is still
       publishing and its data is not zero-variance.
    2. Monitors the state of the physical e-stop button, to ensure it is
       plugged in (has received at least one click) and has not been pressed
       since then (has not received a second click).
This node publishes an output to the /ada_watchdog topic. Any node that moves
the robot should subscribe to this topic and immediately stop if any of the
watchdog conditions fail, or if the watchdog stops publishing.
"""

# Standard imports
from typing import List

# Third-party imports
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

# Local imports
from ada_feeding.watchdog import (
    EStopCondition,
    FTSensorCondition,
    WatchdogCondition,
)


class ADAWatchdog(Node):
    """
    A watchdog node for the ADA robot. This node monitors the state of the
    force-torque sensor and the physical e-stop button (TODO), and publishes its
    output to the watchdog topic.
    """

    # pylint: disable=too-many-instance-attributes
    # Eleven is fine in this case.

    def __init__(self) -> None:
        """
        Initialize the watchdog node.
        """
        super().__init__("ada_watchdog")

        # Load parameters
        self.__load_parameters()

        # Create the conditions
        self.conditions = [
            FTSensorCondition(self),
        ]
        if self.use_estop.value:
            self.conditions.insert(0, EStopCondition(self))
        self.has_passed_startup_conditions = False

        # Create a watchdog publisher
        self.watchdog_publisher = self.create_publisher(
            DiagnosticArray,
            "~/watchdog",
            1,
        )

        # Publish at the specified rate
        timer_period = 1.0 / self.publish_rate_hz.value  # seconds
        self.timer = self.create_timer(timer_period, self.check_conditions)

    def __load_parameters(self) -> None:
        """
        Load parameters from the parameter server.
        """
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
        self.use_estop = self.declare_parameter(
            "use_estop",
            True,
            ParameterDescriptor(
                name="use_estop",
                type=ParameterType.PARAMETER_BOOL,
                description=(
                    "Whether to check the state of the physical e-stop button. "
                    "This should only be set False in sim, since we currently "
                    "have no way of simulating the e-stop button."
                ),
                read_only=True,
            ),
        )

    def __generate_diagnostic_status(
        self, status: bool, condition: WatchdogCondition, message: str, startup=False
    ) -> DiagnosticStatus:
        """
        Generate a diagnostic status message.

        Parameters
        ----------
        status: bool
            The status of the condition.
        condition: WatchdogCondition
            The watchdog condition that was checked.
        message: str
            The message to include in the status.
        startup: bool
            Whether to append " Startup" to the name of the condition.

        Returns
        -------
        DiagnosticStatus
            The diagnostic status message.
        """
        return DiagnosticStatus(
            level=DiagnosticStatus.OK if status else DiagnosticStatus.ERROR,
            name=condition.__class__.__name__ + (" Startup" if startup else ""),
            message=message,
        )

    def __generate_diagnostic_array(
        self, diagnostic_statuses: List[DiagnosticStatus]
    ) -> DiagnosticArray:
        """
        Generate a diagnostic array message.

        Parameters
        ----------
        diagnostic_statuses: List[DiagnosticStatus]
            The diagnostic statuses to include in the message.

        Returns
        -------
        DiagnosticArray
            The diagnostic array message.
        """
        return DiagnosticArray(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id="",
            ),
            status=diagnostic_statuses,
        )

    def check_conditions(self) -> None:
        """
        Checks the watchdog conditions and publishes its output.
        """
        # First, check the startup conditions
        if not self.has_passed_startup_conditions:
            diagnostic_statuses = []
            passed = True
            for condition in self.conditions:
                condition_status = condition.check_startup()
                for status, message in condition_status:
                    passed = passed and status  # Fail as soon as one status is False
                    diagnostic_statuses.append(
                        self.__generate_diagnostic_status(
                            status, condition, message, startup=True
                        )
                    )
            if not passed:  # At least one startup condition failed
                watchdog_output = self.__generate_diagnostic_array(diagnostic_statuses)
                self.watchdog_publisher.publish(watchdog_output)
            else:  # All startup conditions passed
                self.has_passed_startup_conditions = True

        # If the startup conditions have passed, check the status conditions
        if self.has_passed_startup_conditions:
            diagnostic_statuses = []
            passed = True
            for condition in self.conditions:
                condition_status = condition.check_status()
                for status, message in condition_status:
                    passed = passed and status
                    diagnostic_statuses.append(
                        self.__generate_diagnostic_status(status, condition, message)
                    )
            # Publish the watchdog status
            watchdog_output = self.__generate_diagnostic_array(diagnostic_statuses)
            self.watchdog_publisher.publish(watchdog_output)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    # Use a MultiThreadedExecutor to enable processing topics concurrently
    executor = MultiThreadedExecutor()

    # Create and spin the node
    ada_watchdog = ADAWatchdog()
    rclpy.spin(ada_watchdog, executor=executor)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ada_watchdog.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
