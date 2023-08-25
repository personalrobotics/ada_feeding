"""
This module defines the ADAWatchdogListener class, which listens to the watchdog
topic and can be used in two ways:
    A: Query the watchdog listener to determine if the watchdog is `ok()`.
    B: Pass in a callback function to be called when the watchdog status changes.
"""

# Standard imports
from typing import Callable, Optional

# Third-party imports
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time


# pylint: disable=too-few-public-methods
# This class only needs one public method to check if the watchdog is ok.
class ADAWatchdogListener:
    """
    The ADAWatchdogListener class listens to the watchdog topic and can be used
    in two ways:
        A: Query the watchdog listener to determine if the watchdog is `ok()`.
        B: Pass in a callback function to be called when the watchdog status
            changes.
    """

    # pylint: disable=too-many-instance-attributes
    # One extra is fine in this case.

    def __init__(self, node: Node, callback_fn: Optional[Callable] = None) -> None:
        """
        Initialize the watchdog listener.

        Parameters
        ----------
        node: the ROS node that this watchdog listener is associated with.
        callback_fn: If not None, this function will be called when the watchdog
            status changes. The function should take in a single boolean argument
            that is True if the watchdog is ok, else False.
        """
        # Store the node
        self._node = node

        # Read the watchdog_timeout_sec parameter
        watchdog_timeout_sec = self._node.declare_parameter(
            "watchdog_timeout_sec",
            0.5,  # default value
            ParameterDescriptor(
                name="watchdog_timeout_sec",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The maximum time (s) that the watchdog can go without "
                    "publishing before the watchdog fails."
                ),
                read_only=True,
            ),
        )
        self.watchdog_timeout_sec = Duration(seconds=watchdog_timeout_sec.value)

        # Read the watchdog_check_hz parameter
        watchdog_check_hz = self._node.declare_parameter(
            "watchdog_check_hz",
            60.0,  # default value
            ParameterDescriptor(
                name="watchdog_check_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to check the whether the watchdog has failed."
                    "This parameter is only used if a callback function is passed to the"
                    "watchdog listener."
                ),
                read_only=True,
            ),
        )

        # Subscribe to the watchdog topic
        # Initializing `watchdog_failed` to False lets the node wait up to `watchdog_timeout_sec`
        # sec to receive the first message
        self.watchdog_failed = False
        self.last_watchdog_msg_time = self._node.get_clock().now()
        self.watchdog_sub = self._node.create_subscription(
            DiagnosticArray,
            "~/watchdog",
            self.__watchdog_callback,
            1,
        )

        # If a callback function is passed in, check the watchdog at the specified rate
        if callback_fn is not None:
            self.callback_fn = callback_fn
            self._prev_status = None
            timer_period = 1.0 / watchdog_check_hz.value
            self.timer = self._node.create_timer(timer_period, self.__timer_callback)

    def __watchdog_callback(self, msg: DiagnosticArray) -> None:
        """
        Callback function for the watchdog topic. This function checks if the
        watchdog has failed (i.e., if any DiagnosticStatus has a level that is
        not DiagnosticStatus.OK).

        Parameters
        ----------
        msg: The watchdog message.
        """
        watchdog_failed = False
        for status in msg.status:
            if status.level != DiagnosticStatus.OK:
                watchdog_failed = True
                break
        self.watchdog_failed = watchdog_failed

        self.last_watchdog_msg_time = Time.from_msg(msg.header.stamp)

    # pylint: disable=invalid-name
    # This matches the corresponding method name in rclpy.
    def ok(self) -> bool:
        """
        Returns
        -------
        True if the watchdog is OK and has not timed out, else False.
        """
        if self.watchdog_failed:
            self._node.get_logger().error("Watchdog failed!", throttle_duration_sec=1)
            return False
        if (
            self._node.get_clock().now() - self.last_watchdog_msg_time
        ) > self.watchdog_timeout_sec:
            self._node.get_logger().error(
                f"Did not receive a watchdog message for > {self.watchdog_timeout_sec}!",
                throttle_duration_sec=1,
            )
            return False
        return True

    def __timer_callback(self) -> None:
        """
        If the watchdog has failed, call the callback function.
        """
        # Get the watchdog status
        curr_status = self.ok()

        # Check if the watchdog status has changed since the last time this function was called
        if self._prev_status is not None and curr_status != self._prev_status:
            # If it has, call the callback function
            self._node.get_logger().debug(
                f"Watchdog status (whether it is ok) changed to {curr_status}"
            )
            self.callback_fn(curr_status)

        # Update the previous status
        self._prev_status = curr_status
