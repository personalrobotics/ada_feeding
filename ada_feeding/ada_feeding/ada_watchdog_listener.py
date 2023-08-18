"""
This module defines the ADAWatchdogListener class, which listens to the watchdog
topic and can be queried to determine if the watchdog is still alive.
"""

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
    The ADAWatchdogListener class listens to the watchdog topic and can be queried
    to determine if the watchdog is still alive.
    """

    def __init__(self, node: Node):
        """
        Initialize the watchdog listener.

        Parameters
        ----------
        node: the ROS node that this watchdog listener is associated with
        """
        # Store the node
        self.node = node

        # Read the watchdog timeout parameter
        watchdog_timeout = self.node.declare_parameter(
            "watchdog_timeout",
            0.5,  # default value
            ParameterDescriptor(
                name="watchdog_timeout",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The maximum time (s) that the watchdog can go without "
                    "publishing before the watchdog fails."
                ),
                read_only=True,
            ),
        )
        self.watchdog_timeout = Duration(seconds=watchdog_timeout.value)

        # Subscribe to the watchdog topic
        # Initializing `watchdog_failed` to False lets the node wait up to `watchdog_timeout`
        # sec to receive the first message
        self.watchdog_failed = False
        self.last_watchdog_msg_time = self.node.get_clock().now()
        self.watchdog_sub = self.node.create_subscription(
            DiagnosticArray,
            "~/watchdog",
            self.__watchdog_callback,
            1,
        )

    def __watchdog_callback(self, msg: DiagnosticArray) -> None:
        """
        Callback function for the watchdog topic. This function checks if the
        watchdog has failed.

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
        return (
            (not self.watchdog_failed)
            and (
                (self.node.get_clock().now() - self.last_watchdog_msg_time)
                < self.watchdog_timeout
            )
        )
