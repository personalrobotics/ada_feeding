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
import socket
from threading import Lock, Thread

# Third-party imports
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import WrenchStamped
import numpy as np
import pyaudio
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.clock import Clock
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

    def ok(self, now: Time) -> bool:
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


class EStopCondition:
    """
    The EStopCondition class monitors the state of the physical e-stop button.

    Specifically, it monitors the number of times the button has been pressed
    (defined as a reading above/below a certain threshold) and fails if that
    exceeds 1. This is because the first click is used to tell the watchdog
    that the button is plugged in and working, and the watchdog will keep
    failing until the first click is registered. After that, the watchdog will
    pass until the button is clicked a second time, at which point it fails.
    """

    def __init__(
        self,
        clock: Clock,
        initial_wait_secs: float,
        rate: int,
        chunk: int,
        device: int,
        min_threshold: int,
        max_threshold: int,
        acpi_event_name: str,
    ) -> None:
        """
        Initialize the e-stop condition.

        Parameters
        ----------
        clock: the ROS clock. Used to wait for `initial_wait_secs` at the beginning.
        initial_wait_secs: the number of seconds to wait at the beginning. This is
            because there is often initial noise in the audio readings.
        rate: the rate (Hz) at which the audio readings are taken.
        chunk: the number of audio readings to take at a time. If you are getting overflow
            errors, try increasing this number.
        device: the index of the audio device to use. This should usually be 0, for default
            device input.
        min_threshold (int16): the minimum threshold for the audio readings. If an audio reading
            is below this threshold, the e-stop button is considered to be pressed.
        max_threshold (int16): the maximum threshold for the audio readings. If an audio reading
            is above this threshold, the e-stop button is considered to be pressed.
        acpi_event_name: the name of the ACPI event that is published when the e-stop
            button is (un)plugged. To find it, run `acpi_listen` in the terminal and unplug
            the e-stop button. Exclude the suffix " plug"/" unplug".
        """

        # Initialize the parameters
        self.clock = clock
        self.initial_wait_duration = Duration(seconds=initial_wait_secs)
        self.acpi_event_name = acpi_event_name
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Initialize the accumulators
        self.start_time = None
        self.num_clicks = 0
        self.num_clicks_lock = Lock()
        self.is_mic_unplugged = False
        self.is_mic_unplugged_lock = Lock()

        # Start listening for ACPI events on a separate thread
        self.acpi_thread = Thread(target=self.__acpi_listener, daemon=True)
        self.acpi_thread.start()

        # Initialize the pyaudio object
        self.audio = pyaudio.PyAudio()

        # Initialize the stream, with a callback
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,  # The e-stop button is mono
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            input_device_index=device,
            stream_callback=self.__audio_callback,
        )

    def __acpi_listener(self) -> None:
        """
        Listens for ACPI events. If the e-stop button is unplugged, sets the
        `is_mic_unplugged` flag to True.
        """
        PLUG_KEYWORD = "plug"
        UNPLUG_KEYWORD = "unplug"

        # Open a socket to listen for ACPI events
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect("/var/run/acpid.socket")

        # Listen for ACPI events
        while rclpy.ok():
            data_bytes = s.recv(4096)
            data = data_bytes.decode("utf-8")
            for event in data.split("\n"):
                if self.acpi_event_name in event:
                    keyword = event.split(" ")[-1]
                    if UNPLUG_KEYWORD == keyword:
                        with self.is_mic_unplugged_lock:
                            self.is_mic_unplugged = True
                    elif PLUG_KEYWORD == keyword:
                        with self.is_mic_unplugged_lock:
                            self.is_mic_unplugged = False

    def __audio_callback(self, data, frame_count, time_info, status):
        """
        Callback function for the audio stream. This function is called whenever
        the audio stream has new data. This function checks if the e-stop button
        has been pressed, and if so, increments the number of clicks.

        Parameters
        ----------
        data: the audio data, as a byte string
        frame_count: the number of frames in the data
        time_info: the time info
        status: the status
        """
        # Skip the first few seconds of data, to avoid initial noise
        if self.start_time is None:
            self.start_time = self.clock.now()
        if self.clock.now() - self.start_time < self.initial_wait_duration:
            return (data, pyaudio.paContinue)

        # Convert the data to a numpy array
        data_arr = np.frombuffer(data, dtype=np.int16)

        # Check if the e-stop button has been pressed
        if np.any(
            np.logical_or(data_arr < self.min_threshold, data_arr > self.max_threshold)
        ):
            with self.num_clicks_lock:
                self.num_clicks += 1

        # Return the data
        return (data, pyaudio.paContinue)

    def is_plugged_in(self) -> bool:
        """
        Returns True if the e-stop button is plugged in, False otherwise.

        Specifically, this function returns True if the e-stop button has been
        clicked at least once and no ACPI message has been received indicating
        it has been unplugged, False otherwise.
        """
        with self.num_clicks_lock:
            num_clicks = self.num_clicks
        with self.is_mic_unplugged_lock:
            is_mic_unplugged = self.is_mic_unplugged
        return num_clicks > 0 and (not is_mic_unplugged)

    def ok(self) -> bool:
        """
        Returns True if the e-stop button is plugged in (e.g., has been clicked once)
        and hasn't been pressed since then (e.g., hasn't been clicked a second time).

        Returns
        -------
        True if the e-stop button has been pressed exactly once, False otherwise.
        """
        with self.num_clicks_lock:
            num_clicks = self.num_clicks
        with self.is_mic_unplugged_lock:
            is_mic_unplugged = self.is_mic_unplugged
        return num_clicks == 1 and (not is_mic_unplugged)


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
        self.load_parameters()

        # Create a watchdog publisher
        self.watchdog_publisher = self.create_publisher(
            DiagnosticArray,
            "~/watchdog",
            1,
        )

        # Create the e-stop conditions
        self.e_stop_condition = EStopCondition(
            self.get_clock(),
            initial_wait_secs=2.0,
            rate=48000,
            chunk=4800,
            device=0,
            min_threshold=-10000,
            max_threshold=10000,
            acpi_event_name="jack/microphone MICROPHONE",
        )

        # Parameters for the force-torque conditions
        self.ft_sensor_condition = FTSensorCondition(
            Duration(seconds=self.ft_timeout_sec.value)
        )
        self.recv_first_ft_msg = False
        self.ft_sensor_condition_lock = Lock()
        ft_sensor_ok_message = (
            f"Over the last {self.ft_timeout_sec.value} sec, "
            "the force-torque sensor has published data with nonzero variance"
        )
        self.ft_ok_status = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name="~/ft_topic",
            message=ft_sensor_ok_message,
        )
        ft_sensor_error_message = (
            f"Over the last {self.ft_timeout_sec.value} sec, the force-torque sensor "
            "has either not published data or its data is zero-variance"
        )
        self.ft_error_status = DiagnosticStatus(
            level=DiagnosticStatus.ERROR,
            name="~/ft_topic",
            message=ft_sensor_error_message,
        )

        # Create the watchdog output
        self.watchdog_output = DiagnosticArray()

        # Subscribe to the force-torque sensor topic
        self.ft_sensor_subscription = self.create_subscription(
            WrenchStamped,
            "~/ft_topic",
            self.ft_sensor_callback,
            rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value,
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
        self.ft_timeout_sec = self.declare_parameter(
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
        self.get_logger().info(
            f"E-Stop Condition first condition {self.e_stop_condition.is_plugged_in()} continuing condition {self.e_stop_condition.ok()} num clicks {self.e_stop_condition.num_clicks}"
        )
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
            ft_condition = self.ft_sensor_condition.ok(now)

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
