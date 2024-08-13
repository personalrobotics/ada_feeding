#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains EStopCondition, a watchdog condition that listens for the
e-stop button either being clicked or being unplugged and fails if so.
"""

# Standard imports
import math
import socket
import subprocess
from threading import Lock, Thread
from typing import List, Optional, Tuple, Union

# Third party imports
import numpy as np
import numpy.typing as npt
import pyaudio
import pyudev
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import sounddevice  # pylint: disable=unused-import

# Local imports
from ada_feeding.watchdog import WatchdogCondition


class EStopCondition(WatchdogCondition):
    """
    The EStopCondition class monitors the state of the physical e-stop button.

    Specifically, the stratup condition is that the e-stop button is clicked
    once (to ensure it is plugged in). After that, the status condition is that
    it is not clicked anymore. As soon as it is clicked a second time, this
    condition permanently fails.

    Note that clicking the e-stop is defined as TODO.
    """

    # pylint: disable=too-many-instance-attributes
    # We need so many because we allow users to configure the audio stream parameters.

    PYAUDIO_STREAM_TROUBLESHOOTING = (
        "The Pyaudio stream not opening error is often caused by another process using "
        "the microphone and/or audio device. To address this, terminate the code and "
        "try the following:\n"
        "  1. Close all applications (e.g., System Settings) that may be accessing "
        "audio devices.\n"
        "  2. If that still doesn't address it, run `sudo alsa force-reload`.\n"
        "     Wait a few (~5) secs after running this command to restart the node,\n"
        "     and note that you may have to run this command multiple times.\n"
        "Note that until this is addressed, the e-stop button will not be working."
    )

    def __init__(self, node: Node) -> None:
        """
        Initialize the EStopCondition class.

        Parameters
        ----------
        node: Node
            The ROS2 node that this watchdog condition belongs to.
        """
        # Store the node
        self._node = node

        # Load the parameters
        self.__load_parameters()

        # Set system volume
        self.__configure_system_audio()

        # Initialize the accumulators
        self.start_time = None
        self.last_recv_data_time_lock = Lock()
        self.last_recv_data_time = None
        self.curr_data_arr = None
        self.prev_data_arr = None
        self.prev_button_click_start_time = None
        self.num_clicks = 0
        self.num_clicks_lock = Lock()
        self.is_mic_unplugged = False
        self.is_mic_unplugged_lock = Lock()

        # Initialize STD Exponential Moving Average
        self.std_ema = None
        self.std_ema_lock = Lock()

        # Initialize the stream, with a callback
        self.audio = None
        self.stream = None
        self.last_stream_init_time = self._node.get_clock().now()
        self.__init_stream()

    def __load_parameters(self) -> None:
        """
        Load the parameters for this watchdog condition.
        """

        # pylint: disable=too-many-locals

        rate = self._node.declare_parameter(
            "rate",
            48000,
            ParameterDescriptor(
                name="rate",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which audio readings are taken from the microphone "
                    "(where the e-stop button is plugged in)"
                ),
                read_only=True,
            ),
        )
        self.rate = rate.value

        chunk = self._node.declare_parameter(
            "chunk",
            4800,
            ParameterDescriptor(
                name="chunk",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The number of audio samples to take at a time from the microphone "
                    "(where the e-stop button is plugged in). If you get overflow errors, "
                    "try increasing this value."
                ),
                read_only=True,
            ),
        )
        self.chunk = chunk.value

        stream_open_retry_hz = self._node.declare_parameter(
            "stream_open_retry_hz",
            1.0,
            ParameterDescriptor(
                name="stream_open_retry_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to retry opening the audio stream "
                    "if it fails to open."
                ),
                read_only=True,
            ),
        )
        self.stream_open_retry_duration = Duration(
            seconds=1.0 / stream_open_retry_hz.value
        )

        initial_wait_secs = self._node.declare_parameter(
            "initial_wait_secs",
            2.0,
            ParameterDescriptor(
                name="initial_wait_secs",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The number of seconds to wait before checking if the e-stop "
                    "button is plugged in. This is because there is typically initial "
                    "noise from the audio signal."
                ),
                read_only=True,
            ),
        )
        self.initial_wait_duration = Duration(seconds=initial_wait_secs.value)

        min_threshold = self._node.declare_parameter(
            "min_threshold",
            -10000,
            ParameterDescriptor(
                name="min_threshold",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "A falling edge must go below this threshold to be considered "
                    "a button click"
                ),
                read_only=True,
            ),
        )
        self.min_threshold = min_threshold.value

        max_threshold = self._node.declare_parameter(
            "max_threshold",
            10000,
            ParameterDescriptor(
                name="max_threshold",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "A rising edge must go above this threshold to be considered "
                    "a button click"
                ),
                read_only=True,
            ),
        )
        self.max_threshold = max_threshold.value

        time_per_click_sec = self._node.declare_parameter(
            "time_per_click_sec",
            0.5,
            ParameterDescriptor(
                name="time_per_click_sec",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "After the first rising/falling edge is detected, any future "
                    "rising/falling edges detected within these many seconds "
                    "will be considered part of the same button click."
                ),
                read_only=True,
            ),
        )
        self.time_per_click_duration = Duration(seconds=time_per_click_sec.value)

        # Parameters for setting system volume using `amixer`. This is necessary
        # because the e-stop button is a microphone, so the system's microphone
        # volume will impact the amplitude of readings for the e-stop button.
        # See `ada_watchdog.yaml` for instructions on tuning these parameters.
        amixer_configuration_name = self._node.declare_parameter(
            "amixer_configuration_name",
            None,
            ParameterDescriptor(
                name="amixer_configuration_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The name of the configuration to use for `amixer`. "
                    "For configuration name `config_name`, there must be "
                    "parameters `config_name.amixer_mic_toggle_control_name`, "
                    "`config_name.amixer_mic_control_names`, and "
                    "`config_name.amixer_mic_control_percentages`."
                ),
                read_only=True,
            ),
        )
        pactl_mic_name = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.pactl_mic_name",
            None,
            ParameterDescriptor(
                name=f"{amixer_configuration_name.value}.pactl_mic_name",
                type=ParameterType.PARAMETER_STRING,
                description=("The name of the microphone to set as default."),
                read_only=True,
            ),
        )
        amixer_card_num = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.amixer_card_num",
            None,
            ParameterDescriptor(
                name=f"{amixer_configuration_name.value}.amixer_card_num",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The sound card number of the microphone to set as default."
                ),
                read_only=True,
            ),
        )
        amixer_mic_toggle_control_name = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.amixer_mic_toggle_control_name",
            None,
            ParameterDescriptor(
                name=f"{amixer_configuration_name.value}.amixer_mic_toggle_control_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The name of the toggle control to use for `amixer`. "
                    "This is the control that is toggled to mute/unmute the "
                    "microphone."
                ),
                read_only=True,
            ),
        )
        amixer_mic_control_names = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.amixer_mic_control_names",
            None,
            ParameterDescriptor(
                name=f"{amixer_configuration_name.value}.amixer_mic_control_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "The names of the controls to use for `amixer`. "
                    "These are the controls that are set to change the microphone "
                    "volume."
                ),
                read_only=True,
            ),
        )
        amixer_mic_control_percentages = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.amixer_mic_control_percentages",
            None,
            ParameterDescriptor(
                name=f"{amixer_configuration_name.value}.amixer_mic_control_percentages",
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description=(
                    "The percentages to set the controls to for `amixer`. "
                    "These are the percentages that the controls are set to change "
                    "the microphone volume. Must be the same length as "
                    "`amixer_mic_control_names`."
                ),
                read_only=True,
            ),
        )
        is_usb = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.is_usb",
            False,
            ParameterDescriptor(
                name="is_usb",
                type=ParameterType.PARAMETER_BOOL,
                description=(
                    "Whether the microphone is a USB microphone. If True, the "
                    "udev event listener is used to detect it being unplugged, "
                    "else the ACPI event listener is used."
                ),
                read_only=True,
            ),
        )
        acpi_event_name = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.acpi_event_name",
            None,
            ParameterDescriptor(
                name="acpi_event_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The name of the ACPI event that is triggered when the e-stop "
                    "button is (un)plugged. Only applicable if `is_usb` is False."
                ),
                read_only=True,
            ),
        )
        udev_id = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.udev_id",
            None,
            ParameterDescriptor(
                name="udev_id",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The id of the USB device. Only applicable if `is_usb` is True."
                ),
                read_only=True,
            ),
        )
        device_name = self._node.declare_parameter(
            f"{amixer_configuration_name.value}.device_name",
            None,
            ParameterDescriptor(
                name="device_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The name of the audio device. This is used to get the device "
                    "index for `pyaudio`. If unspecified, device index 0 is used, "
                    "which typically works if your device is connected by the 3.5mm aux port."
                ),
                read_only=True,
            ),
        )
        self.audio_configuration = {
            "pactl_mic_name": pactl_mic_name.value,
            "amixer_card_num": amixer_card_num.value,
            "amixer_mic_toggle_control_name": amixer_mic_toggle_control_name.value,
            "amixer_mic_control_names": amixer_mic_control_names.value,
            "amixer_mic_control_percentages": amixer_mic_control_percentages.value,
            "is_usb": is_usb.value,
            "acpi_event_name": acpi_event_name.value,
            "device_name": device_name.value,
            "udev_id": udev_id.value,
        }

        std_ema_min_thresh = self._node.declare_parameter(
            "std_ema_min_thresh",
            0.0,
            ParameterDescriptor(
                name="std_ema_min_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "If the exponential moving average of the standard deviation of microphone "
                    "readings is less than this value, fail the e-stop condition"
                ),
                read_only=True,
            ),
        )
        self.std_ema_min_thresh = std_ema_min_thresh.value

        std_ema_max_thresh = self._node.declare_parameter(
            "std_ema_max_thresh",
            math.inf,
            ParameterDescriptor(
                name="std_ema_max_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "If the exponential moving average of the standard deviation of microphone "
                    "readings is lmoreess than this value, fail the e-stop condition"
                ),
                read_only=True,
            ),
        )
        self.std_ema_max_thresh = std_ema_max_thresh.value

        std_ema_alpha = self._node.declare_parameter(
            "std_ema_alpha",
            0.5,
            ParameterDescriptor(
                name="std_ema_alpha",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The alpha value to use when computing the exponential moving average "
                    "of the standard deviation. Should be in [0, 1]"
                ),
                read_only=True,
            ),
        )
        self.std_ema_alpha = std_ema_alpha.value

        num_secs_threshold = self._node.declare_parameter(
            "num_secs_threshold",
            1.0,
            ParameterDescriptor(
                name="num_secs_threshold",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "If the e-stop button has not sent data for these many secs, fail."
                ),
                read_only=True,
            ),
        )
        self.num_secs_threshold = num_secs_threshold.value

    def __configure_system_audio(self) -> None:
        """
        Configure the system audio. This function sets the default mic and sets
        its volume. This is necessary because the  e-stop button is a microphone,
        so the system's microphone volume will impact the amplitude of readings
        for the e-stop button.

        TODO: Although not crucial, we should consider storing the original
        system volume, and then restoring it when thos watchdog condition is
        terminated.
        """
        # pylint: disable=too-many-branches
        # There are several steps to the configuration, so we need many branches.

        # Get the amixer_configuration
        pactl_mic_name = self.audio_configuration["pactl_mic_name"]
        amixer_card_num = self.audio_configuration["amixer_card_num"]
        if amixer_card_num is None:
            amixer_card_num = "0"
        if not isinstance(amixer_card_num, str):
            amixer_card_num = str(amixer_card_num)
        toggle_control_name = self.audio_configuration["amixer_mic_toggle_control_name"]
        control_names = self.audio_configuration["amixer_mic_control_names"]
        control_percentages = self.audio_configuration["amixer_mic_control_percentages"]
        is_usb = self.audio_configuration["is_usb"]

        # Set the default microphone
        if pactl_mic_name is not None:
            try:
                set_default_mic = subprocess.check_output(
                    ["pactl", "set-default-source", pactl_mic_name]
                )
                if b"Failure" in set_default_mic:
                    self._node.get_logger().error(
                        f"Failed to set default microphone to {pactl_mic_name}:\n{set_default_mic}"
                    )
            except subprocess.CalledProcessError as exc:
                self._node.get_logger().error(
                    f"Error setting default microphone: {exc.output}"
                )

        # Unmute the microphone
        if toggle_control_name is not None:
            try:
                unmute_output = subprocess.check_output(
                    [
                        "amixer",
                        "-c",
                        amixer_card_num,
                        "sset",
                        toggle_control_name,
                        "unmute",
                    ]
                )
                if b"[on]" not in unmute_output:
                    self._node.get_logger().error(
                        f"Microphone remained muted even after unmuting:\n{unmute_output}"
                    )
            except subprocess.CalledProcessError as exc:
                self._node.get_logger().error(
                    f"Error toggling microphone on: {exc.output}"
                )
        else:
            self._node.get_logger().error(
                "toggle_control_name is not set, so the system microphone "
                "cannot be unmuted"
            )

        # Set the microphone volume
        if control_names is not None and control_percentages is not None:
            for i in range(min(len(control_names), len(control_percentages))):
                try:
                    control_name = control_names[i]
                    control_percentage = control_percentages[i]
                    control_output = subprocess.check_output(
                        [
                            "amixer",
                            "-c",
                            amixer_card_num,
                            "sset",
                            control_name,
                            f"{control_percentage}%",
                        ]
                    )
                    if f"[{control_percentage}%]".encode() not in control_output:
                        self._node.get_logger().error(
                            f"Microphone {control_name} volume did not correctly set to "
                            f"{control_percentage}%:\n{control_output}"
                        )
                except subprocess.CalledProcessError as exc:
                    self._node.get_logger().error(
                        f"Error setting microphone volume: {exc.output}"
                    )
        else:
            self._node.get_logger().error(
                "control_names and/or control_percentages "
                "are not set, so the system microphone volume cannot be set"
            )

        # Monitor for the audio device being unplugged
        if is_usb:
            self.udev_context = pyudev.Context()
            self.udev_monitor = pyudev.Monitor.from_netlink(self.udev_context)
            self.udev_monitor.filter_by(subsystem="sound")
            self.udev_observer = pyudev.MonitorObserver(
                self.udev_monitor, self.__udev_listener
            )
            self.udev_observer.start()
        else:
            # Start listening for ACPI events on a separate thread
            self.acpi_thread = Thread(target=self.__acpi_listener, daemon=True)
            self.acpi_thread.start()

    def __init_stream(self) -> None:
        """
        Initialize the audio stream. This function opens the audio stream and
        sets the callback function.
        """
        # Initialize the pyaudio object
        self.audio = pyaudio.PyAudio()

        # Get the device index
        device_i = 0
        device_name = self.audio_configuration["device_name"]
        if device_name is not None:
            device_i = None
            names = []
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    names.append(device_info["name"])
                    if device_name in device_info["name"]:
                        device_i = i
                        break
            if device_i is None:
                self._node.get_logger().error(
                    f"Device name {device_name} not found. Available devices: {repr(names)}. "
                    "Using device index 0."
                )
                device_i = 0

        self.last_stream_init_time = self._node.get_clock().now()
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,  # The e-stop button is mono
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=device_i,
                stream_callback=self.__audio_callback,
            )
        except OSError as exc:
            self._node.get_logger().error(
                (
                    f"Error opening audio device with index {device_i}. "
                    f"{EStopCondition.PYAUDIO_STREAM_TROUBLESHOOTING}\n\n"
                    f"Excpetion: {exc}"
                ),
                throttle_duration_sec=1,
            )

    def __deinit_stream(self) -> None:
        """
        Deinitialize the audio stream. This function stops the audio stream and
        closes it.
        """
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def __udev_listener(self, action: str, device: pyudev.Device) -> None:
        """
        Listens for UDEV events. If the e-stop button is unplugged, sets the
        `is_mic_unplugged` flag to True.
        """
        udev_id = self.audio_configuration["udev_id"]
        device_id = device.get("ID_ID")
        self._node.get_logger().info(f"UDEV event: {action} {device_id}")
        if device_id is not None and udev_id in device_id:
            if action == "remove":
                self._node.get_logger().info("E-Stop button unplugged")
                with self.is_mic_unplugged_lock:
                    self.is_mic_unplugged = True
            # Although there is also an "add" action, that doesn't have the devce ID.
            elif action == "change":
                self._node.get_logger().info("E-Stop button plugged in")
                with self.is_mic_unplugged_lock:
                    self.is_mic_unplugged = False
                self.__deinit_stream()
                self.__init_stream()

    def __acpi_listener(self) -> None:
        """
        Listens for ACPI events. If the e-stop button is unplugged, sets the
        `is_mic_unplugged` flag to True.
        """
        chunk = 4096
        plug_keyword = "plug"
        unplug_keyword = "unplug"
        acpi_event_name = self.audio_configuration["acpi_event_name"]

        # Open a socket to listen for ACPI events
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stream.connect("/var/run/acpid.socket")

        # Listen for ACPI events
        while rclpy.ok():
            data_bytes = stream.recv(chunk)
            data = data_bytes.decode("utf-8")
            for event in data.split("\n"):
                if acpi_event_name in event:
                    keyword = event.split(" ")[-1]
                    if unplug_keyword == keyword:
                        self._node.get_logger().info("E-Stop button unplugged")
                        with self.is_mic_unplugged_lock:
                            self.is_mic_unplugged = True
                    elif plug_keyword == keyword:
                        self._node.get_logger().info("E-Stop button plugged in")
                        with self.is_mic_unplugged_lock:
                            self.is_mic_unplugged = False

    @staticmethod
    def rising_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: Optional[npt.NDArray],
        threshold: Union[int, float],
    ) -> bool:
        """
        Detects whether there is a rising edge in `curr_data_arr` that exceeds
        `threshold`. In other words, this function returns True if there is a
        point in `curr_data_arr` that is greater than `threshold` and the previous
        point is less than `threshold`.

        Although this method of detecting a rising edge is suceptible to noise
        (since it only requires two points to determine an edge), in practice
        the e-stop button's signal has little noise. If noise is an issue
        moving forward, we can add a filter to smoothen the signal, and then
        continue using this detector.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a rising edge

        Returns
        -------
        is_rising_edge: bool
            True if a rising edge was detected, False otherwise
        """
        is_above_threshold = curr_data_arr > threshold
        if np.any(is_above_threshold):
            first_index_above_threshold = np.argmax(is_above_threshold)
            # Get the previous value
            if first_index_above_threshold == 0:
                if prev_data_arr is None:
                    # If the first datapoint is above the threshold, it's not a
                    # rising edge
                    return False
                prev_value = prev_data_arr[-1]
            else:
                prev_value = curr_data_arr[first_index_above_threshold - 1]
            # If the previous value is less than the threshold, it is a rising edge
            return prev_value < threshold
        # If no point is above the threshold, there is no rising edge
        return False

    @staticmethod
    def falling_edge_detector(
        curr_data_arr: npt.NDArray,
        prev_data_arr: Optional[npt.NDArray],
        threshold: Union[int, float],
    ) -> bool:
        """
        Detects whether there is a falling edge in `curr_data_arr` that exceeds
        `threshold`. In other words, this function returns True if there is a
        point in `curr_data_arr` that is less than `threshold` and the previous
        point is greater than `threshold`.

        Parameters
        ----------
        curr_data_arr: npt.NDArray
            The current data array
        prev_data_arr: Optional[npt.NDArray]
            The previous data array
        threshold: Union[int, float]
            The threshold that the data must cross to be considered a falling edge

        Returns
        -------
        is_falling_edge: bool
            True if a falling edge was detected, False otherwise
        """
        # Flip all signs and call the rising edge detector
        return EStopCondition.rising_edge_detector(
            -curr_data_arr,
            None if prev_data_arr is None else -prev_data_arr,
            -threshold,
        )

    # pylint: disable=unused-argument
    # The audio callback function must have this signature
    def __audio_callback(
        self, data: bytes, frame_count: int, time_info: dict, status: int
    ) -> Tuple[bytes, int]:
        """
        Callback function for the audio stream. This function is called whenever
        the audio stream has new data. This function checks if the e-stop button
        has been pressed, and if so, increments the number of clicks.

        This function detects a button press if the signal has either a rising
        edge that goes above `self.max_threshold` or a falling edge that goes
        below `self.min_threshold`. If the signal crosses the threshold multiple
        times within the same `self.time_per_click_duration`, it is considered
        part of the same button click. A rising edge is defined as two consecutive
        points that increase, and a falling edge is defined as two consecutive
        points that decrease.

        Parameters
        ----------
        data: the audio data, as a byte string
        frame_count: the number of frames in the data
        time_info: the time info
        status: the status
        """
        # Update the time it has received data
        with self.last_recv_data_time_lock:
            self.last_recv_data_time = self._node.get_clock().now()

        # Update the previous data array
        self.prev_data_arr = self.curr_data_arr

        # Skip the first few seconds of data, to avoid initial noise
        if self.start_time is None:
            self.start_time = self._node.get_clock().now()
        if self._node.get_clock().now() - self.start_time < self.initial_wait_duration:
            return (data, pyaudio.paContinue)

        # Convert the data to a numpy array
        self.curr_data_arr = np.frombuffer(data, dtype=np.int16)

        # Check if the e-stop button has been pressed
        if EStopCondition.rising_edge_detector(
            self.curr_data_arr,
            self.prev_data_arr,
            self.max_threshold,
        ) or EStopCondition.falling_edge_detector(
            self.curr_data_arr,
            self.prev_data_arr,
            self.min_threshold,
        ):
            # If it has been more than `self.time_per_click_duration`, since the
            # last button click, it is a new button click
            if (self.prev_button_click_start_time is None) or (
                (self._node.get_clock().now() - self.prev_button_click_start_time)
                > self.time_per_click_duration
            ):
                self.prev_button_click_start_time = self._node.get_clock().now()
                with self.num_clicks_lock:
                    self._node.get_logger().info("E-Stop button clicked")
                    self.num_clicks += 1

        # Return the data
        with self.std_ema_lock:
            curr_data_arr_std = np.std(self.curr_data_arr)
            if self.std_ema is None:
                self.std_ema = curr_data_arr_std
            else:
                self.std_ema = self.std_ema * (
                    self.std_ema_alpha
                ) + curr_data_arr_std * (1.0 - self.std_ema_alpha)
        return (data, pyaudio.paContinue)

    def check_startup(self) -> List[Tuple[bool, str, str]]:
        """
        Check if the e-stop button is plugged in. To do so, it checks whether
        the e-stop button has been clicked at least once.

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
        name_1 = "Startup: Pyaudio Stream Started"
        # Attempt to start the stream
        if self.stream is None:
            # Retry opening the stream at the user-specified rate.
            if (
                self._node.get_clock().now() - self.last_stream_init_time
                > self.stream_open_retry_duration
            ):
                self.__init_stream()
        status_1 = self.stream is not None
        condition_1 = f"Pyaudio stream has {'' if status_1 else 'not '}been started. "
        if not status_1:  # Add troubleshooting information
            condition_1 += EStopCondition.PYAUDIO_STREAM_TROUBLESHOOTING

        # Verify that the e-stop button has been clicked at least once
        name_2 = "Startup: E-Stop Button Clicked"
        with self.num_clicks_lock:
            status_2 = self.num_clicks > 0
        condition_2 = "E-stop button must be clicked at least once"

        # Print the std_ema, which can help with debugging
        if not (status_1 and status_2):
            with self.std_ema_lock:
                self._node.get_logger().warning(
                    f"At least one e-stop startup condition failed. std_ema {self.std_ema}",
                    throttle_duration_sec=1,
                )

        return [(status_1, name_1, condition_1), (status_2, name_2, condition_2)]

    def check_status(self) -> List[Tuple[bool, str, str]]:
        """
        Check if the e-stop button has been clicked again. If so, this condition
        fails.

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
        name_1 = "E-Stop Button Not Clicked"
        with self.num_clicks_lock:
            status_1 = self.num_clicks < 2
        condition_1 = (
            f"E-stop button has {'not ' if status_1 else ''}"
            "been clicked since the startup click"
        )

        name_2 = "E-Stop Button Plugged In"
        with self.is_mic_unplugged_lock:
            status_2 = not self.is_mic_unplugged
        condition_2 = f"E-stop button has {'not ' if status_2 else ''}been unplugged"

        name_3 = "E-Stop Button Streaming Data"
        with self.last_recv_data_time_lock:
            if self.last_recv_data_time is None:
                status_3 = False
            else:
                status_3 = (
                    self._node.get_clock().now() - self.last_recv_data_time
                ).nanoseconds <= self.num_secs_threshold * 10**9.0
        condition_3 = f"E-stop button has {'' if status_3 else 'not '}sent data within {self.num_secs_threshold} secs"

        name_4 = "E-Stop Standard Deviation"
        with self.std_ema_lock:
            if self.std_ema is None:
                status_4 = False
            else:
                status_4 = (
                    self.std_ema_min_thresh <= self.std_ema
                    and self.std_ema <= self.std_ema_max_thresh
                )
            condition_4 = (
                f"E-stop button's standard deviation {self.std_ema} is {'' if status_4 else 'not '}"
                f"within expected range [{self.std_ema_min_thresh}, {self.std_ema_max_thresh}]"
            )

        # Print the std_ema, which can help with debugging
        if not (status_1 and status_2 and status_3 and status_4):
            with self.std_ema_lock:
                self._node.get_logger().warning(
                    f"At least one e-stop condition failed. std_ema {self.std_ema}",
                    throttle_duration_sec=1,
                )

        return [
            (status_1, name_1, condition_1),
            (status_2, name_2, condition_2),
            (status_3, name_3, condition_3),
            (status_4, name_4, condition_4),
        ]

    def terminate(self) -> None:
        """
        Terminate the EStop condition. This cleanly closes the pyaudio connection.
        """
        self.__deinit_stream()
