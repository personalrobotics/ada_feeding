#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains EStopCondition, a watchdog condition that listens for the
e-stop button either being clicked or being unplugged and fails if so.
"""

# Standard imports
import socket
from threading import Lock, Thread
from typing import List, Tuple

# Third party imports
import numpy as np
import pyaudio
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

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
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device,
            stream_callback=self.__audio_callback,
        )

    def __load_parameters(self) -> None:
        """
        Load the parameters for this watchdog condition.
        """
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

        device = self._node.declare_parameter(
            "device",
            0,
            ParameterDescriptor(
                name="device",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The index of the audio device to use. This should usually "
                    "be 0, to indicate the default audio device."
                ),
                read_only=True,
            ),
        )
        self.device = device.value

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

        acpi_event_name = self._node.declare_parameter(
            "acpi_event_name",
            "jack/microphone MICROPHONE",
            ParameterDescriptor(
                name="acpi_event_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The name of the ACPI event that is triggered when the e-stop "
                    "button is (un)plugged. Get this value by running `acpi_listener` "
                    "in the terminal and unplugging the e-stop button. Exclude the word "
                    "` plug` or ` unplug` from the end of the event name."
                ),
                read_only=True,
            ),
        )
        self.acpi_event_name = acpi_event_name.value

    def __acpi_listener(self) -> None:
        """
        Listens for ACPI events. If the e-stop button is unplugged, sets the
        `is_mic_unplugged` flag to True.
        """
        CHUNK = 4096
        PLUG_KEYWORD = "plug"
        UNPLUG_KEYWORD = "unplug"

        # Open a socket to listen for ACPI events
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stream.connect("/var/run/acpid.socket")

        # Listen for ACPI events
        while rclpy.ok():
            data_bytes = stream.recv(CHUNK)
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

    # pylint: disable=unused-argument
    # The audio callback function must have this signature
    def __audio_callback(
        self, data: bytes, frame_count: int, time_info: dict, status: int
    ) -> Tuple[bytes, int]:
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
        MIN_THRESHOLD = -10000
        MAX_THRESHOLD = 10000

        # Skip the first few seconds of data, to avoid initial noise
        if self.start_time is None:
            self.start_time = self._node.get_clock().now()
        if self._node.get_clock().now() - self.start_time < self.initial_wait_duration:
            return (data, pyaudio.paContinue)

        # Convert the data to a numpy array
        data_arr = np.frombuffer(data, dtype=np.int16)

        # Check if the e-stop button has been pressed
        if np.any(np.logical_or(data_arr < MIN_THRESHOLD, data_arr > MAX_THRESHOLD)):
            with self.num_clicks_lock:
                self.num_clicks += 1

        # Return the data
        return (data, pyaudio.paContinue)

    def check_startup(self) -> List[Tuple[bool, str]]:
        """
        Check if the e-stop button is plugged in. To do so, it checks whether
        the e-stop button has been clicked at least once.

        Returns
        -------
        startup_status: A list of tuples, where each tuple contains a boolean
            status of a startup condition and a string describing the condition.
            All conditions must be True for the startup condition to be considered
            passed. For example, [(False, "Has received at least one message on
            topic X")] means that the startup condition has not passed because
            the node has not received any messages on topic X yet.
        """
        condition_1 = "E-stop button has been clicked at least once"
        with self.num_clicks_lock:
            status_1 = self.num_clicks > 0

        return [(status_1, condition_1)]

    def check_status(self) -> List[Tuple[bool, str]]:
        """
        Check if the e-stop button has been clicked again. If so, this condition
        fails.

        Returns
        -------
        status: A list of tuples, where each tuple contains a boolean status
            of a condition and a string describing the condition. All conditions
            must be True for the status to be considered True. For example,
            [(True, "Has received a message on topic X within the last Y secs"),
            (False, "Messages on topic X over the last Y secs have non-zero variance")]
            means that the status is False and the watchdog should fail.
        """
        condition_1 = "E-stop button has not been clicked more than once"
        with self.num_clicks_lock:
            status_1 = self.num_clicks < 2

        condition_2 = "E-stop button has not been unplugged"
        with self.is_mic_unplugged_lock:
            status_2 = not self.is_mic_unplugged

        return [(status_1, condition_1), (status_2, condition_2)]
