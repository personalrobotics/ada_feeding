#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains WatchdogCondition, an abstract class that every watchdog
condition must inherit from.
"""

# Standard imports
from abc import ABC, abstractmethod
from typing import List, Tuple


class WatchdogCondition(ABC):
    """
    An abstract class that every watchdog condition must inherit from.

    Every watchdog condition has a startup condition, before which its status
    should not be checked. After the startup condition has passed, the watchdog
    should then subscribe to the condition's status, to determine whether it
    should continue or not.

    Note that each watchdog condition is responsible for initializing its own
    ROS2 subscriptions and parameters.
    """

    @abstractmethod
    def check_startup(self) -> List[Tuple[bool, str, str]]:
        """
        Check whether the startup condition is True. Before this condition is
        True, the watchdog should not call `check_status`. After this condition
        returns True once, the watchdog should call `check_status` to determine
        whether it should continue or not.

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
        raise NotImplementedError("check_startup not implemented")

    @abstractmethod
    def check_status(self) -> List[Tuple[bool, str]]:
        """
        Check whether the status of the condition is True. If this condition
        returns False, the watchdog should call `check_status` again after some
        time. If this condition returns True, the watchdog should continue
        running.

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
        raise NotImplementedError("check_status not implemented")

    @abstractmethod
    def terminate(self) -> None:
        """
        Terminate the watchdog condition. This is called when the watchdog
        terminates.
        """
        raise NotImplementedError("terminate not implemented")
