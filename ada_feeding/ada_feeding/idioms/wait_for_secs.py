#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the wait_for_secs idiom, which returns a behavior that
waits for a specified number of seconds.
"""

# Standard imports

# Third-party imports
import py_trees


def wait_for_secs(
    name: str,
    secs: float,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behavior that returns RUNNING for a specified number of secs
    and then returns SUCCESS.

    Parameters
    ----------
    name: The name of the behavior.
    secs: The number of seconds to wait.
    """

    # pylint: disable=abstract-class-instantiated
    # Creating a Running behavior is not instantiating an abstract class.

    return py_trees.decorators.FailureIsSuccess(
        name=name,
        child=py_trees.decorators.Timeout(
            name=name + "Timeout",
            duration=secs,
            child=py_trees.behaviours.Running(),
        ),
    )
