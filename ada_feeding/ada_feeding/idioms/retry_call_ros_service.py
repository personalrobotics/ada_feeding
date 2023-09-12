#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the retry_call_ros_service idiom, which will call a ROS
service, optionally check the response value, and retry up to a pre-specified
number of times if there is a failure.
"""

# Standard imports
import logging
from typing import Any, List, Optional

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from py_trees_ros.service_clients import FromBlackboard, FromConstant


def retry_call_ros_service(
    name: str,
    service_type: Any,
    service_name: str,
    key_request: Optional[str] = None,
    request: Optional[Any] = None,
    key_response: Optional[str] = None,
    response_checks: Optional[List[py_trees.common.ComparisonExpression]] = None,
    max_retries: int = 3,
    wait_for_server_timeout_sec: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> py_trees.behaviour.Behaviour:
    """
    Creates a behavior that calls a ROS service and optionally checkes the response.
    If there is a failure for any kind (e.g., service is unavailable, doesn't
    return, response does not pass the check, etc.) it retries up to `max_retries`
    times.

    Parameters
    ----------
    name: The name of the behavior.
    service_type: The type of the ROS service.
    service_name: The name of the ROS service.
    key_request: The key for the request in the blackboard. If None, the constant
        value in `request` is used. Note that exactly one of `key_request` and
        `request` must be None.
    request: The request for the ROS service. If None, the value atored in the
        blackboard with key `key_request` will be used. Note that exactly one of
        `key_request` and `request` must be None.
    key_response: The key to write the response to the blackboard. If None, the
        response is not written to the blackboard.
    response_checks: A list of comparison expression to check the response against. If
        None, the response is not checked. Note that if `key_response` is None,
        this must be None.
    max_retries: The maximum number of retries.
    wait_for_server_timeout_sec: The timeout for waiting for the server to be
        available.
    logger: The logger for the tree that this behavior is in.
    """

    # pylint: disable=too-many-arguments, too-many-locals
    # Idioms tend to be hefty, in order to prevent other functions from being hefty.

    # Check the inputs
    if (key_request is None and request is None) or (
        key_request is not None and request is not None
    ):
        raise ValueError("Exactly one of `key_request` and `request` must be None.")
    if key_response is None and response_checks is not None:
        raise ValueError("If `key_response` is None, `response_checks` must be None.")

    # Separate namespaces for child behaviours
    call_ros_service_namespace_prefix = "call_ros_service"
    check_response_namespace_prefix = "check_response"

    # Create the call ROS service behavior
    call_ros_service_behavior_name = Blackboard.separator.join(
        [name, call_ros_service_namespace_prefix]
    )
    call_ros_service_behavior_params = {
        "name": call_ros_service_behavior_name,
        "service_type": service_type,
        "service_name": service_name,
        "key_response": key_response,
        "wait_for_server_timeout_sec": wait_for_server_timeout_sec,
    }
    if key_request is None:
        call_ros_service_behavior = FromConstant(
            **call_ros_service_behavior_params,
            service_request=request,
        )
    else:
        call_ros_service_behavior = FromBlackboard(
            **call_ros_service_behavior_params,
            key_request=key_request,
        )
    if logger is not None:
        call_ros_service_behavior.logger = logger

    # Create the check response behavior
    if response_checks is not None:
        # Add all the response checks
        children = [call_ros_service_behavior]
        for i, response_check in enumerate(response_checks):
            check_response_behavior_name = Blackboard.separator.join(
                [name, check_response_namespace_prefix + str(i)]
            )
            check_response_behavior = py_trees.behaviours.CheckBlackboardVariableValue(
                name=check_response_behavior_name,
                check=response_check,
            )
            if logger is not None:
                check_response_behavior.logger = logger
            children.append(check_response_behavior)

        # Chain the behaviours together in a sequence
        child = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=children,
        )
        if logger is not None:
            child.logger = logger
    else:
        child = call_ros_service_behavior

    # Add a retry decorator on top of the child
    retry_behavior = py_trees.decorators.Retry(
        name=name,
        child=child,
        num_failures=max_retries,
    )
    if logger is not None:
        retry_behavior.logger = logger
    return retry_behavior
