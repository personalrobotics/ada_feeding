#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""
This module contains a node, CreateActionServers, for creating action servers
that wrap behavior trees.
"""

# Standard imports
import collections.abc
import multiprocessing
import os
import threading
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Third-party imports
from ada_watchdog_listener import ADAWatchdogListener
from ament_index_python.packages import get_package_share_directory
import py_trees
from rcl_interfaces.msg import (
    Parameter as ParameterMsg,
    ParameterDescriptor,
    ParameterType,
    ParameterValue,
    SetParametersResult,
)
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.helpers import import_from_string, register_logger
from ada_feeding.visitors import DebugVisitor


class ActionServerParams:
    """
    A class to hold the parameters of an action server that wraps a behavior tree.
    """

    # pylint: disable=too-many-arguments, too-few-public-methods, too-many-instance-attributes
    # This is a data class

    def __init__(
        self,
        server_name: List[str],
        action_type: str,
        tree_class: str,
        tree_kwargs: Optional[Dict[str, Any]] = None,
        tick_rate: int = 30,
    ) -> None:
        """
        Initialize the ActionServerParams class.

        Parameters
        ----------
        server_name: The name of the action server.
        action_type: The type of the action as a str, e.g., ada_feeding_msgs.action.MoveTo.
        tree_class: The class of the behavior tree, e.g., ada_feeding.trees.MoveToConfigurationTree.
        tree_kwargs: The keyword arguments to pass to the behavior tree class.
        tick_rate: The rate at which to tick the behavior tree.
        """
        self.server_name = server_name
        self.action_type = action_type
        self.tree_class = tree_class
        self.tree_kwargs = tree_kwargs
        if tree_kwargs is None:
            tree_kwargs = {}
        self.tick_rate = tick_rate


class CreateActionServers(Node):
    """
    The CreateActionServers node initializes a series of action servers, each
    of which runs a behavior tree (implemented using the py_trees library).
    The mapping between behavior tree files and the parameters of an action
    server are specified in a configuration file.

    This node ensures that only one of the action servers is executing a goal
    at any given time. Consequently, any goal requests received (to any of the
    action servers exposed by this node) while an action server already has a
    goal will be rejected. This is useful e.g., in the case where each action
    servers command a robot to move in a particular way; since the robot can
    only execute one motion at once, all other action servers must reject
    incoming goal requests.
    """

    # pylint: disable=too-many-instance-attributes
    # Fine because of the generic parameters and watchdog capabiltiies.

    # pylint: disable=attribute-defined-outside-init
    # Fine because we are defining these attributes in the `initialize` function.

    DEFAULT_PARAMETER_NAMESPACE = "default"

    def __init__(self) -> None:
        """
        Initialize the CreateActionServers node. This function reads in the
        configuration file and creates the action servers. Note that it does not
        load the behavior tree file associated with each action server; that
        happens when the action server receives a goal request.
        """
        # Although all parameters are decalred, we've noticed an issue where ROS2
        # seems to forget that some nodes were declared. This is a workaround.
        super().__init__("create_action_servers", allow_undeclared_parameters=True)
        register_logger(self.get_logger())

    def initialize(self) -> None:
        """
        Initialize the node. This is a separate function from above so rclpy can
        be spinning while this function is called.
        """
        # Create clients to get and set parameters for the `ada_planning_scene` node.
        # This is necessary to couple custom configurations in this node with custom
        # planning scenes.
        self.planning_scene_get_parameters_client = self.create_client(
            GetParameters,
            "/ada_planning_scene/get_parameters",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.planning_scene_set_parameters_client = self.create_client(
            SetParametersAtomically,
            "/ada_planning_scene/set_parameters_atomically",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Read the parameters that specify what action servers to create.
        self.namespace_to_use = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE
        self.action_server_params = self.read_params()
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Create the watchdog listener. Note that this watchdog listener
        # adds additional parameters -- `watchdog_timeout_sec` and
        # `initial_wait_time_sec` -- and another subscription to `~/watchdog`.
        self.watchdog_listener = ADAWatchdogListener(self)

        # Track the active goal request.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action servers.
        self.create_action_servers(self.action_server_params)

    @staticmethod
    def get_parameter_value(param: Parameter) -> Any:
        """
        Get the value of a parameter, converting a sequence to a list.

        Parameters
        ----------
        param: The parameter to get the value of.
        """
        param_value = param.value
        if not isinstance(param_value, str) and isinstance(
            param_value, collections.abc.Sequence
        ):
            param_value = list(param_value)
        return param_value

    def read_params(self) -> Tuple[Parameter, Parameter, Dict[str, ActionServerParams]]:
        """
        Read the parameters that specify what action servers to create.

        Returns
        -------
        action_server_params: A dict mapping server names to ActionServerParams objects.
        """
        # pylint: disable=too-many-locals
        # Okay because we are providing a lot of generic capabilities through parameters

        default_namespace = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE

        # Read the server names
        server_names = self.declare_parameter(
            f"{default_namespace}.server_names",
            descriptor=ParameterDescriptor(
                name="server_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of action server names to create. "
                    "All names must correspond to their own param namespace within this node."
                ),
                read_only=True,
            ),
        )

        # Read the custom parameter namespaces
        custom_namespaces = self.declare_parameter(
            "custom_namespaces",
            descriptor=ParameterDescriptor(
                name="custom_namespaces",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of custom parameter namespaces. Each one can have its "
                    "own overrides of default parameters."
                ),
                read_only=False,
            ),
        )
        custom_namespaces = (
            []
            if custom_namespaces.value is None
            else [
                namespace for namespace in custom_namespaces.value if len(namespace) > 0
            ]
        )
        self.parameters = {namespace: {} for namespace in custom_namespaces}
        self.parameters[default_namespace] = {}
        namespace_to_use = self.declare_parameter(
            "namespace_to_use",
            value=default_namespace,
            descriptor=ParameterDescriptor(
                name="namespace_to_use",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The custom parameter namespace to use. Must be "
                    f'"{default_namespace}" or one of the namespaces in '
                    "`custom_namespaces`."
                ),
                read_only=False,
            ),
        )
        self.set_namespace_to_use(namespace_to_use.value)

        # Get the planning scene namespace to use
        for namespace in [default_namespace] + custom_namespaces:
            planning_scene_namespace_to_use = self.declare_parameter(
                f"{namespace}.planning_scene_namespace_to_use",
                value="seated",
                descriptor=ParameterDescriptor(
                    name="planning_scene_namespace_to_use",
                    type=ParameterType.PARAMETER_STRING,
                    description=(
                        "The planning scene namespace to use. Must be one of the "
                        "parameters i the ada_planning_scene config file."
                    ),
                    read_only=False,
                ),
            )
            planning_scene_namespace_to_use = planning_scene_namespace_to_use.value
            self.parameters[namespace][
                "planning_scene_namespace_to_use"
            ] = planning_scene_namespace_to_use
            if namespace == self.namespace_to_use:
                self.set_planning_scene_namespace_to_use(
                    planning_scene_namespace_to_use
                )

        # Read each action server's params
        action_server_params = {}
        for server_name in server_names.value:
            # Get the action server's params
            action_type, tree_class, tick_rate = self.declare_parameters(
                "",
                [
                    (
                        f"{default_namespace}.{server_name}.action_type",
                        None,
                        ParameterDescriptor(
                            name="action_type",
                            type=ParameterType.PARAMETER_STRING,
                            description=(
                                "The type of action server to create. "
                                "E.g., ada_feeding_msgs.action.MoveTo"
                            ),
                            read_only=True,
                        ),
                    ),
                    (
                        f"{default_namespace}.{server_name}.tree_class",
                        None,
                        ParameterDescriptor(
                            name="tree_class",
                            type=ParameterType.PARAMETER_STRING,
                            description=(
                                "The class of the behavior tree to run, must "
                                "subclass ActionServerBT. E.g., ada_feeding.behaviors.MoveTo"
                            ),
                            read_only=True,
                        ),
                    ),
                    (
                        f"{default_namespace}.{server_name}.tick_rate",
                        30,
                        ParameterDescriptor(
                            name="tick_rate",
                            type=ParameterType.PARAMETER_INTEGER,
                            description=(
                                "The rate at which the behavior tree should be "
                                "ticked, in Hz."
                            ),
                            read_only=True,
                        ),
                    ),
                ],
            )
            if action_type.value is None or tree_class.value is None:
                self.get_logger().warn(
                    f"Skipping action server {server_name} "
                    "because it has no action type or tree class"
                )
                continue
            tree_kws = self.declare_parameter(
                f"{default_namespace}.{server_name}.tree_kws",
                descriptor=ParameterDescriptor(
                    name="tree_kws",
                    type=ParameterType.PARAMETER_STRING_ARRAY,
                    description=(
                        "List of keywords for custom arguments to be passed "
                        "to the behavior tree during initialization."
                    ),
                    read_only=True,
                ),
            )
            tree_kwargs = {}
            if tree_kws.value is not None:
                for kw in tree_kws.value:
                    full_name = f"{server_name}.tree_kwargs.{kw}"
                    # Get the default value
                    default_value = self.declare_parameter_in_namespace(
                        namespace=default_namespace,
                        full_name=full_name,
                    )
                    default_value = CreateActionServers.get_parameter_value(
                        default_value
                    )
                    if default_value is None:
                        self.get_logger().error(
                            f"tree_kwarg {full_name} must have a non-None value. "
                            "Skipping this tree_kwarg."
                        )
                        continue
                    self.parameters[default_namespace][full_name] = default_value
                    # Get the custom value(s)
                    for namespace in custom_namespaces:
                        custom_value = self.declare_parameter_in_namespace(
                            namespace=namespace,
                            full_name=full_name,
                        )
                        self.parameters[namespace][
                            full_name
                        ] = CreateActionServers.get_parameter_value(custom_value)
                    if self.parameters[self.namespace_to_use][full_name] is not None:
                        tree_kwargs[kw] = self.parameters[self.namespace_to_use][
                            full_name
                        ]
                    else:
                        tree_kwargs[kw] = self.parameters[default_namespace][full_name]

            action_server_params[server_name] = ActionServerParams(
                server_name=server_name,
                action_type=action_type.value,
                tree_class=tree_class.value,
                tree_kwargs=tree_kwargs,
                tick_rate=tick_rate.value,
            )

        return action_server_params

    def declare_namespace_parameters(self, namespace: str) -> None:
        """
        For a given namespace, declares all the parameters in `default` within
        that namespace.

        Parameters
        ----------
        namespace: The namespace to declare parameters for.
        """
        default_namespace = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE

        if namespace not in self.parameters.keys():
            self.parameters[namespace] = {}
        for full_name in self.parameters[default_namespace].keys():
            custom_value = self.declare_parameter_in_namespace(
                namespace=namespace,
                full_name=full_name,
            )
            self.parameters[namespace][
                full_name
            ] = CreateActionServers.get_parameter_value(custom_value)

    def set_namespace_to_use(
        self, namespace_to_use: str, create_if_not_exist: bool = False
    ) -> None:
        """
        Sets the parameter namespace to use for the node.

        Parameters
        ----------
        namespace_to_use: The namespace to use.
        create_if_not_exist: If True, create the namespace if it doesn't exist.
            Else, set the namespace to default if it doesn't exist. Requires
            all default parameters to already be declared.
        """
        default_namespace = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE
        if namespace_to_use in self.parameters.keys():
            self.namespace_to_use = namespace_to_use
        else:
            if create_if_not_exist:
                self.declare_namespace_parameters(namespace_to_use)
                self.namespace_to_use = namespace_to_use
            else:
                self.get_logger().warn(
                    f"Unknown namespace {namespace_to_use}, using {default_namespace}"
                )
                self.namespace_to_use = default_namespace

    def set_planning_scene_namespace_to_use(
        self,
        planning_scene_namespace_to_use: str,
        timeout_secs: float = 10.0,
        rate_hz: float = 10.0,
        reinit_same_namespace: bool = True,
    ) -> bool:
        """
        Sets the planning scene namespace to use for the node. Further, it gets the
        current parameter value for the planning scene node and, if it is different,
        updates the parameter value for that node.

        Parameters
        ----------
        planning_scene_namespace_to_use: The planning scene namespace to use.
        timeout_secs: The timeout in seconds for the service calls.
        rate_hz: The rate at which to check for the parameter value.
        reinit_same_namespace: If True, reinitialize the planning scene node if the
            namespace is the same as the current one. This is useful e.g., to update the
            workspace walls with the new configurations. default: true.

        Returns
        -------
        True if the parameter was set successfully, False otherwise.
        """
        start_time = self.get_clock().now()
        timeout = rclpy.time.Duration(seconds=timeout_secs)
        rate = self.create_rate(rate_hz)

        if not reinit_same_namespace:
            # Wait for the service to be ready
            self.planning_scene_set_parameters_client.wait_for_service(
                (self.get_clock().now() - start_time).nanoseconds / 1.0e9
            )

            # First, get the current value of the parameter
            request = GetParameters.Request()
            request.names = ["namespace_to_use"]
            future = self.planning_scene_get_parameters_client.call_async(request)
            while (
                rclpy.ok()
                and not future.done()
                and (self.get_clock().now() - start_time < timeout)
            ):
                rate.sleep()
            curr_planning_scene_namespace_to_use = None
            if future.done():
                response = future.result()
                if len(response.values) > 0:
                    if response.values[0].type == ParameterType.PARAMETER_STRING:
                        curr_planning_scene_namespace_to_use = response.values[
                            0
                        ].string_value
            if curr_planning_scene_namespace_to_use is None:
                self.get_logger().warn(
                    "Failed to get parameters from ada_planning_scene."
                )
                return False

            # If the parameter is the same, return
            if curr_planning_scene_namespace_to_use == planning_scene_namespace_to_use:
                return True

        # Wait for the service to be ready
        self.planning_scene_set_parameters_client.wait_for_service(
            (self.get_clock().now() - start_time).nanoseconds / 1.0e9
        )

        # Otherwise, set the parameter
        request = SetParametersAtomically.Request()
        request.parameters = [
            ParameterMsg(
                name="namespace_to_use",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_STRING,
                    string_value=planning_scene_namespace_to_use,
                ),
            )
        ]
        future = self.planning_scene_set_parameters_client.call_async(request)
        while (
            rclpy.ok()
            and not future.done()
            and (self.get_clock().now() - start_time < timeout)
        ):
            rate.sleep()
        if future.done():
            response = future.result()
            if response.result.successful:
                self.get_logger().info(
                    f"Successfully set planning scene namespace to {planning_scene_namespace_to_use}"
                )
                return True
        self.get_logger().warn(
            f"Failed to set planning scene namespace to {planning_scene_namespace_to_use}. "
            f"Elapsed time: {(self.get_clock().now() - start_time).nanoseconds / 1.0e9} seconds."
        )
        return False

    def declare_parameter_in_namespace(
        self, namespace: str, full_name: str
    ) -> Parameter:
        """
        This method declares a dynamically-typed parameter in the given namespace.

        Parameters
        ----------
        namespace: The namespace to declare the parameter in.
        full_name: The full name of the parameter, e.g., "{server_name}.tree_kwargs.{kw}".

        Returns
        -------
        The declared parameter.
        """
        read_only = namespace == CreateActionServers.DEFAULT_PARAMETER_NAMESPACE
        name_within_namespace = f"{namespace}.{full_name}"
        self.get_logger().debug(f"Declaring parameter {name_within_namespace}")
        try:
            return self.declare_parameter(
                name_within_namespace,
                descriptor=ParameterDescriptor(
                    name=name_within_namespace,
                    description="Custom parameter for the behavior tree.",
                    dynamic_typing=True,
                    read_only=read_only,
                ),
            )
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            # Sometimes, even if we terminate and reset the node, ROS may remember
            # the previously-declared parameter.
            self.get_logger().warn(
                f"Tried to declare parameter {name_within_namespace}, which was already declared."
            )
            return self.get_parameter(name_within_namespace)

    def parameter_callback(self, params: List[Parameter]) -> SetParametersResult:
        """
        Callback function for when a parameter is changed. Note that in practice,
        only tree_kwargs are not read-only, so we only expect those to be changed.

        Note that we only return failure if there is a type mismatch. That is just
        in case some other code in this file (e.g., the WatchdogListener) needs
        to process the parameter change. This is because rclpy runs all parameter
        callbacks in sequence until one returns failure.
        """
        # pylint: disable=too-many-branches, too-many-statements
        # Necessary for flexible checking of parameters

        self.get_logger().info(
            f"Processing update for parameters {[param.name for param in params]}"
        )

        default_namespace = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE
        updated_parameters = False
        updated_server_names = set()
        for param in params:
            # If we are declaring but not setting a parameter, ignore it
            if param.value is None:
                continue
            # Change the namespace_to_use
            if param.name == "namespace_to_use":
                if len(param.value) == 0:
                    self.get_logger().warn(
                        "namespace_to_use cannot be empty. Skipping this parameter."
                    )
                    continue
                self.set_namespace_to_use(param.value, create_if_not_exist=True)
                updated_parameters = True
                # If this namespace has custom parameters, switch to them
                for full_name, value in self.parameters[self.namespace_to_use].items():
                    # Handle non tree_kwargs
                    if full_name == "planning_scene_namespace_to_use":
                        if value is None:
                            self.get_logger().info(
                                f"Namespace {param.value} has no parameter `{full_name}`. "
                                f"Resorting to default value {self.parameters[default_namespace][full_name]}."
                            )
                            value = self.parameters[default_namespace][full_name]
                        self.set_planning_scene_namespace_to_use(value)

                    # Handle tree_kwargs
                    if "tree_kwargs" not in full_name:
                        self.get_logger().warn(
                            f"Non tree_kwarg parameter {full_name} in self.parameters. "
                            "This should not happen."
                        )
                        continue
                    server_name, _, kw = full_name.split(".")
                    if server_name not in self.action_server_params:
                        self.get_logger().warn(
                            f"Unknown server name {server_name} in self.parameters. "
                            "This should not happen."
                        )
                        continue
                    # If the parameter has a value, update the tree_kwargs
                    action_server_params = self.action_server_params[server_name]
                    if value is not None:
                        action_server_params.tree_kwargs[kw] = value
                    # Else, set it to default values
                    else:
                        action_server_params.tree_kwargs[kw] = self.parameters[
                            default_namespace
                        ][full_name]
                # Update all server names when the namespace changes
                updated_server_names.update(self.action_server_params.keys())
                continue

            # Change the custom namespaces
            if param.name == "custom_namespaces":
                custom_namespaces = param.value
                if custom_namespaces is None:
                    custom_namespaces = []
                if (
                    self.namespace_to_use != default_namespace
                    and self.namespace_to_use not in custom_namespaces
                ):
                    self.get_logger().warn(
                        f"Currently used namespace {self.namespace_to_use} is not in custom_namespaces. "
                        "This may cause issues."
                    )
                for namespace in custom_namespaces:
                    if len(namespace) == 0:
                        self.get_logger().warn(
                            "custom_namespaces cannot have empty strings. Skipping this namespace."
                        )
                        continue
                    if namespace == default_namespace:
                        self.get_logger().warn(
                            f"Namespace {default_namespace} is reserved and cannot "
                            "be used as a custom namespace. Skipping this namespace."
                        )
                        continue
                    if namespace not in self.parameters.keys():
                        self.declare_namespace_parameters(namespace)
                updated_parameters = True
                continue

            # Change the planning_scene_namespace_to_use
            if "planning_scene_namespace_to_use" in param.name:
                # Deconstruct the parameter name
                namespace, full_name = param.name.split(".")
                # Verify it is a valid parameter name
                if namespace not in self.parameters.keys():
                    self.get_logger().warn(
                        f"Unknown namespace {namespace} for parameter {param.name}. "
                        "Skipping this parameter."
                    )
                    continue
                if full_name != "planning_scene_namespace_to_use":
                    self.get_logger().warn(
                        f"Unknown parameter {param.name}. Skipping this parameter."
                    )
                    continue
                param_value = CreateActionServers.get_parameter_value(param)
                if not isinstance(
                    param_value, type(self.parameters[default_namespace][full_name])
                ):
                    self.get_logger().warn(
                        f"Parameter {param.name} must be of type "
                        f"{type(self.parameters[default_namespace][full_name])} "
                        f"but is of type {type(param_value)}"
                    )
                    return SetParametersResult(successful=False, reason="type mismatch")
                # Change the parameter
                self.parameters[namespace][full_name] = param_value
                updated_parameters = True
                # If this is the namespace we're using, update the planning_scene_namespace_to_use
                if namespace == self.namespace_to_use:
                    self.set_planning_scene_namespace_to_use(param_value)
                continue

            # Change a tree_kwarg
            if "tree_kwargs" not in param.name:
                self.get_logger().debug(
                    f"Update for parameter {param.name} cannot be handled by "
                    "CreateActionServers's parameter_callback."
                )
                continue
            # Deconstruct the parameter name
            names = param.name.split(".")
            namespace, server_name, _, kw = names
            full_name = ".".join(names[1:])
            # Verify it is a valid parameter name
            if namespace not in self.parameters.keys():
                self.get_logger().warn(
                    f"Unknown namespace {namespace} for parameter {param.name}. "
                    "Skipping this parameter."
                )
                continue
            if full_name not in self.parameters[namespace]:
                self.get_logger().warn(
                    f"Unknown parameter {param.name}. Skipping this parameter."
                )
                continue
            if server_name not in self.action_server_params:
                self.get_logger().warn(
                    f"Unknown server name {server_name} for parameter {param.name}. "
                    "Skipping this parameter."
                )
                continue
            param_value = CreateActionServers.get_parameter_value(param)
            if not isinstance(
                param_value, type(self.parameters[default_namespace][full_name])
            ):
                self.get_logger().warn(
                    f"Parameter {param.name} must be of type "
                    f"{type(self.parameters[default_namespace][full_name])} "
                    f"but is of type {type(param_value)}"
                )
                return SetParametersResult(successful=False, reason="type mismatch")
            # Change the parameter
            self.parameters[namespace][full_name] = param_value
            updated_parameters = True
            # If this is the namespace we're using, update the tree_kwargs
            if namespace == self.namespace_to_use:
                action_server_params = self.action_server_params[server_name]
                action_server_params.tree_kwargs[kw] = param_value
                updated_server_names.add(server_name)

        # Update the action servers
        if len(updated_server_names) > 0:
            self.get_logger().info(
                f"Restarting action servers {updated_server_names} due to parameter "
                "update."
            )
            # Re-create the trees with updated kwargs
            for server_name in updated_server_names:
                action_server_params = self.action_server_params[server_name]
                self.create_tree(
                    server_name,
                    action_server_params.tree_class,
                    action_server_params.tree_kwargs,
                )
        # Save the updated parameters
        if updated_parameters:
            self.save_custom_parameters()
        return SetParametersResult(successful=True)

    def save_custom_parameters(self) -> None:
        """
        Overrides `ada_feeding_action_servers_custom.yaml` with the non-None
        custom parameters in the non-default namespaces of `self.parameters`.
        """
        default_namespace = CreateActionServers.DEFAULT_PARAMETER_NAMESPACE
        # Convert the parameters to a dictionary of the right form
        params = {}
        params["namespace_to_use"] = self.namespace_to_use
        custom_namespaces = [
            namespace
            for namespace in self.parameters.keys()
            if namespace != default_namespace
        ]
        params["custom_namespaces"] = custom_namespaces
        for namespace in custom_namespaces:
            params[namespace] = {}
            for full_name, value in self.parameters[namespace].items():
                if value is None:
                    continue
                params[namespace][full_name] = value
            if len(params[namespace]) == 0:
                del params[namespace]
        data = {"ada_feeding_action_servers": {"ros__parameters": params}}

        # Write to yaml
        package_path = get_package_share_directory("ada_feeding")
        file_path = os.path.join(
            package_path, "config", "ada_feeding_action_servers_custom.yaml"
        )
        self.get_logger().debug(f"Writing to {file_path} with data {data}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write("# This file is auto-generated by create_action_servers.py\n")
            yaml.dump(data, file)

    def create_action_servers(
        self, action_server_params: Dict[str, ActionServerParams]
    ) -> None:
        """
        Create the action servers specified in the configuration file.

        Parameters
        ----------
        action_server_params: A dict mapping server names to ActionServerParams objects.
        """
        # For every action server specified in the configuration file, create
        # an action server.
        self._action_servers = []
        self._action_types = {}
        self._tree_classes = {}
        self._trees = {}
        self._tree_action_servers = {}
        for params in action_server_params.values():
            self.get_logger().info(
                f"Creating action server {params.server_name} with type {params.action_type}"
            )

            # Import the action type
            self._action_types[params.action_type] = import_from_string(
                params.action_type
            )

            # Import the behavior tree class
            self._tree_classes[params.tree_class] = import_from_string(
                params.tree_class
            )
            try:
                assert issubclass(
                    self._tree_classes[params.tree_class], ActionServerBT
                ), f"Tree {params.tree_class} must subclass ActionServerBT"
            except AssertionError:
                self.get_logger().warn(
                    f"{traceback.format_exc()}\nSKIPPING THIS ACTION SERVER"
                )

            # Create the action server.
            # Note: remapping action names does not work: https://github.com/ros2/ros2/issues/1312
            action_server = ActionServer(
                self,
                self._action_types[params.action_type],
                params.server_name,
                self.get_execute_callback(
                    params.server_name,
                    self._action_types[params.action_type],
                    params.tree_class,
                    params.tree_kwargs,
                    params.tick_rate,
                ),
                goal_callback=self.get_goal_callback(params.server_name),
                cancel_callback=self.get_cancel_callback(params.server_name),
            )
            self._action_servers.append(action_server)

    def get_goal_callback(self, server_name: str) -> Callable[[object], GoalResponse]:
        """
        Returns a callback function that accepts or rejects goal requests based
        on whether the action server already has an active goal.

        Parameters
        ----------
        server_name: The name of the action server.

        Returns
        -------
        goal_callback: The callback function for the action server.
        """

        def goal_callback(goal_request: object) -> GoalResponse:
            """
            Accept a goal if this action does not already have an active goal,
            else reject.

            Parameters
            ----------
            goal_request: The goal request message.
            """
            self.get_logger().info(f"Received goal request for {server_name}")

            # If we don't already have an active goal_request, accept this one
            with self.active_goal_request_lock:
                if self.watchdog_listener.ok() and self.active_goal_request is None:
                    self.get_logger().info(f"Accepting goal request for {server_name}")
                    self.active_goal_request = goal_request
                    return GoalResponse.ACCEPT

                # Otherwise, reject this goal request
                self.get_logger().info(f"Rejecting goal request for {server_name}")
                return GoalResponse.REJECT

        return goal_callback

    def get_cancel_callback(
        self, server_name: str
    ) -> Callable[[ServerGoalHandle], CancelResponse]:
        """
        Returns a callback function that accepts or rejects cancel requests based
        on whether the action server already has an active goal.

        Parameters
        ----------
        server_name: The name of the action server.

        Returns
        -------
        cancel_callback: The callback function for the action server.
        """

        def cancel_callback(_: ServerGoalHandle) -> CancelResponse:
            """
            Always accept client requests to cancel the active goal.

            Parameters
            ----------
            goal_handle: The goal handle.
            """
            self.get_logger().info(
                f"Received cancel request for {server_name}, accepting"
            )
            return CancelResponse.ACCEPT

        return cancel_callback

    def create_tree(
        self,
        server_name: str,
        tree_class: str,
        tree_kwargs: Dict,
    ) -> None:
        """
        Creates the tree_action_server and tree objects for the given action server.

        Parameters
        ----------
        server_name: The name of the action server.
        tree_class: The class of the behavior tree, e.g., ada_feeding.trees.MoveToConfigurationTree.
        tree_kwargs: The keyword arguments to pass to the behavior tree class.
        """
        # Initialize the ActionServerBT object once
        tree_action_server = self._tree_classes[tree_class](self, **tree_kwargs)
        self._tree_action_servers[server_name] = tree_action_server
        # Create and setup the tree once
        tree = tree_action_server.create_tree(server_name)
        self.setup_tree(tree)
        self._trees[server_name] = tree

    def setup_tree(self, tree: py_trees.trees.BehaviourTree) -> None:
        """
        Runs the initial setup on a behavior tree after creating it.

        Specifically, this function: (1) sets every behavior's logger
        to be the node's logger; and (2) calls teh tree's `setup` function.

        Parameters
        ----------
        tree: The behavior tree to setup.
        """

        # Set every behavior's logger to be the node's logger
        for node in tree.root.iterate():
            node.logger = self.get_logger()

        # Add a DebugVisitor to catch behavior debug messages
        # Set --log-level create_action_servers:=info (or higher) to quiet.
        tree.visitors.append(DebugVisitor())

        # Call the tree's setup function
        # TODO: consider adding a timeout here
        tree.setup(node=self)

    def get_execute_callback(
        self,
        server_name: str,
        action_type: type,
        tree_class: str,
        tree_kwargs: Dict,
        tick_rate: int,
    ) -> Callable[[ServerGoalHandle], Awaitable]:
        """
        This is a wrapper function that takes in information about the action
        server and returns a callback function that can be used for that server.
        This is necessary because the callback function must return a Result and
        Feedback consisitent with the action type.

        Parameters
        ----------
        server_name: The name of the action server.
        action_type: The type of the action, as a class.
        tree_class: The class of the behavior tree, e.g., ada_feeding.trees.MoveToConfigurationTree.
        tree_kwargs: The keyword arguments to pass to the behavior tree class.
        tick_rate: The rate at which to tick the behavior tree.

        Returns
        -------
        execute_callback: The callback function for the action server.
        """
        # pylint: disable=too-many-arguments
        # This is appropriate
        # pylint: disable=too-many-statements
        # This is function declares the main execution loop.

        self.create_tree(server_name, tree_class, tree_kwargs)

        async def execute_callback(goal_handle: ServerGoalHandle) -> Awaitable:
            """
            This function is called when a goal is accepted by an action server.
            It loads the behavior tree file associated with the action server
            (if not already loaded) and executes the behavior tree, publishing
            periodic feedback.
            """
            # pylint: disable=too-many-statements
            # This is the main execution loop.

            goal_uuid = "".join(format(x, "02x") for x in goal_handle.goal_id.uuid)
            self.get_logger().info(
                f"{server_name}: "
                f"Executing goal {goal_uuid}"
                # f" with request {goal_handle.request}"
            )

            # pylint: disable=broad-exception-caught
            # All exceptions need printing at shutdown
            try:
                # Get the tree and action server
                tree = self._trees[server_name]
                tree_action_server = self._tree_action_servers[server_name]

                # Send the goal to the behavior tree
                tree_action_server.send_goal(tree, goal_handle.request)

                # Execute the behavior tree
                rate = self.create_rate(tick_rate)
                result = None
                try:
                    while rclpy.ok():
                        # Check if the goal has been canceled
                        if goal_handle.is_cancel_requested:
                            # Note that the body of this conditional may be called
                            # multiple times until the preemption is complete.
                            self.get_logger().info("Waiting for tree to preempt")
                            tree_action_server.preempt_goal(
                                tree
                            )  # blocks until the preempt succeeds
                            goal_handle.canceled()
                            self.get_logger().info("Canceled goal.")

                            try:
                                result = tree_action_server.get_result(
                                    tree, action_type
                                )
                            except KeyError:
                                # The tree didn't get far enough to set a result
                                result = action_type.Result()
                            break

                        # Check if the watchdog has failed
                        if not self.watchdog_listener.ok():
                            self.get_logger().warn("Watchdog failed, aborting goal")
                            tree_action_server.preempt_goal(
                                tree
                            )  # blocks until the preempt succeeds
                            goal_handle.abort()
                            try:
                                result = tree_action_server.get_result(
                                    tree, action_type
                                )
                            except KeyError:
                                # The tree didn't get far enough to set a result
                                result = action_type.Result()
                            break

                        # Tick the tree once and publish feedback
                        tree.tick()
                        feedback_msg = tree_action_server.get_feedback(
                            tree, action_type
                        )
                        goal_handle.publish_feedback(feedback_msg)
                        self.get_logger().debug(f"Publishing feedback {feedback_msg}")

                        # Check the tree status
                        if tree.root.status == py_trees.common.Status.SUCCESS:
                            self.get_logger().info("Tree succeeded")
                            goal_handle.succeed()
                            result = tree_action_server.get_result(tree, action_type)
                            break
                        if tree.root.status in set(
                            (
                                py_trees.common.Status.FAILURE,
                                py_trees.common.Status.INVALID,
                            )
                        ):
                            # Get the name of the behavior that the tree failed on
                            names_of_failed_behavior = []
                            for node in tree.root.iterate():
                                if node.status in set(
                                    (
                                        py_trees.common.Status.FAILURE,
                                        py_trees.common.Status.INVALID,
                                    )
                                ):
                                    names_of_failed_behavior.append(node.name)
                                    if node.status == py_trees.common.Status.FAILURE:
                                        break
                            self.get_logger().info(
                                f"Tree failed at behavior {names_of_failed_behavior}"
                            )
                            goal_handle.abort()
                            try:
                                result = tree_action_server.get_result(
                                    tree, action_type
                                )
                            except KeyError:
                                # The tree didn't get far enough to set a result
                                result = action_type.Result()
                            break

                        # Sleep
                        rate.sleep()
                except KeyboardInterrupt:
                    pass

                # If we have gotten here without a result, that means something
                # went wrong. Abort the goal.
                if result is None:
                    goal_handle.abort()
                    result = action_type.Result()

            except Exception as exc:
                self.get_logger().error(
                    f"Error running tree: \n{traceback.format_exc()}\n{exc}"
                )
                goal_handle.abort()
                result = action_type.Result()

            # Unset the goal and return the result
            with self.active_goal_request_lock:
                self.active_goal_request = None
            return result

        return execute_callback

    def shutdown(self) -> None:
        """
        Shutdown the node.
        """
        self.get_logger().info("Shutting down CreateActionServers")
        for tree in self._trees.values():
            # Shutdown the tree
            tree.shutdown()


def main(args: List = None) -> None:
    """
    Create the ROS2 node and run the action servers.
    """
    rclpy.init(args=args)

    create_action_servers = CreateActionServers()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor(num_threads=multiprocessing.cpu_count() * 2)

    # Spin in the background until the node has initialized
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(create_action_servers,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # pylint: disable=broad-exception-caught
    # All exceptions need printing at shutdown
    try:
        # Initialize the node
        create_action_servers.initialize()

        # Spin in the foreground
        spin_thread.join()
    except Exception:
        traceback.print_exc()

    # Destroy the node explicitly
    create_action_servers.shutdown()
    create_action_servers.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
