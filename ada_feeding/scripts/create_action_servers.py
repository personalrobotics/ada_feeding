#!/usr/bin/env python3
"""
This module contains a node, CreateActionServers, for creating action servers
that wrap behavior trees.
"""

# Standard imports
import collections.abc
import os
import threading
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Third-party imports
from ada_watchdog_listener import ADAWatchdogListener
from ament_index_python.packages import get_package_share_directory
import py_trees
from py_trees.visitors import DebugVisitor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.helpers import import_from_string, register_logger


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
    # Nine is fine if we are keeping watchdog functionality within the base Node

    def __init__(self) -> None:
        """
        Initialize the CreateActionServers node. This function reads in the
        configuration file and creates the action servers. Note that it does not
        load the behavior tree file associated with each action server; that
        happens when the action server receives a goal request.
        """
        super().__init__("create_action_servers")
        register_logger(self.get_logger())

        # Read the parameters that specify what action servers to create.
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

    def read_params(self) -> Tuple[Parameter, Parameter, Dict[str, ActionServerParams]]:
        """
        Read the parameters that specify what action servers to create.

        Returns
        -------
        action_server_params: A dict mapping server names to ActionServerParams objects.
        """
        # Read the server names
        server_names = self.declare_parameter(
            "default.server_names",
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

        # Read the parameters that have been changed from their default values
        overridden_parameters = self.declare_parameter(
            "current.overridden_parameters",
            descriptor=ParameterDescriptor(
                name="overridden_parameters",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "List of parameters that have been changed from their default values. "
                    "These parameters must be set in the `current` param namespace."
                ),
                read_only=True,
            ),
        )
        overridden_parameters = overridden_parameters.value
        self.overridden_parameters = {}
        self.default_parameters = {}

        # Read each action server's params
        action_server_params = {}
        for server_name in server_names.value:
            # Get the action server's params
            action_type, tree_class, tick_rate = self.declare_parameters(
                "",
                [
                    (
                        f"default.{server_name}.action_type",
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
                        f"default.{server_name}.tree_class",
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
                        f"default.{server_name}.tick_rate",
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
            tree_kws = self.declare_parameter(
                f"default.{server_name}.tree_kws",
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
            if tree_kws.value is not None:
                tree_kwargs = {}
                for kw in tree_kws.value:
                    full_name = f"{server_name}.tree_kwargs.{kw}"
                    default_value = self.declare_parameter(
                        f"default.{full_name}",
                        descriptor=ParameterDescriptor(
                            name=kw,
                            description="Custom keyword argument for the behavior tree.",
                            dynamic_typing=True,
                            read_only=True,
                        ),
                    )
                    if isinstance(default_value, collections.abc.Sequence):
                        default_value = list(default_value.value)
                    else:
                        default_value = default_value.value
                    self.default_parameters[full_name] = default_value
                    current_value = self.declare_parameter(
                        f"current.{full_name}",
                        descriptor=ParameterDescriptor(
                            name=kw,
                            description="Custom keyword argument for the behavior tree.",
                            dynamic_typing=True,
                        ),
                    )
                    if isinstance(current_value, collections.abc.Sequence):
                        current_value = list(current_value.value)
                    else:
                        current_value = current_value.value
                    if full_name in overridden_parameters:
                        tree_kwargs[kw] = current_value
                        self.overridden_parameters[full_name] = current_value
                    else:
                        tree_kwargs[kw] = default_value
            else:
                tree_kwargs = {}

            if action_type.value is None or tree_class.value is None:
                self.get_logger().warn(
                    f"Skipping action server {server_name} "
                    "because it has no action type or tree class"
                )
                continue

            action_server_params[server_name] = ActionServerParams(
                server_name=server_name,
                action_type=action_type.value,
                tree_class=tree_class.value,
                tree_kwargs=tree_kwargs,
                tick_rate=tick_rate.value,
            )

        return action_server_params

    def parameter_callback(self, params: List[Parameter]) -> SetParametersResult:
        """
        Callback function for when a parameter is changed. Note that in practice,
        only tree_kwargs are not read-only, so we only expect those to be changed.

        Note that we only return failure if there is a type mismatch. That is just
        in case some other code in this file (e.g., the WatchdogListener) needs
        to process the parameter change. This is because rclpy runs all parameter
        callbacks in sequence until one returns failure.
        """
        num_updated_params = 0
        for param in params:
            names = param.name.split(".")
            if len(names) < 1 or names[0] != "current":
                self.get_logger().warn(f"Parameter {param.name} cannot be changed")
                continue
            full_name = ".".join(names[1:])
            if full_name not in self.default_parameters:
                self.get_logger().warn(f"Unknown parameter {param.name}")
                continue
            if isinstance(param.value, collections.abc.Sequence):
                param_value = list(param.value)
            else:
                param_value = param.value
            if not isinstance(param_value, type(self.default_parameters[full_name])):
                self.get_logger().warn(
                    f"Parameter {param.name} must be of type "
                    f"{type(self.default_parameters[full_name])} "
                    f"but is of type {type(param_value)}"
                )
                return SetParametersResult(successful=False, reason="type mismatch")
            self.overridden_parameters[full_name] = param_value
            num_updated_params += 1
            # If a tree_kwarg was set, re-initialize the tree
            if "tree_kwargs" in param.name:
                server_name = names[1]
                if server_name in self.action_server_params:
                    action_server_params = self.action_server_params[server_name]
                    kw = names[3]
                    action_server_params.tree_kwargs[kw] = param_value
                    # pylint: disable=too-many-function-args
                    self.create_tree(
                        server_name,
                        action_server_params.tree_class,
                        action_server_params.tree_kwargs,
                    )
        if num_updated_params > 0:
            self.save_overridden_parameters()
        return SetParametersResult(successful=True)

    def save_overridden_parameters(self) -> None:
        """
        Overrides `ada_feeding_action_servers_current.yaml` with the current
        values of `self.overridden_parameters`.
        """
        # Convert the parameters to a dictionary of the right form
        params = {}
        params["overridden_parameters"] = list(self.overridden_parameters.keys())
        for full_name, value in self.overridden_parameters.items():
            params[full_name] = value
        data = {"ada_feeding_action_servers": {"ros__parameters": {"current": params}}}

        # Write to yaml
        package_path = get_package_share_directory("ada_feeding")
        file_path = os.path.join(
            package_path, "config", "ada_feeding_action_servers_current.yaml"
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
                goal_callback=self.goal_callback,
                cancel_callback=self.cancel_callback,
            )
            self._action_servers.append(action_server)

    def goal_callback(self, goal_request: object) -> GoalResponse:
        """
        Accept a goal if this action does not already have an active goal,
        else reject.

        Parameters
        ----------
        goal_request: The goal request message.
        """
        self.get_logger().info("Received goal request")

        # If we don't already have an active goal_request, accept this one
        with self.active_goal_request_lock:
            if self.watchdog_listener.ok() and self.active_goal_request is None:
                self.get_logger().info("Accepting goal request")
                self.active_goal_request = goal_request
                return GoalResponse.ACCEPT

            # Otherwise, reject this goal request
            self.get_logger().info("Rejecting goal request")
            return GoalResponse.REJECT

    def cancel_callback(self, _: ServerGoalHandle) -> CancelResponse:
        """
        Always accept client requests to cancel the active goal.

        Parameters
        ----------
        goal_handle: The goal handle.
        """
        self.get_logger().info("Received cancel request, accepting")
        return CancelResponse.ACCEPT

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

    # pylint: disable=too-many-arguments
    # This is appropriate
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
        self.create_tree(server_name, tree_class, tree_kwargs)

        async def execute_callback(goal_handle: ServerGoalHandle) -> Awaitable:
            """
            This function is called when a goal is accepted by an action server.
            It loads the behavior tree file associated with the action server
            (if not already loaded) and executes the behavior tree, publishing
            periodic feedback.
            """

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
                            self.get_logger().info("Goal canceled")
                            tree_action_server.preempt_goal(
                                tree
                            )  # blocks until the preempt succeeds
                            goal_handle.canceled()
                            result = tree_action_server.get_result(tree, action_type)
                            break

                        # Check if the watchdog has failed
                        if not self.watchdog_listener.ok():
                            self.get_logger().warn("Watchdog failed, aborting goal")
                            tree_action_server.preempt_goal(
                                tree
                            )  # blocks until the preempt succeeds
                            goal_handle.abort()
                            result = tree_action_server.get_result(tree, action_type)
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
                            self.get_logger().info("Tree failed")
                            goal_handle.abort()
                            result = tree_action_server.get_result(tree, action_type)
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
    executor = MultiThreadedExecutor()

    # pylint: disable=broad-exception-caught
    # All exceptions need printing at shutdown
    try:
        rclpy.spin(create_action_servers, executor=executor)
    except Exception:
        traceback.print_exc()

    # Destroy the node explicitly
    create_action_servers.shutdown()
    create_action_servers.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
