#!/usr/bin/env python3
"""
This module contains a node, CreateActionServers, for creating action servers
that wrap behavior trees.
"""

# Standard imports
import threading
import traceback
from typing import Any, Awaitable, Callable, Dict, List

# Third-party imports
import py_trees
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Local imports
from ada_feeding import ActionServerBT
from ada_feeding.helpers import import_from_string


class ActionServerParams:
    """
    A class to hold the parameters of an action server that wraps a behavior tree.
    """

    def __init__(
        self,
        server_name: List[str],
        action_type: str,
        tree_class: str,
        tree_kwargs: Dict[str, Any] = {},
        tick_rate: int = 30,
    ) -> None:
        """
        Initialize the ActionServerParams class.
        """
        self.server_name = server_name
        self.action_type = action_type
        self.tree_class = tree_class
        self.tree_kwargs = tree_kwargs
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

    def __init__(self) -> None:
        """
        Initialize the CreateActionServers node. This function reads in the
        configuration file and creates the action servers. Note that it does not
        load the behavior tree file associated with each action server; that
        happens when the action server receives a goal request.
        """
        super().__init__("create_action_servers")

        # Read the parameters that specify what action servers to create.
        action_server_params = self.read_params()

        # Track the active goal request.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action servers.
        self.create_action_servers(action_server_params)

    def read_params(self) -> List[ActionServerParams]:
        """
        Read the parameters that specify what action servers to create.

        Returns
        -------
        action_server_params: A list of ActionServerParams objects.
        """
        # Read the server names
        server_names = self.declare_parameter(
            "server_names",
            descriptor=ParameterDescriptor(
                name="server_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="List of action server names to create. All names must correspond to their own param namespace within this node.",
                read_only=True,
            ),
        )

        # Read each action server's params
        action_server_params = []
        for server_name in server_names.value:
            # Get the action server's params
            action_type, tree_class, tick_rate = self.declare_parameters(
                "",
                [
                    (
                        "%s.action_type" % server_name,
                        None,
                        ParameterDescriptor(
                            name="action_type",
                            type=ParameterType.PARAMETER_STRING,
                            description="The type of action server to create. E.g., ada_feeding_msgs.action.MoveTo",
                            read_only=True,
                        ),
                    ),
                    (
                        "%s.tree_class" % server_name,
                        None,
                        ParameterDescriptor(
                            name="tree_class",
                            type=ParameterType.PARAMETER_STRING,
                            description="The class of the behavior tree to run, must subclass ActionServerBT. E.g., ada_feeding.behaviors.MoveTo",
                            read_only=True,
                        ),
                    ),
                    (
                        "%s.tick_rate" % server_name,
                        30,
                        ParameterDescriptor(
                            name="tick_rate",
                            type=ParameterType.PARAMETER_INTEGER,
                            description="The rate at which the behavior tree should be ticked, in Hz.",
                            read_only=True,
                        ),
                    ),
                ],
            )
            tree_kws = self.declare_parameter(
                "%s.tree_kws" % server_name,
                descriptor=ParameterDescriptor(
                    name="tree_kws",
                    type=ParameterType.PARAMETER_STRING_ARRAY,
                    description="List of keywords for custom arguments to be passed to the behavior tree during initialization.",
                    read_only=True,
                ),
            )
            if tree_kws.value is not None:
                tree_kwargs = {
                    kw: self.declare_parameter(
                        "%s.tree_kwargs.%s" % (server_name, kw),
                        descriptor=ParameterDescriptor(
                            name=kw,
                            description="Custom keyword argument for the behavior tree.",
                            dynamic_typing=True,
                            read_only=True,
                        ),
                    )
                    for kw in tree_kws.value
                }
            else:
                tree_kwargs = {}

            if action_type.value is None or tree_class.value is None:
                self.get_logger().warn(
                    "Skipping action server %s because it has no action type or tree class"
                    % server_name
                )
                continue

            action_server_params.append(
                ActionServerParams(
                    server_name=server_name,
                    action_type=action_type.value,
                    tree_class=tree_class.value,
                    tree_kwargs={kw: arg.value for kw, arg in tree_kwargs.items()},
                    tick_rate=tick_rate.value,
                )
            )

        return action_server_params

    def create_action_servers(
        self, action_server_params: List[ActionServerParams]
    ) -> None:
        """
        Create the action servers specified in the configuration file.

        Parameters
        ----------
        action_server_params: A list of ActionServerParams objects.
        """
        # For every action server specified in the configuration file, create
        # an action server.
        self._action_servers = []
        self._action_types = {}
        self._tree_classes = {}
        for params in action_server_params:
            self.get_logger().info(
                "Creating action server %s with type %s"
                % (params.server_name, params.action_type)
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
                ), ("Tree %s must subclass ActionServerBT" % params.tree_class)
            except Exception:
                self.get_logger().warn(
                    "%s\nSKIPPING THIS ACTION SERVER" % (traceback.format_exc())
                )

            # Create the action server.
            action_server = ActionServer(
                self,
                self._action_types[params.action_type],
                params.server_name,
                self.get_execute_callback(
                    params.server_name,
                    params.action_type,
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
            if self.active_goal_request is None:
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

    def get_execute_callback(
        self,
        server_name: str,
        action_type: str,
        tree_class: str,
        tree_kwargs: Dict,
        tick_rate: int,
    ) -> Callable[[ServerGoalHandle], Awaitable]:
        """
        This is a wrapper function that takes in information about the action
        server and returns a callback function that can be used for that server.
        This is necessary because the callback function must return a Result and
        Feedback consisitent with the action type.
        """
        # Initialize the ActionServerBT object once
        tree_action_server = self._tree_classes[tree_class](**tree_kwargs)

        async def execute_callback(goal_handle: ServerGoalHandle) -> Awaitable:
            """
            This function is called when a goal is accepted by an action server.
            It loads the behavior tree file associated with the action server
            (if not already loaded) and executes the behavior tree, publishing
            periodic feedback.
            """

            self.get_logger().info(
                "%s: Executing goal %s with request %s"
                % (
                    server_name,
                    "".join(format(x, "02x") for x in goal_handle.goal_id.uuid),
                    goal_handle.request,
                )
            )

            # Load the behavior tree class
            tree = tree_action_server.create_tree(server_name, self.get_logger())
            tree.setup()  # TODO: consider adding a timeout here

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
                        with self.active_goal_request_lock:
                            self.active_goal_request = None
                        result = tree_action_server.get_result(tree)
                        break

                    # Tick the tree once and publish feedback
                    tree.tick()
                    feedback_msg = tree_action_server.get_feedback(tree)
                    goal_handle.publish_feedback(feedback_msg)
                    self.get_logger().info("Publishing feedback %s" % feedback_msg)

                    # Check the tree status
                    if tree.root.status == py_trees.common.Status.SUCCESS:
                        self.get_logger().info("Goal succeeded")
                        goal_handle.succeed()
                        with self.active_goal_request_lock:
                            self.active_goal_request = None
                        result = tree_action_server.get_result(tree)
                        break
                    if tree.root.status in set(
                        (py_trees.common.Status.FAILURE, py_trees.common.Status.INVALID)
                    ):
                        self.get_logger().info("Goal failed")
                        goal_handle.abort()
                        with self.active_goal_request_lock:
                            self.active_goal_request = None
                        result = tree_action_server.get_result(tree)
                        break

                    # Sleep
                    rate.sleep()
            except KeyboardInterrupt:
                pass

            # If we have gotten here without a result, that means something
            # went wrong. Abort the goal.
            if result is None:
                goal_handle.abort()
                result = self._action_types[action_type].Result()

            # Shutdown the tree
            try:
                tree.shutdown()
            except Exception as exc:
                self.get_logger().error(
                    "Error shutting down tree: \n%s\n%s" % (traceback.format_exc(), exc)
                )

            # Unset the goal and return the result
            with self.active_goal_request_lock:
                self.active_goal_request = None
            return result

        return execute_callback


def main(args: List = None) -> None:
    """
    Create the ROS2 node and run the action servers.
    """
    rclpy.init(args=args)

    create_action_servers = CreateActionServers()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    try:
        rclpy.spin(create_action_servers, executor=executor)
    except Exception:
        traceback.print_exc()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    create_action_servers.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
