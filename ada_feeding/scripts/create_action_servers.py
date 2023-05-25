#!/usr/bin/env python3

# Standard imports
import os
import pprint
import threading
import traceback
from typing import Awaitable, Callable, Dict, List

# Third-party imports
import py_trees
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import yaml

# Local imports
from ada_feeding import ActionServerBT


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

        # Declare the ROS2 parameters this node will be looking up.
        self.declare_parameter("config_file", rclpy.Parameter.Type.STRING)

        # Get the path to the configuration file.
        config_file = self.get_parameter("config_file").value
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                "Path specified in config_file parameter does not exist: %s"
                % config_file
            )

        # Read the configuration file.
        self.get_logger().info("Loading configuration file: %s" % config_file)
        with open(config_file, "r", encoding="utf-8") as config_file_handle:
            configs = yaml.load(config_file_handle, Loader=yaml.FullLoader)
        self.get_logger().debug("\n" + pprint.pformat(configs, indent=4))

        # Track the active goal request.
        self.active_goal_request_lock = threading.Lock()
        self.active_goal_request = None

        # Create the action servers.
        self.create_action_servers(configs)

    def create_action_servers(self, configs: Dict) -> None:
        """
        Create the action servers specified in the configuration file.

        Parameters
        ----------
        configs: The configuration file.
        """
        # For every action server specified in the configuration file, create
        # an action server.
        self._action_servers = []
        self._action_types = {}
        self._tree_classes = {}
        for action_config in configs["action_servers"]:
            # Load the action server parameters from the configuration file.
            server_name = action_config["server_name"]
            action_type = action_config["action_type"]
            tree_class = action_config["tree_class"]
            if "tree_kwargs" in action_config:
                tree_kwargs = action_config["tree_kwargs"]
            else:
                tree_kwargs = {}
            if "tick_rate" in action_config:
                tick_rate = action_config["tick_rate"]
            else:
                tick_rate = 30  # Hz, default tick rate
            self.get_logger().info(
                "Creating action server %s with type %s" % (server_name, action_type)
            )

            # Import the action type
            try:
                action_package, action_module, action_class = action_type.split(".", 3)
            except Exception as exc:
                raise NameError(
                    'Invalid action type %s. Except "package.module.class" e.g., "ada_feeding_msgs.action.MoveTo"'
                    % action_type
                ) from exc
            try:
                self._action_types[action_type] = getattr(
                    getattr(
                        __import__("%s.%s" % (action_package, action_module)),
                        action_module,
                    ),
                    action_class,
                )
            except Exception as exc:
                raise ImportError("Error importing %s" % (action_type)) from exc

            # Import the behavior tree class
            try:
                tree_package, tree_module, tree_class = tree_class.split(".", 3)
            except Exception as exc:
                raise NameError(
                    'Invalid tree class %s. Except "package.module.class" e.g., "ada_feeding.trees.MoveAbovePlate"'
                    % tree_class
                ) from exc
            try:
                self._tree_classes[tree_class] = getattr(
                    getattr(
                        __import__("%s.%s" % (tree_package, tree_module)),
                        tree_module,
                    ),
                    tree_class,
                )
                assert issubclass(self._tree_classes[tree_class], ActionServerBT), (
                    "Tree %s must subclass ActionServerBT" % tree_class
                )
            except Exception as exc:
                raise ImportError("Error importing %s" % (tree_class)) from exc

            # Create the action server.
            action_server = ActionServer(
                self,
                self._action_types[action_type],
                server_name,
                self.get_execute_callback(
                    server_name, action_type, tree_class, tree_kwargs, tick_rate
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
