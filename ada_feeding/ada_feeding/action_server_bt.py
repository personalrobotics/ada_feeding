"""
This module defines ActionServerBT, an abstract class that links ROS2 action
servers with py_trees.
"""
# Standard imports
from abc import ABC, abstractmethod
import traceback

# Third-party imports
import py_trees
from rclpy.node import Node


class ActionServerBT(ABC):
    """
    An interface for behavior trees to be wrapped in an action server.

    Only subclasses of ActionServerBT can be spun up as an action server in
    `create_action_server.py`
    """

    def __init__(self, node: Node) -> None:
        """
        Store the ROS2 Node that created trees is associated with. Necessary for
        behaviors within the tree to connect to ROS topics/services/actions.

        Subclasses should add kwargs with tree-agnostic
        input parameters.
        """
        self._node = node

    @abstractmethod
    def create_tree(self, name: str) -> py_trees.trees.BehaviourTree:
        """
        Create the behavior tree that will be executed by this action server.
        Note that subclasses of ActionServerBT can decide whether they want
        to cache the tree or create a new one each time.

        Parameters
        ----------
        name: The name of the behavior tree.
        """
        raise NotImplementedError("create_tree not implemented")

    @abstractmethod
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        """
        Sends the goal of the action to the behavior tree.

        Parameters
        ----------
        tree: The behavior tree that is being executed.
        goal: The ROS goal message sent to the action server.

        Returns
        -------
        success: Whether the goal was successfully sent to the behavior tree.
        """
        raise NotImplementedError("send_goal not implemented")

    def preempt_goal(self, tree: py_trees.trees.BehaviourTree) -> bool:
        """
        Preempts the currently running goal on the behavior tree and blocks until
        the preemption has been completed.

        The default behavior of this function calls the `stop` method on the
        root of the behavior tree. This should block until all nodes of the
        behavior tree have succesfully terminated. Subclasses can override this
        method if they want to implement a different behavior.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        success: Whether the goal was successfully preempted.
        """
        # pylint: disable=broad-exception-caught
        # All exceptions need printing when stopping a tree
        try:
            tree.root.stop(py_trees.common.Status.INVALID)
            return True
        except Exception:
            tree.root.logger.warn(f"Failed to preempt goal \n{traceback.format_exc()}")
            return False

    @abstractmethod
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        """
        Creates the ROS feedback message corresponding to this action.

        This function will be passed as a post_tick_handler to the behavior tree.

        Parameters
        ----------
        tree: The behavior tree that is being executed.
        action_type: the type for the action, as a class

        Returns
        -------
        The ROS feedback message to be sent to the action client, type action_type.Feedback()
        """
        raise NotImplementedError("get_feedback not implemented")

    @abstractmethod
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        """
        Creates the ROS result message corresponding to this action.

        This function is called after the tree terminates.

        Parameters
        ----------
        tree: The behavior tree that is being executed.
        action_type: the type for the action, as a class

        Returns
        -------
        The ROS result message to be sent to the action client, type action_type.Result()
        """
        raise NotImplementedError("get_result not implemented")
