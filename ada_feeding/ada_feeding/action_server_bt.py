"""
This module defines ActionServerBT, an abstract class that links ROS2 action
servers with py_trees.
"""
# Standard imports
from abc import ABC, abstractmethod
import logging
import traceback

# Third-party imports
import py_trees


class ActionServerBT(ABC):
    """
    An interface for behavior trees to be wrapped in an action server.

    Only subclasses of ActionServerBT can be spun up as an action server in
    `create_action_server.py`
    """

    @abstractmethod
    def create_tree(
        self, name: str, logger: logging.Logger
    ) -> py_trees.trees.BehaviourTree:
        """
        Create the behavior tree that will be executed by this action server.
        Note that subclasses of ActionServerBT can decide whether they want
        to cache the tree or create a new one each time.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.
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
        try:
            tree.root.stop(py_trees.common.Status.INVALID)
            return True
        except Exception:
            tree.root.logger.warn(
                "Failed to preempt goal \n%s" % traceback.format_exc()
            )
            return False

    @abstractmethod
    def get_feedback(self, tree: py_trees.trees.BehaviourTree) -> object:
        """
        Creates the ROS feedback message corresponding to this action.

        This function will be passed as a post_tick_handler to the behavior tree.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        feedback: The ROS feedback message to be sent to the action client.
        """
        raise NotImplementedError("get_feedback not implemented")

    @abstractmethod
    def get_result(self, tree: py_trees.trees.BehaviourTree) -> object:
        """
        Creates the ROS result message corresponding to this action.

        This function is called after the tree terminates.

        Parameters
        ----------
        tree: The behavior tree that is being executed.

        Returns
        -------
        result: The ROS result message to be sent to the action client.
        """
        raise NotImplementedError("get_result not implemented")
