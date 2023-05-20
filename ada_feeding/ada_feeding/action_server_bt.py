from abc import ABC, abstractmethod
import py_trees
from typing import Self

class ActionServerBT(ABC):
    """
    An interface for behavior trees to be wrapped in an action server.

    Only subclasses of ActionServerBT can be spun up as an action server in
    `create_action_server.py`
    """

    @abstractmethod
    def create_tree(self: Self) -> py_trees.trees.BehaviourTree:
        """
        Create the behavior tree that will be executed by this action server.
        """
        raise NotImplementedError("create_tree not implemented")
    
    @abstractmethod
    def gen_feedback(self: Self, tree: py_trees.trees.BehaviourTree) -> object:
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
        raise NotImplementedError("gen_feedback not implemented")

    @abstractmethod
    def gen_result(self: Self, tree: py_trees.trees.BehaviourTree) -> object:
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
        raise NotImplementedError("gen_result not implemented")