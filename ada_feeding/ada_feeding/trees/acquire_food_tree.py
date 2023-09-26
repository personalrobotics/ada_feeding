#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the AcquireFood behavior tree and provides functions to
wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from overrides import override

# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding import ActionServerBT


class AcquireFoodTree(MoveToTree):
    """
    A behvaior tree to select and execute an acquisition
    action (see ada_feeding_msgs.action.AcquisitionSchema)
    for a given food mask in ada_feeding_msgs.action.AcquireFood.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    def __init__(
        self,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        """
        # Initialize MoveToTree
        super().__init__()

        # Store the parameters
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time

    # pylint: disable=too-many-locals
    # Unfortunately, many local variables are required here to isolate the keys
    # of similar constraints in the blackboard.
    def create_move_to_tree(
        self,
        name: str,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the AcquireFood behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that acquires food
        """

        # TODO: Define full tree
        pass

    # Override result to handle timing outside MoveTo Behaviors
    @override
    def get_feedback(self, tree: py_trees.trees.BehaviourTree) -> object:
        # Docstring copied by @override
        feedback_msg = super().get_feedback(tree)

        # TODO: fix is_planning / planning_time in non-MoveTo Behavior
        return feedback_msg

    # Override result to add other elements to result msg
    @override
    def get_result(self, tree: py_trees.trees.BehaviourTree) -> object:
        # Docstring copied by @override
        result_msg = super().get_feedback(tree)

        # TODO: add action_index, posthoc, action_select_hash
        return result_msg
