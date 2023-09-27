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
from ada_feeding.behaviors.acquisition import ComputeFoodFrame
from ada_feeding.helpers import BlackboardKey
from ada_feeding.trees import MoveToTree
from ada_feeding_msgs.action import AcquireFood


class AcquireFoodTree(MoveToTree):
    """
    A behvaior tree to select and execute an acquisition
    action (see ada_feeding_msgs.action.AcquisitionSchema)
    for a given food mask in ada_feeding_msgs.action.AcquireFood.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.

    # pylint: disable=too-many-locals
    # Unfortunately, many local variables are required here to isolate the keys
    # of similar constraints in the blackboard.
    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the AcquireFood behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree. (DEPRECATED)
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables. (DEPRECATED)
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that acquires food
        """
        # TODO: remove name and tree_root_name
        # We have access to them implicitly via self.blackboard / self.blackboard_tree_root

        ### Inputs we expect on the blackboard on initialization
        # Note: WRITE access since send_goal could write to these
        # Mask for ComputeFoodFrame
        self.blackboard.register_key(key="mask", access=py_trees.common.Access.WRITE)
        # CameraInfo for ComputeFoodFrame
        self.blackboard.register_key(
            key="camera_info", access=py_trees.common.Access.WRITE
        )

        ### Define Tree Leaves and Subtrees

        # Add ComputeFoodFrame
        compute_food_frame = ComputeFoodFrame(
            "ComputeFoodFrame",
            self.blackboard.namespace,
            inputs={
                "ros2_node": node,
                "camera_info": BlackboardKey("camera_info"),
                "mask": BlackboardKey("mask")
                # Default world_frame = "world"
                # Default debug_tf_frame = "food"
            },
            outputs={
                "action_select_request": None,
                "food_frame": None,
                "debug_tf_publisher": BlackboardKey("debug_tf_publisher")
                # ^^ Set to None to disable
            },
        )

        ### Define Tree Logic

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name="RootSequence",
            memory=True,
            children=[
                compute_food_frame,
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)

    # Override goal to read arguments into local blackboard
    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied by @override
        super().send_goal(tree, goal)

        # Check goal type
        if not isinstance(goal, AcquireFood.Goal):
            return False

        # Write tree inputs to blackboard
        self.blackboard.mask = goal.detected_food
        self.blackboard.camera_info = goal.camera_info

        return True

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
        result_msg = super().get_result(tree)

        # TODO: add action_index, posthoc, action_select_hash
        return result_msg
