#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the abstract TriggerTree class, which is a behavior tree
that implements the `Trigger.action` interface defined in `ada_feeding_msgs`.
"""

# Standard imports
from abc import ABC
import time

# Third-party imports
from overrides import override
import py_trees
from builtin_interfaces.msg import Duration

# Local imports
from ada_feeding import ActionServerBT


class TriggerTree(ActionServerBT, ABC):
    """
    An abstract behvaior tree for any behavior that should get triggered and send
    feedback and results according to the `Trigger.action` interface defined in
    `ada_feeding_msgs`.
    """

    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # We use Python time as opposed to ROS time because the behavior tree
        # won't necessarily have access to the node's clock.
        self.start_time = time.time()  # pylint: disable=attribute-defined-outside-init
        return True

    @override
    def get_feedback(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        feedback_msg = action_type.Feedback()
        elapsed_time = time.time() - self.start_time
        secs = int(elapsed_time)
        nsecs = int((elapsed_time - secs) * 1e9)
        feedback_msg.elapsed_time = Duration(sec=secs, nanosec=nsecs)
        return feedback_msg

    @override
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        result = action_type.Result()
        result.success = tree.root.status == py_trees.common.Status.SUCCESS
        # TODO: Consider adding a result message that consists of the last
        # behavior that finished ticking.
        return result
