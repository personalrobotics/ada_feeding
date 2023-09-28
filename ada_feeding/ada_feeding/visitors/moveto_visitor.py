#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveIt2 Visitor.

This generally collects variables used for feedback/result in
all actions based on MoveTo.action.
"""

# Standard imports
from overrides import override

# Third-party imports
import py_trees
from py_trees.visitors import VisitorBase
from rclpy.node import Node

# Local imports
import ada_feeding.behaviors
from ada_feeding_msgs.action import MoveTo


class MoveToVisitor(VisitorBase):
    """
    A BT Visitor that computes the feedback used in MoveTo.action.
    Can be used in all actions that return similar information.
    """

    def __init__(self, node: Node) -> None:
        super().__init__(full=False)

        # Just need the node's clock for timing
        self.node = node

        # Used for planning/motion time calculations
        self.start_time = None

        # To return with get_feedback
        self.feedback = MoveTo.Feedback()
        self.feedback.is_planning = True

    def reinit(self) -> None:
        """
        Reset all local variables.
        Can be called if a tree is run again.
        """
        self.start_time = None
        self.feedback = MoveTo.Feedback()
        self.feedback.is_planning = True

    @override
    def run(self, behaviour: py_trees.behaviour.Behaviour) -> None:
        # Docstring copied by @override

        # Record Start Time
        if self.start_time is None:
            self.start_time = self.node.get_clock().now()

        if isinstance(behaviour, ada_feeding.behaviors.MoveTo):
            # If in MoveTo action, copy from there
            self.feedback.motion_initial_distance = (
                behaviour.tree_blackboard.motion_initial_distance
            )
            self.feedback.motion_curr_distance = (
                behaviour.tree_blackboard.motion_curr_distance
            )

            # Check for flip between planning/motion
            if behaviour.tree_blackboard.is_planning != self.feedback.is_planning:
                self.start_time = self.node.get_clock().now()
                self.feedback.is_planning = behaviour.tree_blackboard.is_planning
        else:
            # Else just update planning time
            if not self.feedback.is_planning:
                self.start_time = self.node.get_clock().now()
                self.feedback.is_planning = True

        # Calculate updated planning/motion time
        if self.feedback.is_planning:
            self.feedback.planning_time = (
                self.node.get_clock().now() - self.start_time
            ).to_msg()
        else:
            self.feedback.motion_time = (
                self.node.get_clock().now() - self.start_time
            ).to_msg()

    def get_feedback(self) -> MoveTo.Feedback:
        """

        Returns
        -------
        MoveTo Feedback message, see MoveTo.action for more info.
        """
        return self.feedback
