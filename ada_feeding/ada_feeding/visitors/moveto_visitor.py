#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveIt2 Visitor.

This generally collects variables used for feedback/result in
all actions based on MoveTo.action.
"""

# Standard imports
from copy import deepcopy
import math
from overrides import override

# Third-party imports
import py_trees
from py_trees.composites import Composite
from py_trees.decorators import Decorator
from py_trees.visitors import VisitorBase
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

# Local imports
import ada_feeding.behaviors
from ada_feeding.behaviors.moveit2 import MoveIt2Execute
from ada_feeding.helpers import get_moveit2_object
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

        # Used for get_distance_to_goal
        self.aligned_joint_indices = None
        self.traj_i = 0

    def reinit(self) -> None:
        """
        Reset all local variables.
        Can be called if a tree is run again.
        """
        self.start_time = None
        self.feedback = MoveTo.Feedback()
        self.feedback.is_planning = True

    @staticmethod
    def joint_position_dist(point1: float, point2: float) -> float:
        """
        Given two joint positions in radians, this function computes the
        distance between then, accounting for rotational symmetry.

        Parameters
        ----------
        point1: The first joint position, in radians.
        point2: The second joint position, in radians.
        """
        abs_dist = abs(point1 - point2) % (2 * math.pi)
        return min(abs_dist, 2 * math.pi - abs_dist)

    def get_distance_to_goal(self, traj: JointTrajectory) -> float:
        """
        Returns the remaining distance to the goal.

        In practice, it keeps track of what joint state along the trajectory the
        robot is currently in, and returns the number of remaining joint states. As
        a result, this is not technically a measure of either distance or time, but
        should give some intuition of how much of the trajectory is left to execute.
        """
        # If the trajectory has not been set, something is wrong.
        if traj is None:
            self.node.get_logger().error(
                "[MoveIt2Visitor::get_distance_to_goal()] Trajectory is None!"
            )
            return 0.0

        # Get the latest joint state of the robot
        moveit2, lock = get_moveit2_object(
            py_trees.blackboard.Client(name="visitor", namespace="/"), self.node
        )
        with lock:
            curr_joint_state = moveit2.joint_state

        # Lazily align the joints between the trajectory and the joint states message
        if self.aligned_joint_indices is None:
            self.aligned_joint_indices = []
            for joint_traj_i, joint_name in enumerate(traj.joint_names):
                if joint_name in curr_joint_state.name:
                    joint_state_i = curr_joint_state.name.index(joint_name)
                    self.aligned_joint_indices.append(
                        (joint_name, joint_state_i, joint_traj_i)
                    )
                else:
                    self.node.get_logger().error(
                        f"[MoveIt2Visitor::get_distance_to_goal()] Joint {joint_name} not in "
                        "current joint state, despite being in the trajectory. Skipping this joint "
                        "in distance to goal calculation."
                    )

        # Get the point along the trajectory closest to the robot's current joint state
        min_dist = None
        for i in range(self.traj_i, len(traj.points)):
            # Compute the distance between the current joint state and the
            # ujoint state at index i.
            traj_joint_state = traj.points[i]
            dist = sum(
                MoveToVisitor.joint_position_dist(
                    curr_joint_state.position[joint_state_i],
                    traj_joint_state.positions[joint_traj_i],
                )
                for (_, joint_state_i, joint_traj_i) in self.aligned_joint_indices
            )

            # If the distance is increasing, we've found the local min.
            if min_dist is not None:
                if dist >= min_dist:
                    self.traj_i = i - 1
                    return float(len(traj.points) - self.traj_i)

            min_dist = dist

        # If the distance never increased, we are nearing the final waypoint.
        # Because the robot may still have slight motion even after this point,
        # we conservatively return 1.0.
        return 1.0

    @override
    def run(self, behaviour: py_trees.behaviour.Behaviour) -> None:
        # Docstring copied by @override

        # pylint: disable=too-many-branches

        # We only care about leaf nodes
        if isinstance(behaviour, (Composite, Decorator)):
            return

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
        elif (
            isinstance(behaviour, MoveIt2Execute)
            and behaviour.status == py_trees.common.Status.RUNNING
        ):
            # Flip to execute
            if self.feedback.is_planning:
                self.start_time = self.node.get_clock().now()
                self.feedback.is_planning = False
                self.traj_i = 0
            if behaviour.blackboard_exists("trajectory"):
                traj = behaviour.blackboard_get("trajectory")
                if traj is not None:
                    self.feedback.motion_initial_distance = float(len(traj.points))
                    self.feedback.motion_curr_distance = self.get_distance_to_goal(traj)
        else:
            # Catch end of MoveIt2Execute behavior
            if behaviour.status == py_trees.common.Status.SUCCESS:
                self.feedback.motion_curr_distance = 0.0
            # Switch to planning start time
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
        return deepcopy(self.feedback)
