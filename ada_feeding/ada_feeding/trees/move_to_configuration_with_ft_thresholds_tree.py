#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithFTThresholdsTree behavior
tree and provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
from typing import List, Optional, Set

# Third-party imports
from overrides import override
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding_msgs.action import MoveToConfiguration
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2JointConstraint,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config, scoped_behavior
from ada_feeding.idioms.bite_transfer import get_toggle_watchdog_listener_behavior
from ada_feeding.trees import MoveToTree


class MoveToConfigurationWithFTThresholdsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration, after
    re-taring the FT sensor and setting specific FT thresholds.

    TODO: Add the ability to pass force-torque thresholds to revert after the
    motion, and then set the force-torque thresholds in the scoped behavior.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        node: Node,
        # Required parameters for moving to a configuration
        joint_positions: Optional[List[float]] = None,
        # Optional parameters for moving to a configuration
        tolerance_joint: float = 0.001,
        weight_joint: float = 1.0,
        pipeline_id: str = "ompl",
        planner_id: str = "RRTConnectkConfigDefault",
        allowed_planning_time: float = 0.5,
        max_velocity_scaling_factor: float = 0.1,
        max_acceleration_scaling_factor: float = 0.1,
        # Optional parameters for the FT thresholds
        re_tare: bool = True,
        toggle_watchdog_listener: bool = True,
        f_mag: float = 0.0,
        f_x: float = 0.0,
        f_y: float = 0.0,
        f_z: float = 0.0,
        t_mag: float = 0.0,
        t_x: float = 0.0,
        t_y: float = 0.0,
        t_z: float = 0.0,
        keys_to_not_write_to_blackboard: Set[str] = set(),
        clear_constraints: bool = True,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        joint_positions: The joint positions for the goal constraint. If None,
            the action type must be MoveToConfiguration, so the joint_positions
            are passed in with the goal.
        tolerance_joint: The tolerance for the joint goal constraint.
        weight_joint: The weight for the joint goal constraint.
        pipeline_id: The pipeline ID to use for MoveIt2 motion planning.
        planner_id: The planner ID to use for MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
        max_velocity_scaling_factor: The maximum velocity scaling factor for the
            MoveIt2 motion planner.
        max_acceleration_scaling_factor: The maximum acceleration scaling factor for the
            MoveIt2 motion planner.
        re_tare: Whether to re-tare the force-torque sensor.
        toggle_watchdog_listener: Whether to toggle the watchdog listener on and off.
            In practice, if the watchdog listener is on, you should toggle it.
        f_mag: The magnitude of the overall force threshold. No threshold if 0.0.
        f_x: The magnitude of the x component of the force threshold. No threshold if 0.0.
        f_y: The magnitude of the y component of the force threshold. No threshold if 0.0.
        f_z: The magnitude of the z component of the force threshold. No threshold if 0.0.
        t_mag: The magnitude of the overall torque threshold. No threshold if 0.0.
        t_x: The magnitude of the x component of the torque threshold. No threshold if 0.0.
        t_y: The magnitude of the y component of the torque threshold. No threshold if 0.0.
        t_z: The magnitude of the z component of the torque threshold. No threshold if 0.0.
        keys_to_not_write_to_blackboard: the keys to not write to the blackboard.
            Note that the keys need to be exact e.g., "move_to.cartesian,"
            "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
            etc.
        clear_constraints: Whether or not to put a ClearConstraints decorator at the top
            of this branch. If you will be adding additional Constraints on top of this
            tree, this should be False. Else (e.g., if this is a standalone tree), True.
        """

        # pylint: disable=too-many-arguments, too-many-function-args, too-many-locals
        # Many arguments is fine for this class since it has to be able to configure all parameters
        # of its constraints.
        # pylint: disable=dangerous-default-value
        # A mutable default value is okay since we don't change it in this function.

        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters for the joint goal constraint
        self.joint_positions = joint_positions
        self.tolerance_joint = tolerance_joint
        self.weight_joint = weight_joint
        self.pipeline_id = pipeline_id
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.max_acceleration_scaling_factor = max_acceleration_scaling_factor

        # Store the parameters for the FT threshold
        self.re_tare = re_tare
        self.toggle_watchdog_listener = toggle_watchdog_listener
        self.f_mag = f_mag
        self.f_x = f_x
        self.f_y = f_y
        self.f_z = f_z
        self.t_mag = t_mag
        self.t_x = t_x
        self.t_y = t_y
        self.t_z = t_z

        self.keys_to_not_write_to_blackboard = keys_to_not_write_to_blackboard
        self.clear_constraints = clear_constraints

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # Write joint_positions to blackboard. This is done on the blackboard as
        # opposed to with a hardcoded input so that it can be overridden by
        # `send_goal` if the action type is MoveToConfiguration, not MoveTo.
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(
            key="joint_positions", access=py_trees.common.Access.WRITE
        )
        blackboard.joint_positions = self.joint_positions

        turn_watchdog_listener_on_prefix = "turn_watchdog_listener_on"

        # First, create the MoveToConfiguration behavior tree, in the same
        # namespace as this tree
        move_to_configuration_root = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                MoveIt2JointConstraint(
                    name="JointConstraint",
                    ns=name,
                    inputs={
                        "joint_positions": BlackboardKey("joint_positions"),
                        "tolerance": self.tolerance_joint,
                        "weight": self.weight_joint,
                    },
                    outputs={
                        "constraints": BlackboardKey("goal_constraints"),
                    },
                ),
                py_trees.decorators.Timeout(
                    name="MoveIt2PlanTimeout",
                    # Increase allowed_planning_time to account for ROS2 overhead and MoveIt2 setup and such
                    duration=10.0 * self.allowed_planning_time,
                    child=MoveIt2Plan(
                        name="MoveToConfigurationPlan",
                        ns=name,
                        inputs={
                            "goal_constraints": BlackboardKey("goal_constraints"),
                            "pipeline_id": self.pipeline_id,
                            "planner_id": self.planner_id,
                            "allowed_planning_time": self.allowed_planning_time,
                            "max_velocity_scale": self.max_velocity_scaling_factor,
                            "max_acceleration_scale": self.max_acceleration_scaling_factor,
                        },
                        outputs={"trajectory": BlackboardKey("trajectory")},
                    ),
                ),
                MoveIt2Execute(
                    name="MoveToConfigurationExecute",
                    ns=name,
                    inputs={"trajectory": BlackboardKey("trajectory")},
                    outputs={},
                ),
            ],
        )

        # Add the re-taring and FT thresholds
        pre_moveto_behavior = pre_moveto_config(
            name=name,
            re_tare=self.re_tare,
            toggle_watchdog_listener=self.toggle_watchdog_listener,
            f_mag=self.f_mag,
            f_x=self.f_x,
            f_y=self.f_y,
            f_z=self.f_z,
            t_mag=self.t_mag,
            t_x=self.t_x,
            t_y=self.t_y,
            t_z=self.t_z,
        )

        if self.toggle_watchdog_listener:
            # If there was a failure in the main tree, we want to ensure to turn
            # the watchdog listener back on
            # pylint: disable=duplicate-code, too-many-function-args
            # This is similar to any other tree that needs to cleanup pre_moveto_config
            turn_watchdog_listener_on = get_toggle_watchdog_listener_behavior(
                name,
                turn_watchdog_listener_on_prefix,
                True,
            )

            # Create the main tree
            root = scoped_behavior(
                name=name + " ToggleWatchdogListenerOffScope",
                pre_behavior=pre_moveto_behavior,
                workers=[move_to_configuration_root],
                post_behavior=turn_watchdog_listener_on,
            )
        else:
            # Combine them in a sequence with memory
            root = py_trees.composites.Sequence(
                name=name,
                memory=True,
                children=[pre_moveto_behavior, move_to_configuration_root],
            )

        tree = py_trees.trees.BehaviourTree(root)
        return tree

    # Override goal to read arguments into local blackboard
    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree

        # If it is a MoveToConfiguration.Goal, override the joint_positions
        if isinstance(goal, MoveToConfiguration.Goal):
            # Write tree inputs to blackboard
            name = tree.root.name
            blackboard = py_trees.blackboard.Client(name=name, namespace=name)
            blackboard.register_key(
                key="joint_positions", access=py_trees.common.Access.WRITE
            )
            blackboard.joint_positions = goal.joint_positions
        else:
            assert (
                self.joint_positions is not None
            ), "For action MoveTo, must provide hardcoded joint_positions"
            assert (
                len(self.joint_positions) == 6
            ), "For action MoveTo, must provide 6 joint positions"

        # Adds MoveToVisitor for Feedback
        return super().send_goal(tree, goal)
