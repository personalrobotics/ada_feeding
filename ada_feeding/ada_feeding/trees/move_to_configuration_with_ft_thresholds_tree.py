#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToConfigurationWithFTThresholdsTree behavior
tree and provides functions to wrap that behavior tree in a ROS2 action server.
"""

# Standard imports
import logging
from typing import List, Set

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.behaviors import ToggleCollisionObject
from ada_feeding.idioms import pre_moveto_config
from ada_feeding.trees import MoveToTree, MoveToConfigurationTree


class MoveToConfigurationWithFTThresholdsTree(MoveToTree):
    """
    A behavior tree that moves the robot to a specified configuration, after
    re-taring the FT sensor and setting specific FT thresholds. This tree also
    provides the option to specify a list of collision object IDs that should
    be ignored during the motion (e.g., collisions between the robot and those
    objects will be allowed before the motion begins and disallowed after the
    motion ends or if it fails).
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Many arguments is fine for this class since it has to be able to configure all parameters
    # of its constraints.
    # pylint: disable=dangerous-default-value
    # A mutable default value is okay since we don't change it in this function.
    def __init__(
        self,
        # Required parameters for moving to a configuration
        joint_positions: List[float],
        # Optional parameters for moving to a configuration
        tolerance_joint: float = 0.001,
        weight_joint: float = 1.0,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
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
        collision_object_ids: List[str] = [],
        keys_to_not_write_to_blackboard: Set[str] = set(),
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        joint_positions: The joint positions for the goal constraint.
        tolerance_joint: The tolerance for the joint goal constraint.
        weight_joint: The weight for the joint goal constraint.
        planner_id: The planner ID to use for MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time for the MoveIt2 motion
            planner.
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
        collision_object_ids: The collision object IDs to ignore during the motion.
            Collisions between the robot and these objects will be allowed in the
            planning scene before the robot moves, and disallowed after the robot
            stops moving or if the motion failed.
        keys_to_not_write_to_blackboard: the keys to not write to the blackboard.
            Note that the keys need to be exact e.g., "move_to.cartesian,"
            "position_goal_constraint.tolerance," "orientation_goal_constraint.tolerance,"
            etc.
        """

        # pylint: disable=too-many-locals
        # One over is okay

        # Initialize MoveToTree
        super().__init__()

        # Store the parameters for the joint goal constraint
        self.joint_positions = joint_positions
        assert len(self.joint_positions) == 6, "Must provide 6 joint positions"
        self.tolerance_joint = tolerance_joint
        self.weight_joint = weight_joint
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time

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

        # Optional parameter to ignore collisions with certain objects
        self.collision_object_ids = collision_object_ids

        self.keys_to_not_write_to_blackboard = keys_to_not_write_to_blackboard

    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToConfigurationWithFTThresholdsTree behavior tree.

        Parameters
        ----------
        name: The name of the behavior tree.
        logger: The logger to use for the behavior tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviors within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behavior tree that moves the robot above the plate.
        """

        # pylint: disable=too-many-locals
        # Unfortunately, many local variables are required here since this tree
        # is exposing many different options.

        children = []

        # Create the re-taring and setting FT threshold behaviors that should
        # run before the MoveTo action
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
            logger=logger,
        )
        children.append(pre_moveto_behavior)

        # Create the behavior to allow collisions with certain objects
        if len(self.collision_object_ids) > 0:
            for collision_object_id in self.collision_object_ids:
                allow_collision_prefix = f"allow_collision_with_{collision_object_id}"
                allow_collision = ToggleCollisionObject(
                    name=Blackboard.separator.join([name, allow_collision_prefix]),
                    node=node,
                    collision_object_id=collision_object_id,
                    allow=True,
                )
                allow_collision.logger = logger
                children.append(allow_collision)

        # Create the MoveToConfiguration behavior tree, in the same
        # namespace as this tree
        move_to_configuration_root = (
            MoveToConfigurationTree(
                joint_positions=self.joint_positions,
                tolerance=self.tolerance_joint,
                weight=self.weight_joint,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time,
                keys_to_not_write_to_blackboard=self.keys_to_not_write_to_blackboard,
            )
            .create_tree(name, self.action_type, tree_root_name, logger, node)
            .root
        )
        children.append(move_to_configuration_root)

        # Create the behavior to disallow collisions with certain objects
        def gen_disallow_collision_children():
            disallow_collision_children = []
            if len(self.collision_object_ids) > 0:
                for collision_object_id in self.collision_object_ids:
                    disallow_collision_prefix = (
                        f"disallow_collision_with_{collision_object_id}"
                    )
                    disallow_collision = ToggleCollisionObject(
                        name=Blackboard.separator.join(
                            [name, disallow_collision_prefix]
                        ),
                        node=node,
                        collision_object_id=collision_object_id,
                        allow=False,
                    )
                    disallow_collision.logger = logger
                    disallow_collision_children.append(disallow_collision)
            return disallow_collision_children

        children.extend(gen_disallow_collision_children())

        # Combine them in a sequence with memory
        main_behavior = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=children,
        )
        main_behavior.logger = logger

        # If any part of the main behavior fails, we need to clean it up by still
        # allowing the collision.
        if len(self.collision_object_ids) > 0:
            # pylint: disable=duplicate-code
            # The cleanup across trees will look similar.
            root = py_trees.composites.Selector(
                name=name,
                memory=True,
                children=[
                    main_behavior,
                    # Even though we are cleaning up the tree, it should still
                    # pass the failure up.
                    py_trees.decorators.SuccessIsFailure(
                        name + " Cleanup",
                        py_trees.composites.Sequence(
                            name=name,
                            memory=True,
                            children=gen_disallow_collision_children(),
                        ),
                    ),
                ],
            )
            root.logger = logger
        else:
            root = main_behavior

        tree = py_trees.trees.BehaviourTree(root)
        return tree
