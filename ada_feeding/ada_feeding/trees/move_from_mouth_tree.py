#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveFromMouthTree behaviour tree and provides functions to
wrap that behaviour tree in a ROS2 action server.
"""
# pylint: disable=duplicate-code
# MoveFromMouth and MoveToMouth are inverses of each other, so it makes sense
# that they have similar code.

# Standard imports
from functools import partial
import logging
from typing import List, Tuple

# Third-party imports
import py_trees
from py_trees.blackboard import Blackboard
from rclpy.node import Node

# Local imports
from ada_feeding.idioms import pre_moveto_config
from ada_feeding.idioms.bite_transfer import (
    get_toggle_collision_object_behavior,
    get_toggle_watchdog_listener_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
    MoveToConfigurationWithPosePathConstraintsTree,
    MoveToPoseWithPosePathConstraintsTree,
)


class MoveFromMouthTree(MoveToTree):
    """
    A behaviour tree that moves the robot back to the staging location in a
    cartesian motion, and then moves the robot to a specified end configuration.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals
    # Bite transfer is a big part of the robot's behavior, so it makes
    # sense that it has lots of attributes/arguments.

    def __init__(
        self,
        staging_configuration_position: Tuple[float, float, float],
        staging_configuration_quat_xyzw: Tuple[float, float, float, float],
        end_configuration: List[float],
        staging_configuration_tolerance: float = 0.001,
        end_configuration_tolerance: float = 0.001,
        orientation_constraint_to_staging_configuration_quaternion: List[float] = None,
        orientation_constraint_to_staging_configuration_tolerances: List[float] = None,
        orientation_constraint_to_end_configuration_quaternion: List[float] = None,
        orientation_constraint_to_end_configuration_tolerances: List[float] = None,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time_to_staging_configuration: float = 0.5,
        allowed_planning_time_to_end_configuration: float = 0.5,
        wheelchair_collision_object_id: str = "wheelchair_collision",
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        staging_configuration_position: The position of the staging
            configuration.
        staging_configuration_quat_xyzw: The quaternion of the staging
            configuration.
        end_configuration: The joint positions to move the robot arm to after
            going to the staging configuration.
        staging_configuration_tolerance: The tolerance for the joint positions.
        end_configuration_tolerance: The tolerance for the joint positions.
        orientation_constraint_to_staging_configuration_quaternion: The
            quaternion for the orientation constraint to the staging
            configuration.
        orientation_constraint_to_staging_configuration_tolerances: The
            tolerances for the orientation constraint to the staging
            configuration.
        orientation_constraint_to_end_configuration_quaternion: The
            quaternion for the orientation constraint to the end configuration.
        orientation_constraint_to_end_configuration_tolerances: The
            tolerances for the orientation constraint to the end configuration.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time_to_staging_configuration: The allowed planning
            time for the MoveIt2 motion planner to move to the staging config.
        allowed_planning_time_to_end_configuration: The allowed planning
            time for the MoveIt2 motion planner to move to the end config.
        wheelchair_collision_object_id: The ID of the wheelchair collision object
            in the MoveIt2 planning scene.
        force_threshold: The force threshold (N) for the ForceGateController.
            For now, the same threshold is used to move to the staging location
            and to the mouth.
        torque_threshold: The torque threshold (N*m) for the ForceGateController.
            For now, the same threshold is used to move to the staging location
            and to the mouth.
        """
        # Initialize MoveToTree
        super().__init__()

        # Store the parameters
        self.staging_configuration_position = staging_configuration_position
        self.staging_configuration_quat_xyzw = staging_configuration_quat_xyzw
        self.end_configuration = end_configuration
        assert len(self.end_configuration) == 6, "Must provide 6 joint positions"
        self.staging_configuration_tolerance = staging_configuration_tolerance
        self.end_configuration_tolerance = end_configuration_tolerance
        self.orientation_constraint_to_staging_configuration_quaternion = (
            orientation_constraint_to_staging_configuration_quaternion
        )
        self.orientation_constraint_to_staging_configuration_tolerances = (
            orientation_constraint_to_staging_configuration_tolerances
        )
        self.orientation_constraint_to_end_configuration_quaternion = (
            orientation_constraint_to_end_configuration_quaternion
        )
        self.orientation_constraint_to_end_configuration_tolerances = (
            orientation_constraint_to_end_configuration_tolerances
        )
        self.planner_id = planner_id
        self.allowed_planning_time_to_staging_configuration = (
            allowed_planning_time_to_staging_configuration
        )
        self.allowed_planning_time_to_end_configuration = (
            allowed_planning_time_to_end_configuration
        )
        self.wheelchair_collision_object_id = wheelchair_collision_object_id
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold

    def create_move_to_tree(
        self,
        name: str,
        tree_root_name: str,
        logger: logging.Logger,
        node: Node,
    ) -> py_trees.trees.BehaviourTree:
        """
        Creates the MoveToMouth behaviour tree.

        Parameters
        ----------
        name: The name of the behaviour tree.
        tree_root_name: The name of the tree. This is necessary because sometimes
            trees create subtrees, but still need to track the top-level tree
            name to read/write the correct blackboard variables.
        logger: The logger to use for the behaviour tree.
        node: The ROS2 node that this tree is associated with. Necessary for
            behaviours within the tree connect to ROS topics/services/actions.

        Returns
        -------
        tree: The behaviour tree that moves the robot above the plate.
        """

        # Separate the namespace of each sub-behavior
        pre_moveto_config_prefix = "pre_moveto_config"
        allow_wheelchair_collision_prefix = "allow_wheelchair_collision"
        move_to_staging_configuration_prefix = "move_to_staging_configuration"
        disallow_wheelchair_collision_prefix = "disallow_wheelchair_collision"
        move_to_end_configuration_prefix = "move_to_end_configuration"
        turn_watchdog_listener_on_prefix = "turn_watchdog_listener_on"

        # Configure the force-torque sensor and thresholds before moving
        pre_moveto_config_name = Blackboard.separator.join(
            [name, pre_moveto_config_prefix]
        )
        pre_moveto_config_behavior = pre_moveto_config(
            name=pre_moveto_config_name,
            f_mag=self.force_threshold,
            t_mag=self.torque_threshold,
            logger=logger,
        )

        # Create the behavior to allow collisions between the robot and the
        # wheelchair collision object. The wheelchair collision object is
        # intentionally expanded to nsure the robot gets nowhere close to the
        # user during acquisition, but during transfer the robot must get close
        # to the user so the wheelchair collision object must be allowed.
        allow_wheelchair_collision = get_toggle_collision_object_behavior(
            name,
            allow_wheelchair_collision_prefix,
            node,
            self.wheelchair_collision_object_id,
            True,
            logger,
        )

        # Create the behaviour to move the robot to the staging configuration
        move_to_staging_configuration_name = Blackboard.separator.join(
            [name, move_to_staging_configuration_prefix]
        )
        move_to_staging_configuration = (
            MoveToPoseWithPosePathConstraintsTree(
                position_goal=self.staging_configuration_position,
                quat_xyzw_goal=self.staging_configuration_quat_xyzw,
                tolerance_position_goal=self.staging_configuration_tolerance,
                tolerance_orientation_goal=(0.6, 0.5, 0.5),
                parameterization_orientation_goal=1,  # Rotation vector
                cartesian=True,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time_to_staging_configuration,
                quat_xyzw_path=self.orientation_constraint_to_staging_configuration_quaternion,
                tolerance_orientation_path=(
                    self.orientation_constraint_to_staging_configuration_tolerances
                ),
                parameterization_orientation_path=1,  # Rotation vector
            )
            .create_tree(
                move_to_staging_configuration_name,
                self.action_type,
                tree_root_name,
                logger,
                node,
            )
            .root
        )

        # Create the behavior to disallow collisions between the robot and the
        # wheelchair collision object.
        gen_disallow_wheelchair_collision = partial(
            get_toggle_collision_object_behavior,
            name,
            disallow_wheelchair_collision_prefix,
            node,
            self.wheelchair_collision_object_id,
            False,
            logger,
        )

        # Create the behaviour to move the robot to the end configuration
        move_to_end_configuration_name = Blackboard.separator.join(
            [name, move_to_end_configuration_prefix]
        )
        move_to_end_configuration = (
            MoveToConfigurationWithPosePathConstraintsTree(
                joint_positions_goal=self.end_configuration,
                tolerance_joint_goal=self.end_configuration_tolerance,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time_to_end_configuration,
                quat_xyzw_path=self.orientation_constraint_to_end_configuration_quaternion,
                tolerance_orientation_path=(
                    self.orientation_constraint_to_end_configuration_tolerances
                ),
                parameterization_orientation_path=1,  # Rotation vector
            )
            .create_tree(
                move_to_end_configuration_name,
                self.action_type,
                tree_root_name,
                logger,
                node,
            )
            .root
        )

        # Link all the behaviours together in a sequence with memory
        move_from_mouth = py_trees.composites.Sequence(
            name=name + " Main",
            memory=True,
            children=[
                # For now, we only re-tare the F/T sensor once, since no large forces
                # are expected during transfer.
                pre_moveto_config_behavior,
                allow_wheelchair_collision,
                move_to_staging_configuration,
                gen_disallow_wheelchair_collision(),
                move_to_end_configuration,
            ],
        )
        move_from_mouth.logger = logger

        # If there was a failure in the main tree, we want to ensure to turn
        # the watchdog listener back on
        turn_watchdog_listener_on = get_toggle_watchdog_listener_behavior(
            name,
            turn_watchdog_listener_on_prefix,
            True,
            logger,
        )

        # Create a cleanup branch for the behaviors that should get executed if
        # the main tree has a failure
        cleanup_tree = py_trees.composites.Sequence(
            name=name + " Cleanup",
            memory=True,
            children=[
                gen_disallow_wheelchair_collision(),
                turn_watchdog_listener_on,
            ],
        )

        # If move_from_mouth fails, we still want to do some cleanup (e.g., turn
        # face detection off).
        root = py_trees.composites.Selector(
            name=name,
            memory=True,
            children=[
                move_from_mouth,
                # Even though we are cleaning up the tree, it should still
                # pass the failure up.
                py_trees.decorators.SuccessIsFailure(
                    name + " Cleanup Root", cleanup_tree
                ),
            ],
        )
        root.logger = logger

        # raise Exception(py_trees.display.unicode_blackboard())

        tree = py_trees.trees.BehaviourTree(root)
        return tree
