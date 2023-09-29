#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the ComputeFK behavior, which computes the forward kinematics
for the specified robot links and writes the results to the blackboard.
"""

# Standard imports
from typing import List, Optional, Union

# Third-party imports
from moveit_msgs.msg import MoveItErrorCodes
import py_trees
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Local imports
from ada_feeding.helpers import get_moveit2_object


class ComputeFK(py_trees.behaviour.Behaviour):
    """
    ComputeFK is a behavior that computes the forward kinematics for the
    specified robot links and writes the results to the blackboard.
    """

    def __init__(
        self,
        name: str,
        node: Node,
        poses_output_key: str,
        links_output_key: str,
        joint_state: Optional[Union[JointState, List[float]]] = None,
        link_names: Optional[List[str]] = None,
    ):
        """
        Initializes the behavior.

        Parameters
        ----------
        name: The name of the behavior.
        node: The ROS node to associate the service call with.
        poses_output_key: The key for the poses output on the blackboard.
        links_output_key: The key to write the links that the poses correspond to.
        joint_state: The joint state to use for the forward kinematics computation.
            If None, then the current joint state is used.
        link_names: The names of the links to compute the forward kinematics for.
            If None, then the end effector link is used.
        """
        # pylint: disable=too-many-arguments
        # All arguments are necessary.

        # Initiatilize the behavior
        super().__init__(name=name)

        # Store parameters
        self.poses_output_key = poses_output_key
        self.links_output_key = links_output_key
        self.joint_state = joint_state
        self.link_names = link_names

        # Configure the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " ComputeFK", namespace=name
        )
        self.blackboard.register_key(
            key=poses_output_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.register_key(
            key=links_output_key,
            access=py_trees.common.Access.WRITE,
        )

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            node,
        )

    # pylint: disable=attribute-defined-outside-init
    # For attributes that are only used during the execution of the tree
    # and get reset before the next execution, it is reasonable to define
    # them in `initialise`.
    def initialise(self) -> None:
        """
        Reset the service_future.
        """
        self.logger.info(f"{self.name} [ComputeFK::initialise()]")

        self.service_future = None

    def update(self) -> py_trees.common.Status:
        """
        Compute the forward kinematics for the specified link names.
        """
        self.logger.info(f"{self.name} [ComputeFK::update()]")

        # Compute the forward kinematics
        if self.service_future is None:
            with self.moveit2_lock:
                service_future = self.moveit2.compute_fk_async(
                    joint_state=self.joint_state,
                    link_names=self.link_names,
                )
            if service_future is None:
                self.logger.warning(
                    f"{self.name} [ComputeFK::update()] Failed to compute FK"
                )
                return py_trees.common.Status.FAILURE
            self.service_future = service_future

        # Check if the service is done
        if self.service_future.done():
            result = self.service_future.result()
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.logger.warning(
                    f"{self.name} [ComputeFK::update()] Failed to compute FK "
                    f"with error code: {result.error_code.val}"
                )
                return py_trees.common.Status.FAILURE
            self.blackboard.set(self.poses_output_key, result.pose_stamped)
            self.blackboard.set(self.links_output_key, result.fk_link_names)

        # If not, keep waiting
        return py_trees.common.Status.RUNNING
