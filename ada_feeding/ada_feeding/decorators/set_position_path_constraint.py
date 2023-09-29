#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the SetPositionPathConstraint decorator, which adds a path
constraint that keeps a specified frame within a secified tolerance of a
specified position.
"""
# Third-party imports
import py_trees
from rclpy.node import Node

# Local imports
from ada_feeding.decorators import MoveToConstraint
from ada_feeding.helpers import get_from_blackboard_with_default, get_moveit2_object

# pylint: disable=duplicate-code
# All the constraints have similar code when registering and setting blackboard
# keys, since the parameters for constraints are similar. This is not a problem.


class SetPositionPathConstraint(MoveToConstraint):
    """
    SetPositionPathConstraint adds a path constraint that keeps a specified frame
    within a secified tolerance of a specified position.
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
        node: Node,
        fk_poses_key: Optional[str] = None,
        fk_links_key: Optional[str] = None,
    ):
        """
        Initialize the MoveToConstraint decorator.

        Parameters
        ----------
        name: The name of the behavior.
        child: The child behavior.
        node: The ROS node to associate the service call with.
        fk_poses_key: The key for the forward kinematics poses.
        fk_links_key: The key for the forward kinematics links.
        """
        # Initiatilize the decorator
        super().__init__(name=name, child=child)
        self.fk_poses_key = fk_poses_key
        self.fk_links_key = fk_links_key

        # Define inputs from the blackboard
        self.blackboard = self.attach_blackboard_client(
            name=name + " SetPositionPathConstraint", namespace=name
        )
        self.blackboard.register_key(key="position", access=py_trees.common.Access.READ)
        self.blackboard.register_key(key="frame_id", access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key="target_link", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(
            key="tolerance", access=py_trees.common.Access.READ
        )
        self.blackboard.register_key(key="weight", access=py_trees.common.Access.READ)
        if self.fk_poses_key is not None and self.fk_links_key is not None:
            self.blackboard.register_key(
                key=self.fk_poses_key, access=py_trees.common.Access.READ
            )
            self.blackboard.register_key(
                key=self.fk_links_key, access=py_trees.common.Access.READ
            )

        # Get the MoveIt2 object.
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            node,
        )

    def is_constraint_satisfied_at_start(
        self,
        position: Tuple[float, float, float],
        frame_id: str,
        target_link: str,
        tolerance: float,
    ) -> bool:
        """
        Checks whether the constraint is satisfied by the start configuration.

        Parameters
        ----------
        position: The position to check.
        frame_id: The frame ID for the position.
        target_link: The target link for the constraint.
        tolerance: The tolerance for the constraint.

        Returns
        -------
        Returns True if the starting configuration satisfies the constraint,
        and False if it does not or if this function cannot determine whether
        it does.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # All arguments are necessary.

        # Check whether the user asked to check the starting FK.
        if self.fk_poses_key is None or self.fk_links_key is None:
            self.logger.debug(
                f"{self.name} [SetPositionPathConstraint::"
                "is_constraint_satisfied_at_start()] "
                "Will not check whether the constraint is satisfied at start, "
                "as specified by the parameters."
            )
            return False

        # Get the FK links and poses from the balckboard
        try:
            fk_links = self.blackboard.get(self.fk_links_key)
            fk_poses = self.blackboard.get(self.fk_poses_key)
        except KeyError:
            self.logger.warning(
                f"{self.name} [SetPositionPathConstraint::is_constraint_satisfied_at_start()] "
                "The forward kinematics links and poses were not found on the "
                "blackboard. Setting the position path constraint regardless "
                "of the starting FK."
            )
            return False

        # Get the index of the target link in the forward kinematics links.
        try:
            target_link_index = fk_links.index(target_link)
        except ValueError:
            self.logger.warning(
                f"{self.name} [SetPositionPathConstraint::is_constraint_satisfied_at_start()] "
                "The target link was not found in the forward kinematics "
                "links. Setting the position path constraint regardless "
                "of the starting FK."
            )
            return False

        # Verify that we have a pose for the link
        if target_link_index >= len(fk_poses):
            self.logger.warning(
                f"{self.name} [SetPositionPathConstraint::is_constraint_satisfied_at_start()] "
                "The target link index is out of bounds. Setting the "
                "position path constraint regardless of the starting FK."
            )
            return False

        # Get the pose of the target link
        target_link_pose = fk_poses[target_link_index]

        # Verify that the pose is in the right frame_id
        if frame_id is None:
            with self.moveit2_lock:
                frame_id = self.moveit2.base_link_name
        if target_link_pose.header.frame_id != frame_id:
            self.logger.warning(
                f"{self.name} [SetPositionPathConstraint::is_constraint_satisfied_at_start()] "
                f"The target link pose is in frame {target_link_pose.header.frame_id}, "
                f"not the requested frame {frame_id}. Setting the position path "
                "constraint regardless of the starting FK."
            )
            return False

    def set_constraint(self) -> None:
        """
        Sets the position path constraint.
        """
        self.logger.info(f"{self.name} [SetPositionPathConstraint::set_constraint()]")

        # Get all parameters for planning, resorting to default values if unset.
        position = self.blackboard.position  # required
        frame_id = get_from_blackboard_with_default(self.blackboard, "frame_id", None)
        target_link = get_from_blackboard_with_default(
            self.blackboard, "target_link", None
        )
        tolerance = get_from_blackboard_with_default(
            self.blackboard, "tolerance", 0.001
        )
        weight = get_from_blackboard_with_default(self.blackboard, "weight", 1.0)

        # Check whether the constraint is satisfied by the start configuration.
        constraint_satisfied_at_start = self.is_constraint_satisfied_at_start(
            position=position,
            frame_id=frame_id,
            target_link=target_link,
            tolerance=tolerance,
        )

        # Set the constraint
        if constraint_satisfied_at_start:
        with self.moveit2_lock:
            self.moveit2.set_path_position_constraint(
                position=position,
                frame_id=frame_id,
                target_link=target_link,
                tolerance=tolerance,
                weight=weight,
            )
