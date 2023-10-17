#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines behaviors that take an AcquisitionSchema.msg object
(or return from AcquisitionSelect.srv) and computes the outputs
needed to send MoveIt2Plan calls.
"""

# Standard imports
from copy import deepcopy
from typing import Union, Optional

# Third-party imports
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, TransformStamped
import numpy as np
from overrides import override
import py_trees
import rclpy
import ros2_numpy
from scipy.spatial.transform import Rotation as R

# Local imports
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
    get_tf_object,
    set_static_tf,
)
from ada_feeding.idioms.pre_moveto_config import create_ft_thresh_request
from ada_feeding_msgs.msg import AcquisitionSchema
from ada_feeding_msgs.srv import AcquisitionSelect


class ComputeActionConstraints(BlackboardBehavior):
    """
    Checks AcquisitionSelect response, implements stochastic
    policy choice, and then decomposes into individual
    BlackboardKey objects for MoveIt2 Behaviors.

    Also sets static TF from food -> approach frame
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    # pylint: disable=too-many-arguments
    # These are effectively config definitions
    # They require a lot of arguments.

    def blackboard_inputs(
        self,
        action_select_response: Union[BlackboardKey, AcquisitionSelect.Response],
        move_above_dist_m: Union[BlackboardKey, float] = 0.05,
        food_frame_id: Union[BlackboardKey, str] = "food",
        approach_frame_id: Union[BlackboardKey, str] = "approach",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        action_select_response: response from AcquisitionSelect.srv
        move_above_dist_m: how far from the food to start
        food_frame_id: food frame defined in AcquisitionSchema.msg
        approach_frame_id: approach frame defined in AcquisitionSchema.msg
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        move_above_pose: Optional[BlackboardKey],  # Pose, in Food Frame
        move_into_pose: Optional[BlackboardKey],  # Pose, in Food Frame
        approach_thresh: Optional[BlackboardKey],  # SetParameters.Request
        action: Optional[BlackboardKey],  # AcquisitionSchema.msg
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        move_above_pose: Pose constraint when moving above food
        move_into_pose: Pose constraint when moving into food
        approach_thresh: SetParameters request to set thresholds pre-approach
        action: AcquisitionSchema object to use in later computations
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists("action_select_response"):
            self.logger.error("Missing Action Select Response")
            return py_trees.common.Status.FAILURE
        response = self.blackboard_get("action_select_response")
        if response.status != "Success":
            self.logger.error(f"Bad Action Select Response: {response.status}")
            return py_trees.common.Status.FAILURE
        prob = np.array(response.probabilities)
        if (
            len(response.actions) == 0
            or len(response.actions) != len(response.probabilities)
            or not np.isclose(np.sum(prob), 1.0)
        ):
            self.logger.error(f"Malformed action select response: {response}")
            return py_trees.common.Status.FAILURE

        # Sample Action
        index = np.random.choice(np.arange(prob.size), p=prob)
        action = response.actions[index]

        ### Construct Constraints

        # Re-scale pre-transform to move_above_dist_m
        position = ros2_numpy.numpify(action.pre_transform.position)
        if np.isclose(np.linalg.norm(position), 0.0):
            self.logger.error(
                f"Malformed action pre_transform: {action.pre_transform.position}"
            )
            return py_trees.common.Status.FAILURE
        position = (
            position
            * self.blackboard_get("move_above_dist_m")
            / np.linalg.norm(position)
        )
        action.pre_transform.position = ros2_numpy.msgify(Point, position)
        self.blackboard_set("move_above_pose", action.pre_transform)

        # Calculate Approach Target (in food frame)
        move_into_pose = Pose()
        move_into_pose.orientation = deepcopy(action.pre_transform.orientation)
        offset = ros2_numpy.numpify(action.pre_offset)
        move_into_pose.position = ros2_numpy.msgify(Point, offset)
        self.blackboard_set("move_into_pose", move_into_pose)

        ### Calculate Approach Frame
        approach_vec = offset - position
        approach_frame = TransformStamped()
        approach_mat = np.eye(4)
        if not np.all(np.isclose(approach_vec[:2], np.zeros(2))):
            approach_mat[:3, :3] = R.from_rotvec(
                np.arctan2(approach_vec[1], approach_vec[0]) * np.array([0, 0, 1])
            ).as_matrix()
        approach_frame.transform = ros2_numpy.msgify(Transform, approach_mat)
        approach_frame.header.stamp = self.node.get_clock().now().to_msg()
        approach_frame.header.frame_id = "food"
        approach_frame.child_frame_id = "approach"
        set_static_tf(approach_frame, self.blackboard, self.node)

        ### Calculate F/T Thresholds
        self.blackboard_set(
            "approach_thresh",
            create_ft_thresh_request(action.pre_force, action.pre_torque),
        )
        self.blackboard_set(
            "grasp_thresh",
            create_ft_thresh_request(action.grasp_force, action.grasp_torque),
        )
        self.blackboard_set(
            "ext_thresh",
            create_ft_thresh_request(action.ext_force, action.ext_torque),
        )

        ### Final write to Blackboard
        self.blackboard_set("action", action)
        return py_trees.common.Status.SUCCESS


class ComputeActionTwist(BlackboardBehavior):
    """
    DEPRECATED
    Decomposes AcquisitionSchema msg into individual
    BlackboardKey objects for MoveIt2 Extraction Behaviors.
    """


class ComputeExtractConstraints(BlackboardBehavior):
    """
    DEPRECATED
    Decomposes AcquisitionSchema msg into individual
    BlackboardKey objects for MoveIt2 Extraction Behaviors.
    """

    # pylint: disable=arguments-differ
    # We *intentionally* violate Liskov Substitution Princple
    # in that blackboard config (inputs + outputs) are not
    # meant to be called in a generic setting.

    # pylint: disable=too-many-arguments
    # These are effectively config definitions
    # They require a lot of arguments.

    def blackboard_inputs(
        self,
        action: Union[BlackboardKey, AcquisitionSchema],
        approach_frame_id: Union[BlackboardKey, str] = "approach",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        action: AcquisitionSchema msg object
        approach_frame_id: approach frame defined in AcquisitionSchema.msg
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        extract_position: Optional[BlackboardKey],  # Position, in approach frame
        extract_orientation: Optional[BlackboardKey],  # Quaternion, in forkTip frame
        ft_thresh: Optional[BlackboardKey],  # SetParameters.Request
        ee_frame_id: Optional[BlackboardKey] = None,  # str
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        extract_position: Postition constraint when moving out of food
        extract_orientation: Orientation constraint when moving out of food
        ft_thresh: SetParameters request to set thresholds pre-extraction
        ee_frame_id: end-effector frame for the extract_orientation constraint
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_outputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    @override
    def setup(self, **kwargs):
        # Docstring copied from @override

        # pylint: disable=attribute-defined-outside-init
        # It is okay for attributes in behaviors to be
        # defined in the setup / initialise functions.

        # Get Node from Kwargs
        self.node = kwargs["node"]

        # Get TF Listener from blackboard
        # For transform approach -> end_effector_frame
        self.tf_buffer, _, self.tf_lock = get_tf_object(self.blackboard, self.node)

        # Get the MoveIt2 object.
        # To get the end_effector_link name (and frame)
        self.moveit2, self.moveit2_lock = get_moveit2_object(
            self.blackboard,
            self.node,
        )

    @override
    def update(self) -> py_trees.common.Status:
        # Docstring copied from @override

        # Input Validation
        if not self.blackboard_exists("action"):
            self.logger.error("Missing AcquisitionSchema action")
            return py_trees.common.Status.FAILURE
        action = self.blackboard_get("action")
        approach_frame_id = self.blackboard_get("approach_frame_id")

        ### Lock used objects
        if self.tf_lock.locked() or self.moveit2_lock.locked():
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING
        with self.tf_lock, self.moveit2_lock:
            # Get TF EE frame -> approach frame
            if not self.tf_buffer.can_transform(
                approach_frame_id,
                self.moveit2.end_effector_name,
                rclpy.time.Time(),
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                return py_trees.common.Status.RUNNING
            utensil_to_approach_transform = self.tf_buffer.lookup_transform(
                approach_frame_id,
                self.moveit2.end_effector_name,
                rclpy.time.Time(),
            )
            ee_position_np = ros2_numpy.numpify(
                utensil_to_approach_transform.transform.translation
            )
            self.blackboard_set("ee_frame_id", self.moveit2.end_effector_name)

            ### Construct Constraints

            # Calculate Extract position (in approach frame)
            position = ros2_numpy.numpify(action.ext_linear)
            dur_s = float(action.ext_duration.sec) + (
                float(action.ext_duration.nanosec) / 10e9
            )
            position = position * dur_s
            position = position + ee_position_np  # Offset from current position
            extract_position = ros2_numpy.msgify(Point, position[:3])
            self.blackboard_set("extract_position", extract_position)

            # Calculate extract orientation (in forktip frame)
            rot_vec = ros2_numpy.numpify(action.ext_angular)
            rot_vec = rot_vec * dur_s
            extract_orientation = ros2_numpy.msgify(
                Quaternion, R.from_rotvec(rot_vec).as_quat()
            )
            self.blackboard_set("extract_orientation", extract_orientation)

            # Calculate Approach F/T Thresholds
            self.blackboard_set(
                "ft_thresh",
                create_ft_thresh_request(action.ext_force, action.ext_torque),
            )

        return py_trees.common.Status.SUCCESS