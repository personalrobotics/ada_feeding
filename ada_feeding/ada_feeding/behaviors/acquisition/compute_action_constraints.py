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
from geometry_msgs.msg import (
    Point,
    Pose,
    Transform,
    TransformStamped,
    TwistStamped,
    Vector3Stamped,
)
import numpy as np
from overrides import override
import py_trees
import rclpy
import ros2_numpy
from scipy.spatial.transform import Rotation as R

# Local imports
from ada_feeding_msgs.msg import AcquisitionSchema
from ada_feeding_msgs.srv import AcquisitionSelect
from ada_feeding.behaviors import BlackboardBehavior
from ada_feeding.helpers import (
    BlackboardKey,
    get_moveit2_object,
    get_tf_object,
    set_static_tf,
)
from ada_feeding.idioms.pre_moveto_config import create_ft_thresh_request


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
        action: Union[BlackboardKey, Optional[AcquisitionSchema]] = None,
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        action_select_response: response from AcquisitionSelect.srv
        move_above_dist_m: how far from the food to start
        food_frame_id: food frame defined in AcquisitionSchema.msg
        approach_frame_id: approach frame defined in AcquisitionSchema.msg
        action: which action has been chosen in the initial pi-symmetry break
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
        grasp_thresh: Optional[BlackboardKey],  # SetParameters.Request
        ext_thresh: Optional[BlackboardKey],  # SetParameters.Request
        action: Optional[BlackboardKey],  # AcquisitionSchema.msg
        action_index: Optional[BlackboardKey],  # int
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        move_above_pose: Pose constraint when moving above food
        move_into_pose: Pose constraint when moving into food
        approach_thresh: SetParameters request to set thresholds pre-approach
        grasp_thresh: SetParameters request to set thresholds in grasp
        ext_thresh: SetParameters request to set thresholds for extraction
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
        self.logger.info(f"{self.name} [{self.__class__.__name__}::update()]")

        # Input Validation
        if not self.blackboard_exists("action_select_response"):
            self.logger.error("Missing Action Select Response")
            return py_trees.common.Status.FAILURE
        response = self.blackboard_get("action_select_response")
        if response.status != "Success":
            self.logger.error(f"Bad Action Select Response: {response.status}")
            return py_trees.common.Status.FAILURE
        prob = np.array(response.probabilities)
        self.logger.debug(f"Action Probabilities: {prob} (size {prob.size})")
        if (
            len(response.actions) == 0
            or len(response.actions) != len(response.probabilities)
            or not np.isclose(np.sum(prob), 1.0)
        ):
            self.logger.error(f"Malformed action select response: {response}")
            return py_trees.common.Status.FAILURE

        action_set = self.blackboard_get("action")
        if action_set is None:
            # Sample Action
            index = np.random.choice(np.arange(prob.size), p=prob)
            action = response.actions[index]
            self.logger.info(f"Chosen action index: {index}")

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
                create_ft_thresh_request(
                    f_mag=action.pre_force, t_mag=action.pre_torque
                ),
            )
            self.blackboard_set(
                "grasp_thresh",
                create_ft_thresh_request(
                    f_mag=action.grasp_force, t_mag=action.grasp_torque
                ),
            )
            self.blackboard_set(
                "ext_thresh",
                create_ft_thresh_request(
                    f_mag=action.ext_force, t_mag=action.ext_torque
                ),
            )

            ### Final write to Blackboard
            self.blackboard_set("action", action)
            self.blackboard_set("action_index", index)
        return py_trees.common.Status.SUCCESS


class ComputeActionTwist(BlackboardBehavior):
    """
    Converts grasp/extraction parameters into a single
    TwistStamped for use with moveit2_servo
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
        is_grasp: Union[BlackboardKey, bool] = True,
        approach_frame_id: Union[BlackboardKey, str] = "approach",
    ) -> None:
        """
        Blackboard Inputs

        Parameters
        ----------
        action: AcquisitionSchema msg object
        is_grasp: if true, use the grasp action elements, else use extraction
        approach_frame_id: approach frame defined in AcquisitionSchema.msg
        """
        # pylint: disable=unused-argument, duplicate-code
        # Arguments are handled generically in base class.
        super().blackboard_inputs(
            **{key: value for key, value in locals().items() if key != "self"}
        )

    def blackboard_outputs(
        self,
        twist: Optional[BlackboardKey],  # TwistStamped
        duration: Optional[BlackboardKey],  # float
    ) -> None:
        """
        Blackboard Outputs
        By convention (to avoid collisions), avoid non-None default arguments.

        Parameters
        ----------
        twist: twist to send to MoveIt2 Servo
        duration: how long to twist in seconds
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
        if not self.blackboard_exists(["action", "is_grasp", "approach_frame_id"]):
            self.logger.error("Missing AcquisitionSchema action")
            return py_trees.common.Status.FAILURE
        action = self.blackboard_get("action")
        linear = action.ext_linear
        angular = action.ext_angular
        duration = action.ext_duration
        if self.blackboard_get("is_grasp"):
            linear = action.grasp_linear
            angular = action.grasp_angular
            duration = action.grasp_duration
        approach_frame_id = self.blackboard_get("approach_frame_id")

        ### Lock used objects
        if self.tf_lock.locked() or self.moveit2_lock.locked():
            # Not yet, wait for it
            # Use a Timeout decorator to determine failure.
            return py_trees.common.Status.RUNNING
        with self.tf_lock, self.moveit2_lock:
            twist = TwistStamped()
            twist.header.stamp = self.node.get_clock().now().to_msg()
            twist.header.frame_id = self.moveit2.base_link_name
            ### Move Linear to Base Link Frame
            # Get TF approach frame -> base link frame
            if not self.tf_buffer.can_transform(
                self.moveit2.base_link_name,
                approach_frame_id,
                rclpy.time.Time(),
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                return py_trees.common.Status.RUNNING
            linear_stamped = Vector3Stamped()
            linear_stamped.header.frame_id = approach_frame_id
            linear_stamped.vector = linear
            twist.twist.linear = self.tf_buffer.transform(
                linear_stamped, self.moveit2.base_link_name
            ).vector

            ### Move Angular to Base Link Frame
            # Get TF EE frame -> base link frame
            if not self.tf_buffer.can_transform(
                self.moveit2.base_link_name,
                self.moveit2.end_effector_name,
                rclpy.time.Time(),
            ):
                # Not yet, wait for it
                # Use a Timeout decorator to determine failure.
                return py_trees.common.Status.RUNNING

            angular_stamped = Vector3Stamped()
            angular_stamped.header.frame_id = self.moveit2.end_effector_name
            angular_stamped.vector = angular
            twist.twist.angular = self.tf_buffer.transform(
                angular_stamped, self.moveit2.base_link_name
            ).vector
            self.blackboard_set("twist", twist)

            # Compute Duration
            dur_s = float(duration.sec) + (float(duration.nanosec) / 1e9)
            self.blackboard_set("duration", dur_s)
        return py_trees.common.Status.SUCCESS
