#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToMouthTree behaviour tree and provides functions to
wrap that behaviour tree in a ROS2 action server.
"""
# pylint: disable=duplicate-code
# MoveFromMouth and MoveToMouth are inverses of each other, so it makes sense
# that they have similar code.

# Standard imports
from typing import List, Tuple

# Third-party imports
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.node import Node

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding.behaviors import (
    ComputeMoveToMouthPosition,
    ModifyCollisionObject,
    ModifyCollisionObjectOperation,
)
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2JointConstraint,
    MoveIt2PositionConstraint,
    MoveIt2OrientationConstraint,
)
from ada_feeding.helpers import (
    BlackboardKey,
    POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
)
from ada_feeding.idioms import pre_moveto_config, scoped_behavior
from ada_feeding.idioms.bite_transfer import (
    get_add_in_front_of_wheelchair_wall_behavior,
    get_toggle_collision_object_behavior,
    get_toggle_face_detection_behavior,
    get_remove_in_front_of_wheelchair_wall_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
)


class MoveToMouthTree(MoveToTree):
    """
    A behaviour tree that toggles face detection on, moves the robot to the
    staging configuration, detects a face, checks whether it is within a
    distance threshold from the camera, and does a cartesian motion to the
    face.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Bite transfer is a big part of the robot's behavior, so it makes
    # sense that it has lots of attributes/arguments.

    def __init__(
        self,
        node: Node,
        staging_configuration: List[float],
        staging_configuration_tolerance: float = 0.001,
        mouth_pose_tolerance: float = 0.001,
        orientation_constraint_quaternion: List[float] = None,
        orientation_constraint_tolerances: List[float] = None,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time_to_staging_configuration: float = 0.5,
        allowed_planning_time_to_mouth: float = 0.5,
        max_velocity_scaling_factor_to_staging_configuration: float = 0.1,
        max_velocity_scaling_factor_to_mouth: float = 0.1,
        cartesian_jump_threshold_to_mouth: float = 0.0,
        cartesian_max_step_to_mouth: float = 0.0025,
        head_object_id: str = "head",
        wheelchair_collision_object_id: str = "wheelchair_collision",
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
        allowed_face_distance: Tuple[float, float] = (0.4, 1.5),
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        staging_configuration: The joint positions to move the robot arm to.
            The user's face should be visible in this configuration.
        staging_configuration_tolerance: The tolerance for the joint positions.
        mouth_pose_tolerance: The tolerance for the movement to the mouth pose.
        orientation_constraint_quaternion: The quaternion for the orientation
            constraint. If None, the orientation constraint is not used.
        orientation_constraint_tolerances: The tolerances for the orientation
            constraint, as a 3D rotation vector. If None, the orientation
            constraint is not used.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time_to_staging_configuration: The allowed planning
            time for the MoveIt2 motion planner to move to the staging config.
        allowed_planning_time_to_mouth: The allowed planning time for the MoveIt2
            motion planner to move to the user's mouth.
        max_velocity_scaling_factor_to_staging_configuration: The maximum
            velocity scaling factor for the MoveIt2 motion planner to move to
            the staging config.
        max_velocity_scaling_factor_to_mouth: The maximum velocity scaling
            factor for the MoveIt2 motion planner to move to the user's mouth.
        cartesian_jump_threshold_to_mouth: The maximum allowed jump in the
            cartesian space for the MoveIt2 motion planner to move to the user's
            mouth.
        cartesian_max_step_to_mouth: The maximum allowed step in the cartesian
            space for the MoveIt2 motion planner to move to the user's mouth.
        head_object_id: The ID of the head collision object in the MoveIt2
            planning scene.
        wheelchair_collision_object_id: The ID of the wheelchair collision object
            in the MoveIt2 planning scene.
        force_threshold: The force threshold (N) for the ForceGateController.
            For now, the same threshold is used to move to the staging location
            and to the mouth.
        torque_threshold: The torque threshold (N*m) for the ForceGateController.
            For now, the same threshold is used to move to the staging location
            and to the mouth.
        allowed_face_distance: The min and max distance (m) between a face and the
            **camera's optical frame** for the robot to move towards the face.
        """

        # pylint: disable=too-many-locals
        # These are all necessary due to all the behaviors MoveToMouth contains

        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters
        self.staging_configuration = staging_configuration
        assert len(self.staging_configuration) == 6, "Must provide 6 joint positions"
        self.staging_configuration_tolerance = staging_configuration_tolerance
        self.mouth_pose_tolerance = mouth_pose_tolerance
        self.orientation_constraint_quaternion = orientation_constraint_quaternion
        self.orientation_constraint_tolerances = orientation_constraint_tolerances
        self.planner_id = planner_id
        self.allowed_planning_time_to_staging_configuration = (
            allowed_planning_time_to_staging_configuration
        )
        self.allowed_planning_time_to_mouth = allowed_planning_time_to_mouth
        self.max_velocity_scaling_factor_to_staging_configuration = (
            max_velocity_scaling_factor_to_staging_configuration
        )
        self.max_velocity_scaling_factor_to_mouth = max_velocity_scaling_factor_to_mouth
        self.cartesian_jump_threshold_to_mouth = cartesian_jump_threshold_to_mouth
        self.cartesian_max_step_to_mouth = cartesian_max_step_to_mouth
        self.head_object_id = head_object_id
        self.wheelchair_collision_object_id = wheelchair_collision_object_id
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.allowed_face_distance = allowed_face_distance

    def check_face_msg(self, msg: FaceDetection, _: FaceDetection) -> bool:
        """
        Checks if a face is detected in the message and the face is within
        the required distance from the camera optical frame.

        Parameters
        ----------
        msg: The message that was written to the blackboard
        _: The comparison message.

        Returns
        -------
        True if a face is detected within the required distance, False otherwise.
        """
        if msg.is_face_detected:
            # Check the distance between the face and the camera optical frame
            # The face detection message is in the camera optical frame
            # The camera optical frame is the parent frame of the face detection
            # frame, so we can just use the translation of the face detection
            # frame to get the distance between the face and the camera optical
            # frame.
            distance = (
                msg.detected_mouth_center.point.x**2.0
                + msg.detected_mouth_center.point.y**2.0
                + msg.detected_mouth_center.point.z**2.0
            ) ** 0.5
            if (
                self.allowed_face_distance[0]
                <= distance
                <= self.allowed_face_distance[1]
            ):
                return True
        return False

    @override
    def create_tree(
        self,
        name: str,
        tree_root_name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # TODO: Remove all code in this block
        self.blackboard = py_trees.blackboard.Client(
            name=name + " Tree", namespace=name
        )
        add_wheelchair_wall_prefix = "add_wheelchair_wall"
        remove_wheelchair_wall_prefix = "remove_wheelchair_wall"
        turn_face_detection_on_prefix = "turn_face_detection_on"
        target_position_output_key = "/" + Blackboard.separator.join(
            [
                name,
                "move_to_target_pose",
                POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
                "position",
            ]
        )
        move_head_prefix = "move_head"
        frame_id_key = Blackboard.separator.join([move_head_prefix, "frame_id"])
        self.blackboard.register_key(
            key=frame_id_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.set(frame_id_key, "j2n6s200_link_base")
        quat_xyzw_key = Blackboard.separator.join([move_head_prefix, "quat_xyzw"])
        self.blackboard.register_key(
            key=quat_xyzw_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.set(
            quat_xyzw_key, (-0.0616284, -0.0616284, -0.704416, 0.704416)
        )
        allow_wheelchair_collision_prefix = "allow_wheelchair_collision"
        disallow_wheelchair_collision_prefix = "disallow_wheelchair_collision"
        turn_face_detection_off_prefix = "turn_face_detection_off"

        ### Define Tree Logic

        in_front_of_wheelchair_wall_id = "in_front_of_wheelchair_wall"
        # The target position is 5cm away from the mouth center, in the direction
        # away from the wheelchair backrest.
        target_position_offset = (0.0, -0.05, 0.0)

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                # Retare the F/T sensor and set the F/T Thresholds
                pre_moveto_config(
                    name=name + "PreMoveToConfig",
                    toggle_watchdog_listener=False,
                    f_mag=self.force_threshold,
                    t_mag=self.torque_threshold,
                ),
                # Add a wall in front of the wheelchair to prevent the robot
                # from moving unnecessarily close to the user.
                scoped_behavior(
                    name=name + " InFrontOfWheelchairWallScope",
                    pre_behavior=get_add_in_front_of_wheelchair_wall_behavior(
                        name,
                        add_wheelchair_wall_prefix,
                        in_front_of_wheelchair_wall_id,
                        self._node,
                        self.blackboard,
                    ),
                    # Move to the staging configuration
                    workers=[
                        # Goal configuration: staging configuration
                        MoveIt2JointConstraint(
                            name="StagingConfigurationGoalConstraint",
                            ns=name,
                            inputs={
                                "joint_positions": self.staging_configuration,
                                "tolerance": self.staging_configuration_tolerance,
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        # Orientation path constraint to keep the fork straight
                        MoveIt2OrientationConstraint(
                            name="KeepForkStraightPathConstraint",
                            ns=name,
                            inputs={
                                "quat_xyzw": self.orientation_constraint_quaternion,
                                "tolerance": self.orientation_constraint_tolerances,
                                "parameterization": 1,  # Rotation vector
                            },
                            outputs={
                                "constraints": BlackboardKey("path_constraints"),
                            },
                        ),
                        # Plan
                        MoveIt2Plan(
                            name="MoveToStagingConfigurationPlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "path_constraints": BlackboardKey("path_constraints"),
                                "planner_id": self.planner_id,
                                "allowed_planning_time": (
                                    self.allowed_planning_time_to_staging_configuration
                                ),
                                "max_velocity_scale": (
                                    self.max_velocity_scaling_factor_to_staging_configuration
                                ),
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        # Execute
                        MoveIt2Execute(
                            name="MoveToStagingConfigurationExecute",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
                            outputs={},
                        ),
                    ],
                    # Remove the wall in front of the wheelchair
                    post_behavior=get_remove_in_front_of_wheelchair_wall_behavior(
                        name,
                        remove_wheelchair_wall_prefix,
                        in_front_of_wheelchair_wall_id,
                        self._node,
                    ),
                ),
                # Turn face detection on until a face is detected
                scoped_behavior(
                    name=name + " FaceDetectionOnScope",
                    pre_behavior=get_toggle_face_detection_behavior(
                        name,
                        turn_face_detection_on_prefix,
                        True,
                    ),
                    workers=[
                        py_trees.composites.Sequence(
                            name=name,
                            memory=False,
                            children=[
                                # Get the detected face
                                py_trees_ros.subscribers.ToBlackboard(
                                    name=name + " GetFace",
                                    topic_name="~/face_detection",
                                    topic_type=FaceDetection,
                                    qos_profile=py_trees_ros.utilities.qos_profile_unlatched(),
                                    blackboard_variables={
                                        "face_detection": None,
                                    },
                                    initialise_variables={
                                        "face_detection": FaceDetection(),
                                    },
                                ),
                                # Check whether the face is within the required distance
                                # TODO: This can potentially block the tree forever,
                                # e.g., if the face is not visible. Change it!
                                py_trees.decorators.FailureIsRunning(
                                    name=name + " CheckFaceWrapper",
                                    child=py_trees.behaviours.CheckBlackboardVariableValue(
                                        name=name + " CheckFace",
                                        check=py_trees.common.ComparisonExpression(
                                            variable="face_detection",
                                            value=FaceDetection(),
                                            operator=self.check_face_msg,
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ],
                    # Turn off face detection
                    post_behavior=get_toggle_face_detection_behavior(
                        name,
                        turn_face_detection_off_prefix,
                        False,
                    ),
                ),
                # Compute the goal constraints of the target pose
                ComputeMoveToMouthPosition(
                    name=name + " ComputeTargetPosition",
                    node=self._node,
                    face_detection_input_key="/face_detection",
                    target_position_output_key=target_position_output_key,
                    target_position_frame_id="j2n6s200_link_base",
                    target_position_offset=target_position_offset,
                ),
                # Move the head to the detected face pose
                ModifyCollisionObject(
                    name=Blackboard.separator.join([name, move_head_prefix]),
                    node=self._node,
                    operation=ModifyCollisionObjectOperation.MOVE,
                    collision_object_id=self.head_object_id,
                    collision_object_position_input_key=target_position_output_key,
                    collision_object_orientation_input_key="quat_xyzw",
                    position_offset=tuple(-1.0 * p for p in target_position_offset),
                ),
                # Allow collisions with the expanded wheelchair collision box
                scoped_behavior(
                    name=name + " AllowWheelchairCollisionScope",
                    pre_behavior=get_toggle_collision_object_behavior(
                        name,
                        allow_wheelchair_collision_prefix,
                        self._node,
                        [self.wheelchair_collision_object_id],
                        True,
                    ),
                    # Move to the target pose
                    workers=[
                        # Goal configuration: target position
                        MoveIt2PositionConstraint(
                            name="MoveToTargetPosePositionGoalConstraint",
                            ns=name,
                            inputs={
                                "position": BlackboardKey(target_position_output_key),
                                "tolerance": self.mouth_pose_tolerance,
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        # Goal configuration: target orientation
                        MoveIt2OrientationConstraint(
                            name="MoveToTargetPoseOrientationGoalConstraint",
                            ns=name,
                            inputs={
                                "quat_xyzw": (
                                    0.0,
                                    0.7071068,
                                    0.7071068,
                                    0.0,
                                ),  # Point to wheelchair backrest
                                "tolerance": (0.6, 0.5, 0.5),
                                "parameterization": 1,  # Rotation vector
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        # Orientation path constraint to keep the fork straight
                        MoveIt2OrientationConstraint(
                            name="KeepForkStraightPathConstraint",
                            ns=name,
                            inputs={
                                "quat_xyzw": self.orientation_constraint_quaternion,
                                "tolerance": self.orientation_constraint_tolerances,
                                "parameterization": 1,  # Rotation vector
                            },
                            outputs={
                                "constraints": BlackboardKey("path_constraints"),
                            },
                        ),
                        # Plan
                        MoveIt2Plan(
                            name="MoveToTargetPosePlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "path_constraints": BlackboardKey("path_constraints"),
                                "planner_id": self.planner_id,
                                "allowed_planning_time": self.allowed_planning_time_to_mouth,
                                "max_velocity_scale": (
                                    self.max_velocity_scaling_factor_to_mouth
                                ),
                                "cartesian": True,
                                "cartesian_jump_threshold": self.cartesian_jump_threshold_to_mouth,
                                "cartesian_fraction_threshold": 0.60,
                                "cartesian_max_step": self.cartesian_max_step_to_mouth,
                            },
                            outputs={"trajectory": BlackboardKey("trajectory")},
                        ),
                        # Execute
                        MoveIt2Execute(
                            name="MoveToTargetPoseExecute",
                            ns=name,
                            inputs={"trajectory": BlackboardKey("trajectory")},
                            outputs={},
                        ),
                    ],
                    # Disallow collisions with the expanded wheelchair collision
                    # box.
                    post_behavior=get_toggle_collision_object_behavior(
                        name,
                        disallow_wheelchair_collision_prefix,
                        self._node,
                        [self.wheelchair_collision_object_id],
                        False,
                    ),
                ),
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)
