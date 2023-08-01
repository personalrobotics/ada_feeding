#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the MoveToMouthTree behaviour tree and provides functions to
wrap that behaviour tree in a ROS2 action server.
"""

# Standard imports
import logging
import operator
from typing import List

# Third-party imports
from ada_feeding_msgs.msg import FaceDetection
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding.behaviors import ComputeMoveToMouthPosition, MoveCollisionObject
from ada_feeding.helpers import (
    POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
)
from ada_feeding.idioms import retry_call_ros_service
from ada_feeding.trees import (
    MoveToTree,
    MoveToConfigurationWithPosePathConstraintsTree,
    MoveToPoseWithPosePathConstraintsTree,
)


class MoveToMouthTree(MoveToTree):
    """
    A behaviour tree that moves the robot to a specified configuration.
    """

    def __init__(
        self,
        action_type_class_str: str,
        staging_configuration: List[float],
        staging_configuration_tolerance: float = 0.001,
        mouth_pose_tolerance: float = 0.001,
        orientation_constraint_quaternion: List[float] = None,
        orientation_constraint_tolerances: List[float] = None,
        planner_id: str = "RRTstarkConfigDefault",
        toggle_face_detection_service_name: str = "/toggle_face_detection",
        face_detection_topic_name: str = "/face_detection",
        head_object_id: str = "head",
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        action_type_class_str: The type of action that this tree is implementing,
            e.g., "ada_feeding_msgs.action.MoveTo". The input of this action
            type can be anything, but the Feedback and Result must at a minimum
            include the fields of ada_feeding_msgs.action.MoveTo
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
        toggle_face_detection_service_name: The name of the service to toggle
            face detection on and off.
        face_detection_topic_name: The name of the topic that publishes the
            face detection results.
        head_object_id: The ID of the head collision object in the MoveIt2
            planning scene.
        """
        # Initialize MoveToTree
        super().__init__(action_type_class_str)

        # Store the parameters
        self.action_type_class_str = action_type_class_str
        self.staging_configuration = staging_configuration
        assert len(self.staging_configuration) == 6, "Must provide 6 joint positions"
        self.staging_configuration_tolerance = staging_configuration_tolerance
        self.mouth_pose_tolerance = mouth_pose_tolerance
        self.orientation_constraint_quaternion = orientation_constraint_quaternion
        self.orientation_constraint_tolerances = orientation_constraint_tolerances
        self.planner_id = planner_id
        self.toggle_face_detection_service_name = toggle_face_detection_service_name
        self.face_detection_topic_name = face_detection_topic_name
        self.head_object_id = head_object_id

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
        turn_face_detection_on_prefix = "turn_face_detection_on"
        move_to_staging_configuration_prefix = "move_to_staging_configuration"
        get_face_prefix = "get_face"
        check_face_prefix = "check_face"
        compute_target_position_prefix = "compute_target_position"
        move_to_target_pose_prefix = "move_to_target_pose"
        turn_face_detection_off_prefix = "turn_face_detection_off"

        # Create the behaviour to turn face detection on
        turn_face_detection_on_name = Blackboard.separator.join(
            [name, turn_face_detection_on_prefix]
        )
        turn_face_detection_on_key_response = Blackboard.separator.join(
            [turn_face_detection_on_name, "response"]
        )
        turn_face_detection_on = retry_call_ros_service(
            name=turn_face_detection_on_name,
            service_type=SetBool,
            service_name=self.toggle_face_detection_service_name,
            key_request=None,
            request=SetBool.Request(data=True),
            key_response=turn_face_detection_on_key_response,
            response_check=py_trees.common.ComparisonExpression(
                variable=turn_face_detection_on_key_response + ".success",
                value=True,
                operator=operator.eq,
            ),
            logger=logger,
        )

        # Create the behaviour to move the robot to the staging configuration
        move_to_staging_configuration_name = Blackboard.separator.join(
            [name, move_to_staging_configuration_prefix]
        )
        move_to_staging_configuration = (
            MoveToConfigurationWithPosePathConstraintsTree(
                action_type_class_str=self.action_type_class_str,
                joint_positions_goal=self.staging_configuration,
                tolerance_joint_goal=self.staging_configuration_tolerance,
                planner_id=self.planner_id,
                quat_xyzw_path=self.orientation_constraint_quaternion,
                tolerance_orientation_path=self.orientation_constraint_tolerances,
                parameterization_orientation_path=1,  # Rotation vector
            )
            .create_tree(
                move_to_staging_configuration_name, tree_root_name, logger, node
            )
            .root
        )

        # Create the behaviour to subscribe to the face detection topic until
        # a face is detected
        get_face_name = Blackboard.separator.join([name, get_face_prefix])
        get_face = py_trees_ros.subscribers.ToBlackboard(
            name=get_face_name,
            topic_name=self.face_detection_topic_name,
            topic_type=FaceDetection,
            qos_profile=py_trees_ros.utilities.qos_profile_unlatched(),
            blackboard_variables={
                "face_detection": None,
            },
            initialise_variables={
                "face_detection": FaceDetection(),
            },
        )
        get_face.logger = logger
        check_face_name = Blackboard.separator.join([name, check_face_prefix])
        check_face = py_trees.decorators.FailureIsRunning(
            name=check_face_name,
            child=py_trees.behaviours.CheckBlackboardVariableValue(
                name=check_face_name,
                check=py_trees.common.ComparisonExpression(
                    variable="face_detection.is_face_detected",
                    value=True,
                    operator=operator.eq,
                ),
            ),
        )
        check_face.logger = logger
        detect_face = py_trees.composites.Sequence(
            name=name,
            memory=False,
            children=[
                get_face,
                check_face,
            ],
        )
        detect_face.logger = logger

        # Create the behaviour to compute the target pose for the robot's end
        # effector
        compute_target_position_name = Blackboard.separator.join(
            [name, compute_target_position_prefix]
        )
        target_position_output_key = "/" + Blackboard.separator.join(
            [
                name,
                move_to_target_pose_prefix,
                POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
                "position",
            ]
        )
        # The target position is 5cm away from the mouth
        target_position_offset = (0.0, -0.05, 0.0)
        compute_target_position = ComputeMoveToMouthPosition(
            name=compute_target_position_name,
            node=node,
            face_detection_input_key="/face_detection",
            target_position_output_key=target_position_output_key,
            target_position_frame_id="j2n6s200_link_base",
            target_position_offset=target_position_offset,
        )
        compute_target_position.logger = logger

        # Create the behavior to move the head in the collision scene to the mouth
        # position. For now, assume the head is always perpendicular to the back
        # of the wheelchair.
        move_head_prefix = "move_head"
        move_head = MoveCollisionObject(
            name=Blackboard.separator.join([name, move_head_prefix]),
            node=node,
            collision_object_id=self.head_object_id,
            collision_object_position_input_key=target_position_output_key,
            collision_object_orientation_input_key="quat_xyzw",
            reverse_position_offset=target_position_offset,
        )
        move_head.logger = logger
        # The frame_id for the position outputted by the ComputeMoveToMouthPosition
        # behaviour is the base frame of the robot.
        frame_id_key = Blackboard.separator.join([move_head_prefix, "frame_id"])
        self.blackboard.register_key(
            key=frame_id_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.set(frame_id_key, "j2n6s200_link_base")
        # Hardcode head orientation to be perpendiculaf to the back of the wheelchair.
        quat_xyzw_key = Blackboard.separator.join([move_head_prefix, "quat_xyzw"])
        self.blackboard.register_key(
            key=quat_xyzw_key,
            access=py_trees.common.Access.WRITE,
        )
        self.blackboard.set(
            quat_xyzw_key, (-0.0616284, -0.0616284, -0.704416, 0.704416)
        )

        # Create the behaviour to move the robot to the target pose
        # We want to add a position goal, but it should come from the
        # blackboard instead of being hardcoded. For now the orientation goal
        # has the fork facing in the +y direction of the base line (towards the
        # back of the wheelchair), but eventually this should be variable
        # based on the `target_position_offset`, so the fork is always facing
        # the mouth. We also add an orientation path constraint (e.g., to keep
        # the fork straight).
        move_to_target_pose_name = Blackboard.separator.join(
            [name, move_to_target_pose_prefix]
        )
        move_to_target_pose = (
            MoveToPoseWithPosePathConstraintsTree(
                action_type_class_str=self.action_type_class_str,
                position_goal=(0.0, 0.0, 0.0),
                quat_xyzw_goal=(0.0, 0.7071068, 0.7071068, 0.0),
                tolerance_position_goal=self.mouth_pose_tolerance,
                tolerance_orientation_goal=(0.6, 0.5, 0.5),
                parameterization_orientation_goal=1,  # Rotation vector
                cartesian=False,
                planner_id=self.planner_id,
                quat_xyzw_path=self.orientation_constraint_quaternion,
                tolerance_orientation_path=self.orientation_constraint_tolerances,
                parameterization_orientation_path=1,  # Rotation vector
                keys_to_not_write_to_blackboard={
                    Blackboard.separator.join(
                        [POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX, "position"],
                    ),
                },
            )
            .create_tree(move_to_target_pose_name, tree_root_name, logger, node)
            .root
        )

        # Create the behaviour to turn face detection off
        turn_face_detection_off_name = Blackboard.separator.join(
            [name, turn_face_detection_off_prefix]
        )
        turn_face_detection_off_key_response = Blackboard.separator.join(
            [turn_face_detection_off_name, "response"]
        )
        turn_face_detection_off = retry_call_ros_service(
            name=turn_face_detection_off_name,
            service_type=SetBool,
            service_name=self.toggle_face_detection_service_name,
            key_request=None,
            request=SetBool.Request(data=False),
            key_response=turn_face_detection_off_key_response,
            response_check=py_trees.common.ComparisonExpression(
                variable=turn_face_detection_off_key_response + ".success",
                value=True,
                operator=operator.eq,
            ),
            logger=logger,
        )

        # Link all the behaviours together in a sequence with memory
        root = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                turn_face_detection_on,
                move_to_staging_configuration,
                detect_face,
                compute_target_position,
                move_head,
                move_to_target_pose,
                turn_face_detection_off,
            ],
        )
        root.logger = logger

        # raise Exception(py_trees.display.unicode_blackboard())

        tree = py_trees.trees.BehaviourTree(root)
        return tree
