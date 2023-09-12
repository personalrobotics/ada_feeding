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
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.node import Node
from std_srvs.srv import SetBool

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding.behaviors import (
    ComputeMoveToMouthPosition,
    MoveCollisionObject,
    ToggleCollisionObject,
)
from ada_feeding.helpers import (
    POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX,
)
from ada_feeding.idioms import pre_moveto_config, retry_call_ros_service
from ada_feeding.trees import (
    MoveToTree,
    MoveToConfigurationWithPosePathConstraintsTree,
    MoveToPoseWithPosePathConstraintsTree,
)


class MoveToMouthTree(MoveToTree):
    """
    A behaviour tree that moves the robot to a specified configuration.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Bite transfer is a big part of the robot's behavior, so it makes
    # sense that it has lots of attributes/arguments.

    def __init__(
        self,
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
        head_object_id: str = "head",
        wheelchair_collision_object_id: str = "wheelchair_collision",
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
        allowed_face_distance: float = 1.5,
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
        allowed_face_distance: The maximum distance (m) between a face and the
            **camera's optical frame** for the robot to move towards the face.
        """

        # pylint: disable=too-many-locals
        # These are all necessary due to all the behaviors MoveToMouth contains

        # Initialize MoveToTree
        super().__init__()

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
            if distance <= self.allowed_face_distance:
                return True
        return False

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
        # pylint: disable=too-many-locals, too-many-statements
        # This function creates all the behaviors of the tree, which is why
        # it has so many locals/statements.
        # TODO: consider separating each behavior into its own function to simplify this.

        # Separate the namespace of each sub-behavior
        turn_face_detection_on_prefix = "turn_face_detection_on"
        pre_moveto_config_prefix = "pre_moveto_config"
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
            service_name="~/toggle_face_detection",
            key_request=None,
            request=SetBool.Request(data=True),
            key_response=turn_face_detection_on_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=turn_face_detection_on_key_response + ".success",
                    value=True,
                    operator=operator.eq,
                )
            ],
            logger=logger,
        )

        # Configure the force-torque sensor and thresholds before moving
        pre_moveto_config_name = Blackboard.separator.join(
            [name, pre_moveto_config_prefix]
        )
        pre_moveto_config_behavior = pre_moveto_config(
            name=pre_moveto_config_name,
            toggle_watchdog_listener=False,
            f_mag=self.force_threshold,
            t_mag=self.torque_threshold,
            logger=logger,
        )

        # Create the behaviour to move the robot to the staging configuration
        move_to_staging_configuration_name = Blackboard.separator.join(
            [name, move_to_staging_configuration_prefix]
        )
        move_to_staging_configuration = (
            MoveToConfigurationWithPosePathConstraintsTree(
                joint_positions_goal=self.staging_configuration,
                tolerance_joint_goal=self.staging_configuration_tolerance,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time_to_staging_configuration,
                max_velocity_scaling_factor=(
                    self.max_velocity_scaling_factor_to_staging_configuration
                ),
                quat_xyzw_path=self.orientation_constraint_quaternion,
                tolerance_orientation_path=self.orientation_constraint_tolerances,
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

        # Create the behaviour to subscribe to the face detection topic until
        # a face is detected
        get_face_name = Blackboard.separator.join([name, get_face_prefix])
        get_face = py_trees_ros.subscribers.ToBlackboard(
            name=get_face_name,
            topic_name="~/face_detection",
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
                    variable="face_detection",
                    value=FaceDetection(),
                    operator=self.check_face_msg,
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

        # Create the behavior to allow collisions between the robot and the
        # wheelchair collision object. The wheelchair collision object is
        # intentionally expanded to nsure the robot gets nowhere close to the
        # user during acquisition, but during transfer the robot must get close
        # to the user so the wheelchair collision object must be allowed.
        allow_wheelchair_collision_prefix = "allow_wheelchair_collision"
        allow_wheelchair_collision = ToggleCollisionObject(
            name=Blackboard.separator.join([name, allow_wheelchair_collision_prefix]),
            node=node,
            collision_object_id=self.wheelchair_collision_object_id,
            allow=True,
        )
        allow_wheelchair_collision.logger = logger

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
                position_goal=(0.0, 0.0, 0.0),
                quat_xyzw_goal=(0.0, 0.7071068, 0.7071068, 0.0),
                tolerance_position_goal=self.mouth_pose_tolerance,
                tolerance_orientation_goal=(0.6, 0.5, 0.5),
                parameterization_orientation_goal=1,  # Rotation vector
                cartesian=False,
                planner_id=self.planner_id,
                allowed_planning_time=self.allowed_planning_time_to_mouth,
                max_velocity_scaling_factor=self.max_velocity_scaling_factor_to_mouth,
                quat_xyzw_path=self.orientation_constraint_quaternion,
                tolerance_orientation_path=self.orientation_constraint_tolerances,
                parameterization_orientation_path=1,  # Rotation vector
                keys_to_not_write_to_blackboard={
                    Blackboard.separator.join(
                        [POSITION_GOAL_CONSTRAINT_NAMESPACE_PREFIX, "position"],
                    ),
                },
            )
            .create_tree(
                move_to_target_pose_name, self.action_type, tree_root_name, logger, node
            )
            .root
        )

        # # NOTE: The below is commented out for the time being, because if
        # # we disallow collisions between the robot and the wheelchair
        # # collision object, then the robot will not be able to do any motions
        # # after this (since it is in collision with the wheelchair collision
        # # object at the end of transfer).
        # #
        # # The right solution is to create a separate `MoveAwayFromMouth` tree
        # # that moves the robot back to the staging location, and then disallows
        # # this collision before moving back to the staging location.

        # # Create the behavior to disallow collisions between the robot and the
        # # wheelchair collision object.
        # disallow_wheelchair_collision_prefix = "disallow_wheelchair_collision"
        # disallow_wheelchair_collision = ToggleCollisionObject(
        #     name=Blackboard.separator.join([name, disallow_wheelchair_collision_prefix]),
        #     node=node,
        #     collision_object_id=self.wheelchair_collision_object_id,
        #     allow=False,
        # )
        # disallow_wheelchair_collision.logger = logger

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
            service_name="~/toggle_face_detection",
            key_request=None,
            request=SetBool.Request(data=False),
            key_response=turn_face_detection_off_key_response,
            response_checks=[
                py_trees.common.ComparisonExpression(
                    variable=turn_face_detection_off_key_response + ".success",
                    value=True,
                    operator=operator.eq,
                )
            ],
            logger=logger,
        )

        # Link all the behaviours together in a sequence with memory
        move_to_mouth = py_trees.composites.Sequence(
            name=name + " Main",
            memory=True,
            children=[
                turn_face_detection_on,
                # For now, we only re-tare the F/T sensor once, since no large forces
                # are expected during transfer.
                pre_moveto_config_behavior,
                move_to_staging_configuration,
                detect_face,
                compute_target_position,
                move_head,
                allow_wheelchair_collision,
                move_to_target_pose,
                # disallow_wheelchair_collision,
                turn_face_detection_off,
            ],
        )
        move_to_mouth.logger = logger

        # If move_to_mouth fails, we still want to do some cleanup (e.g., turn
        # face detection off).
        root = py_trees.composites.Selector(
            name=name,
            memory=True,
            children=[
                move_to_mouth,
                # Even though we are cleaning up the tree, it should still
                # pass the failure up.
                py_trees.decorators.SuccessIsFailure(
                    name + " Cleanup", turn_face_detection_off
                ),
            ],
        )
        root.logger = logger

        # raise Exception(py_trees.display.unicode_blackboard())

        tree = py_trees.trees.BehaviourTree(root)
        return tree
