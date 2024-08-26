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
from collections.abc import Sequence
from typing import Annotated, Tuple

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
)
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from std_srvs.srv import Empty

# Local imports
from ada_feeding_msgs.action import MoveToMouth
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding.behaviors.ros import (
    GetTransform,
    SetStaticTransform,
    ApplyTransform,
    CreatePoseStamped,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import (
    pre_moveto_config,
    retry_call_ros_service,
    scoped_behavior,
    servo_until_pose,
    wait_for_secs,
)
from ada_feeding.idioms.bite_transfer import (
    get_toggle_collision_object_behavior,
    get_toggle_face_detection_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
)
from .activate_controller import ActivateControllerTree


class MoveToMouthTree(MoveToTree):
    """
    A behaviour tree that gets the user's mouth pose and moves the robot to it.
    Note that to get the mouth pose, the robot executes these in-order until one
    succeeds:
    1. If the face detection message included in the goal is not stale, use it.
    2. Else, attempt to detect a face from the robot's current pose and use that.
    3. Else, if there is a cached detected mouth pose on the static transform
       tree, use it.
    4. Else, fail. The user can go back to the staging configuration and re-detect
       the face.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    # Bite transfer is a big part of the robot's behavior, so it makes
    # sense that it has lots of attributes/arguments.

    def __init__(
        self,
        node: Node,
        mouth_position_tolerance: float = 0.0075,
        relaxed_mouth_position_tolerance: float = 0.025,
        head_object_id: str = "head",
        max_linear_speed: float = 0.05,
        max_angular_speed: float = 0.15,
        linear_speed_near_mouth: float = 0.025,
        angular_speed_near_mouth: float = 0.075,
        wheelchair_collision_object_id: str = "body",
        force_threshold: float = 1.0,
        torque_threshold: float = 1.0,
        allowed_face_distance: Tuple[float, float] = (0.4, 1.25),
        face_detection_msg_timeout: float = 5.0,
        face_detection_timeout: float = 2.5,
        plan_distance_from_mouth: Annotated[Sequence[float], 3] = (0.025, 0.0, -0.01),
        fork_target_orientation_from_mouth: Tuple[float, float, float, float] = (
            0.5,
            -0.5,
            -0.5,
            0.5,
        ),
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        mouth_position_tolerance: The tolerance for the movement to the mouth pose.
        relaxed_mouth_position_tolerance: Although the robot will keep moving until
            it gets to within `mouth_position_tolerance`, if it stops early (e.g., the
            user leaned forward), it will still return success within `relaxed_mouth_position_tolerance`.
        head_object_id: The ID of the head collision object in the MoveIt2
            planning scene.
        max_linear_speed: The maximum linear speed (m/s) for the motion.
        max_angular_speed: The maximum angular speed (rad/s) for the motion.
        linear_speed_near_mouth: The robot will slow down as it approaches the
            user's mouth, reaching this speed when it is at their mouth.
        angular_speed_near_mouth: The robot will slow down as it approaches the
            user's mouth, reaching this speed when it is at their mouth.
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
        face_detection_msg_timeout: If the timestamp on the face detection message
            is older than these many seconds, don't use it.
        face_detection_timeout: If the robot has been trying to detect a face for
            more than these many seconds, timeout.
        plan_distance_from_mouth: The distance (m) to plan from the mouth center.
        fork_target_orientation_from_mouth: The fork's target orientation, in *mouth*
            frame. Pointing straight to the mouth is (0.5, -0.5, -0.5, 0.5).
        """

        # pylint: disable=too-many-locals
        # These are all necessary due to all the behaviors MoveToMouth contains

        # TODO: Consider modifying feedback to return whether it is perceiving
        # the face right now. Not crucial, but may be nice to have.

        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters
        self.mouth_position_tolerance = mouth_position_tolerance
        self.relaxed_mouth_position_tolerance = relaxed_mouth_position_tolerance
        self.head_object_id = head_object_id
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.linear_speed_near_mouth = linear_speed_near_mouth
        self.angular_speed_near_mouth = angular_speed_near_mouth
        self.wheelchair_collision_object_id = wheelchair_collision_object_id
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.allowed_face_distance = allowed_face_distance
        self.face_detection_msg_timeout = Duration(seconds=face_detection_msg_timeout)
        self.face_detection_timeout = face_detection_timeout
        self.plan_distance_from_mouth = plan_distance_from_mouth
        self.fork_target_orientation_from_mouth = fork_target_orientation_from_mouth

        self.face_detection_relative_blackboard_key = "face_detection"

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
        # Check if a face is detected
        logger = rclpy.logging.get_logger("MoveToMouth check_face_msg")
        if not msg.is_face_detected:
            logger.warn(f"Rejecting face message {msg} because face not detected")
            return False
        # Check if the message is stale
        timestamp = Time.from_msg(msg.detected_mouth_center.header.stamp)
        elapsed_time = self._node.get_clock().now() - timestamp
        if elapsed_time > self.face_detection_msg_timeout:
            logger.warn(
                f"Rejecting face message {msg} because too much time has elapsed {elapsed_time}"
            )
            return False
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
            distance < self.allowed_face_distance[0]
            or distance > self.allowed_face_distance[1]
        ):
            logger.warn(
                f"Rejecting face message {msg} since its distance {distance} is not in the range"
            )
            return False
        return True

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        # TODO: Consider moving move head and scale wheelchair_collision logic into
        # ada_planning_scene.py, to consolidate all planning scene updates.

        ### Define Tree Logic

        face_detection_absolute_key = Blackboard.separator.join(
            [name, self.face_detection_relative_blackboard_key]
        )

        # Use a custom speed profile to do angular motions at the end.
        max_pose_distance = 0.3

        def speed(post_stamped: PoseStamped) -> Tuple[float, float]:
            """
            Linearly interpolate the speed between max_{linear/angular}_speed and
            {linear/angular}_speed_near_mouth as the robot moves closer to the mouth.
            """
            nonlocal max_pose_distance
            pose_distance = (
                post_stamped.pose.position.x**2.0
                + post_stamped.pose.position.y**2.0
                + post_stamped.pose.position.z**2.0
            ) ** 0.5
            if pose_distance > max_pose_distance:
                max_pose_distance = pose_distance
            prop = (max_pose_distance - pose_distance) / max_pose_distance  # ** 0.5
            return (
                self.max_linear_speed * (1.0 - prop)
                + self.linear_speed_near_mouth * prop,
                self.max_angular_speed * (1.0 - prop)
                + self.angular_speed_near_mouth * prop,
            )

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
                # NOTE: `get_result` relies on "FaceDetection" only being in the
                # names of perception behaviors.
                py_trees.composites.Selector(
                    name=name + " FaceDetectionSelector",
                    memory=True,
                    children=[
                        # Try to detect the face and then convert the detected face pose
                        # to the base frame.
                        py_trees.composites.Sequence(
                            name=name + " FaceDetectionSequence",
                            memory=True,
                            children=[
                                py_trees.composites.Selector(
                                    name=name + " FaceDetectionSelector",
                                    memory=True,
                                    children=[
                                        # Check if the face detection message is not stale
                                        # and close enough to the camera
                                        py_trees.behaviours.CheckBlackboardVariableValue(
                                            name=name + " CheckFaceDetectionMsg",
                                            check=py_trees.common.ComparisonExpression(
                                                variable=face_detection_absolute_key,
                                                value=FaceDetection(),
                                                operator=self.check_face_msg,
                                            ),
                                        ),
                                        # If the above didn't work, turn face detection on until
                                        # a face is detected, or until timeout
                                        scoped_behavior(
                                            name=name + " FaceDetectionOnScope",
                                            pre_behavior=get_toggle_face_detection_behavior(
                                                name + "TurnFaceDetectionOn",
                                                True,
                                            ),
                                            # Turn off face detection
                                            post_behavior=get_toggle_face_detection_behavior(
                                                name + "TurnFaceDetectionOff",
                                                False,
                                            ),
                                            workers=[
                                                py_trees.decorators.Timeout(
                                                    name=name + " FaceDetectionTimeout",
                                                    duration=self.face_detection_timeout,
                                                    child=py_trees.composites.Sequence(
                                                        name=name,
                                                        memory=False,
                                                        children=[
                                                            # Get the detected face
                                                            py_trees_ros.subscribers.ToBlackboard(
                                                                name=name
                                                                + " GetFaceDetectionMsg",
                                                                topic_name="~/face_detection",
                                                                topic_type=FaceDetection,
                                                                qos_profile=(
                                                                    py_trees_ros.utilities.qos_profile_unlatched()
                                                                ),
                                                                blackboard_variables={
                                                                    face_detection_absolute_key: None,
                                                                },
                                                                initialise_variables={
                                                                    face_detection_absolute_key: FaceDetection(),
                                                                },
                                                            ),
                                                            # Check whether the face is within the required distance
                                                            py_trees.decorators.FailureIsRunning(
                                                                name=name
                                                                + " CheckFaceDetectionWrapper",
                                                                child=py_trees.behaviours.CheckBlackboardVariableValue(
                                                                    name=name
                                                                    + " CheckFaceDetectionMsg",
                                                                    check=py_trees.common.ComparisonExpression(
                                                                        variable=face_detection_absolute_key,
                                                                        value=FaceDetection(),
                                                                        operator=self.check_face_msg,
                                                                    ),
                                                                ),
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                # Convert `face_detection` to `mouth_position` in the
                                # base frame.
                                ApplyTransform(
                                    name=name + " ConvertFaceDetectionToBaseFrame",
                                    ns=name,
                                    inputs={
                                        "stamped_msg": BlackboardKey(
                                            self.face_detection_relative_blackboard_key
                                            + ".detected_mouth_center"
                                        ),
                                        "target_frame": "j2n6s200_link_base",
                                    },
                                    outputs={
                                        "transformed_msg": BlackboardKey(
                                            "mouth_position"
                                        ),  # PointStamped
                                    },
                                ),
                                # Convert `mouth_position` into a mouth pose using
                                # a fixed quaternion
                                CreatePoseStamped(
                                    name=name + " FaceDetectionToPose",
                                    ns=name,
                                    inputs={
                                        "position": BlackboardKey("mouth_position"),
                                        "quaternion": [
                                            0.0,
                                            0.0,
                                            -0.7071068,
                                            0.7071068,
                                        ],  # Facing away from wheelchair backrest
                                    },
                                    outputs={
                                        "pose_stamped": BlackboardKey(
                                            "mouth_pose"
                                        ),  # PostStamped
                                    },
                                ),
                                # Cache the mouth pose on the static TF tree
                                SetStaticTransform(
                                    name=name + " SetFaceDetectionPoseOnTF",
                                    ns=name,
                                    inputs={
                                        "transform": BlackboardKey("mouth_pose"),
                                        "child_frame_id": "mouth",
                                    },
                                ),
                                # Add a slight delay to allow the static transform
                                # to be registered
                                wait_for_secs(
                                    name=name + " FaceDetectionWaitForStaticTransform",
                                    secs=0.25,
                                ),
                            ],
                        ),
                        # If there is a cached detected mouth pose on the static
                        # transform tree, use it.
                        GetTransform(
                            name=name + " GetCachedFaceDetection",
                            ns=name,
                            inputs={
                                "target_frame": "j2n6s200_link_base",
                                "source_frame": "mouth",
                                "new_type": PoseStamped,
                            },
                            outputs={
                                "transform": BlackboardKey("mouth_pose"),
                            },
                        ),
                    ],
                ),
                # Note that `ada_planning_scene.py` should have already updated
                # the head and wheelchair collision locations in the planning scene.
                # Clear the Octomap. This is far enough before motion to mouth
                # that the Octomap should still get populated before motion
                # begins.
                retry_call_ros_service(
                    name=name + "ClearOctomap",
                    service_type=Empty,
                    service_name="~/clear_octomap",
                    key_request=None,
                    request=Empty.Request(),
                ),
                # The goal constraint of the fork is the mouth pose,
                # translated `self.plan_distance_from_mouth` in front of the mouth,
                # and rotated to match the forkTip orientation.
                ApplyTransform(
                    name=name + " ComputeMoveToMouthPose",
                    ns=name,
                    inputs={
                        "stamped_msg": PoseStamped(
                            header=Header(
                                stamp=Time().to_msg(),
                                frame_id="mouth",
                            ),
                            pose=Pose(
                                position=Point(
                                    x=self.plan_distance_from_mouth[0],
                                    y=self.plan_distance_from_mouth[1],
                                    z=self.plan_distance_from_mouth[2],
                                ),
                                orientation=Quaternion(
                                    x=self.fork_target_orientation_from_mouth[0],
                                    y=self.fork_target_orientation_from_mouth[1],
                                    z=self.fork_target_orientation_from_mouth[2],
                                    w=self.fork_target_orientation_from_mouth[3],
                                ),
                            ),
                        ),
                        "target_frame": "j2n6s200_link_base",
                    },
                    outputs={
                        "transformed_msg": BlackboardKey("goal_pose"),  # PoseStamped
                    },
                ),
                # Retare the F/T sensor and set the F/T Thresholds
                pre_moveto_config(
                    name=name + "PreMoveToConfig",
                    toggle_watchdog_listener=False,
                    f_mag=self.force_threshold,
                    t_mag=self.torque_threshold,
                    param_service_name="~/set_servo_controller_parameters",
                ),
                # Allow collisions with the expanded wheelchair collision box
                scoped_behavior(
                    name=name + " AllowWheelchairCollisionScopeAndStartServo",
                    pre_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            get_toggle_collision_object_behavior(
                                name + "AllowWheelchairCollisionScopePre",
                                [self.wheelchair_collision_object_id],
                                True,
                            ),
                            ActivateControllerTree(self._node)
                            .create_tree(name=name + "ActivateCartesianController")
                            .root,
                        ],
                    ),
                    # Disallow collisions with the expanded wheelchair collision
                    # box.
                    post_behavior=py_trees.composites.Sequence(
                        name=name,
                        memory=True,
                        children=[
                            ActivateControllerTree(
                                self._node, controller_to_activate=None
                            )
                            .create_tree(name=name + "DeactivateCartesianController")
                            .root,
                            get_toggle_collision_object_behavior(
                                name + "DisallowWheelchairCollisionScopePost",
                                [self.wheelchair_collision_object_id],
                                False,
                            ),
                            pre_moveto_config(
                                name=name + "PreMoveToConfigScopePost",
                                re_tare=False,
                                f_mag=1.0,
                                param_service_name="~/set_servo_controller_parameters",
                            ),
                        ],
                    ),
                    # Move to the target pose
                    workers=[
                        servo_until_pose(
                            name=name + " MoveToMouth",
                            ns=name,
                            target_pose_stamped_key=BlackboardKey("goal_pose"),
                            tolerance_position=self.mouth_position_tolerance,
                            tolerance_orientation=0.09,
                            relaxed_tolerance_position=self.relaxed_mouth_position_tolerance,
                            relaxed_tolerance_orientation=0.15,
                            duration=10.0,
                            round_decimals=3,
                            speed=speed,
                            ignore_orientation=True,
                            subscribe_to_servo_status=False,
                            pub_topic="~/cartesian_twist_cmds",
                        )
                    ],
                ),
            ],
        )

        ### Return tree
        return py_trees.trees.BehaviourTree(root_seq)

    # Override goal to read arguments into local blackboard
    @override
    def send_goal(self, tree: py_trees.trees.BehaviourTree, goal: object) -> bool:
        # Docstring copied by @override
        # Note: if here, tree is root, not a subtree

        # Check goal type
        if not isinstance(goal, MoveToMouth.Goal):
            return False

        # Write tree inputs to blackboard
        name = tree.root.name
        blackboard = py_trees.blackboard.Client(name=name, namespace=name)
        blackboard.register_key(
            key=self.face_detection_relative_blackboard_key,
            access=py_trees.common.Access.WRITE,
        )
        blackboard.face_detection = goal.face_detection

        # Adds MoveToVisitor for Feedback
        return super().send_goal(tree, goal)

    @override
    def get_result(
        self, tree: py_trees.trees.BehaviourTree, action_type: type
    ) -> object:
        # Docstring copied by @override
        result_msg = super().get_result(tree, action_type)

        # If the standard result determines there was a planning failure,
        # we check whether that was actually a perception failure.
        if result_msg.status == result_msg.STATUS_PLANNING_FAILED:
            tip = tree.tip()
            if tip is not None and "FaceDetection" in tip.name:
                result_msg.status = result_msg.STATUS_PERCEPTION_FAILED

        return result_msg
