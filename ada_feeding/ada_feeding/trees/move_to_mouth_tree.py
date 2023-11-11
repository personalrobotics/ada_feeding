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
import operator
from typing import Tuple

# Third-party imports
from geometry_msgs.msg import (
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from overrides import override
import py_trees
from py_trees.blackboard import Blackboard
import py_trees_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header

# Local imports
from ada_feeding_msgs.action import MoveToMouth
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding.behaviors.moveit2 import (
    MoveIt2Plan,
    MoveIt2Execute,
    MoveIt2PoseConstraint,
    ModifyCollisionObject,
    ModifyCollisionObjectOperation,
)
from ada_feeding.behaviors import (
    GetTransform,
    SetStaticTransform,
    ApplyTransform,
    CreatePoseStamped,
)
from ada_feeding.helpers import BlackboardKey
from ada_feeding.idioms import pre_moveto_config, scoped_behavior
from ada_feeding.idioms.bite_transfer import (
    get_toggle_collision_object_behavior,
    get_toggle_face_detection_behavior,
)
from ada_feeding.trees import (
    MoveToTree,
)


class MoveToMouthTree(MoveToTree):
    """
    A behaviour tree that gets the user's mouth pose and moves the robot to it.
    Note that to get the mouth pose, the robot executes these in-order until one
    succeeds:
    1. If `use_face_detection_msg` and the face detection message included in the
       goal is not stale, use it.
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
        mouth_position_tolerance: float = 0.001,
        planner_id: str = "RRTstarkConfigDefault",
        allowed_planning_time: float = 0.5,
        head_object_id: str = "head",
        max_velocity_scaling_factor: float = 0.1,
        cartesian_jump_threshold: float = 0.0,
        cartesian_max_step: float = 0.0025,
        wheelchair_collision_object_id: str = "wheelchair_collision",
        force_threshold: float = 4.0,
        torque_threshold: float = 4.0,
        allowed_face_distance: Tuple[float, float] = (0.4, 1.5),
        face_detection_msg_timeout: float = 5.0,
        face_detection_timeout: float = 2.5,
    ):
        """
        Initializes tree-specific parameters.

        Parameters
        ----------
        mouth_position_tolerance: The tolerance for the movement to the mouth pose.
        planner_id: The planner ID to use for the MoveIt2 motion planning.
        allowed_planning_time: The allowed planning time.
        head_object_id: The ID of the head collision object in the MoveIt2
            planning scene.
        max_velocity_scaling_factor: The maximum velocity scaling
            factor for the MoveIt2 motion planner to move to the user's mouth.
        cartesian_jump_threshold: The maximum allowed jump in the
            cartesian space for the MoveIt2 motion planner to move to the user's
            mouth.
        cartesian_max_step: The maximum allowed step in the cartesian
            space for the MoveIt2 motion planner to move to the user's mouth.
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
        """

        # pylint: disable=too-many-locals
        # These are all necessary due to all the behaviors MoveToMouth contains

        # TODO: This class should override get_result and get_feedback to return
        # perception failures, and tell the user it is attempting perception,
        # while it is waiting for the user's face.

        # Initialize MoveToTree
        super().__init__(node)

        # Store the parameters
        self.mouth_position_tolerance = mouth_position_tolerance
        self.planner_id = planner_id
        self.allowed_planning_time = allowed_planning_time
        self.head_object_id = head_object_id
        self.max_velocity_scaling_factor = max_velocity_scaling_factor
        self.cartesian_jump_threshold = cartesian_jump_threshold
        self.cartesian_max_step = cartesian_max_step
        self.wheelchair_collision_object_id = wheelchair_collision_object_id
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.allowed_face_distance = allowed_face_distance
        self.face_detection_msg_timeout = Duration(seconds=face_detection_msg_timeout)
        self.face_detection_timeout = face_detection_timeout

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
        if not msg.is_face_detected:
            return False
        # Check if the message is stale
        timestamp = Time.from_msg(msg.detected_mouth_center.header.stamp)
        if self._node.get_clock().now() - timestamp > self.face_detection_msg_timeout:
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
            return False
        return True

    @override
    def create_tree(
        self,
        name: str,
    ) -> py_trees.trees.BehaviourTree:
        # Docstring copied from @override

        ### Define Tree Logic

        face_detection_absolute_key = Blackboard.separator.join(
            [name, self.face_detection_relative_blackboard_key]
        )
        use_face_detection_msg_absolute_key = Blackboard.separator.join(
            [name, "use_face_detection_msg"]
        )

        # Root Sequence
        root_seq = py_trees.composites.Sequence(
            name=name,
            memory=True,
            children=[
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
                                        # If the client said to use the face detection message and
                                        # it is not stale, use it.
                                        py_trees.composites.Sequence(
                                            name=name,
                                            memory=True,
                                            children=[
                                                # Check if the `use_face_detection_msg` field of the
                                                # goal was set to True
                                                py_trees.behaviours.CheckBlackboardVariableValue(
                                                    name=name
                                                    + " CheckUseFaceDetectionMsg",
                                                    check=py_trees.common.ComparisonExpression(
                                                        variable=use_face_detection_msg_absolute_key,
                                                        value=True,
                                                        operator=operator.eq,
                                                    ),
                                                ),
                                                # Check if the face detection message is not stale
                                                # and close enough to the camera
                                                py_trees.behaviours.CheckBlackboardVariableValue(
                                                    name=name + " CheckFaceMsg",
                                                    check=py_trees.common.ComparisonExpression(
                                                        variable=face_detection_absolute_key,
                                                        value=FaceDetection(),
                                                        operator=self.check_face_msg,
                                                    ),
                                                ),
                                            ],
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
                                                                name=name + " GetFace",
                                                                topic_name="~/face_detection",
                                                                topic_type=FaceDetection,
                                                                qos_profile=py_trees_ros.utilities.qos_profile_unlatched(),
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
                                                                + " CheckFaceWrapper",
                                                                child=py_trees.behaviours.CheckBlackboardVariableValue(
                                                                    name=name
                                                                    + " CheckFaceMsg",
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
                                    name=name + " FaceDetectionToMouthPosition",
                                    ns=name,
                                    inputs={
                                        "stamped_msg": BlackboardKey(
                                            self.face_detection_relative_blackboard_key + ".detected_mouth_center"
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
                                    name=name + " MouthPositionToMouthPose",
                                    ns=name,
                                    inputs={
                                        "position": BlackboardKey("mouth_position"),
                                        "quaternion": [
                                            -0.0616284,
                                            -0.0616284,
                                            -0.704416,
                                            0.704416,
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
                                    name=name + " SetMouthPose",
                                    ns=name,
                                    inputs={
                                        "transform": BlackboardKey("mouth_pose"),
                                        "child_frame_id": "mouth",
                                    },
                                ),
                            ],
                        ),
                        # If there is a cached detected mouth pose on the static
                        # transform tree, use it.
                        GetTransform(
                            name=name + " GetCachedMouthPose",
                            ns=name,
                            inputs={
                                "target_frame": "j2n6s200_link_base",
                                "source_frame": "mouth",
                            },
                            outputs={
                                "transform": BlackboardKey("mouth_pose"),
                            },
                        ),
                    ],
                ),
                # At this stage of the tree, we are guarenteed to have a
                # `mouth_pose` on the blackboard -- else the tree would have failed.
                # Move the head to the detected mouth pose.
                ModifyCollisionObject(
                    name=name + " MoveHead",
                    ns=name,
                    inputs={
                        "operation": ModifyCollisionObjectOperation.MOVE,
                        "collision_object_id": self.head_object_id,
                        "collision_object_position": BlackboardKey(
                            "mouth_pose.pose.position"
                        ),
                        "collision_object_orientation": BlackboardKey(
                            "mouth_pose.pose.orientation"
                        ),
                    },
                ),
                # The goal constraint of the fork is the mouth pose,
                # translated 5cm in the +y direction, rotated to face the mouth
                ApplyTransform(
                    name=name + " ComputeMoveToMouthPose",
                    ns=name,
                    inputs={
                        "stamped_msg": BlackboardKey("mouth_pose"),
                        "target_frame": None,
                        "transform": TransformStamped(
                            child_frame_id="j2n6s200_link_base",
                            transform=Transform(
                                translation=Vector3(
                                    x=0.0,
                                    y=-0.05,
                                    z=0.0,
                                ),
                                rotation=Quaternion(
                                    x=0.0,
                                    y=0.0,
                                    z=0.0,
                                    w=1.0,
                                ),
                            ),
                        ),
                    },
                    outputs={
                        "transformed_msg": BlackboardKey("goal_pose"),  # PoseStamped
                    },
                ),
                # Cache the mouth pose on the static TF tree
                # TODO: remove!
                SetStaticTransform(
                    name=name + " SetMouthPose",
                    ns=name,
                    inputs={
                        "transform": BlackboardKey("goal_pose"),
                        "child_frame_id": "goal",
                    },
                ),
                # Retare the F/T sensor and set the F/T Thresholds
                pre_moveto_config(
                    name=name + "PreMoveToConfig",
                    toggle_watchdog_listener=False,
                    f_mag=self.force_threshold,
                    t_mag=self.torque_threshold,
                ),
                # Allow collisions with the expanded wheelchair collision box
                scoped_behavior(
                    name=name + " AllowWheelchairCollisionScope",
                    pre_behavior=get_toggle_collision_object_behavior(
                        name + " AllowWheelchairCollisionScopePre",
                        [self.wheelchair_collision_object_id],
                        True,
                    ),
                    # Move to the target pose
                    workers=[
                        # Goal configuration
                        MoveIt2PoseConstraint(
                            name="MoveToTargetPosePoseGoalConstraint",
                            ns=name,
                            inputs={
                                "pose": BlackboardKey("goal_pose"),
                                "tolerance_position": self.mouth_position_tolerance,
                                "tolerance_orientation": (0.6, 0.5, 0.5),
                                "parameterization": 1,  # Rotation vector
                            },
                            outputs={
                                "constraints": BlackboardKey("goal_constraints"),
                            },
                        ),
                        # Plan
                        MoveIt2Plan(
                            name="MoveToTargetPosePlan",
                            ns=name,
                            inputs={
                                "goal_constraints": BlackboardKey("goal_constraints"),
                                "planner_id": self.planner_id,
                                "allowed_planning_time": self.allowed_planning_time,
                                "max_velocity_scale": (
                                    self.max_velocity_scaling_factor
                                ),
                                "cartesian": True,
                                "cartesian_jump_threshold": self.cartesian_jump_threshold,
                                "cartesian_fraction_threshold": 0.60,
                                "cartesian_max_step": self.cartesian_max_step,
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
                        name + " AllowWheelchairCollisionScopePost",
                        [self.wheelchair_collision_object_id],
                        False,
                    ),
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
            key="use_face_detection_msg", access=py_trees.common.Access.WRITE
        )
        blackboard.use_face_detection_msg = goal.use_face_detection_msg
        blackboard.register_key(
            key=self.face_detection_relative_blackboard_key,
            access=py_trees.common.Access.WRITE,
        )
        blackboard.face_detection = goal.face_detection

        # Adds MoveToVisitor for Feedback
        return super().send_goal(tree, goal)
