#!/usr/bin/env python3
"""
This module defines the ADAPlanningScene node, which populates the
robot's planning scene with arbitrary STL meshes (passed in as parameters).

In practice, this node is used to add the wheelchair, table, and user's face.
"""

# Standard imports
from collections import namedtuple
from os import path
import threading
import time
from typing import List

# Third-party imports
import numpy as np
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    QuaternionStamped,
    Transform,
    Vector3,
)
from moveit_msgs.msg import CollisionObject, PlanningScene
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_geometry_msgs import PointStamped, PoseStamped, TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from transforms3d._gohlketransforms import quaternion_multiply

# Local imports
from ada_feeding_msgs.msg import FaceDetection

CollisionObjectParams = namedtuple(
    "CollisionObjectParams",
    [
        "filepath",
        "primitive_type",
        "primitive_dims",
        "position",
        "quat_xyzw",
        "frame_id",
        "attached",
        "touch_links",
    ],
)


class ADAPlanningScene(Node):
    """
    A node that initially populates the robot's planning scene with arbitrary
    STL meshes (passed in as parameters).

    In practice, this node is used to add the wheelchair, table, and user's face.
    """

    # pylint: disable=duplicate-code
    # The MoveIt2 object will have similar code in any file it is created.
    # pylint: disable=too-many-instance-attributes
    # The number of attributes is necessary for this node.
    def __init__(self) -> None:
        """
        Initialize the planning scene.
        """
        super().__init__("ada_planning_scene")

        # Load the parameters
        self.load_parameters()

        # Initialize the TF listeners and broadcasters
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Subscribe to the monitored planning scene to determine when an object
        # has been added to the planning scene.
        self.collision_object_ids_lock = threading.Lock()
        self.collision_object_ids = set()
        self.attached_collision_object_ids = set()
        self.monitored_planning_scene_sub = self.create_subscription(
            PlanningScene,
            "~/monitored_planning_scene",
            self.monitored_planning_scene_callback,
            1,
        )

        # Initialize the MoveIt2 interface
        # Using ReentrantCallbackGroup to align with the examples from pymoveit2.
        # TODO: Assess whether ReentrantCallbackGroup is necessary for MoveIt2.
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name="jaco_arm",
            callback_group=callback_group,
        )

        # Subscribe to the face detection topic
        self.face_detection_sub = self.create_subscription(
            FaceDetection,
            "~/face_detection",
            self.face_detection_callback,
            1,
        )
        self.latest_face_detection = None
        self.latest_face_detection_lock = threading.Lock()

        # Create a timer to update planning scene for face detection
        self.face_detection_timer = self.create_timer(
            1.0 / self.update_face_hz, self.update_face_detection
        )

        # Subscribe to the table detection topic
        self.table_detection_sub = self.create_subscription(
            PoseStamped,
            "~/table_detection",
            self.table_detection_callback,
            1,
        )
        self.latest_table_detection = None
        self.latest_table_detection_lock = threading.Lock()

        # Create a timer to update planning scene for table detection
        self.table_detection_timer = self.create_timer(
            1.0 / self.update_table_hz, self.update_table_detection
        )

    def load_parameters(self) -> None:
        """
        Load the parameters for the planning scene.
        """
        # pylint: disable=too-many-locals
        # The number of parameters is necessary for this node.

        # At what frequency (Hz) to check whether the `/collision_object`
        # topic is available (to call to add to the planning scene)
        wait_for_moveit_hz = self.declare_parameter(
            "wait_for_moveit_hz",
            10.0,  # default value
            ParameterDescriptor(
                name="wait_for_moveit_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to check the whether the "
                    "`/collision_object` topic is available (i.e., MoveIt is running)."
                ),
                read_only=True,
            ),
        )
        self.wait_for_moveit_hz = wait_for_moveit_hz.value

        # How long to sleep (in seconds) after the `/collision_object` topic
        # is available, to account for dropped messages when a topic is first
        # advertised.
        wait_for_moveit_sleep = self.declare_parameter(
            "wait_for_moveit_sleep",
            2.5,  # default value
            ParameterDescriptor(
                name="wait_for_moveit_sleep",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "How long to sleep (in seconds) after the `/collision_object` "
                    "topic is available, to account for dropped messages when a topic "
                    "is first advertised."
                ),
                read_only=True,
            ),
        )
        self.wait_for_moveit_sleep = wait_for_moveit_sleep.value

        # If all the collision objects have not been succesfully added to the
        # planning scene within this time, stop initialization. This is necessary
        # because if a collision object ID was previously removed from the planning
        # scene, MoveIt may not accept a request to add it back in.
        initialization_timeout_secs = self.declare_parameter(
            "initialization_timeout_secs",
            20.0,  # default value
            ParameterDescriptor(
                name="initialization_timeout_secs",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "If all the collision objects have not been succesfully added to the "
                    "planning scene within this time, stop initialization."
                ),
                read_only=True,
            ),
        )
        self.initialization_timeout_secs = initialization_timeout_secs.value

        ## The rate (Hz) at which to publish each planning scene object
        publish_hz = self.declare_parameter(
            "publish_hz",
            10.0,  # default value
            ParameterDescriptor(
                name="publish_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to publish each planning scene object."
                ),
                read_only=True,
            ),
        )
        self.publish_hz = publish_hz.value

        # Read the assets directory path
        assets_dir = self.declare_parameter(
            "assets_dir",
            descriptor=ParameterDescriptor(
                name="assets_dir",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The absolute path to the directory to find the assets "
                    "(e.g., STL mesh files)."
                ),
                read_only=True,
            ),
        )

        # Read the object IDs
        object_ids = self.declare_parameter(
            "object_ids",
            descriptor=ParameterDescriptor(
                name="object_ids",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="List of the object IDs to add to the planning scene.",
                read_only=True,
            ),
        )
        self.objects = {}

        # Read the object parameters
        for object_id in object_ids.value:
            filename = self.declare_parameter(
                f"{object_id}.filename",
                None,
                descriptor=ParameterDescriptor(
                    name="filename",
                    type=ParameterType.PARAMETER_STRING,
                    description=(
                        f"The filename of the mesh for the '{object_id}' object. "
                        "Either this or `primitive_type` and `primitive_dims` must "
                        "be specified."
                    ),
                    read_only=True,
                ),
            )
            primitive_type = self.declare_parameter(
                f"{object_id}.primitive_type",
                None,
                descriptor=ParameterDescriptor(
                    name="primitive_type",
                    type=ParameterType.PARAMETER_INTEGER,
                    description=(
                        f"The primitive type of the '{object_id}' object. "
                        "Either this and `primitive_dims` must be defined, or `filename`."
                    ),
                    read_only=True,
                ),
            )
            primitive_dims = self.declare_parameter(
                f"{object_id}.primitive_dims",
                None,
                descriptor=ParameterDescriptor(
                    name="primitive_dims",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=(
                        f"The dimensions of the '{object_id}' object. "
                        "Either this and `primitive_type` must be defined, or `filename`."
                    ),
                    read_only=True,
                ),
            )
            position = self.declare_parameter(
                f"{object_id}.position",
                None,
                descriptor=ParameterDescriptor(
                    name="position",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=f"The position of the '{object_id}' object in the planning scene.",
                    read_only=True,
                ),
            )
            quat_xyzw = self.declare_parameter(
                f"{object_id}.quat_xyzw",
                None,
                descriptor=ParameterDescriptor(
                    name="quat_xyzw",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=(
                        f"The orientation of the '{object_id}'"
                        " object in the planning scene."
                    ),
                    read_only=True,
                ),
            )
            frame_id = self.declare_parameter(
                f"{object_id}.frame_id",
                None,
                descriptor=ParameterDescriptor(
                    name="frame_id",
                    type=ParameterType.PARAMETER_STRING,
                    description=("The frame ID that the pose is in."),
                    read_only=True,
                ),
            )
            attached = self.declare_parameter(
                f"{object_id}.attached",
                False,
                descriptor=ParameterDescriptor(
                    name="attached",
                    type=ParameterType.PARAMETER_BOOL,
                    description=(
                        "Whether the object is attached to the robot. "
                        "If True, frame_id must be a robot link."
                    ),
                    read_only=True,
                ),
            )
            touch_links = self.declare_parameter(
                f"{object_id}.touch_links",
                None,
                descriptor=ParameterDescriptor(
                    name="touch_links",
                    type=ParameterType.PARAMETER_STRING_ARRAY,
                    description=(
                        "The links that the object is allowed to touch. "
                        "Only applies if `attached` is True."
                    ),
                    read_only=True,
                ),
            )

            # Add the object to the list of objects
            filepath = (
                None
                if filename.value is None
                else path.join(assets_dir.value, filename.value)
            )
            touch_links = [] if touch_links.value is None else touch_links.value
            self.objects[object_id] = CollisionObjectParams(
                filepath=filepath,
                primitive_type=primitive_type.value,
                primitive_dims=primitive_dims.value,
                position=position.value,
                quat_xyzw=quat_xyzw.value,
                frame_id=frame_id.value,
                attached=attached.value,
                touch_links=touch_links,
            )

        table_detection_offsets = self.declare_parameter(
            "table_detection_offsets",
            [-0.20, -0.25, -0.79],  # default value
            descriptor=ParameterDescriptor(
                name="table_detection_offsets",
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description=(
                    "The offset values for the center coordinates "
                    "of the table object."
                ),
                read_only=True,
            ),
        )
        self.table_detection_offsets = table_detection_offsets.value

        table_quat_dist_thresh = self.declare_parameter(
            "table_quat_dist_thresh",
            np.pi / 6.0,  # default value
            descriptor=ParameterDescriptor(
                name="table_quat_dist_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The threshold for the angular distance between "
                    "the latest detected table quaternion "
                    "and the previously detected table quaternion."
                ),
                read_only=True,
            ),
        )
        self.table_quat_dist_thresh = table_quat_dist_thresh.value

        table_pos_dist_thresh = self.declare_parameter(
            "table_pos_dist_thresh",
            0.5,  # default value
            descriptor=ParameterDescriptor(
                name="table_pos_dist_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The threshold for the linear distance between "
                    "the latest detected table position "
                    "and the previously detected table position."
                ),
                read_only=True,
            ),
        )
        self.table_pos_dist_thresh = table_pos_dist_thresh.value

        update_face_hz = self.declare_parameter(
            "update_face_hz",
            3.0,  # default value
            descriptor=ParameterDescriptor(
                name="update_face_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to update the planning scene based on the results "
                    "of face detection."
                ),
                read_only=True,
            ),
        )
        self.update_face_hz = update_face_hz.value

        update_table_hz = self.declare_parameter(
            "update_table_hz",
            3.0,  # default value
            descriptor=ParameterDescriptor(
                name="update_table_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to update the planning scene based on the results "
                    "of table detection."
                ),
                read_only=True,
            ),
        )
        self.update_table_hz = update_table_hz.value

        head_object_id = self.declare_parameter(
            "head_object_id",
            "head",
            descriptor=ParameterDescriptor(
                name="head_object_id",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The object ID of the head in the planning scene. "
                    "This is used to move the head based on face detection."
                ),
                read_only=True,
            ),
        )
        self.head_object_id = head_object_id.value

        head_distance_threshold = self.declare_parameter(
            "head_distance_threshold",
            0.5,
            descriptor=ParameterDescriptor(
                name="head_distance_threshold",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "Reject any mouth positions that are greater than the distance "
                    "threshold away from the default head position, in m."
                ),
                read_only=True,
            ),
        )
        self.head_distance_threshold = head_distance_threshold.value

        body_object_id = self.declare_parameter(
            "body_object_id",
            "wheelchair_collision",
            descriptor=ParameterDescriptor(
                name="body_object_id",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The object ID of the body in the planning scene. "
                    "This is used to move and scale the body based on face detection."
                ),
                read_only=True,
            ),
        )
        self.body_object_id = body_object_id.value

        table_object_id = self.declare_parameter(
            "table_object_id",
            "table",
            descriptor=ParameterDescriptor(
                name="table_object_id",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The object ID of the table in the planning scene. "
                    "This is used to move the table based on table detection."
                ),
                read_only=True,
            ),
        )
        self.table_object_id = table_object_id.value

    def wait_for_moveit(self) -> None:
        """
        Wait for the MoveIt2 interface to be ready. Specifically, it waits
        until the `/collision_object` topic has at least one subscriber.
        """
        rate = self.create_rate(self.wait_for_moveit_hz)
        while rclpy.ok():
            # pylint: disable=protected-access
            # This is necessary. Ideally, the service would not be protected.
            if self.moveit2._get_planning_scene_service.service_is_ready():
                break
            rate.sleep()

        # Sleep to avoid the period after the topic is advertised but before
        # is it processing messages.
        time.sleep(self.wait_for_moveit_sleep)

    def monitored_planning_scene_callback(self, msg: PlanningScene) -> None:
        """
        Callback for the monitored planning scene topic. Track the IDs for the
        (attached) collision objects in the planning scene.
        """
        with self.collision_object_ids_lock:
            # Update collision objects
            for collision_object in msg.world.collision_objects:
                object_id = collision_object.id
                if collision_object.operation == CollisionObject.REMOVE:
                    self.collision_object_ids.discard(object_id)
                else:
                    self.collision_object_ids.add(object_id)

            # Update attached collision objects
            for attached_collision_object in msg.robot_state.attached_collision_objects:
                object_id = attached_collision_object.object.id
                if attached_collision_object.object.operation == CollisionObject.REMOVE:
                    self.attached_collision_object_ids.discard(object_id)
                else:
                    self.attached_collision_object_ids.add(object_id)

    def initialize_planning_scene(self) -> None:
        """
        Initialize the planning scene with the objects.
        """
        initialization_start_time = self.get_clock().now()
        rate = self.create_rate(self.publish_hz)

        # If the planning scene already has the collision object, /monitored_planning_scene
        # may not be notified. So, we need to invoke the planning scene service to
        # get the initial planning scene.
        while self.moveit2.planning_scene is None:
            if self.moveit2.update_planning_scene():
                break
            self.get_logger().info(
                "Waiting for the initial planning scene...", throttle_duration_sec=1.0
            )
            rate.sleep()
        # Although this overuses a subscription callback, it does exactly what
        # we want by updating the collision objects.
        self.monitored_planning_scene_callback(self.moveit2.planning_scene)
        self.get_logger().info("Got the initial planning scene...")

        # Check if the node is still okay and within the initialization timeout
        def check_ok() -> bool:
            return (
                rclpy.ok()
                and (self.get_clock().now() - initialization_start_time).nanoseconds
                / 1e9
                <= self.initialization_timeout_secs
            )

        # First, add *all* collision objects to the planning scene
        object_ids = set(self.objects.keys()) - self.collision_object_ids
        while check_ok() and len(object_ids) > 0:
            self.get_logger().debug(
                f"Adding these objects to the planning scene: {object_ids}",
                throttle_duration_sec=1.0,
            )
            # Add each object to the planning scene
            for object_id in object_ids:
                if not check_ok():
                    break
                params = self.objects[object_id]
                if params.primitive_type is None:
                    self.moveit2.add_collision_mesh(
                        id=object_id,
                        filepath=params.filepath,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                    )
                else:
                    self.moveit2.add_collision_primitive(
                        id=object_id,
                        primitive_type=params.primitive_type,
                        dimensions=params.primitive_dims,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                    )
                rate.sleep()
                # Remove object_ids that have been added to the planning scene
                with self.collision_object_ids_lock:
                    object_ids = object_ids - self.collision_object_ids

        if len(object_ids) > 0:
            self.get_logger().error(
                "Initialization timed out. May not have added these objects to "
                f"the planning scene: {object_ids}"
            )

        # Next, attach any objects that need to be attached
        attached_object_ids = {
            object_id for object_id, params in self.objects.items() if params.attached
        }
        while check_ok() and len(attached_object_ids) > 0:
            self.get_logger().debug(
                f"Attaching these objects to the robot: {attached_object_ids}"
            )
            # Attach each object to the robot
            for object_id in attached_object_ids:
                if not check_ok():
                    break
                params = self.objects[object_id]
                self.moveit2.attach_collision_object(
                    id=object_id,
                    link_name=params.frame_id,
                    touch_links=params.touch_links,
                )
                rate.sleep()
                # Remove attached_object_ids that have been attached to the robot
                with self.collision_object_ids_lock:
                    attached_object_ids = (
                        attached_object_ids - self.attached_collision_object_ids
                    )

        if len(attached_object_ids) > 0:
            self.get_logger().error(
                "Initialization timed out. May not have attached these objects to "
                f"the robot: {attached_object_ids}"
            )

    def face_detection_callback(self, msg: FaceDetection) -> None:
        """
        Callback for the face detection topic.
        """
        with self.latest_face_detection_lock:
            self.latest_face_detection = msg

    def update_face_detection(self) -> None:
        """
        First, get the latest face detection message. Then, transform the
        detected mouth center from the camera frame into the base frame. Add a
        fixed quaternion to the mouth center to create a pose. Finally, move the
        head in the planning scene to that pose.
        """
        with self.latest_face_detection_lock:
            if (
                self.latest_face_detection is None
                or not self.latest_face_detection.is_face_detected
            ):
                return
            msg = self.latest_face_detection
            self.latest_face_detection = None

        base_frame = "root"

        # Transform the detected mouth center from the camera frame into the base frame
        try:
            detected_mouth_center = self.tf_buffer.transform(
                msg.detected_mouth_center,
                base_frame,
                rclpy.duration.Duration(seconds=0.5 / self.update_face_hz),
            )
        except TransformException as e:
            self.get_logger().error(
                f"Failed to transform the detected mouth center: {e}"
            )
            return

        # Get the original head pose
        original_head_pose = Pose(
            position=Point(
                x=self.objects[self.head_object_id].position[0],
                y=self.objects[self.head_object_id].position[1],
                z=self.objects[self.head_object_id].position[2],
            ),
            orientation=Quaternion(
                x=self.objects[self.head_object_id].quat_xyzw[0],
                y=self.objects[self.head_object_id].quat_xyzw[1],
                z=self.objects[self.head_object_id].quat_xyzw[2],
                w=self.objects[self.head_object_id].quat_xyzw[3],
            ),
        )

        # Reject any head that is too far from the original head pose
        dist = (
            (original_head_pose.position.x - detected_mouth_center.point.x) ** 2.0
            + (original_head_pose.position.y - detected_mouth_center.point.y) ** 2.0
            + (original_head_pose.position.z - detected_mouth_center.point.z) ** 2.0
        ) ** 0.5
        if dist > self.head_distance_threshold:
            self.get_logger().error(
                f"Detected face in position {detected_mouth_center.point} is {dist}m "
                f"away from the default position {original_head_pose.position}. "
                f"Rejecting since it is greater than the threshold {self.head_distance_threshold}m."
            )
            return

        # Convert to a pose
        detected_mouth_pose = PoseStamped()
        detected_mouth_pose.header = detected_mouth_center.header
        detected_mouth_pose.pose.position = detected_mouth_center.point
        # Fixed orientation facing away from the wheelchair backrest
        detected_mouth_pose.pose.orientation.x = 0.0
        detected_mouth_pose.pose.orientation.y = 0.0
        detected_mouth_pose.pose.orientation.z = -0.7071068
        detected_mouth_pose.pose.orientation.w = 0.7071068

        # Move the head in the planning scene to that pose
        # NOTE: I've seen a bug where sometimes MoveIt2 doesn't
        # receive or process this update. Unclear why.
        self.moveit2.move_collision(
            id=self.head_object_id,
            position=detected_mouth_pose.pose.position,
            quat_xyzw=detected_mouth_pose.pose.orientation,
            frame_id=base_frame,
        )

        # Add the static mouth pose to TF. Although it is done once in MoveToTree,
        # doing it here as well enables it to keep updating, effectively having
        # the robot visual servo to the user's mouth.
        self.tf_broadcaster.sendTransform(
            TransformStamped(
                header=detected_mouth_pose.header,
                child_frame_id="mouth",
                transform=Transform(
                    translation=Vector3(
                        x=detected_mouth_pose.pose.position.x,
                        y=detected_mouth_pose.pose.position.y,
                        z=detected_mouth_pose.pose.position.z,
                    ),
                    rotation=detected_mouth_pose.pose.orientation,
                ),
            )
        )

        # Scale the wheelchair collision object based on the user's head
        # pose.
        if (
            detected_mouth_pose.header.frame_id
            != self.objects[self.head_object_id].frame_id
            or detected_mouth_pose.header.frame_id
            != self.objects[self.body_object_id].frame_id
        ):
            self.get_logger().error(
                "The detected mouth pose frame_id does not match the expected frame_id."
            )
            return
        original_wheelchair_collision_pose = Pose(
            position=Point(
                x=self.objects[self.body_object_id].position[0],
                y=self.objects[self.body_object_id].position[1],
                z=self.objects[self.body_object_id].position[2],
            ),
            orientation=Quaternion(
                x=self.objects[self.body_object_id].quat_xyzw[0],
                y=self.objects[self.body_object_id].quat_xyzw[1],
                z=self.objects[self.body_object_id].quat_xyzw[2],
                w=self.objects[self.body_object_id].quat_xyzw[3],
            ),
        )

        # Compute the new wheelchair collision pose
        wheelchair_position = PointStamped(
            header=detected_mouth_pose.header,
            point=original_wheelchair_collision_pose.position,
        )
        wheelchair_orientation = QuaternionStamped(
            header=detected_mouth_pose.header,
            quaternion=original_wheelchair_collision_pose.orientation,
        )
        wheelchair_scale = (
            1.0,
            1.0,
            (
                detected_mouth_pose.pose.position.z
                - original_wheelchair_collision_pose.position.z
            )
            / (
                original_head_pose.position.z
                - original_wheelchair_collision_pose.position.z
            ),
        )

        # We have to re-add the mesh because the scale changed
        params = self.objects[self.body_object_id]
        self.moveit2.add_collision_mesh(
            id=self.body_object_id,
            filepath=params.filepath,
            position=wheelchair_position.point,
            quat_xyzw=wheelchair_orientation.quaternion,
            frame_id=params.frame_id,
            scale=wheelchair_scale,
        )

    def table_detection_callback(self, msg: PoseStamped) -> None:
        """
        Callback for the table detection topic.
        """
        with self.latest_table_detection_lock:
            self.latest_table_detection = msg

    def update_table_detection(self) -> None:
        """
        Transform the table center detected from the camera frame into the base
        frame. Then, move the table in the planning scene to the position
        received in the latest table detection message. If the angular distance
        between the latest and default table orientations is within a threshold,
        rotate the table object to the latest detected orientation. Otherwise,
        rotate the table object to the default table orientation.
        """
        # Get the latest table detection message
        with self.latest_table_detection_lock:
            if self.latest_table_detection is None:
                return
            msg = self.latest_table_detection
            self.latest_table_detection = None

        base_frame = self.objects[self.table_object_id].frame_id

        # Transform the detected table pose from the camera frame into the base frame
        try:
            detected_table_pose = self.tf_buffer.transform(
                msg,
                base_frame,
                rclpy.duration.Duration(seconds=0.5 / self.update_table_hz),
            )
        except TransformException as e:
            self.get_logger().error(
                f"Failed to transform the detected table center: {e}"
            )
            return

        # Translate detected position of table into table's origin
        detected_table_pose.pose.position.x += self.table_detection_offsets[0]
        detected_table_pose.pose.position.y += self.table_detection_offsets[1]
        detected_table_pose.pose.position.z += self.table_detection_offsets[2]

        # Reject the table if its detected position is too far from the default
        detected_table_pos = np.array(
            [
                detected_table_pose.pose.position.x,
                detected_table_pose.pose.position.y,
                detected_table_pose.pose.position.z,
            ]
        )
        default_table_pos = np.array(self.objects[self.table_object_id].position)
        pos_dist = np.linalg.norm(detected_table_pos - default_table_pos)
        if pos_dist > self.table_pos_dist_thresh:
            self.get_logger().warn(
                f"Rejecting detected table because its position {detected_table_pos} "
                f" is too far from the default {default_table_pos} ({pos_dist} > {self.table_pos_dist_thresh})"
            )
            return

        # Reject the table if its detected quaternion is too far from the default.
        # Because the table is 180 degrees symmetric around the z-axis, we try
        # two rotations and reject the table if the min one is too large.
        # Note that the library we use for quaternion distance represents
        # quaternions as [w, x, y, z]
        detected_table_quat = np.array(
            [
                detected_table_pose.pose.orientation.w,
                detected_table_pose.pose.orientation.x,
                detected_table_pose.pose.orientation.y,
                detected_table_pose.pose.orientation.z,
            ]
        )
        default_table_quat = np.array(
            [
                self.objects[self.table_object_id].quat_xyzw[3],
                self.objects[self.table_object_id].quat_xyzw[0],
                self.objects[self.table_object_id].quat_xyzw[1],
                self.objects[self.table_object_id].quat_xyzw[2],
            ]
        )
        quat_dist = None
        for rotation in [[1, 0, 0, 0], [0, 0, 0, 1]]:
            rotated_quat = quaternion_multiply(rotation, detected_table_quat)
            # Formula from https://math.stackexchange.com/questions/90081/quaternion-distance/90098#90098
            dist = np.arccos(2 * (np.dot(default_table_quat, rotated_quat) ** 2) - 1)
            if quat_dist is None or dist < quat_dist:
                quat_dist = dist
        if quat_dist > self.table_quat_dist_thresh:
            self.get_logger().warn(
                f"Rejecting detected table because its orientation {detected_table_quat} "
                f" is too far from the default {default_table_quat} ({quat_dist} > {self.table_quat_dist_thresh})"
            )
            return

        # Move the table object in the planning scene to the determined pose
        self.moveit2.move_collision(
            id=self.table_object_id,
            position=detected_table_pose.pose.position,
            quat_xyzw=detected_table_pose.pose.orientation,
            frame_id=base_frame,
        )


def main(args: List = None) -> None:
    """
    Create the ROS2 node and run the action servers.
    """
    rclpy.init(args=args)

    ada_planning_scene = ADAPlanningScene()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    # Spin in the background so that the messages to populate the planning scene
    # are processed.
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(ada_planning_scene,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Wait for the MoveIt2 interface to be ready
    ada_planning_scene.get_logger().info("Waiting for MoveIt2 interface...")
    ada_planning_scene.wait_for_moveit()

    # Initialize the planning scene
    ada_planning_scene.get_logger().info("Initializing planning scene...")
    ada_planning_scene.initialize_planning_scene()
    ada_planning_scene.get_logger().info("Planning scene initialized.")

    # # Sleep to allow the messages to go through
    # time.sleep(10.0)

    # # Terminate this node
    # ada_planning_scene.destroy_node()
    # rclpy.shutdown()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
