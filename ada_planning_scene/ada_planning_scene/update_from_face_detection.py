"""
This module contains the UpdateFromFaceDetection class, which subscribes to the
output of face detection, moves the head to the detected position, and scales the
body hull accordingly. This behavior gets implicitly toggled on and off when face
detection gets toggled on and off.
"""

# Standard imports
from collections import namedtuple
from threading import Lock
from typing import Dict, List

# Third-party imports
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from tf2_geometry_msgs import TransformStamped
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams


# Define a namedtuple to store latest the joint state
UpdateFromFaceDetectionParams = namedtuple(
    "UpdateFromFaceDetectionParams",
    [
        "head_object_id",
        "body_object_id",
        "head_distance_threshold",
        "mouth_frame_id",
        "update_body",
    ],
)


class UpdateFromFaceDetection:
    """
    This class subscribes to the output of face detection, moves the head to the
    detected position, and scales the body hull accordingly. It only does so within
    a range of the default head position (i.e., rejecting faces that are too far).
    """

    # pylint: disable=too-many-instance-attributes
    # Fine for this class.
    # pylint: disable=too-few-public-methods
    # This class updates the face internally, so doesn't need many public methods.

    def __init__(
        self,
        node: Node,
        collision_object_manager: CollisionObjectManager,
        objects: Dict[str, Dict[str, CollisionObjectParams]],
        base_frame_id: str,
        tf_buffer: Buffer,
        tf_broadcaster: StaticTransformBroadcaster,
        namespaces: List[str],
        namespace_to_use: str,
    ):
        """
        Initialize the UpdateFromFaceDetection.

        Parameters
        ----------
        node: The ROS 2 node.
        collision_object_manager: The collision object manager.
        objects: The collision objects in the planning scene.
        base_frame_id: The base frame ID.
        tf_buffer: The TF buffer.
        tf_broadcaster: The TF static transform broadcaster.
        namespaces: The list of namespaces to search for parameters.
        namespace_to_use: The namespace to use for the parameters.
        """
        # pylint: disable=too-many-arguments
        # This class needs a lot of objects passed from the main node.
        # pylint: disable=duplicate-code
        # Many of the classes in ada_planning scene have similar initializations.
        self.__node = node
        self.__collision_object_manager = collision_object_manager
        self.__base_frame_id = base_frame_id
        self.__objects = objects
        self.__tf_buffer = tf_buffer
        self.__tf_broadcaster = tf_broadcaster
        self.__namespaces = namespaces
        self.__namespace_to_use = namespace_to_use

        # Load the parameters
        self.__load_parameters()

        # Get the default head and body params and poses, to avoid recomputing
        # them every time we update the head and body.
        self.__default_head_params = {}
        self.__default_body_params = {}
        self.__default_head_poses = {}
        self.__default_body_poses = {}
        for namespace in self.__namespaces:
            head_object_id = self.__namespace_to_params[namespace].head_object_id
            body_object_id = self.__namespace_to_params[namespace].body_object_id

            self.__default_head_params[namespace] = self.__objects[namespace][
                head_object_id
            ]
            self.__default_body_params[namespace] = self.__objects[namespace][
                body_object_id
            ]
            self.__default_head_poses[namespace] = Pose(
                position=Point(
                    x=self.__default_head_params[namespace].position[0],
                    y=self.__default_head_params[namespace].position[1],
                    z=self.__default_head_params[namespace].position[2],
                ),
                orientation=Quaternion(
                    x=self.__default_head_params[namespace].quat_xyzw[0],
                    y=self.__default_head_params[namespace].quat_xyzw[1],
                    z=self.__default_head_params[namespace].quat_xyzw[2],
                    w=self.__default_head_params[namespace].quat_xyzw[3],
                ),
            )
            self.__default_body_poses[namespace] = Pose(
                position=Point(
                    x=self.__default_body_params[namespace].position[0],
                    y=self.__default_body_params[namespace].position[1],
                    z=self.__default_body_params[namespace].position[2],
                ),
                orientation=Quaternion(
                    x=self.__default_body_params[namespace].quat_xyzw[0],
                    y=self.__default_body_params[namespace].quat_xyzw[1],
                    z=self.__default_body_params[namespace].quat_xyzw[2],
                    w=self.__default_body_params[namespace].quat_xyzw[3],
                ),
            )

        # Subscribe to the face detection topic
        # pylint: disable=unused-private-member
        self.__latest_face_detection = None
        self.__latest_face_detection_lock = Lock()
        self.__face_detection_sub = self.__node.create_subscription(
            FaceDetection,
            "~/face_detection",
            self.__face_detection_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create a timer to update planning scene for face detection
        self.__face_detection_timer = self.__node.create_timer(
            1.0 / self.__update_face_hz,
            self.__update_face_detection,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def __load_parameters(self) -> None:
        """
        Load the parameters. Note that this class is re-initialized when the
        namespace changes, which requires us to handle ParameterAlreadyDeclaredException.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # How often to update the planning scene based on face detection
        try:
            update_face_hz = self.__node.declare_parameter(
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
        except ParameterAlreadyDeclaredException:
            update_face_hz = self.__node.get_parameter("update_face_hz")
        self.__update_face_hz = update_face_hz.value

        # Load the parameters within each namespace
        self.__namespace_to_params = {}
        for namespace in self.__namespaces:
            try:
                # The object ID of the head in the planning scene
                head_object_id = self.__node.declare_parameter(
                    f"{namespace}.head_object_id",
                    "head",
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.head_object_id",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The object ID of the head in the planning scene. "
                            "This is used to move the head based on face detection."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                head_object_id = self.__node.get_parameter(
                    f"{namespace}.head_object_id"
                )

            try:
                # The object ID of the body in the planning scene
                body_object_id = self.__node.declare_parameter(
                    f"{namespace}.body_object_id",
                    "body",
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.body_object_id",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The object ID of the body in the planning scene. "
                            "This is used to move and scale the body based on face detection."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                body_object_id = self.__node.get_parameter(
                    f"{namespace}.body_object_id"
                )

            try:
                # If the head is farther than this distance from the default position,
                # use the default position instead
                head_distance_threshold = self.__node.declare_parameter(
                    f"{namespace}.head_distance_threshold",
                    0.5,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.head_distance_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=(
                            "Reject any mouth positions that are greater than the distance "
                            "threshold away from the default head position, in m."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                head_distance_threshold = self.__node.get_parameter(
                    f"{namespace}.head_distance_threshold"
                )

            try:
                # The TF frame to use for the mouth
                mouth_frame_id = self.__node.declare_parameter(
                    f"{namespace}.mouth_frame_id",
                    "mouth",
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.mouth_frame_id",
                        type=ParameterType.PARAMETER_STRING,
                        description=("The name of the frame to use for the mouth."),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                mouth_frame_id = self.__node.get_parameter(
                    f"{namespace}.mouth_frame_id"
                )

            try:
                # Whether to move the body as well as the head
                update_body = self.__node.declare_parameter(
                    f"{namespace}.update_body",
                    False,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.update_body",
                        type=ParameterType.PARAMETER_BOOL,
                        description=(
                            "Whether to update the body as well as the head based on face detection."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                update_body = self.__node.get_parameter(f"{namespace}.update_body")

            self.__namespace_to_params[namespace] = UpdateFromFaceDetectionParams(
                head_object_id=head_object_id.value,
                body_object_id=body_object_id.value,
                head_distance_threshold=head_distance_threshold.value,
                mouth_frame_id=mouth_frame_id.value,
                update_body=update_body.value,
            )

    def __face_detection_callback(self, msg: FaceDetection) -> None:
        """
        Callback for the face detection topic.
        """
        with self.__latest_face_detection_lock:
            self.__latest_face_detection = msg

    def __update_face_detection(self) -> None:
        """
        First, get the latest face detection message. Then, transform the
        detected mouth center from the camera frame into the base frame. Add a
        fixed quaternion to the mouth center to create a pose. Finally, move the
        head in the planning scene to that pose.
        """
        with self.__latest_face_detection_lock:
            if (
                self.__latest_face_detection is None
                or not self.__latest_face_detection.is_face_detected
            ):
                return
            msg = self.__latest_face_detection
            self.__latest_face_detection = None

        # Transform the detected mouth center from the camera frame into the base frame
        try:
            msg.detected_mouth_center.header.stamp = Time(
                sec=0, nanosec=0
            )  # Get latest transform
            detected_mouth_center = self.__tf_buffer.transform(
                msg.detected_mouth_center,
                self.__base_frame_id,
                rclpy.duration.Duration(seconds=0.5 / self.__update_face_hz),
            )
        except tf2.TransformException as e:
            self.__node.get_logger().error(
                f"Failed to transform the detected mouth center: {e}"
            )
            return

        # Get the parameters for the namespace we are using
        head_distance_threshold = self.__namespace_to_params[
            self.__namespace_to_use
        ].head_distance_threshold
        head_object_id = self.__namespace_to_params[
            self.__namespace_to_use
        ].head_object_id
        mouth_frame_id = self.__namespace_to_params[
            self.__namespace_to_use
        ].mouth_frame_id
        body_object_id = self.__namespace_to_params[
            self.__namespace_to_use
        ].body_object_id
        update_body = self.__namespace_to_params[self.__namespace_to_use].update_body
        default_head_pose = self.__default_head_poses[self.__namespace_to_use]
        default_head_params = self.__default_head_params[self.__namespace_to_use]
        default_body_pose = self.__default_body_poses[self.__namespace_to_use]
        default_body_params = self.__default_body_params[self.__namespace_to_use]

        # Reject any head that is too far from the original head pose
        dist = (
            (default_head_pose.position.x - detected_mouth_center.point.x) ** 2.0
            + (default_head_pose.position.y - detected_mouth_center.point.y) ** 2.0
            + (default_head_pose.position.z - detected_mouth_center.point.z) ** 2.0
        ) ** 0.5
        if dist > head_distance_threshold:
            self.__node.get_logger().error(
                f"Detected face in position {detected_mouth_center.point} is {dist}m "
                f"away from the default position {default_head_pose.position}. "
                f"Rejecting since it is greater than the threshold {head_distance_threshold}m."
            )
            return

        # Convert to a pose
        detected_mouth_pose = PoseStamped()
        detected_mouth_pose.header = detected_mouth_center.header
        detected_mouth_pose.pose.position = detected_mouth_center.point
        # Fixed orientation
        detected_mouth_pose.pose.orientation = default_head_pose.orientation

        self.__collision_object_manager.move_collision_objects(
            objects=CollisionObjectParams(
                object_id=head_object_id,
                frame_id=default_head_params.frame_id,
                position=[
                    detected_mouth_pose.pose.position.x,
                    detected_mouth_pose.pose.position.y,
                    detected_mouth_pose.pose.position.z,
                ],
                quat_xyzw=[
                    detected_mouth_pose.pose.orientation.x,
                    detected_mouth_pose.pose.orientation.y,
                    detected_mouth_pose.pose.orientation.z,
                    detected_mouth_pose.pose.orientation.w,
                ],
            ),
            rate_hz=self.__update_face_hz * 3.0,
        )

        # Add the static mouth pose to TF.
        self.__tf_broadcaster.sendTransform(
            TransformStamped(
                header=detected_mouth_pose.header,
                child_frame_id=mouth_frame_id,
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

        # Scale the body object based on the user's head pose.
        if update_body:
            if (
                detected_mouth_pose.header.frame_id != default_head_params.frame_id
                or detected_mouth_pose.header.frame_id != default_head_params.frame_id
            ):
                self.__node.get_logger().error(
                    "The detected mouth pose frame_id does not match the expected frame_id."
                )
                return

            # Compute the new body scale
            body_scale = (
                1.0,
                1.0,
                (detected_mouth_pose.pose.position.z - default_body_pose.position.z)
                / (default_head_pose.position.z - default_body_pose.position.z),
            )

            # We have to re-add it because the scale changed.
            self.__collision_object_manager.add_collision_objects(
                objects=CollisionObjectParams(
                    object_id=body_object_id,
                    frame_id=default_body_params.frame_id,
                    position=default_body_params.position,
                    quat_xyzw=default_body_params.quat_xyzw,
                    mesh_filepath=default_body_params.mesh_filepath,
                    mesh=default_body_params.mesh,
                    mesh_scale=body_scale,
                ),
                rate_hz=self.__update_face_hz * 3.0,
            )

    # pylint: disable=duplicate-code
    # Many of the classes in ada_planning_scene have this property and setter.
    @property
    def namespace_to_use(self) -> str:
        """
        Get the namespace to use for the parameters.
        """
        return self.__namespace_to_use

    @namespace_to_use.setter
    def namespace_to_use(self, namespace_to_use: str) -> None:
        """
        Set the namespace to use for the parameters.
        """
        if namespace_to_use not in self.__namespaces:
            raise ValueError(
                f"Namespace '{namespace_to_use}' not in the list of namespaces {self.__namespaces}."
            )
        self.__namespace_to_use = namespace_to_use
