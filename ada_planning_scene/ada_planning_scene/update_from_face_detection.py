"""
This module contains the UpdateFromFaceDetection class, which subscribes to the
output of face detection, moves the head to the detected position, and scales the
body hull accordingly. This behavior gets implicitly toggled on and off when face
detection gets toggled on and off.
"""

# Standard imports
from threading import Lock
from typing import Dict

# Third-party imports
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, Vector3
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tf2_geometry_msgs import TransformStamped
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams


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
        objects: Dict[str, CollisionObjectParams],
        base_frame_id: str,
        tf_buffer: Buffer,
        tf_broadcaster: StaticTransformBroadcaster,
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
        """
        # pylint: disable=too-many-arguments
        # This class needs a lot of objects passed from the main node.
        # pylint: disable=duplicate-code
        # Many of the classes in ada_planning scene have similar initializations.
        self.__node = node
        self.__collision_object_manager = collision_object_manager
        self.__objects = objects
        self.__base_frame_id = base_frame_id
        self.__tf_buffer = tf_buffer
        self.__tf_broadcaster = tf_broadcaster

        # Load the parameters
        self.__load_parameters()

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
        Load the parameters.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # How often to update the planning scene based on face detection
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
        self.__update_face_hz = update_face_hz.value

        # The object ID of the head in the planning scene
        head_object_id = self.__node.declare_parameter(
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
        self.__head_object_id = head_object_id.value

        # The object ID of the body in the planning scene
        body_object_id = self.__node.declare_parameter(
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
        self.__body_object_id = body_object_id.value

        # If the head is farther than this distance from the default position,
        # use the default position instead
        head_distance_threshold = self.__node.declare_parameter(
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
        self.__head_distance_threshold = head_distance_threshold.value

        # The TF frame to use for the mouth
        mouth_frame_id = self.__node.declare_parameter(
            "mouth_frame_id",
            "mouth",
            descriptor=ParameterDescriptor(
                name="mouth_frame_id",
                type=ParameterType.PARAMETER_STRING,
                description=("The name of the frame to use for the mouth."),
                read_only=True,
            ),
        )
        self.__mouth_frame_id = mouth_frame_id.value

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

        # Get the original head pose
        original_head_pose = Pose(
            position=Point(
                x=self.__objects[self.__head_object_id].position[0],
                y=self.__objects[self.__head_object_id].position[1],
                z=self.__objects[self.__head_object_id].position[2],
            ),
            orientation=Quaternion(
                x=self.__objects[self.__head_object_id].quat_xyzw[0],
                y=self.__objects[self.__head_object_id].quat_xyzw[1],
                z=self.__objects[self.__head_object_id].quat_xyzw[2],
                w=self.__objects[self.__head_object_id].quat_xyzw[3],
            ),
        )

        # Reject any head that is too far from the original head pose
        dist = (
            (original_head_pose.position.x - detected_mouth_center.point.x) ** 2.0
            + (original_head_pose.position.y - detected_mouth_center.point.y) ** 2.0
            + (original_head_pose.position.z - detected_mouth_center.point.z) ** 2.0
        ) ** 0.5
        if dist > self.__head_distance_threshold:
            self.__node.get_logger().error(
                f"Detected face in position {detected_mouth_center.point} is {dist}m "
                f"away from the default position {original_head_pose.position}. "
                f"Rejecting since it is greater than the threshold {self.__head_distance_threshold}m."
            )
            return

        # Convert to a pose
        detected_mouth_pose = PoseStamped()
        detected_mouth_pose.header = detected_mouth_center.header
        detected_mouth_pose.pose.position = detected_mouth_center.point
        # Fixed orientation
        detected_mouth_pose.pose.orientation = original_head_pose.orientation

        # Move the head in the planning scene to that pose
        self.__objects[self.__head_object_id].position = [
            detected_mouth_pose.pose.position.x,
            detected_mouth_pose.pose.position.y,
            detected_mouth_pose.pose.position.z,
        ]
        # Fixed orientation
        self.__objects[self.__head_object_id].quat_xyzw = [
            detected_mouth_pose.pose.orientation.x,
            detected_mouth_pose.pose.orientation.y,
            detected_mouth_pose.pose.orientation.z,
            detected_mouth_pose.pose.orientation.w,
        ]
        self.__collision_object_manager.move_collision_objects(
            objects={self.__head_object_id: self.__objects[self.__head_object_id]},
            rate_hz=self.__update_face_hz * 3.0,
        )

        # Add the static mouth pose to TF.
        self.__tf_broadcaster.sendTransform(
            TransformStamped(
                header=detected_mouth_pose.header,
                child_frame_id=self.__mouth_frame_id,
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

        # Scale the body object based on the user's head
        # pose.
        if (
            detected_mouth_pose.header.frame_id
            != self.__objects[self.__head_object_id].frame_id
            or detected_mouth_pose.header.frame_id
            != self.__objects[self.__body_object_id].frame_id
        ):
            self.__node.get_logger().error(
                "The detected mouth pose frame_id does not match the expected frame_id."
            )
            return
        original_body_pose = Pose(
            position=Point(
                x=self.__objects[self.__body_object_id].position[0],
                y=self.__objects[self.__body_object_id].position[1],
                z=self.__objects[self.__body_object_id].position[2],
            ),
            orientation=Quaternion(
                x=self.__objects[self.__body_object_id].quat_xyzw[0],
                y=self.__objects[self.__body_object_id].quat_xyzw[1],
                z=self.__objects[self.__body_object_id].quat_xyzw[2],
                w=self.__objects[self.__body_object_id].quat_xyzw[3],
            ),
        )

        # Compute the new body scale
        body_scale = (
            1.0,
            1.0,
            (detected_mouth_pose.pose.position.z - original_body_pose.position.z)
            / (original_head_pose.position.z - original_body_pose.position.z),
        )

        # Modify the wheelbodychair collision object in the planning scene
        self.__objects[self.__body_object_id].position = [
            original_body_pose.position.x,
            original_body_pose.position.y,
            original_body_pose.position.z,
        ]
        self.__objects[self.__body_object_id].quat_xyzw = [
            original_body_pose.orientation.x,
            original_body_pose.orientation.y,
            original_body_pose.orientation.z,
            original_body_pose.orientation.w,
        ]
        self.__objects[self.__body_object_id].mesh_scale = body_scale
        # We have to re-add it because the scale changed.
        self.__collision_object_manager.add_collision_objects(
            objects={self.__body_object_id: self.__objects[self.__body_object_id]},
            rate_hz=self.__update_face_hz * 3.0,
        )
