"""
This module contains the UpdateFromTableDetection class, which updates the pose
of the table based on the results of table detection.
"""

# Standard imports
from collections import namedtuple
from threading import Lock
from typing import Dict, List

# Third-party imports
from geometry_msgs.msg import PoseStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from transforms3d._gohlketransforms import quaternion_multiply

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams

# Define a namedtuple to store latest the joint state
UpdateFromTableDetectionParams = namedtuple(
    "UpdateFromTableDetectionParams",
    [
        "disable_table_detection",
        "table_object_id",
        "table_origin_offset",
        "table_distance_threshold",
        "table_rotation_threshold",
    ],
)


class UpdateFromTableDetection:
    """
    The UpdateFromTableDetection class updates the pose of the table based on the
    results of table detection.
    """

    # pylint: disable=too-many-instance-attributes
    # Fine for this class.
    # pylint: disable=too-few-public-methods
    # This class updates the table internally, so doesn't need many public methods.

    def __init__(
        self,
        node: Node,
        collision_object_manager: CollisionObjectManager,
        objects: Dict[str, Dict[str, CollisionObjectParams]],
        base_frame_id: str,
        tf_buffer: Buffer,
        namespaces: List[str],
        namespace_to_use: str,
    ):
        """
        Initialize the UpdateFromTableDetection object.

        Parameters
        ----------
        node: the ROS2 node
        collision_object_manager: the CollisionObjectManager object
        objects: the dictionary of collision objects
        base_frame_id: the base frame ID
        tf_buffer: the TF buffer
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
        self.__tf_buffer = tf_buffer
        self.__namespaces = namespaces
        self.__namespace_to_use = namespace_to_use

        # Load the parameters
        self.__load_parameters()

        # Get the relevant default table parameters, to avoid recomputing them
        # every time we update the table
        self.__default_table_params = {}
        self.__default_table_position = {}
        self.__default_table_quat_wxyz = {}
        for namespace in self.__namespaces:
            table_object_id = self.__namespace_to_params[namespace].table_object_id

            self.__default_table_params[namespace] = objects[namespace][table_object_id]
            self.__default_table_position[namespace] = np.array(
                self.__default_table_params[namespace].position
            )
            self.__default_table_quat_wxyz[namespace] = np.array(
                [
                    self.__default_table_params[namespace].quat_xyzw[3],  # w
                    self.__default_table_params[namespace].quat_xyzw[0],  # x
                    self.__default_table_params[namespace].quat_xyzw[1],  # y
                    self.__default_table_params[namespace].quat_xyzw[2],  # z
                ]
            )

        # Subscribe to the table detection topic
        # pylint: disable=unused-private-member
        self.__latest_table_detection = None
        self.__latest_table_detection_lock = Lock()
        self.__table_detection_sub = self.__node.create_subscription(
            PoseStamped,
            "~/table_detection",
            self.__table_detection_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create a timer to update planning scene for table detection
        self.__table_detection_timer = self.__node.create_timer(
            1.0 / self.__update_table_hz,
            self.__update_table_detection,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def __load_parameters(self) -> None:
        """
        Load the parameters. Note that this class is re-initialized when the
        namespace changes, which requires us to handle ParameterAlreadyDeclaredException.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        try:
            # How often to update the planning scene based on table detection
            update_table_hz = self.__node.declare_parameter(
                "update_table_hz",
                2.0,  # default value
                descriptor=ParameterDescriptor(
                    name="update_table_hz",
                    type=ParameterType.PARAMETER_DOUBLE,
                    description=(
                        "The rate (Hz) at which to update the planning scene based on "
                        "the results of table detection."
                    ),
                    read_only=True,
                ),
            )
        except ParameterAlreadyDeclaredException:
            update_table_hz = self.__node.get_parameter("update_table_hz")
        self.__update_table_hz = update_table_hz.value

        # Load the parameters within each namespace
        self.__namespace_to_params = {}
        for namespace in self.__namespaces:
            try:
                # The object ID of the table in the planning scene
                table_object_id = self.__node.declare_parameter(
                    f"{namespace}.table_object_id",
                    "table",
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.table_object_id",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The object ID of the table in the planning scene. "
                            "This is used to move the table based on table detection."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                table_object_id = self.__node.get_parameter(
                    f"{namespace}.table_object_id"
                )

            try:
                # Where the origin of the table is expected to be relative to the detected point
                table_origin_offset = self.__node.declare_parameter(
                    f"{namespace}.table_origin_offset",
                    [-0.20, -0.25, -0.79],  # default value
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.table_origin_offset",
                        type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                        description=(
                            "(x, y, z) values to add to the detected table pose (e.g., plate "
                            "center) to get the table origin."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                table_origin_offset = self.__node.get_parameter(
                    f"{namespace}.table_origin_offset"
                )

            try:
                # If the detected table has moved more than this relative to the default,
                # publish the default pose instead.
                table_distance_threshold = self.__node.declare_parameter(
                    f"{namespace}.table_distance_threshold",
                    0.5,  # default value
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.table_distance_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=(
                            "The threshold for the distance (m) between "
                            "the latest detected table position "
                            "and the default table position."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                table_distance_threshold = self.__node.get_parameter(
                    f"{namespace}.table_distance_threshold"
                )

            try:
                # If the detected table has rotated more than this relative to the default,
                # publish the default pose instead.
                table_rotation_threshold = self.__node.declare_parameter(
                    f"{namespace}.table_rotation_threshold",
                    np.pi / 6.0,  # default value
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.table_rotation_threshold",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=(
                            "The threshold for the angular distance between "
                            "the latest detected table quaternion "
                            "and the previously detected table quaternion."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                table_rotation_threshold = self.__node.get_parameter(
                    f"{namespace}.table_rotation_threshold"
                )

            try:
                disable_table_detection = self.__node.declare_parameter(
                    f"{namespace}.disable_table_detection",
                    False,  # default value
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.disable_table_detection",
                        type=ParameterType.PARAMETER_BOOL,
                        description=(
                            "Whether to disable table detection in this namespace."
                        ),
                        read_only=True,
                    ),
                )
            except ParameterAlreadyDeclaredException:
                disable_table_detection = self.__node.get_parameter(
                    f"{namespace}.disable_table_detection"
                )

            # Store the parameters
            self.__namespace_to_params[namespace] = UpdateFromTableDetectionParams(
                disable_table_detection=disable_table_detection.value,
                table_object_id=table_object_id.value,
                table_origin_offset=table_origin_offset.value,
                table_distance_threshold=table_distance_threshold.value,
                table_rotation_threshold=table_rotation_threshold.value,
            )

    def __table_detection_callback(self, msg: PoseStamped) -> None:
        """
        Callback for the table detection topic.
        """
        with self.__latest_table_detection_lock:
            self.__latest_table_detection = msg

    def __update_table_detection(self) -> None:
        """
        Transform the table center detected from the camera frame into the base
        frame. Move the table to that pose unless either the detection position
        or orientation differs too much from the default.
        """
        # pylint: disable=too-many-locals
        # This is where the main work happens.
        if self.__namespace_to_params[self.__namespace_to_use].disable_table_detection:
            return

        # Get the latest table detection message
        with self.__latest_table_detection_lock:
            if self.__latest_table_detection is None:
                return
            msg = self.__latest_table_detection
            self.__latest_table_detection = None

        # Transform the detected table pose from the camera frame into the base frame
        try:
            detected_table_pose = self.__tf_buffer.transform(
                msg,
                self.__base_frame_id,
                rclpy.duration.Duration(seconds=0.5 / self.__update_table_hz),
            )
        except tf2.TransformException as e:
            self.__node.get_logger().error(
                f"Failed to transform the detected table center: {e}"
            )
            return

        # Get the parameters for the namespace we are using
        table_origin_offset = self.__namespace_to_params[
            self.__namespace_to_use
        ].table_origin_offset
        table_distance_threshold = self.__namespace_to_params[
            self.__namespace_to_use
        ].table_distance_threshold
        table_rotation_threshold = self.__namespace_to_params[
            self.__namespace_to_use
        ].table_rotation_threshold
        default_table_params = self.__default_table_params[self.__namespace_to_use]
        default_table_position = self.__default_table_position[self.__namespace_to_use]
        default_table_quat_wxyz = self.__default_table_quat_wxyz[
            self.__namespace_to_use
        ]
        self.__node.get_logger().debug(
            f"Detected table position: {detected_table_pose.pose.position}"
        )

        # Translate detected position of table into table's origin
        detected_table_pose.pose.position.x += table_origin_offset[0]
        detected_table_pose.pose.position.y += table_origin_offset[1]
        detected_table_pose.pose.position.z += table_origin_offset[2]

        # Reject the table if its detected position is too far from the default
        detected_table_posision = np.array(
            [
                detected_table_pose.pose.position.x,
                detected_table_pose.pose.position.y,
                detected_table_pose.pose.position.z,
            ]
        )
        position_dist = np.linalg.norm(detected_table_posision - default_table_position)
        if position_dist > table_distance_threshold:
            self.__node.get_logger().warn(
                f"Rejecting detected table because its position {detected_table_posision} "
                f" is too far from the default {default_table_position} ({position_dist} > "
                f"{table_distance_threshold})"
            )
            return

        # Reject the table if its detected quaternion is too far from the default.
        # Because the table is 180 degrees symmetric around the z-axis, we try
        # two rotations and reject the table if the min one is too large.
        # Note that the library we use for quaternion distance represents
        # quaternions as [w, x, y, z]
        detected_table_quat_wxyz = np.array(
            [
                detected_table_pose.pose.orientation.w,
                detected_table_pose.pose.orientation.x,
                detected_table_pose.pose.orientation.y,
                detected_table_pose.pose.orientation.z,
            ]
        )
        min_quat_dist = None
        min_quat_wxyz = None
        for rotation in [[1, 0, 0, 0], [0, 0, 0, 1]]:
            rotated_quat = quaternion_multiply(rotation, detected_table_quat_wxyz)
            # Formula from https://math.stackexchange.com/questions/90081/quaternion-distance/90098#90098
            dist = np.arccos(
                2 * (np.dot(default_table_quat_wxyz, rotated_quat) ** 2) - 1
            )
            if min_quat_dist is None or dist < min_quat_dist:
                min_quat_dist = dist
                min_quat_wxyz = rotated_quat
        if min_quat_dist > table_rotation_threshold:
            self.__node.get_logger().warn(
                f"Rejecting detected table because its orientation {detected_table_quat_wxyz} "
                f" is too far from the default {default_table_quat_wxyz} ({min_quat_dist} > "
                f"{table_rotation_threshold})"
            )
            return

        # Move the table object in the planning scene
        self.__collision_object_manager.move_collision_objects(
            objects=CollisionObjectParams(
                object_id=default_table_params.object_id,
                position=detected_table_posision,
                quat_xyzw=(
                    min_quat_wxyz[1],
                    min_quat_wxyz[2],
                    min_quat_wxyz[3],
                    min_quat_wxyz[0],
                ),
                frame_id=default_table_params.frame_id,
            ),
            rate_hz=self.__update_table_hz * 3.0,
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
