"""
This module contains the UpdateFromTableDetection class, which updates the pose
of the table based on the results of table detection.
"""

# Standard imports
from threading import Lock
from typing import Dict

# Third-party imports
from geometry_msgs.msg import PoseStamped
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
import tf2_py as tf2
from tf2_ros.buffer import Buffer
from transforms3d._gohlketransforms import quaternion_multiply

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams


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
        objects: Dict[str, CollisionObjectParams],
        base_frame_id: str,
        tf_buffer: Buffer,
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

        # Load the parameters
        self.__load_parameters()

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
        Load the parameters.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # How often to update the planning scene based on table detection
        update_table_hz = self.__node.declare_parameter(
            "update_table_hz",
            3.0,  # default value
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
        self.__update_table_hz = update_table_hz.value

        # The object ID of the table in the planning scene
        table_object_id = self.__node.declare_parameter(
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
        self.__table_object_id = table_object_id.value

        # Where the origin of the table is expected to be relative to the detected point
        table_origin_offset = self.__node.declare_parameter(
            "table_origin_offset",
            [-0.20, -0.25, -0.79],  # default value
            descriptor=ParameterDescriptor(
                name="table_origin_offset",
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description=(
                    "(x, y, z) values to add to the detected table pose (e.g., plate "
                    "center) to get the table origin."
                ),
                read_only=True,
            ),
        )
        self.__table_origin_offset = table_origin_offset.value

        # If the detected table has moved more than this relative to the default,
        # publish the default pose instead.
        table_distance_threshold = self.__node.declare_parameter(
            "table_distance_threshold",
            0.5,  # default value
            descriptor=ParameterDescriptor(
                name="table_distance_threshold",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The threshold for the distance (m) between "
                    "the latest detected table position "
                    "and the default table position."
                ),
                read_only=True,
            ),
        )
        self.__table_distance_threshold = table_distance_threshold.value

        # If the detected table has rotated more than this relative to the default,
        # publish the default pose instead.
        table_rotation_threshold = self.__node.declare_parameter(
            "table_rotation_threshold",
            np.pi / 6.0,  # default value
            descriptor=ParameterDescriptor(
                name="table_rotation_threshold",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The threshold for the angular distance between "
                    "the latest detected table quaternion "
                    "and the previously detected table quaternion."
                ),
                read_only=True,
            ),
        )
        self.__table_rotation_threshold = table_rotation_threshold.value

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

        # Translate detected position of table into table's origin
        detected_table_pose.pose.position.x += self.__table_origin_offset[0]
        detected_table_pose.pose.position.y += self.__table_origin_offset[1]
        detected_table_pose.pose.position.z += self.__table_origin_offset[2]

        # Reject the table if its detected position is too far from the default
        detected_table_posision = np.array(
            [
                detected_table_pose.pose.position.x,
                detected_table_pose.pose.position.y,
                detected_table_pose.pose.position.z,
            ]
        )
        default_table_position = np.array(
            self.__objects[self.__table_object_id].position
        )
        position_dist = np.linalg.norm(detected_table_posision - default_table_position)
        if position_dist > self.__table_distance_threshold:
            self.__node.get_logger().warn(
                f"Rejecting detected table because its position {detected_table_posision} "
                f" is too far from the default {default_table_position} ({position_dist} > "
                f"{self.__table_distance_threshold})"
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
                self.__objects[self.__table_object_id].quat_xyzw[3],
                self.__objects[self.__table_object_id].quat_xyzw[0],
                self.__objects[self.__table_object_id].quat_xyzw[1],
                self.__objects[self.__table_object_id].quat_xyzw[2],
            ]
        )
        quat_dist = None
        for rotation in [[1, 0, 0, 0], [0, 0, 0, 1]]:
            rotated_quat = quaternion_multiply(rotation, detected_table_quat)
            # Formula from https://math.stackexchange.com/questions/90081/quaternion-distance/90098#90098
            dist = np.arccos(2 * (np.dot(default_table_quat, rotated_quat) ** 2) - 1)
            if quat_dist is None or dist < quat_dist:
                quat_dist = dist
        if quat_dist > self.__table_rotation_threshold:
            self.__node.get_logger().warn(
                f"Rejecting detected table because its orientation {detected_table_quat} "
                f" is too far from the default {default_table_quat} ({quat_dist} > {self.__table_rotation_threshold})"
            )
            return

        # Move the table object in the planning scene
        self.__collision_object_manager.move_collision_objects(
            objects={self.__table_object_id: self.__objects[self.__table_object_id]},
            rate_hz=self.__update_table_hz * 3.0,
        )
