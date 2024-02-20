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
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener

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
    def __init__(self) -> None:
        """
        Initialize the planning scene.
        """
        # pylint: disable=too-many-instance-attributes
        # The number of attributes is necessary for this node.

        super().__init__("ada_planning_scene")

        # Load the parameters
        self.load_parameters()

        # Initialize the TF listeners and broadcasters
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # Initialize the MoveIt2 interface
        # Using ReentrantCallbackGroup to align with the examples from pymoveit2.
        # TODO: Assess whether ReentrantCallbackGroup is necessary for MoveIt2.
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name=kinova.MOVE_GROUP_ARM,
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

    def initialize_planning_scene(self) -> None:
        """
        Initialize the planning scene with the objects.
        """
        rate = self.create_rate(self.publish_hz)
        # Add each object to the planning scene
        for object_id, params in self.objects.items():
            if not rclpy.ok():
                break
            if params.primitive_type is None:
                if params.attached:
                    self.moveit2.add_attached_collision_mesh(
                        id=object_id,
                        filepath=params.filepath,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        link_name=params.frame_id,
                        touch_links=params.touch_links,
                    )
                else:
                    self.moveit2.add_collision_mesh(
                        id=object_id,
                        filepath=params.filepath,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                    )
            else:
                if params.attached:
                    self.moveit2.add_attached_collision_primitive(
                        id=object_id,
                        prim_type=params.primitive_type,
                        dims=params.primitive_dims,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        link_name=params.frame_id,
                        touch_links=params.touch_links,
                    )
                else:
                    self.moveit2.add_collision_primitive(
                        id=object_id,
                        prim_type=params.primitive_type,
                        dims=params.primitive_dims,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                    )
            rate.sleep()

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
            if self.latest_face_detection is None:
                return
            msg = self.latest_face_detection
            self.latest_face_detection = None

        base_frame = "j2n6s200_link_base"

        # Transform the detected mouth center from the camera frame into the base frame
        detected_mouth_center = self.tf_buffer.transform(
            msg.detected_mouth_center,
            base_frame,
            0.5 / self.update_face_hz,
        )

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
        self.moveit2.move_collision(
            id=self.head_object_id,
            position=detected_mouth_pose.pose.position,
            quat_xyzw=detected_mouth_pose.pose.orientation,
            frame_id=base_frame,
        )

        # TODO: Add the static mouth pose to TF
        # TODO: Scale the wheelchair collision object based on the user's head
        # pose.


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
