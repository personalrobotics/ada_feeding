"""
This module defines the PlanningSceneInitializer class, which reads the configuration
for static mesh and primitive collision objects from the parameters and adds them to
the planning scene.
"""

# Standard imports
from os import path

# Third-party imports
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams, get_remaining_time


class PlanningSceneInitializer:
    """
    The PlanningSceneInitializer reads the configuration for static mesh and primitive
    collision objects from the parameters and adds them to the planning scene.
    """

    def __init__(self, node: Node, collision_object_manager: CollisionObjectManager):
        """
        Initialize the PlanningSceneInitializer.

        Parameters
        ----------
        node: The ROS2 node.
        collision_object_manager: The CollisionObjectManager.
        """
        self.__node = node
        self.__collision_object_manager = collision_object_manager

        self.load_parameters()

    def load_parameters(self) -> None:
        """
        Load the parameters for the planning scene.
        """
        self.load_time_parameters()
        self.load_objects_parameters()

    def load_time_parameters(self) -> None:
        """
        Load parameters related to time during initialization.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # At what frequency (Hz) to check whether the `/get_planning_scene`
        # service is ready before starting to populate the planning scene
        wait_for_moveit_hz = self.__node.declare_parameter(
            "wait_for_moveit_hz",
            10.0,  # default value
            ParameterDescriptor(
                name="wait_for_moveit_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to check the whether the "
                    "`/get_planning_scene` service is ready (i.e., MoveIt is running)."
                ),
                read_only=True,
            ),
        )
        self.wait_for_moveit_hz = wait_for_moveit_hz.value

        # If all the collision objects have not been succesfully added to the
        # planning scene within this time, stop initialization.
        initialization_timeout_secs = self.__node.declare_parameter(
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
        self.initialization_timeout = Duration(
            seconds=initialization_timeout_secs.value
        )

        # The rate (Hz) at which to publish each planning scene object
        publish_hz = self.__node.declare_parameter(
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

    def load_objects_parameters(self) -> None:
        """
        Load parameters related to objects to add to the planning scene.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # Read the assets directory path
        assets_dir = self.__node.declare_parameter(
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
        object_ids = self.__node.declare_parameter(
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
            filename = self.__node.declare_parameter(
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
            primitive_type = self.__node.declare_parameter(
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
            primitive_dims = self.__node.declare_parameter(
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
            position = self.__node.declare_parameter(
                f"{object_id}.position",
                None,
                descriptor=ParameterDescriptor(
                    name="position",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description=f"The position of the '{object_id}' object in the planning scene.",
                    read_only=True,
                ),
            )
            quat_xyzw = self.__node.declare_parameter(
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
            frame_id = self.__node.declare_parameter(
                f"{object_id}.frame_id",
                None,
                descriptor=ParameterDescriptor(
                    name="frame_id",
                    type=ParameterType.PARAMETER_STRING,
                    description=("The frame ID that the pose is in."),
                    read_only=True,
                ),
            )
            within_workspace_walls = self.__node.declare_parameter(
                f"{object_id}.within_workspace_walls",
                False,
                descriptor=ParameterDescriptor(
                    name="within_workspace_walls",
                    type=ParameterType.PARAMETER_BOOL,
                    description=("Whether the object is within the workspace walls."),
                    read_only=True,
                ),
            )
            attached = self.__node.declare_parameter(
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
            touch_links = self.__node.declare_parameter(
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
                object_id=object_id,
                position=position.value,
                quat_xyzw=quat_xyzw.value,
                frame_id=frame_id.value,
                mesh_filepath=filepath,
                primitive_type=primitive_type.value,
                primitive_dims=primitive_dims.value,
                within_workspace_walls=within_workspace_walls.value,
                attached=attached.value,
                touch_links=touch_links,
            )

    def wait_for_moveit(self) -> None:
        """
        Wait for the MoveIt2 interface to be ready. Specifically, it waits
        until the `/get_planning_scene` service is ready.
        """
        rate = self.__node.create_rate(self.wait_for_moveit_hz)
        while rclpy.ok():
            # pylint: disable=protected-access
            # This is necessary. Ideally, the service would not be protected.
            if (
                self.__collision_object_manager.moveit2._get_planning_scene_service.service_is_ready()
            ):
                break
            rate.sleep()

    def initialize(self) -> None:
        """
        First, wait for the MoveIt2 interface to be ready. Then, get the objects
        that are already in the planning scene. Then, add the objects to the
        planning scene.
        """
        # Wait for the MoveIt2 interface to be ready
        self.__node.get_logger().info("Waiting for MoveIt2 interface...")
        self.wait_for_moveit()
        self.__node.get_logger().info("...MoveIt2 is ready.")

        # Start time
        start_time = self.__node.get_clock().now()

        # Get the objects that are already in the planning scene
        self.__node.get_logger().info(
            "Getting objects currently in the planning scene..."
        )
        self.__collision_object_manager.get_global_collision_objects(
            rate_hz=self.publish_hz,
            timeout=get_remaining_time(
                self.__node, start_time, self.initialization_timeout
            ),
        )
        self.__node.get_logger().info("...got planning scene objects.")

        # Add the objects to the planning scene
        self.__node.get_logger().info("Adding objects to the planning scene...")
        self.__collision_object_manager.add_collision_objects(
            objects=self.objects,
            rate_hz=self.publish_hz,
            timeout=get_remaining_time(
                self.__node, start_time, self.initialization_timeout
            ),
            ignore_existing=True,
        )
        self.__node.get_logger().info("Initialized planning scene.")
