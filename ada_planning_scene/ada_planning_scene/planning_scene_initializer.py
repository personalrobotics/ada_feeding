"""
This module defines the PlanningSceneInitializer class, which reads the configuration
for static mesh and primitive collision objects from the parameters and adds them to
the planning scene.
"""

# Standard imports
from os import path
from typing import List

# Third-party imports
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.duration import Duration
from rclpy.node import Node

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import (
    check_ok,
    CollisionObjectParams,
    get_remaining_time,
)


class PlanningSceneInitializer:
    """
    The PlanningSceneInitializer reads the configuration for static mesh and primitive
    collision objects from the parameters and adds them to the planning scene.
    """

    # pylint: disable=too-few-public-methods
    # This class only exists to intialize the planning scene, hence it only needs one
    # public method

    def __init__(
        self,
        node: Node,
        collision_object_manager: CollisionObjectManager,
        namespaces: List[str],
        namespace_to_use: str,
    ):
        """
        Initialize the PlanningSceneInitializer.

        Parameters
        ----------
        node: The ROS2 node.
        collision_object_manager: The CollisionObjectManager.
        namespaces: The list of namespaces to search for parameters.
        namespace_to_use: The namespace to use for the parameters.
        """
        self.__node = node
        self.__collision_object_manager = collision_object_manager
        self.__namespaces = namespaces
        self.__namespace_to_use = namespace_to_use

        self.objects = (
            {}
        )  # namespace (str) -> object_id (str) -> params (CollisionObjectParams)
        self.__load_parameters()

    def __load_parameters(self) -> None:
        """
        Load the parameters for the planning scene.
        """
        self.__load_time_parameters()
        self.__load_objects_parameters()

    def __load_time_parameters(self) -> None:
        """
        Load parameters related to time during initialization.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # At what frequency (Hz) to check whether the `/get_planning_scene`
        # service is ready before starting to populate the planning scene
        __wait_for_moveit_hz = self.__node.declare_parameter(
            "__wait_for_moveit_hz",
            10.0,  # default value
            ParameterDescriptor(
                name="__wait_for_moveit_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to check the whether the "
                    "`/get_planning_scene` service is ready (i.e., MoveIt is running)."
                ),
                read_only=True,
            ),
        )
        self.__wait_for_moveit_hz = __wait_for_moveit_hz.value

    def __load_objects_parameters(self) -> None:
        """
        Load parameters related to objects to add to the planning scene.
        """
        # pylint: disable=too-many-locals
        # This method needs to read many parameters

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

        # Read the object parameters
        for namespace in self.__namespaces:
            if namespace not in self.objects:
                self.objects[namespace] = {}

            # Read the object IDs
            object_ids = self.__node.declare_parameter(
                f"{namespace}.object_ids",
                descriptor=ParameterDescriptor(
                    name=f"{namespace}.object_ids",
                    type=ParameterType.PARAMETER_STRING_ARRAY,
                    description="List of the object IDs to add to the planning scene.",
                    read_only=True,
                ),
            )
            object_ids = [] if object_ids.value is None else object_ids.value

            # Read the object parameters
            for object_id in object_ids:
                filename = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.filename",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.filename",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            f"The filename of the mesh for the '{object_id}' object. "
                            "Either this or `primitive_type` and `primitive_dims` must "
                            "be specified."
                        ),
                        read_only=True,
                    ),
                )
                mesh_scale = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.mesh_scale",
                    1.0,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.mesh_scale",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=(
                            f"The scale of the mesh for the '{object_id}' object."
                        ),
                        read_only=True,
                    ),
                )
                primitive_type = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.primitive_type",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.primitive_type",
                        type=ParameterType.PARAMETER_INTEGER,
                        description=(
                            f"The primitive type of the '{object_id}' object. "
                            "Either this and `primitive_dims` must be defined, or `filename`."
                        ),
                        read_only=True,
                    ),
                )
                primitive_dims = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.primitive_dims",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.primitive_dims",
                        type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                        description=(
                            f"The dimensions of the '{object_id}' object. "
                            "Either this and `primitive_type` must be defined, or `filename`."
                        ),
                        read_only=True,
                    ),
                )
                position = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.position",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.position",
                        type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                        description=f"The position of the '{object_id}' object in the planning scene.",
                        read_only=True,
                    ),
                )
                quat_xyzw = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.quat_xyzw",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.quat_xyzw",
                        type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                        description=(
                            f"The orientation of the '{object_id}'"
                            " object in the planning scene."
                        ),
                        read_only=True,
                    ),
                )
                frame_id = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.frame_id",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.frame_id",
                        type=ParameterType.PARAMETER_STRING,
                        description=("The frame ID that the pose is in."),
                        read_only=True,
                    ),
                )
                within_workspace_walls = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.within_workspace_walls",
                    False,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.within_workspace_walls",
                        type=ParameterType.PARAMETER_BOOL,
                        description=(
                            "Whether the object is within the workspace walls."
                        ),
                        read_only=True,
                    ),
                )
                attached = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.attached",
                    False,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.attached",
                        type=ParameterType.PARAMETER_BOOL,
                        description=(
                            "Whether the object is attached to the robot. "
                            "If True, frame_id must be a robot link."
                        ),
                        read_only=True,
                    ),
                )
                touch_links = self.__node.declare_parameter(
                    f"{namespace}.{object_id}.touch_links",
                    None,
                    descriptor=ParameterDescriptor(
                        name=f"{namespace}.{object_id}.touch_links",
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
                self.objects[namespace][object_id] = CollisionObjectParams(
                    object_id=object_id,
                    position=position.value,
                    quat_xyzw=quat_xyzw.value,
                    frame_id=frame_id.value,
                    mesh_filepath=filepath,
                    mesh_scale=mesh_scale.value,
                    primitive_type=primitive_type.value,
                    primitive_dims=primitive_dims.value,
                    within_workspace_walls=within_workspace_walls.value,
                    attached=attached.value,
                    touch_links=touch_links,
                )

    def __wait_for_moveit(self, timeout: Duration(seconds=10.0)) -> bool:
        """
        Wait for the MoveIt2 interface to be ready. Specifically, it waits
        until the `/get_planning_scene` service is ready.

        Parameters
        ----------
        timeout: The maximum time to wait for the MoveIt2 interface to be ready.
        """
        # Get the start time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(self.__wait_for_moveit_hz)

        while check_ok(self.__node, start_time, timeout):
            # pylint: disable=protected-access
            # This is necessary. Ideally, the service would not be protected.
            if (
                self.__collision_object_manager.moveit2._get_planning_scene_service.service_is_ready()
            ):
                return True
            rate.sleep()

        return False

    def initialize(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=10.0)
    ) -> bool:
        """
        First, wait for the MoveIt2 interface to be ready. Then, get the objects
        that are already in the planning scene. Then, add the objects to the
        planning scene.
        """
        # Get the start time
        start_time = self.__node.get_clock().now()

        # Wait for the MoveIt2 interface to be ready
        self.__node.get_logger().info("Waiting for MoveIt2 interface...")
        success = self.__wait_for_moveit(
            timeout=get_remaining_time(self.__node, start_time, timeout)
        )
        if not success:
            self.__node.get_logger().error(
                "MoveIt2 interface is not ready. Exiting initialization."
            )
            return False
        self.__node.get_logger().info("...MoveIt2 is ready.")

        # Clear the planning scene
        self.__node.get_logger().info("Clearing planning scene...")
        success = self.__collision_object_manager.clear_all_collision_objects(
            rate_hz=rate_hz,
            timeout=get_remaining_time(self.__node, start_time, timeout),
        )
        if success:
            self.__node.get_logger().info("...cleared planning scene.")
        else:
            self.__node.get_logger().error(
                "...failed to clear planning scene. Continuing."
            )

        # # Get the objects that are already in the planning scene
        # self.__node.get_logger().info(
        #     "Getting objects currently in the planning scene..."
        # )
        # success = self.__collision_object_manager.get_global_collision_objects(
        #     rate_hz=rate_hz,
        #     timeout=get_remaining_time(self.__node, start_time, timeout),
        # )
        # if not success:
        #     self.__node.get_logger().error(
        #         "Failed to get objects in the planning scene. Exiting initialization."
        #     )
        #     return False
        # self.__node.get_logger().info("...got planning scene objects.")

        # Add the objects to the planning scene
        self.__node.get_logger().info("Adding objects to the planning scene...")
        success = self.__collision_object_manager.add_collision_objects(
            objects=self.objects[self.__namespace_to_use],
            rate_hz=rate_hz,
            timeout=get_remaining_time(self.__node, start_time, timeout),
            ignore_existing=True,
        )
        if not success:
            self.__node.get_logger().error(
                "Failed to add objects to the planning scene. Exiting initialization."
            )
            return False
        self.__node.get_logger().info("Initialized planning scene.")

        return True

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
