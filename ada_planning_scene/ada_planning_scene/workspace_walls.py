"""
This module contains the class `WorkspaceWalls` for managing the workspace walls
in ADA's planning scene.
"""

# Standard imports
import copy
from typing import Dict, List, Optional, Tuple

# Third-party imports
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import (
    Point,
    Quaternion,
)
from lxml import etree
import numpy as np
import numpy.typing as npt
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import PoseStamped
import tf2_py as tf2
from tf2_ros import TypeException
from tf2_ros.buffer import Buffer
from transforms3d._gohlketransforms import quaternion_matrix
import trimesh
from yourdfpy import URDF

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import (
    check_ok,
    CollisionObjectParams,
    get_remaining_time,
)


class WorkspaceWalls:
    """
    This class manages the workspace walls in ADA's planning scene. Specifically,
    it exposes a method to initialize the workspace walls, and a service to trigger
    re-computation of the workspace walls.
    """

    # pylint: disable=too-many-instance-attributes
    # We need to keep track of a lot to compute and update workspace walls.
    # pylint: disable=too-few-public-methods
    # This class updates workspace walls internally, so doesn't need many public methods.

    def __init__(
        self,
        node: Node,
        collision_object_manager: CollisionObjectManager,
        objects: Dict[str, CollisionObjectParams],
        base_frame_id: str,
        tf_buffer: Buffer,
    ):
        """
        Initialize the workspace walls.

        Parameters
        ----------
        node: The ROS 2 node.
        collision_object_manager: The object to manage collision objects.
        objects: The collision objects in the planning scene.
        base_frame_id: The base frame ID. Walls will be published in this frame.
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

        # Load parameters
        self.__load_parameters()

        # Check if the necessary parameters are set to use the robot model
        self.__use_robot_model = True
        if (
            self.__get_urdf_parameter_service_name is None
            or self.__get_robot_configurations_parameter_service_name is None
            or len(self.__robot_configurations_parameter_names) == 0
        ):
            self.__node.get_logger().warn(
                "Not using robot model because the necessary parameters are not set."
            )
            self.__use_robot_model = False

        if self.__use_robot_model:
            # The service to load the robot's URDF
            self.__robot_model = None
            self.__get_urdf_parameter_service = self.__node.create_client(
                GetParameters,
                self.__get_urdf_parameter_service_name,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

            # The service to load the robot arm configurations that should be within
            # the workspace walls
            self.__get_robot_configurations_parameter_service = (
                self.__node.create_client(
                    GetParameters,
                    self.__get_robot_configurations_parameter_service_name,
                    callback_group=MutuallyExclusiveCallbackGroup(),
                )
            )

        # Get the bounds of all the objects that are within the workspace walls.
        # As of now, these are only computed once.
        self.__per_object_bounds = {}
        self.__compute_object_bounds()

    def __load_parameters(self):
        """
        Load parameters relevant to the workspace walls.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # The margin (m) to add between workspace wall and objects within the workspace
        workspace_wall_margin = self.__node.declare_parameter(
            "workspace_wall_margin",
            0.1,  # default value
            ParameterDescriptor(
                name="workspace_wall_margin",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The margin (m) to add between workspace wall and objects within the workspace."
                ),
                read_only=True,
            ),
        )
        self.__workspace_wall_margin = workspace_wall_margin.value

        # The thickness (m) of the workspace walls
        workspace_wall_thickness = self.__node.declare_parameter(
            "workspace_wall_thickness",
            0.01,  # default value
            ParameterDescriptor(
                name="workspace_wall_thickness",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The thickness (m) of the workspace walls.",
                read_only=True,
            ),
        )
        self.__workspace_wall_thickness = workspace_wall_thickness.value

        # The name of the service to get the URDF parameter
        get_urdf_parameter_service_name = self.__node.declare_parameter(
            "get_urdf_parameter_service_name",
            None,  # default value
            ParameterDescriptor(
                name="get_urdf_parameter_service_name",
                type=ParameterType.PARAMETER_STRING,
                description="The name of the service to get the URDF parameter.",
                read_only=True,
            ),
        )
        self.__get_urdf_parameter_service_name = get_urdf_parameter_service_name.value

        # The parameter to request from the URDF parameter service
        urdf_parameter_name = self.__node.declare_parameter(
            "urdf_parameter_name",
            "robot_description",  # default value
            ParameterDescriptor(
                name="urdf_parameter_name",
                type=ParameterType.PARAMETER_STRING,
                description="The parameter to request from the URDF parameter service.",
                read_only=True,
            ),
        )
        self.__urdf_parameter_name = urdf_parameter_name.value

        # The name of the service to get the robot configuration parameters.
        get_robot_configurations_parameter_service_name = self.__node.declare_parameter(
            "get_robot_configurations_parameter_service_name",
            None,  # default value
            ParameterDescriptor(
                name="get_robot_configurations_parameter_service_name",
                type=ParameterType.PARAMETER_STRING,
                description="The name of the service to get the robot configuration parameters.",
                read_only=True,
            ),
        )
        self.__get_robot_configurations_parameter_service_name = (
            get_robot_configurations_parameter_service_name.value
        )

        # The parameter that contains the namespace to use. The value for this parameter,
        # if set, will be prepended to the parameter names for the robot configuration,
        # followed by a period.
        namespace_to_use_parameter_name = self.__node.declare_parameter(
            "namespace_to_use_parameter_name",
            "namespace_to_use",  # default value
            ParameterDescriptor(
                name="namespace_to_use_parameter_name",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The parameter that contains the namespace to use. The value for this "
                    "parameter, if set, will be prepended to the parameter names for the "
                    "robot configuration, followed by a period."
                ),
                read_only=True,
            ),
        )
        self.__namespace_to_use_parameter_name = namespace_to_use_parameter_name.value

        # The names of the parameters that contain robot joint configurations that should
        # be contained within the workspace walls.
        robot_configurations_parameter_names = self.__node.declare_parameter(
            "robot_configurations_parameter_names",
            None,  # default value
            ParameterDescriptor(
                name="robot_configurations_parameter_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "The names of the parameters that contain robot joint configurations "
                    "that should be contained within the workspace walls."
                ),
                read_only=True,
            ),
        )
        self.__robot_configurations_parameter_names = (
            robot_configurations_parameter_names.value
        )
        if self.__robot_configurations_parameter_names is None:
            self.__robot_configurations_parameter_names = []

        # The names and values of the fixed joints in the robot's full URDF.
        fixed_joint_names = self.__node.declare_parameter(
            "fixed_joint_names",
            [
                "robot_tilt",
                "j2n6s200_joint_finger_1",
                "j2n6s200_joint_finger_2",
            ],  # default value
            ParameterDescriptor(
                name="fixed_joint_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="The names of the fixed joints in the robot's full URDF.",
                read_only=True,
            ),
        )
        self.__fixed_joint_names = fixed_joint_names.value
        fixed_joint_values = self.__node.declare_parameter(
            "fixed_joint_values",
            [0.0, 1.33, 1.33],  # default value
            ParameterDescriptor(
                name="fixed_joint_values",
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="The values of the fixed joints in the robot's full URDF.",
                read_only=True,
            ),
        )
        self.__fixed_joint_values = fixed_joint_values.value
        min_len = min(len(self.__fixed_joint_names), len(self.__fixed_joint_values))
        self.__fixed_joint_names = self.__fixed_joint_names[:min_len]
        self.__fixed_joint_values = self.__fixed_joint_values[:min_len]

        # The name of the articulated joints in the robot's full URDF. The order
        # of these must match the order ot joints in the robot configuration parameters.
        articulated_joint_names = self.__node.declare_parameter(
            "articulated_joint_names",
            [
                "j2n6s200_joint_1",
                "j2n6s200_joint_2",
                "j2n6s200_joint_3",
                "j2n6s200_joint_4",
                "j2n6s200_joint_5",
                "j2n6s200_joint_6",
            ],  # default value
            ParameterDescriptor(
                name="articulated_joint_names",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="The names of the articulated joints in the robot's full URDF.",
                read_only=True,
            ),
        )
        self.__articulated_joint_names = articulated_joint_names.value

    def __get_homogenous_transform_in_base_frame(
        self,
        position: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
        frame_id: str,
        timeout: Duration = Duration(seconds=0.5),
    ) -> Optional[npt.NDArray]:
        """
        Transforms the position and quaternion in frame_id into base_frame. Returns
        The resulting pose represented as a homogenous transformation matrix. In other
        words, the return value times (0, 0, 0, 1) is the position in the base frame.

        Parameters
        ----------
        position: The position of the object.
        quat_xyzw: The orientation of the object.
        frame_id: The frame ID of the object.
        base_frame_id: The base frame ID.
        timeout: The timeout for the transform.

        Returns
        -------
        The homogenous transformation matrix that takes a point in the object's frame
        and converts it to the base frame. None if the transform fails.
        """
        # Get the pose as a PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position = Point(x=position[0], y=position[1], z=position[2])
        pose.pose.orientation = Quaternion(
            x=quat_xyzw[0],
            y=quat_xyzw[1],
            z=quat_xyzw[2],
            w=quat_xyzw[3],
        )

        # Transform the PoseStamped to the base frame
        try:
            pose = self.__tf_buffer.transform(pose, self.__base_frame_id, timeout)
        except (
            tf2.ConnectivityException,
            tf2.ExtrapolationException,
            tf2.InvalidArgumentException,
            tf2.LookupException,
            tf2.TimeoutException,
            tf2.TransformException,
            TypeException,
        ) as error:
            self.__node.get_logger().error(f"Failed to transform the pose: {error}")
            return None

        # Covert the pose to a homogenous transformation matrix
        pose_matrix = quaternion_matrix(
            [
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
            ]
        )
        pose_matrix[0, 3] = pose.pose.position.x
        pose_matrix[1, 3] = pose.pose.position.y
        pose_matrix[2, 3] = pose.pose.position.z

        return pose_matrix

    def __get_mesh_bounds(
        self,
        params: CollisionObjectParams,
    ) -> npt.NDArray:
        """
        Get the bounds of a mesh file.

        Parameters
        ----------
        params: The parameters of the collision object.

        Returns
        -------
        The bounds of the mesh file.
        """
        # Get the mesh
        if params.mesh is None:
            params.mesh = trimesh.load(params.mesh_filepath)
            # Once we load the mesh, we don't need the filepath anymore
            params.mesh_filepath = None
        mesh = params.mesh

        # Get the transformation matrix
        transform = self.__get_homogenous_transform_in_base_frame(
            position=params.position,
            quat_xyzw=params.quat_xyzw,
            frame_id=params.frame_id,
        )

        # Transform the mesh
        mesh_transformed = copy.deepcopy(mesh)
        mesh_transformed.apply_transform(transform)

        # Get the bounds
        bounds = mesh_transformed.bounds

        return bounds

    def __get_primitive_bounds(
        self,
        params: CollisionObjectParams,
        n_points: int = 20,
    ) -> npt.NDArray:
        """
        Get the bounds of a primitive object.

        Parameters
        ----------
        params: The parameters of the collision object.
        n_points: The number of points to use to approximate the circle(s) on the
            cylinder or cone primitives. Ignored for other primitives.

        Returns
        -------
        The bounds of the primitive object.
        """
        # Get the transformation matrix
        transform = self.__get_homogenous_transform_in_base_frame(
            position=params.position,
            quat_xyzw=params.quat_xyzw,
            frame_id=params.frame_id,
        )

        # Get the points we care about, as a (n, 3) np.ndarray
        # The origin of all primitives is at the center of the object
        if params.primitive_type == SolidPrimitive.BOX:
            len_x, len_y, len_z = params.primitive_dims
            points = np.array(
                [
                    [-len_x / 2, -len_y / 2, -len_z / 2],
                    [len_x / 2, -len_y / 2, -len_z / 2],
                    [-len_x / 2, len_y / 2, -len_z / 2],
                    [len_x / 2, len_y / 2, -len_z / 2],
                    [-len_x / 2, -len_y / 2, len_z / 2],
                    [len_x / 2, -len_y / 2, len_z / 2],
                    [-len_x / 2, len_y / 2, len_z / 2],
                    [len_x / 2, len_y / 2, len_z / 2],
                ]
            )
        elif params.primitive_type == SolidPrimitive.SPHERE:
            # For a sphere, the only point we care about is the center,
            # because it spreads equally in all directions
            points = np.array([[0, 0, 0]])
        elif params.primitive_type in {SolidPrimitive.CYLINDER, SolidPrimitive.CONE}:
            height, radius = params.primitive_dims
            # Get n_points (x, y) points along the circle
            circle_points = np.array(
                [
                    [radius * np.cos(theta), radius * np.sin(theta)]
                    for theta in np.linspace(0, 2 * np.pi, n_points)
                ]
            )
            points = [[x, y, -height / 2] for x, y in circle_points]
            if params.primitive_type == SolidPrimitive.CYLINDER:
                # For a cylinder, add the top circle
                points += [[x, y, height / 2] for x, y in circle_points]
            else:
                # For a cone, add the top point
                points += [[0, 0, height / 2]]
            points = np.array(points)

        # Transform the points
        points_homogenous = np.hstack([points, np.ones((points.shape[0], 1))])
        points_transformed = np.dot(transform, points_homogenous.T).T[:, :3]

        # Get the bounds as a (2, 3) array
        if params.primitive_type == SolidPrimitive.SPHERE:
            center = points_transformed[0]
            radius = params.primitive_dims[0]
            bounds = np.array(
                [
                    [center[0] - radius, center[1] - radius, center[2] - radius],
                    [center[0] + radius, center[1] + radius, center[2] + radius],
                ]
            )
        else:
            bounds = np.array(
                [
                    np.min(points_transformed, axis=0),
                    np.max(points_transformed, axis=0),
                ]
            )

        return bounds

    def __compute_object_bounds(self):
        """
        Compute the bounds of all objects that are within the workspace walls.
        """
        for object_id, params in self.__objects.items():
            if params.within_workspace_walls:
                if params.primitive_type is None:
                    bounds = self.__get_mesh_bounds(params)
                else:
                    bounds = self.__get_primitive_bounds(params)
                self.__per_object_bounds[object_id] = bounds

    def __compute_workspace_bounds(self) -> npt.NDArray:
        """
        Compute the workspace bounds based on the bounds of all objects within the
        workspace walls.
        """
        # Get the bounds of all objects within the workspace walls
        all_bounds = np.array(list(self.__per_object_bounds.values()))

        # Get the min and max bounds
        min_bounds = np.min(all_bounds[:, 0, :], axis=0)
        max_bounds = np.max(all_bounds[:, 1, :], axis=0)

        # Add margin
        min_bounds -= self.__workspace_wall_margin
        max_bounds += self.__workspace_wall_margin

        return np.array([min_bounds, max_bounds])

    def __compute_workspace_walls(self) -> Dict[str, CollisionObjectParams]:
        """
        Compute the workspace walls as collision objects.
        """
        workspace_walls = {}

        # Compute the bounds of the workspace
        bounds = self.__compute_workspace_bounds()

        # Add new objects for each workspace walls, with labels where left is +x,
        # top is +z, and front is -y. j corresponds to the dimension and i corresponds
        # to whether we're taking the lower or upper bound.
        labels = {
            (0, 0): "right",
            (0, 1): "front",
            (0, 2): "bottom",
            (1, 0): "left",
            (1, 1): "back",
            (1, 2): "top",
        }
        primitive_type = SolidPrimitive.BOX
        quat_xyzw = [0.0, 0.0, 0.0, 1.0]
        for i in range(2):
            for j in range(3):
                if (i, j) not in labels:
                    continue
                # Compute the position and dimensions
                position = [
                    bounds[i, k] if k == j else (bounds[0, k] + bounds[1, k]) / 2
                    for k in range(3)
                ]
                primitive_dims = [
                    self.__workspace_wall_thickness
                    if k == j
                    else bounds[1, k] - bounds[0, k]
                    for k in range(3)
                ]

                # Add the workspace wall
                object_id = f"workspace_wall_{labels[(i, j)]}"
                workspace_walls[object_id] = CollisionObjectParams(
                    object_id=object_id,
                    primitive_type=primitive_type,
                    primitive_dims=primitive_dims,
                    position=position,
                    quat_xyzw=quat_xyzw,
                    frame_id=self.__base_frame_id,
                )

        return workspace_walls

    @staticmethod
    def urdf_replace_package_paths(
        urdf: str,
    ) -> str:
        """
        Takes in a URDF string with package paths represented as `package://{name}/`
        and replaces them with the absolute path to the package.

        Parameters
        ----------
        urdf: The URDF string.

        Returns
        -------
        The URDF string with package paths replaced.
        """
        keyword = "package://"

        while urdf.find(keyword) != -1:
            start = urdf.find(keyword)
            end = urdf.find("/", start + len(keyword))
            package_name = urdf[start + len(keyword) : end]
            package_path = get_package_share_directory(package_name)
            urdf = urdf[:start] + package_path + urdf[end:]

        return urdf

    def __get_robot_model(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> bool:
        """
        Get the robot's model.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        True if successful, False otherwise.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Wait until the service is ready
        while not self.__get_urdf_parameter_service.service_is_ready():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while waiting for the get URDF parameter service."
                )
                return False
            rate.sleep()

        # Get the robot's URDF
        request = GetParameters.Request()
        request.names = [self.__urdf_parameter_name]
        future = self.__get_urdf_parameter_service.call_async(request)
        while not future.done():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while getting the robot's URDF."
                )
                return False
            rate.sleep()

        # Get the response
        try:
            response = future.result()
        except Exception as error:  # pylint: disable=broad-except
            self.__node.get_logger().error(f"Failed to get the robot's URDF: {error}")
            return False
        if (
            len(response.values) == 0
            or response.values[0].type != ParameterType.PARAMETER_STRING
        ):
            return False
        robot_urdf = response.values[0].string_value
        robot_urdf = WorkspaceWalls.urdf_replace_package_paths(robot_urdf)

        # Load the URDF. Note that this is a blocking operation, so we can't
        # check timeout during it.
        parser = etree.XMLParser(remove_blank_text=True)
        xml_root = etree.fromstring(robot_urdf, parser=parser)
        # pylint: disable=protected-access
        # `yourdfpy` only allows loading URDF from file, so we bypass its default load.
        self.__robot_model = URDF(robot=URDF._parse_robot(xml_element=xml_root))

        return True

    def __get_parameter_prefix(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> Tuple[bool, str]:
        """
        Get the namespace to use.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        success: True if successful, False otherwise.
        prefix: The prefix to add to the parameter name.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Wait for the service to be ready
        while not self.__get_robot_configurations_parameter_service.service_is_ready():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while waiting for the get robot configurations parameter service."
                )
                return False, ""
            rate.sleep()

        # Get the value of the namespace_to_use parameter
        request = GetParameters.Request()
        request.names = [self.__namespace_to_use_parameter_name]
        future = self.__get_robot_configurations_parameter_service.call_async(request)
        while not future.done():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while getting the namespace to use."
                )
                return False, ""
            rate.sleep()

        # Get the response
        try:
            response = future.result()
        except Exception as error:  # pylint: disable=broad-except
            self.__node.get_logger().error(
                f"Failed to get the namespace to use: {error}"
            )
            return False, ""
        if (
            len(response.values) == 0
            or response.values[0].type != ParameterType.PARAMETER_STRING
        ):
            prefix = ""
        else:
            prefix = response.values[0].string_value + "."

        return True, prefix

    def __get_robot_configurations(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> Tuple[bool, Dict[str, List[float]]]:
        """
        Get the robot's configurations.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        success: True if successful, False otherwise.
        robot_configurations: A map from the parameter name to the configuration.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Get the prefix
        success, prefix = self.__get_parameter_prefix(
            rate_hz, get_remaining_time(self.__node, start_time, timeout)
        )
        if not success:
            self.__node.get_logger().error("Failed to get the parameter prefix.")
            return False, {}

        # Wait for the service to be ready
        while not self.__get_robot_configurations_parameter_service.service_is_ready():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while waiting for the get robot configurations parameter service."
                )
                return False, {}
            rate.sleep()

        # Get the robot configurations
        robot_configurations = {}
        request = GetParameters.Request()
        request.names = [
            prefix + name for name in self.__robot_configurations_parameter_names
        ]
        future = self.__get_robot_configurations_parameter_service.call_async(request)
        while not future.done():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timeout while getting the robot configurations."
                )
                return False, {}
            rate.sleep()

        # Get the response
        try:
            response = future.result()
        except Exception as error:  # pylint: disable=broad-except
            self.__node.get_logger().error(
                f"Failed to get robot configurations: {error}"
            )
            return False, {}
        for i, param in enumerate(response.values):
            if param.type != ParameterType.PARAMETER_DOUBLE_ARRAY:
                continue
            robot_configurations[self.__robot_configurations_parameter_names[i]] = list(
                param.double_array_value
            )

        return True, robot_configurations

    def __update_robot_configuration_bounds(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> bool:
        """
        Updates the robot configuration bounds that must be contained within
        the workspace walls.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        True if successful, False otherwise.
        """
        # Start the time
        start_time = self.__node.get_clock().now()

        # Get the robot configurations
        success, robot_configurations = self.__get_robot_configurations(
            rate_hz, get_remaining_time(self.__node, start_time, timeout)
        )
        if not success:
            self.__node.get_logger().error("Failed to get robot configurations.")
            return False

        # Get each configuration, get the bounds
        for name, configuration in robot_configurations.items():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    f"Timeout while getting robot configuration {name}."
                )
                return False
            if len(configuration) != len(self.__articulated_joint_names):
                self.__node.get_logger().error(
                    f"Configuration {name} has the wrong number of joints."
                )
                continue

            # Get the joint values
            joint_values = dict(
                zip(self.__fixed_joint_names, self.__fixed_joint_values)
            )
            joint_values.update(zip(self.__articulated_joint_names, configuration))

            # Set the configuration in the URDF
            self.__robot_model.update_cfg(joint_values)

            # Get the bounds
            bounds = self.__robot_model.scene.bounds

            # Store the bounds
            self.__per_object_bounds[name] = bounds

        return True

    def __compute_and_add_workspace_walls(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ):
        """
        Recomputes workspace walls and adds them to the planning scene.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        True if successful, False otherwise.
        """
        # Compute the workspace walls. We don't pass the rate or timeout to
        # this function because it is relatively fast and doesn't have asynchronous
        # components.
        workspace_walls = self.__compute_workspace_walls()

        # Add the workspace walls to the planning scene
        success = self.__collision_object_manager.add_collision_objects(
            workspace_walls, rate_hz, timeout
        )
        return success

    def initialize(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> bool:
        """
        Initialize the workspace walls.

        Parameters
        ----------
        rate_hz: The rate at which to call the service.
        timeout: The timeout for the service.

        Returns
        -------
        True if successful, False otherwise.
        """
        start_time = self.__node.get_clock().now()

        # Load the robot's URDF. We do this in `initialize` as opposed to `__init__`
        # because the MoveGroup has to be running to get the paramter.
        if self.__use_robot_model:
            # Get the robot model (may take <= 10 secs)
            self.__node.get_logger().info("Loading robot model.")
            success = self.__get_robot_model(
                rate_hz=rate_hz,
                timeout=get_remaining_time(self.__node, start_time, timeout),
            )
            if not success:
                self.__node.get_logger().error("Failed to load robot model.")
                return False
            self.__node.get_logger().info("Loaded robot model.")

            # Get the different arm configurations and their corresponding
            # bounds that must be contained within the workspace walls.
            self.__node.get_logger().info("Updating robot configuration bounds.")
            success = self.__update_robot_configuration_bounds(
                rate_hz=rate_hz,
                timeout=get_remaining_time(self.__node, start_time, timeout),
            )
            if not success:
                self.__node.get_logger().info(
                    "Failed to update robot configuration bounds."
                )
                return False

        success = self.__compute_and_add_workspace_walls(
            rate_hz, get_remaining_time(self.__node, start_time, timeout)
        )
        if not success:
            self.__node.get_logger().error("Failed to compute and add workspace walls.")
            return False
        self.__node.get_logger().info("Initialized workspace walls.")

        return True
