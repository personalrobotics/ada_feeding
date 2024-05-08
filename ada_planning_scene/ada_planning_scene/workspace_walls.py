"""
This module contains the class `WorkspaceWalls` for managing the workspace walls
in ADA's planning scene.
"""

# Standard imports
import copy
from typing import Dict, Optional, Tuple

# Third-party imports
from geometry_msgs.msg import (
    Point,
    Quaternion,
)
import numpy as np
import numpy.typing as npt
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.duration import Duration
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_geometry_msgs import PoseStamped
import tf2_py as tf2
from tf2_ros import TypeException
from tf2_ros.buffer import Buffer
from transforms3d._gohlketransforms import quaternion_matrix
import trimesh

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import CollisionObjectParams


class WorkspaceWalls:
    """
    This class manages the workspace walls in ADA's planning scene. Specifically,
    it exposes a method to initialize the workspace walls, and a service to trigger
    re-computation of the workspace walls.
    """

    # pylint: disable=too-many-instance-attributes
    # We need to keep track of a lot to compute and update workspace walls.

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

    def initialize(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=5)
    ) -> None:
        """
        Initialize the workspace walls.
        """
        # Compute the workspace walls
        workspace_walls = self.__compute_workspace_walls()

        # Add the workspace walls to the planning scene
        self.__collision_object_manager.add_collision_objects(
            workspace_walls, rate_hz, timeout
        )

        self.__node.get_logger().info("Initialized workspace walls.")
