"""
This module contains helper functions for populating and maintaining the planning
scene.
"""

# Standard imports
from typing import List, Optional, Tuple, Union

# Third-party imports
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import trimesh

# Local imports


class CollisionObjectParams:
    """
    The CollisionObjectParams stores a superset of the parameters we need to
    add a collision object to the planning scene.
    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes, too-few-public-methods
    # This class is a data class and should have many attributes

    def __init__(
        self,
        # All collision objects need an ID, position, orientation, and frame ID
        object_id: str,
        position: Tuple[float, float, float],
        quat_xyzw: Tuple[float, float, float, float],
        frame_id: str,
        # Optional parameters for specifying a mesh
        mesh_filepath: Optional[str] = None,
        mesh: Optional[trimesh.Trimesh] = None,
        mesh_scale: Union[float, Tuple[float, float, float]] = 1.0,
        # Optional parameters for specifying a primitive
        primitive_type: Optional[int] = None,
        primitive_dims: Optional[List[float]] = None,
        # Whether the collision object is within the workspace walls
        within_workspace_walls: bool = False,
        # Optional parameters for attaching the collision object
        attached: bool = False,
        touch_links: Optional[List[str]] = None,
    ):
        """
        Initialize the CollisionObjectParams.

        Parameters
        ----------
        object_id: The ID of the collision object.
        position: The position of the collision object.
        quat_xyzw: The orientation of the collision object.
        frame_id: The frame ID of the collision object.
        mesh_filepath: The filepath to the mesh of the collision object.
        mesh: The mesh of the collision object. If passed, mesh_filepath is not loaded.
        mesh_scale: The scale of the mesh.
        primitive_type: The type of the primitive.
        primitive_dims: The dimensions of the primitive.
        within_workspace_walls: Whether the collision object is within the workspace walls.
        attached: Whether the collision object is attached to the robot.
        touch_links: The touch links of the collision object.
        """
        # Store the parameters
        self.object_id = object_id
        self.position = position
        self.quat_xyzw = quat_xyzw
        self.frame_id = frame_id
        self.mesh_filepath = mesh_filepath
        self.mesh = mesh
        self.mesh_scale = mesh_scale
        self.primitive_type = primitive_type
        self.primitive_dims = primitive_dims
        self.within_workspace_walls = within_workspace_walls
        self.attached = attached
        self.touch_links = [] if touch_links is None else touch_links


def duration_minus(
    duration1: Duration,
    duration2: Duration,
) -> Duration:
    """
    Subtract two durations.

    Parameters
    ----------
    duration1: The first duration.
    duration2: The second duration.

    Returns
    -------
    The difference between the two durations.
    """
    return Duration(nanoseconds=duration1.nanoseconds - duration2.nanoseconds)


def get_remaining_time(
    node: Node,
    start_time: Time,
    timeout: Duration,
) -> Duration:
    """
    Get the remaining time until timeout.

    Parameters
    ----------
    node: The ROS 2 node.
    start_time: The start time.
    timeout: The timeout duration.

    Returns
    -------
    The remaining time.
    """
    current_time = node.get_clock().now()
    elapsed_time = duration_minus(current_time, start_time)
    remaining_time = duration_minus(timeout, elapsed_time)
    return remaining_time


def has_timed_out(
    node: Node,
    start_time: Time,
    timeout: Duration,
) -> bool:
    """
    Check if the timeout has been reached.

    Parameters
    ----------
    node: The ROS 2 node.
    start_time: The start time.
    timeout: The timeout duration.

    Returns
    -------
    True if the timeout has been reached, False otherwise.
    """
    remaining_time = get_remaining_time(
        node=node,
        start_time=start_time,
        timeout=timeout,
    )
    return remaining_time.nanoseconds <= 0
