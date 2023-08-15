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
from typing import List

# Third-party imports
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

CollisionMeshParams = namedtuple(
    "CollisionMeshParams", ["filepath", "position", "quat_xyzw"]
)


class ADAPlanningScene(Node):
    """
    A node that initially populates the robot's planning scene with arbitrary
    STL meshes (passed in as parameters).

    In practice, this node is used to add the wheelchair, table, and user's face.
    """

    def __init__(self) -> None:
        """
        Initialize the planning scene.
        """
        super().__init__("ada_planning_scene")

        # Load the parameters
        self.load_parameters()

        # Initialize the MoveIt2 interface
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name=kinova.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

    def load_parameters(self) -> None:
        """
        Load the parameters for the planning scene.
        """
        # Read the assets directory path
        assets_dir = self.declare_parameter(
            "assets_dir",
            descriptor=ParameterDescriptor(
                name="assets_dir",
                type=ParameterType.PARAMETER_STRING,
                description="The absolute path to the directory to find the assets (e.g., STL mesh files).",
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
                "%s.filename" % object_id,
                descriptor=ParameterDescriptor(
                    name="filename",
                    type=ParameterType.PARAMETER_STRING,
                    description="The filename of the mesh for the '%s' object."
                    % object_id,
                    read_only=True,
                ),
            )
            position = self.declare_parameter(
                "%s.position" % object_id,
                descriptor=ParameterDescriptor(
                    name="position",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description="The position of the '%s' object in the planning scene."
                    % object_id,
                    read_only=True,
                ),
            )
            quat_xyzw = self.declare_parameter(
                "%s.quat_xyzw" % object_id,
                descriptor=ParameterDescriptor(
                    name="quat_xyzw",
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                    description="The orientation of the '%s' object in the planning scene."
                    % object_id,
                    read_only=True,
                ),
            )

            # Add the object to the list of objects
            self.objects[object_id] = CollisionMeshParams(
                filepath=path.join(assets_dir.value, filename.value),
                position=position.value,
                quat_xyzw=quat_xyzw.value,
            )

    def initialize_planning_scene(self) -> None:
        """
        Initialize the planning scene with the objects.
        """
        # Add each object to the planning scene
        for object_id, params in self.objects.items():
            self.moveit2.add_collision_mesh(
                id=object_id,
                filepath=params.filepath,
                position=params.position,
                quat_xyzw=params.quat_xyzw,
            )


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

    # Initialize the planning scene
    ada_planning_scene.initialize_planning_scene()
    ada_planning_scene.get_logger().info("Planning scene initialized.")

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
