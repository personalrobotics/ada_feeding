#!/usr/bin/env python3
"""
This module contains the main node for populating and maintaining ADA's planning scene.
"""

# Standard imports
import threading
from typing import List

# Third-party imports
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.planning_scene_initializer import PlanningSceneInitializer
from ada_planning_scene.update_from_face_detection import UpdateFromFaceDetection

# from ada_planning_scene.update_from_table_detection import UpdateFromTableDetection
from ada_planning_scene.workspace_walls import WorkspaceWalls


class ADAPlanningScene(Node):
    """
    This node does the following:
      1. Initializes the planning scene with statically-configured collision
         objects and dynamically-configured workspace walls.
      2. Updates the head and body pose based on the results of face detection.
      3. Updates the table pose based on the results of table detection.
      4. Exposes a service that can be used to trigger re-computation of the
         workspace walls.
    """

    # pylint: disable=too-many-instance-attributes
    # Fine for this class.

    def __init__(self):
        """
        Initialize the planning scene.
        """
        super().__init__("ada_planning_scene")

        # Load the parameters. Note that each of the other classes below may
        # initialize their own parameters, all of which will be in the same
        # namespace as the parameters loaded here.
        self.__load_parameters()

        # Create an object to add collision objects to the planning scene
        self.__collision_object_manager = CollisionObjectManager(node=self)

        # Create the initializer
        self.initializer = PlanningSceneInitializer(
            node=self,
            collision_object_manager=self.__collision_object_manager,
        )
        self.__objects = self.initializer.objects

        # Initialize the TF listeners and broadcasters
        self.__tf_buffer = Buffer()
        # pylint: disable=unused-private-member
        self.__tf_listener = TransformListener(self.__tf_buffer, self)
        self.__tf_broadcaster = StaticTransformBroadcaster(self)

        # Create an object to manage the workspace walls
        self.__workspace_walls = None
        if self.__use_workspace_walls:
            self.__workspace_walls = WorkspaceWalls(
                node=self,
                collision_object_manager=self.__collision_object_manager,
                objects=self.__objects,
                base_frame_id=self.__base_frame,
                tf_buffer=self.__tf_buffer,
            )

    def __load_parameters(self):
        """
        Load parameters from the parameter server.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # Whether or not to add workspace walls to the planning scene
        use_workspace_walls = self.declare_parameter(
            "use_workspace_walls",
            True,  # default value
            ParameterDescriptor(
                name="use_workspace_walls",
                type=ParameterType.PARAMETER_BOOL,
                description=(
                    "Whether or not to add workspace walls to the planning scene."
                ),
                read_only=True,
            ),
        )
        self.__use_workspace_walls = use_workspace_walls.value

        # Get the base_frame. This is the frame that workspace walls are published in.
        base_frame = self.declare_parameter(
            "base_frame",
            "root",
            ParameterDescriptor(
                name="base_frame",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The base frame. Workspace walls and other dynamically added "
                    "objects are published in this frame."
                ),
                read_only=True,
            ),
        )
        self.__base_frame = base_frame.value

    def initialize(self):
        """
        Initialize the planning scene.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # Initialize the planning scene
        self.initializer.initialize()
        if self.__use_workspace_walls:
            self.__workspace_walls.initialize()

        # pylint: disable=unused-private-member
        # Update attributes contain subscribers and automatically perform work
        # even if not used.

        # Create an object to process planning scene updates from face detection
        self.__update_from_face_detection = UpdateFromFaceDetection(
            node=self,
            collision_object_manager=self.__collision_object_manager,
            objects=self.__objects,
            base_frame_id=self.__base_frame,
            tf_buffer=self.__tf_buffer,
            tf_broadcaster=self.__tf_broadcaster,
        )

        # # Create an object to process planning scene updates from table detection
        # self.__update_from_table_detection = UpdateFromTableDetection(
        #     node=self,
        #     collision_object_manager=self.__collision_object_manager,
        #     objects=self.__objects,
        #     base_frame_id=self.__base_frame,
        #     tf_buffer=self.__tf_buffer,
        # )


def main(args: List = None) -> None:
    """
    Create the ROS2 node.
    """
    rclpy.init(args=args)

    ada_planning_scene = ADAPlanningScene()

    # Use a MultiThreadedExecutor to enable processing goals concurrently
    executor = MultiThreadedExecutor()

    # Spin in the background so that initialization can occur.
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(ada_planning_scene,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Initialize the planning scene
    ada_planning_scene.initialize()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
