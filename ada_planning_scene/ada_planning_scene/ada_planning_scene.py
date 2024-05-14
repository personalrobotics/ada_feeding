#!/usr/bin/env python3
"""
This module contains the main node for populating and maintaining ADA's planning scene.
"""

# Standard imports
import threading
from typing import List

# Third-party imports
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from tf2_ros.buffer import Buffer
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.helpers import get_remaining_time
from ada_planning_scene.planning_scene_initializer import PlanningSceneInitializer
from ada_planning_scene.update_from_face_detection import UpdateFromFaceDetection
from ada_planning_scene.update_from_table_detection import UpdateFromTableDetection
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
        self.__initializer = PlanningSceneInitializer(
            node=self,
            collision_object_manager=self.__collision_object_manager,
            namespaces=self.__namespaces,
            namespace_to_use=self.__namespace_to_use,
        )
        self.__objects = (
            self.__initializer.objects
        )  # namespace (str) -> object_id (str) -> params (CollisionObjectParams)
        # Initialize the TF listeners and broadcasters
        self.__tf_buffer = Buffer()
        # pylint: disable=unused-private-member
        self.__tf_listener = TransformListener(self.__tf_buffer, self)
        self.__tf_broadcaster = StaticTransformBroadcaster(self)

        # Create an object to manage the workspace walls
        self.__workspace_walls = WorkspaceWalls(
            node=self,
            collision_object_manager=self.__collision_object_manager,
            objects=self.__objects,
            base_frame_id=self.__base_frame,
            tf_buffer=self.__tf_buffer,
            namespaces=self.__namespaces,
            namespace_to_use=self.__namespace_to_use,
        )

        # Add a callback to update the namespace to use
        self.add_on_set_parameters_callback(self.parameter_callback)

    def __load_parameters(self):
        """
        Load parameters from the parameter server.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # Get the list of namespaces that parameters are nested within.
        # This is to allow different preset planning scenes to co-exist.
        namespaces = self.declare_parameter(
            "namespaces",
            None,
            ParameterDescriptor(
                name="namespaces",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "The list of namespaces that parameters are nested within. "
                    "This is to allow different preset planning scenes to co-exist."
                ),
                read_only=True,
            ),
        )
        self.__namespaces = [] if namespaces.value is None else namespaces.value

        # Get the namespace that is currently being used. Note that this is the only
        # changeable parameter in this node.
        namespace_to_use = self.declare_parameter(
            "namespace_to_use",
            None,
            ParameterDescriptor(
                name="namespace_to_use",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The namespace that is currently being used. Note that this is the only "
                    "changeable parameter in this node."
                ),
                read_only=False,
            ),
        )
        self.__namespace_to_use = namespace_to_use.value
        if self.__namespace_to_use not in self.__namespaces:
            raise ValueError(
                "The `namespace_to_use` parameter must be included in the `namespaces` parameter."
            )

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

        # If all the collision objects have not been succesfully added to the
        # planning scene within this time, stop initialization.
        initialization_timeout_secs = self.declare_parameter(
            "initialization_timeout_secs",
            40.0,  # default value
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
        self.__initialization_timeout = Duration(
            seconds=initialization_timeout_secs.value
        )

        # The rate (Hz) at which to publish each planning scene object
        initialization_hz = self.declare_parameter(
            "initialization_hz",
            10.0,  # default value
            ParameterDescriptor(
                name="initialization_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to publish each planning scene object during initialization."
                ),
                read_only=True,
            ),
        )
        self.__initialization_hz = initialization_hz.value

    def initialize(self) -> bool:
        """
        Initialize the planning scene.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        # Get the start time
        start_time = self.get_clock().now()

        # Initialize the planning scene
        success = self.__initializer.initialize(
            rate_hz=self.__initialization_hz,
            timeout=get_remaining_time(self, start_time, self.__initialization_timeout),
        )
        if not success:
            self.get_logger().error("Failed to initialize the planning scene.")
            return False
        success = self.__workspace_walls.initialize(
            rate_hz=self.__initialization_hz,
            timeout=get_remaining_time(self, start_time, self.__initialization_timeout),
        )
        if not success:
            self.get_logger().error("Failed to initialize the workspace walls.")
            return False

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
            namespaces=self.__namespaces,
            namespace_to_use=self.__namespace_to_use,
        )

        # Create an object to process planning scene updates from table detection
        self.__update_from_table_detection = UpdateFromTableDetection(
            node=self,
            collision_object_manager=self.__collision_object_manager,
            objects=self.__objects,
            base_frame_id=self.__base_frame,
            tf_buffer=self.__tf_buffer,
            namespaces=self.__namespaces,
            namespace_to_use=self.__namespace_to_use,
        )

        self.get_logger().info("Finished initializing the planning scene.")
        return True

    def parameter_callback(self, params: List[Parameter]) -> SetParametersResult:
        """
        Callback for when parameters are set.
        """
        # pylint: disable=attribute-defined-outside-init
        # Fine for this method

        for param in params:
            if param.name == "namespace_to_use":
                # Check the parameter
                namespace_to_use = param.value
                if namespace_to_use not in self.__namespaces:
                    self.get_logger().error(
                        "The `namespace_to_use` parameter must be part of the `namespaces` parameter."
                    )
                    return SetParametersResult(successful=False)

                # Set the parameter
                self.__namespace_to_use = namespace_to_use
                self.__initializer.namespace_to_use = namespace_to_use
                self.__workspace_walls.namespace_to_use = namespace_to_use
                self.__update_from_face_detection.namespace_to_use = namespace_to_use
                self.__update_from_table_detection.namespace_to_use = namespace_to_use

                # Re-initialize the planning scene
                self.initialize()

        return SetParametersResult(successful=True)


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
    success = ada_planning_scene.initialize()

    if not success:
        ada_planning_scene.get_logger().error("Exiting node.")
        # If initialization fails, stop the node
        rclpy.shutdown()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()