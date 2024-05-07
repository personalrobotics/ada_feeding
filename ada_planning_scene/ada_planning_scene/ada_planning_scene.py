#!/usr/bin/env python3
"""
This module contains the main node for populating and maintaining ADA's planning scene.
"""

# Standard imports
import threading
from typing import List

# Third-party imports
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Local imports
from ada_planning_scene.collision_object_manager import CollisionObjectManager
from ada_planning_scene.planning_scene_initializer import PlanningSceneInitializer


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

    def __init__(self):
        """
        Initialize the planning scene.
        """
        super().__init__("ada_planning_scene")

        # Create an object to add collision objects to the planning scene
        self.collision_object_manager = CollisionObjectManager(node=self)

        # Create the initializer
        self.initializer = PlanningSceneInitializer(
            node=self,
            collision_object_manager=self.collision_object_manager,
        )

    def initialize(self):
        """
        Initialize the planning scene.
        """
        self.initializer.initialize()


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
