"""
This module defines the CollisionObjectManager class, which tracks the IDs of the
(attached) collision objects in the planning scene and allows users to add collision
objects to the planning scene.
"""

# Standard imports
from threading import Lock
from typing import Callable, Dict, Optional, Union

# Third-party imports
from moveit_msgs.msg import CollisionObject, PlanningScene
from pymoveit2 import MoveIt2
from pymoveit2.robots import kinova
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node

# Local imports
from ada_planning_scene.helpers import check_ok, CollisionObjectParams


class CollisionObjectManager:
    """
    The CollisionObjectManager class does the following:
    1. Maintains a global list of the IDs of (attached) collision objects in the
       planning scene.
    2. Allows users to add collision objects to the planning scene. Collision objects
       are added in "batches," where the colllision object manager keeps trying until
       MoveIt2 has confirmed that the collision object has been added.
    """

    __GLOBAL_BATCH_ID = "global"
    __BATCH_ID_FORMAT = "batch_{:d}"

    def __init__(self, node: Node):
        """
        Initialize the CollisionObjectManager.

        Parameters
        ----------
        node: The ROS 2 node.
        """
        self.__node = node

        # Initialize the MoveIt2 interface
        # Using ReentrantCallbackGroup to align with the examples from pymoveit2.
        # TODO: Assess whether ReentrantCallbackGroup is necessary for MoveIt2.
        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self.__node,
            joint_names=kinova.joint_names(),
            base_link_name=kinova.base_link_name(),
            end_effector_name="forkTip",
            group_name="jaco_arm",
            callback_group=callback_group,
        )

        # Add the parameters to store the (attached) collision objects that have
        # been added since the start of each batch.
        self.__collision_objects_lock = Lock()
        self.__n_batches = 0
        self.__collision_objects_per_batch = {
            self.__GLOBAL_BATCH_ID: set(),
        }
        self.__attached_collision_objects_per_batch = {
            self.__GLOBAL_BATCH_ID: set(),
        }

        # Subscribe to the monitored planning scene to get updates on which
        # CollisionObject messages got processed.
        # pylint: disable=unused-private-member
        self.__monitored_planning_scene_sub = self.__node.create_subscription(
            PlanningScene,
            "~/monitored_planning_scene",
            self.__monitored_planning_scene_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def __process_planning_scene(
        self,
        planning_scene: PlanningScene,
        batch_id_to_update: Optional[str] = None,
    ) -> None:
        """
        Takes in a planning_scene message and updates the (attached) collision objects.

        Parameters
        ----------
        planning_scene: The planning scene message.
        batch_id_to_update: The batch ID to update. If None, update all the batch IDs.
        """
        with self.__collision_objects_lock:
            # Get the batch_ids to update
            batch_ids = (
                self.__collision_objects_per_batch.keys()
                if batch_id_to_update is None
                else [batch_id_to_update]
            )

            # If the message is not a diff, clear the collision objects
            if not planning_scene.is_diff:
                for batch_id in batch_ids:
                    self.__collision_objects_per_batch[batch_id].clear()
                    self.__attached_collision_objects_per_batch[batch_id].clear()

            # Update the (attached) collision objects
            for batch_id in batch_ids:
                # Update collision objects
                for collision_object in planning_scene.world.collision_objects:
                    if collision_object.operation == CollisionObject.REMOVE:
                        self.__collision_objects_per_batch[batch_id].discard(
                            collision_object.id
                        )
                    else:
                        self.__collision_objects_per_batch[batch_id].add(
                            collision_object.id
                        )

                # Update attached collision objects
                for (
                    attached_collision_object
                ) in planning_scene.robot_state.attached_collision_objects:
                    if attached_collision_object.operation == CollisionObject.REMOVE:
                        self.__attached_collision_objects_per_batch[batch_id].discard(
                            attached_collision_object.object.id
                        )
                    else:
                        self.__attached_collision_objects_per_batch[batch_id].add(
                            attached_collision_object.object.id
                        )

    def __monitored_planning_scene_callback(self, msg: PlanningScene) -> None:
        """
        Callback for the monitored planning scene.

        Parameters
        ----------
        msg: The PlanningScene message.
        """
        self.__process_planning_scene(planning_scene=msg)

    def get_global_collision_objects(
        self,
        rate_hz: float = 1.0,
        timeout: Duration = Duration(seconds=10.0),
    ) -> bool:
        """
        This method gets the global list of (attached) collision objects, by
        invoking MoveIt's `/get_planning_scene` service to get the entire planning
        scene. This is intended to be used during initialization, and overrides
        the list of (attached) collision objects in the global batch ID.

        Parameters
        ----------
        rate_hz: The rate at which to check for the planning scene.
        timeout: The maximum amount of time to wait for the planning scene.

        Returns
        -------
        True if the global list of (attached) collision objects was successfully
        retrieved, False otherwise.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Get the planning scene
        while self.moveit2.planning_scene is None:
            # Check if the node is still OK and if the timeout has been reached
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timed out while getting the planning scene."
                )
                return False

            # Attempt to get the planning scene
            success = self.moveit2.update_planning_scene()
            if success:
                break
            self.__node.get_logger().info(
                "Waiting for the planning scene...", throttle_duration_sec=1.0
            )

            # Sleep
            rate.sleep()

        # Process the planning scene
        self.__process_planning_scene(
            planning_scene=self.moveit2.planning_scene,
            batch_id_to_update=self.__GLOBAL_BATCH_ID,
        )

        return True

    def add_collision_objects(
        self,
        objects: Union[CollisionObjectParams, Dict[str, CollisionObjectParams]],
        rate_hz: float = 10.0,
        timeout: Duration = Duration(seconds=10.0),
        ignore_existing: bool = False,
        publish_feedback: Optional[Callable[[], None]] = None,
        retry_until_added: bool = True,
    ) -> bool:
        """
        Add collision objects to the planning scene.

        Parameters
        ----------
        objects: A map from object ID to CollisionObjectParams for collision objects to add.
        rate_hz: The rate at which to publish messages to add collision objects.
        timeout: The maximum amount of time to wait for the collision objects to be added.
        ignore_existing: If True, ignore the existing collision objects.
        publish_feedback: If specified, invoke this function periodically.
        retry_until_added: If True, keep retrying until all collision objects are added.

        Returns
        -------
        True if the collision objects were successfully added, False otherwise.
        """
        # pylint: disable=too-many-arguments, too-many-branches, too-many-statements
        # This is the main bread and butter of adding to the planning scene,
        # so is expected to be complex.
        self.__node.get_logger().info(
            "Adding collision objects to the planning scene..."
        )

        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Check if the objects are a single object
        if isinstance(objects, CollisionObjectParams):
            objects = {objects.object_id: objects}

        # Create a new batch for this add_collision_objects operation
        if retry_until_added:
            with self.__collision_objects_lock:
                batch_id = self.__BATCH_ID_FORMAT.format(self.__n_batches)
                if ignore_existing:
                    self.__collision_objects_per_batch[batch_id] = set()
                    self.__attached_collision_objects_per_batch[batch_id] = set()
                else:
                    self.__collision_objects_per_batch[
                        batch_id
                    ] = self.__collision_objects_per_batch[
                        self.__GLOBAL_BATCH_ID
                    ].copy()
                    self.__attached_collision_objects_per_batch[
                        batch_id
                    ] = self.__attached_collision_objects_per_batch[
                        self.__GLOBAL_BATCH_ID
                    ].copy()
                self.__n_batches += 1

        # First, try to add all the collision objects
        collision_object_ids = set(objects.keys())
        i = -1
        while len(collision_object_ids) > 0:
            i += 1
            # Check if the node is still OK and if the timeout has been reached
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timed out while adding collision objects. "
                    f"May not have added {collision_object_ids}."
                )
                return False

            # Remove any collision objects that have already been added
            if retry_until_added:
                with self.__collision_objects_lock:
                    if ignore_existing or i > 0:
                        collision_object_ids -= self.__collision_objects_per_batch[
                            batch_id
                        ]
                if len(collision_object_ids) == 0:
                    break

            # Add the collision objects
            self.__node.get_logger().info(
                f"Adding these objects to the planning scene: {collision_object_ids}",
                throttle_duration_sec=5.0,
            )
            for object_id in collision_object_ids:
                if not check_ok(self.__node, start_time, timeout):
                    break

                # Publish feedback
                if publish_feedback is not None:
                    publish_feedback()

                params = objects[object_id]
                # Collision mesh
                if params.primitive_type is None:
                    self.moveit2.add_collision_mesh(
                        id=object_id,
                        filepath=params.mesh_filepath if params.mesh is None else None,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                        mesh=params.mesh,
                        scale=params.mesh_scale,
                    )
                # Collision primitive
                else:
                    self.moveit2.add_collision_primitive(
                        id=object_id,
                        primitive_type=params.primitive_type,
                        dimensions=params.primitive_dims,
                        position=params.position,
                        quat_xyzw=params.quat_xyzw,
                        frame_id=params.frame_id,
                    )
                rate.sleep()
            if not retry_until_added:
                break

        # Second, attach all collision objects that need to be attached
        attached_collision_object_ids = {
            object_id for object_id, params in objects.items() if params.attached
        }
        i = -1
        while len(attached_collision_object_ids) > 0:
            i += 1
            # Check if the node is still OK and if the timeout has been reached
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timed out while attaching collision objects. "
                    f"May not have attached {attached_collision_object_ids}."
                )
                return False

            # Remove any attached collision objects that have already been attached
            if retry_until_added:
                with self.__collision_objects_lock:
                    if ignore_existing or i > 0:
                        attached_collision_object_ids -= (
                            self.__attached_collision_objects_per_batch[batch_id]
                        )
                if len(attached_collision_object_ids) == 0:
                    break

            # Attach the collision objects
            self.__node.get_logger().info(
                f"Attaching these objects to the robot: {attached_collision_object_ids}",
                throttle_duration_sec=5.0,
            )
            for object_id in attached_collision_object_ids:
                # Publish feedback
                if publish_feedback is not None:
                    publish_feedback()

                if not check_ok(self.__node, start_time, timeout):
                    break
                params = objects[object_id]
                self.moveit2.attach_collision_object(
                    id=object_id,
                    link_name=params.frame_id,
                    touch_links=params.touch_links,
                )
                rate.sleep()
            if not retry_until_added:
                break

        # Remove the batch that corresponds to this add_collision_objects
        # operation
        if retry_until_added:
            with self.__collision_objects_lock:
                self.__collision_objects_per_batch.pop(batch_id)
                self.__attached_collision_objects_per_batch.pop(batch_id)

        return True

    def move_collision_objects(
        self,
        objects: Union[CollisionObjectParams, Dict[str, CollisionObjectParams]],
        rate_hz: float = 10.0,
        timeout: Duration = Duration(seconds=10.0),
    ) -> bool:
        """
        Move collision objects in the planning scene. Note that moving attached
        collision objects is not yet implemented.

        Parameters
        ----------
        objects: A map from object ID to CollisionObjectParams for collision objects to move.
        rate_hz: The rate at which to publish messages to move collision objects.
        timeout: The maximum amount of time to wait for the collision objects to be moved.

        Returns
        -------
        True if the collision objects were successfully moved, False otherwise.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Check if the objects are a single object
        if isinstance(objects, CollisionObjectParams):
            objects = {objects.object_id: objects}

        # Create a new batch for this add_collision_objects operation
        with self.__collision_objects_lock:
            batch_id = self.__BATCH_ID_FORMAT.format(self.__n_batches)
            self.__collision_objects_per_batch[batch_id] = set()
            self.__attached_collision_objects_per_batch[batch_id] = set()
            self.__n_batches += 1

        # Store whether the collision objects were successfully moved
        success = True

        # Move all collision objects
        collision_object_ids = set(objects.keys())
        i = -1
        while len(collision_object_ids) > 0:
            i += 1
            # Check if the node is still OK and if the timeout has been reached
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timed out while moving collision objects. "
                    f"May not have moved {collision_object_ids}."
                )
                success = False
                break

            # Remove any collision objects that have already been moved
            with self.__collision_objects_lock:
                if i > 0:
                    collision_object_ids -= self.__collision_objects_per_batch[batch_id]
            if len(collision_object_ids) == 0:
                break

            # Move the collision objects
            self.__node.get_logger().info(
                f"Moving these objects in the planning scene: {collision_object_ids}",
                throttle_duration_sec=1.0,
            )
            for object_id in collision_object_ids:
                if not check_ok(self.__node, start_time, timeout):
                    break
                params = objects[object_id]
                if params.attached:
                    self.__node.get_logger().warn(
                        (
                            "Moving attached collision objects is not yet implemented. "
                            "Skipping this object."
                        ),
                        throttle_duration_sec=1.0,
                    )
                    continue
                # Move the object
                self.moveit2.move_collision(
                    id=object_id,
                    position=params.position,
                    quat_xyzw=params.quat_xyzw,
                    frame_id=params.frame_id,
                )
                rate.sleep()

        # Remove the batch that corresponds to this move_collision_object operation
        with self.__collision_objects_lock:
            self.__collision_objects_per_batch.pop(batch_id)
            self.__attached_collision_objects_per_batch.pop(batch_id)

        return success

    def clear_all_collision_objects(
        self, rate_hz: float = 10.0, timeout: Duration = Duration(seconds=10.0)
    ) -> bool:
        """
        Remove all attached and unattached collision objects from the planning scene.

        Parameters
        ----------
        rate_hz: The rate at which to check for the planning scene.
        timeout: The maximum amount of time to wait for the planning scene.

        Returns
        -------
        True if the planning scene was successfully cleared, False otherwise.
        """
        # Start the time
        start_time = self.__node.get_clock().now()
        rate = self.__node.create_rate(rate_hz)

        # Clear the planning scene
        future = self.moveit2.clear_all_collision_objects()
        if future is None:
            self.__node.get_logger().error(
                "Could not clear planning scene; service is not ready"
            )
            return False

        while not future.done():
            if not check_ok(self.__node, start_time, timeout):
                self.__node.get_logger().error(
                    "Timed out while clearing the planning scene."
                )
                self.moveit2.cancel_clear_all_collision_objects_future(future)
                return False

            # Sleep
            rate.sleep()

        return self.moveit2.process_clear_all_collision_objects_future(future)

    def remove_collision_object(
        self,
        object_id: str,
    ) -> bool:
        """
        Remove a specific collision object from the planning scene.

        Parameters
        ----------
        object_id: The ID of the collision object to remove.

        Returns
        -------
        True if the collision object was successfully removed, False otherwise.
        """
        # TODO: Extend this to verify that the object was removed in a closed-loop
        # fashion, as `add_collision_objects` does.
        self.__node.get_logger().info(f"Removing collision object {object_id}...")
        self.moveit2.remove_collision_object(id=object_id)
        return True
