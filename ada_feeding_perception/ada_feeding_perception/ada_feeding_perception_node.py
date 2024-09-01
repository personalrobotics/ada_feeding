"""
This module contains the ADAFeedingPerceptionNode class, which is used as a component
of all perception nodes in the ADA Feeding project. Specifically, by storing all
functionality to get camera images (RGB and depth) and info in this node, it makes
it easier to combine one or more perception functionalities into a single node,
which will reduce the number of parallel subscriptions the ROS2 middleware has to
manage.
"""
# Standard imports
from collections import deque
from functools import partial
from threading import Lock, Thread
from typing import Any, Optional, Type, Union

# Third-party imports
import rclpy
from rclpy.callback_groups import CallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription

# Local imports


class ADAFeedingPerceptionNode(Node):
    """
    Class that contains all functionality to get camera images (RGB and depth) and
    camera info. This class is meant to consolidate the number of parallel subscriptions
    the ROS2 middleware has to manage.
    """

    def __init__(self, name: str):
        """
        Constructor for the ADAFeedingPerceptionNode class.

        Parameters
        ----------
        name : str
            The name of the node.
        """
        super().__init__(name)
        self.__msg_locks: dict[str, Lock] = {}
        self.__latest_msgs: dict[str, deque[Any]] = {}
        self.__subs: dict[str, Subscription] = {}

    # pylint: disable=too-many-arguments
    # These are fine to mimic ROS2 API
    def add_subscription(
        self,
        msg_type: Type,
        topic: str,
        qos_profile: Union[QoSProfile, int] = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT
        ),
        callback_group: Optional[CallbackGroup] = MutuallyExclusiveCallbackGroup(),
        num_msgs: int = 1,
    ) -> None:
        """
        Adds a subscription to this node.

        Parameters
        ----------
        msg_type : Type
            The type of message to subscribe to.
        topic : str
            The name of the topic to subscribe to.
        qos_profile : Union[QoSProfile, int], optional
            The quality of service profile to use for the subscription, by default
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT).
        callback_group : Optional[CallbackGroup], optional
            The callback group to use for the subscription, by default
            MutuallyExclusiveCallbackGroup().
        num_msgs : int, optional
            The number of messages to store in the subscription, by default 1.
        """
        if topic in self.__msg_locks:
            with self.__msg_locks[topic]:
                if num_msgs > self.__latest_msgs[topic].maxlen:
                    # Grow the deque
                    self.__latest_msgs[topic] = deque(
                        self.__latest_msgs[topic], maxlen=num_msgs
                    )
            return
        self.__msg_locks[topic] = Lock()
        self.__latest_msgs[topic] = deque(maxlen=num_msgs)
        self.__subs[topic] = self.create_subscription(
            msg_type=msg_type,
            topic=topic,
            callback=partial(self.__callback, topic=topic),
            qos_profile=qos_profile,
            callback_group=callback_group,
        )

    def get_latest_msg(self, topic: str) -> Optional[Any]:
        """
        Returns the latest message from the specified topic.

        Parameters
        ----------
        topic : str
            The name of the topic to get the latest message from.

        Returns
        -------
        Optional[Any]
            The latest message from the specified topic, or None if no message has been
            received.
        """
        if topic not in self.__msg_locks:
            self.get_logger().error(f"Topic '{topic}' not found.")
            return None
        with self.__msg_locks[topic]:
            if len(self.__latest_msgs[topic]) == 0:
                self.get_logger().error(f"No message received from topic '{topic}'.")
                return None
            return self.__latest_msgs[topic][-1]

    def get_all_msgs(self, topic: str, copy: bool = True) -> Optional[deque[Any]]:
        """
        Returns all messages from the specified topic.

        Parameters
        ----------
        topic : str
            The name of the topic to get the messages from.
        copy : bool, optional
            Whether to return a copy of the messages, by default True.

        Returns
        -------
        Optional[deque[Any]]
            All messages from the specified topic, or None if no messages have been
            received.
        """
        if topic not in self.__msg_locks:
            self.get_logger().error(f"Topic '{topic}' not found.")
            return None
        with self.__msg_locks[topic]:
            if len(self.__latest_msgs[topic]) == 0:
                self.get_logger().error(f"No message received from topic '{topic}'.")
                return None
            if copy:
                return deque(
                    self.__latest_msgs[topic], maxlen=self.__latest_msgs[topic].maxlen
                )
            return self.__latest_msgs[topic]

    def __callback(self, msg: Any, topic: str) -> None:
        """
        Callback function for the subscription.

        Parameters
        ----------
        msg : Any
            The message received from the subscription.
        topic : str
            The name of the topic the message was received from.
        """
        with self.__msg_locks[topic]:
            self.__latest_msgs[topic].append(msg)


# pylint: disable=too-many-locals
def main(args=None):
    """
    Launch the ROS node and spin.
    """
    # Import the necessary modules
    # pylint: disable=import-outside-toplevel
    from ada_feeding_perception.face_detection import FaceDetectionNode
    from ada_feeding_perception.food_on_fork_detection import FoodOnForkDetectionNode
    from ada_feeding_perception.segment_from_point import SegmentFromPointNode
    from ada_feeding_perception.table_detection import TableDetectionNode

    rclpy.init(args=args)

    node = ADAFeedingPerceptionNode("ada_feeding_perception")
    face_detection = FaceDetectionNode(node)
    food_on_fork_detection = FoodOnForkDetectionNode(node)
    segment_from_point = SegmentFromPointNode(node)  # pylint: disable=unused-variable
    table_detection = TableDetectionNode(node)
    executor = MultiThreadedExecutor(num_threads=16)

    # Spin in the background initially
    spin_thread = Thread(
        target=rclpy.spin,
        args=(node,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run the perception nodes
    def face_detection_run():
        try:
            face_detection.run()
        except KeyboardInterrupt:
            pass

    def food_on_fork_detection_run():
        try:
            food_on_fork_detection.run()
        except KeyboardInterrupt:
            pass

    def table_detection_run():
        try:
            table_detection.run()
        except KeyboardInterrupt:
            pass

    face_detection_thread = Thread(target=face_detection_run, daemon=True)
    face_detection_thread.start()
    food_on_fork_detection_thread = Thread(
        target=food_on_fork_detection_run, daemon=True
    )
    food_on_fork_detection_thread.start()
    table_detection_thread = Thread(target=table_detection_run, daemon=True)
    table_detection_thread.start()

    # Spin in the foreground
    spin_thread.join()

    # Terminate this node
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
