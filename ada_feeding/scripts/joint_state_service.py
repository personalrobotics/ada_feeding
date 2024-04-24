#!/usr/bin/env python3
"""
A node that subscribes to the joint states, keeps track of the most up-to-date
state for each joint, and exposes a service of type GetJointState.srv that
returns the most up-to-date state for requested joints.
"""

# Standard imports
from collections import namedtuple

# Third-party imports
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState

# Local imports
from ada_feeding_msgs.srv import GetJointState

# Define a namedtuple to store latest the joint state
SingleJointState = namedtuple(
    "SingleJointState", ["header", "position", "velocity", "effort"]
)


class JointStateService(Node):
    """
    The JointStateService class subscribes to the joint states, keeps track of
    the most up-to-date state for each joint, and exposes a service that returns
    the most up-to-date state for requested joints.
    """

    def __init__(self) -> None:
        """
        Initialize the JointStateService node.
        """
        super().__init__("joint_state_service")

        # Initialize the latest joint states
        self.latest_joint_states = {}

        # Create a subscriber to the joint states topic
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_states_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create a service that returns the most up-to-date state for requested joints
        self.create_service(
            GetJointState,
            "get_joint_state",
            self.get_joint_state_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states subscriber. Updates the latest
        joint states.

        Parameters
        ----------
        msg : JointState
            The joint states message.
        """
        for name, position, velocity, effort in zip(
            msg.name, msg.position, msg.velocity, msg.effort
        ):
            self.latest_joint_states[name] = SingleJointState(
                header=msg.header,
                position=position,
                velocity=velocity,
                effort=effort,
            )

    def get_joint_state_callback(self, request, response) -> GetJointState.Response:
        """
        Callback function for the get_joint_state service. Returns the most
        up-to-date state for requested joints.

        Parameters
        ----------
        request : GetJointState.Request
            The request object.
        response : GetJointState.Response
            The response object.

        Returns
        -------
        GetJointState.Response
            The response object.
        """
        self.get_logger().info(
            f"Received request for joint states {self.latest_joint_states}"
        )
        # Initialize the response
        response.joint_state.name = []
        response.joint_state.position = []
        response.joint_state.velocity = []
        response.joint_state.effort = []

        # Get the most up-to-date state for each requested joint
        oldest_header = None
        for joint_name in request.joint_names:
            if joint_name in self.latest_joint_states:
                single_joint_state = self.latest_joint_states[joint_name]
                response.joint_state.name.append(joint_name)
                response.joint_state.position.append(single_joint_state.position)
                response.joint_state.velocity.append(single_joint_state.velocity)
                response.joint_state.effort.append(single_joint_state.effort)
                if oldest_header is None or Time.from_msg(
                    single_joint_state.header.stamp
                ) < Time.from_msg(oldest_header.stamp):
                    oldest_header = single_joint_state.header
        if oldest_header is not None:
            response.joint_state.header = oldest_header
        self.get_logger().info("Sending response for joint states")
        return response


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    # Initialize the ROS context
    rclpy.init(args=args)

    # Create the JointStateService node
    joint_state_service = JointStateService()

    # Use a MultiThreadedExecutor to make the subscriber and service concurrent
    executor = MultiThreadedExecutor()

    # Spin the node
    rclpy.spin(joint_state_service, executor=executor)

    # Destroy the node
    joint_state_service.destroy_node()

    # Shutdown the ROS context
    rclpy.shutdown()


if __name__ == "__main__":
    main()
