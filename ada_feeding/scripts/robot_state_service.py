#!/usr/bin/env python3
"""
A node that subscribes to the joint states, keeps track of the most up-to-date
state for each joint, and exposes a service of type GetRobotState.srv that
returns the most up-to-date state for requested joints.
"""

# Standard imports
from collections import namedtuple

# Third-party imports
from geometry_msgs.msg import Point, Pose, Quaternion
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_geometry_msgs import PoseStamped
import tf2_py as tf2
import tf2_ros

# Local imports
from ada_feeding_msgs.srv import GetRobotState

# Define a namedtuple to store latest the joint state
SingleJointState = namedtuple(
    "SingleJointState", ["header", "position", "velocity", "effort"]
)


class RobotStateService(Node):
    """
    The RobotStateService class subscribes to the joint states, keeps track of
    the most up-to-date state for each joint, and exposes a service that returns
    the most up-to-date state for requested joints.
    """

    def __init__(self) -> None:
        """
        Initialize the RobotStateService node.
        """
        super().__init__("robot_state_service")

        # Read the parameters
        self.read_params()

        # Initialize the latest joint states
        self.latest_joint_states = {}

        # Create the TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
            GetRobotState,
            "get_robot_state",
            self.get_robot_state_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def read_params(self) -> None:
        """
        Read parameters from the ROS parameter server.
        """
        # pylint: disable=attribute-defined-outside-init
        timeout_secs = self.declare_parameter(
            "timeout_secs",
            0.1,
            descriptor=ParameterDescriptor(
                name="timeout",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The timeout in seconds for waiting for a transform from the TF tree. "
                    "If the timeout is exceeded, the transform will not be returned; "
                    "the PoseStamped at that index have no frame_id."
                ),
                read_only=True,
            ),
        )
        self.timeout_secs = Duration(seconds=timeout_secs.value)

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Callback function for the joint states subscriber. Updates the latest
        joint states.

        Parameters
        ----------
        msg : RobotState
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

    def get_robot_state_callback(self, request, response) -> GetRobotState.Response:
        """
        Callback function for the get_robot_state service. Returns the most
        up-to-date state for requested joints and/or the poses for the requested
        frames.

        Parameters
        ----------
        request : GetRobotState.Request
            The request object.
        response : GetRobotState.Response
            The response object.

        Returns
        -------
        GetRobotState.Response
            The response object.
        """
        self.get_logger().info(f"Received request for robot state(s) {request}")
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

        # Get the most up-to-date pose for each requested frame
        zero_pose_stamped = PoseStamped(
            pose=Pose(
                position=Point(x=0.0, y=0.0, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
        for i in range(min(len(request.child_frames), len(request.parent_frames))):
            # Compute the transform from the TF tree
            zero_pose_stamped.header.frame_id = request.child_frames[i]
            try:
                transformed_pose_stamped = self.tf_buffer.transform(
                    zero_pose_stamped,
                    request.parent_frames[i],
                    self.timeout_secs,
                )
            except (
                tf2.ConnectivityException,
                tf2.ExtrapolationException,
                tf2.InvalidArgumentException,
                tf2.LookupException,
                tf2.TimeoutException,
                tf2.TransformException,
                tf2_ros.TypeException,
            ) as error:
                self.get_logger().error(
                    f"Could not get pose of {request.child_frames[i]} in {request.parent_frames[i]}. "
                    f"Error: {error}"
                )
                transformed_pose_stamped = PoseStamped()
            response.poses.append(transformed_pose_stamped)

        self.get_logger().info("Sending response for joint states")
        return response


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    # Initialize the ROS context
    rclpy.init(args=args)

    # Create the RobotStateService node
    robot_state_service = RobotStateService()

    # Use a MultiThreadedExecutor to make the subscriber and service concurrent
    executor = MultiThreadedExecutor()

    # Spin the node
    rclpy.spin(robot_state_service, executor=executor)

    # Destroy the node
    robot_state_service.destroy_node()

    # Shutdown the ROS context
    rclpy.shutdown()


if __name__ == "__main__":
    main()
