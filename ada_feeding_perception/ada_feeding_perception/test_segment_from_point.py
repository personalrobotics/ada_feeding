#!/usr/bin/env python3
"""

"""

# Standard imports
from asyncio import Future
import threading

# Third-party imports
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Local Imports
from ada_feeding_msgs.action import SegmentFromPoint
from ada_feeding_perception.helpers import overlay_mask_on_image


class TestSegmentFromPoint(Node):
    """ """

    def __init__(self) -> None:
        """ """
        super().__init__("test_segment_from_point")

        # Read parameters
        self.read_params()

        # Store the future for the action server responses
        self._send_goal_future = Future()  # When the goal is accepted/rejected
        self._send_goal_future.set_result(None)
        self._get_result_future = Future()  # When we've gotten the result of the goal
        self._get_result_future.set_result(None)

        # Create the action client
        self._action_client = ActionClient(
            self, SegmentFromPoint, self.action_server_name.value
        )

        # Configure online mode
        if self.mode.value == "online":
            # Enable interactive mode
            plt.ion()

            # Create the figure to display the image and get coordinates
            self.fig, self.axs = plt.subplots(2, 3, sharex=True, sharey=True)
            self.axs[0, 0].set_title("Live video. Click to start.")
            self.axs[0, 1].set_title("Segmented Image")
            self.axs[0, 2].set_title("Contender Mask 1")
            self.axs[1, 0].set_title("Contender Mask 2")
            self.axs[1, 1].set_title("Contender Mask 3")
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            plt.show()

            # Convert between ROS and CV images
            self.bridge = CvBridge()

            # Whether we are waiting for a goal to return or not
            self.waiting_for_goal_lock = threading.Lock()
            self.waiting_for_goal = False

            # Store images while the action server is running
            self.stored_image_msgs_lock = threading.Lock()
            self.stored_image_msgs = []

            # Store the segmented image
            self.segmented_image_lock = threading.Lock()
            self.segmented_image = None

            # Store the segmentation_result
            self.segmentation_results_lock = threading.Lock()
            self.segmentation_results = None

            # Store the clicked point
            self.input_point_lock = threading.Lock()
            self.input_point = None

            # Subscribe to the image topic
            self.latest_img_msg_lock = threading.Lock()
            self.latest_img_msg = None
            self.create_subscription(
                Image,
                self.image_topic.value,
                self.image_callback,
                1,
            )

    def read_params(self) -> None:
        """ """
        # Get the required parameters, mode and action_server_name
        self.mode, self.action_server_name = self.declare_parameters(
            "",
            [
                (
                    "mode",
                    None,
                    ParameterDescriptor(
                        name="mode",
                        type=ParameterType.PARAMETER_STRING,
                        description="What mode to run the node in. Options are 'online' and 'offline'",
                        read_only=True,
                    ),
                ),
                (
                    "action_server_name",
                    None,
                    ParameterDescriptor(
                        name="action_server_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the action server to connect to.",
                        read_only=True,
                    ),
                ),
            ],
        )

        # Get the appropriate parameters for the mode
        if self.mode.value == "online":
            self.image_topic = self.declare_parameter(
                "online.image_topic",
                descriptor=ParameterDescriptor(
                    name="image_topic",
                    type=ParameterType.PARAMETER_STRING,
                    description="The topic to subscribe to for images.",
                    read_only=True,
                ),
            )
        elif self.mode.value == "offline":
            self.images, self.point_x, self.point_y = self.declare_parameters(
                "",
                [
                    (
                        "offline.images",
                        None,
                        ParameterDescriptor(
                            name="images",
                            type=ParameterType.PARAMETER_STRING_ARRAY,
                            description="The paths to the images to segment.",
                            read_only=True,
                        ),
                    ),
                    (
                        "offline.point_x",
                        None,
                        ParameterDescriptor(
                            name="point_x",
                            type=ParameterType.PARAMETER_INTEGER_ARRAY,
                            description="The x coordinates of the points to segment.",
                            read_only=True,
                        ),
                    ),
                    (
                        "offline.point_y",
                        None,
                        ParameterDescriptor(
                            name="point_y",
                            type=ParameterType.PARAMETER_INTEGER_ARRAY,
                            description="The y coordinates of the points to segment.",
                            read_only=True,
                        ),
                    ),
                ],
            )
        else:
            raise ValueError(
                "Invalid mode parameter. Must be either 'online' or 'offline'"
            )

    def image_callback(self, msg: Image) -> None:
        """ """
        # Store the latest image
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg

        # If we are waiting for a goal, accumulate the image in a list
        with self.waiting_for_goal_lock:
            waiting_for_goal = self.waiting_for_goal
        if waiting_for_goal:
            with self.stored_image_msgs_lock:
                self.stored_image_msgs.append(msg)

    def render_canvas(self) -> None:
        """ """
        r = self.create_rate(10)
        while rclpy.ok():
            # Display the live video feed
            img = None
            with self.latest_img_msg_lock:
                if self.latest_img_msg is not None:
                    # Convert the image to a CV image
                    img = self.bridge.imgmsg_to_cv2(
                        self.latest_img_msg, desired_encoding="bgr8"
                    )
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                # Display it
                self.axs[0, 0].imshow(img)

            # Display the image that was segmented
            with self.segmented_image_lock:
                # Display it
                if self.segmented_image is not None:
                    self.axs[0, 1].imshow(self.segmented_image)

            # Display the segmentation results
            with self.segmentation_results_lock:
                # Display it
                if self.segmentation_results is not None:
                    if len(self.segmentation_results) > 0:
                        self.axs[0, 2].imshow(self.segmentation_results[0])
                    if len(self.segmentation_results) > 1:
                        self.axs[1, 0].imshow(self.segmentation_results[1])
                    if len(self.segmentation_results) > 2:
                        self.axs[1, 1].imshow(self.segmentation_results[2])

            # Update the canvas
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Sleep
            r.sleep()

    def on_click(self, event: MouseEvent) -> None:
        """ """
        # Only register clicks on the first plot
        if event.inaxes != self.axs[0, 0]:
            return

        # Get the x, y of the click
        x, y = event.xdata, event.ydata
        self.get_logger().info(f"Clicked at {x}, {y}")
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg
            self.get_logger().info(
                "Latest image header: {}".format(self.latest_img_msg.header)
            )

        # Call the action server if we aren't waiting for the result of another call
        with self.waiting_for_goal_lock:
            was_waiting_for_goal = self.waiting_for_goal
            self.waiting_for_goal = True

        if not was_waiting_for_goal:
            # Clear the stored image cache
            with self.stored_image_msgs_lock:
                self.stored_image_msgs = [latest_img_msg]

            # Store the clicked point
            with self.input_point_lock:
                self.input_point = (x, y)

            # Create the goal message
            goal_msg = SegmentFromPoint.Goal()
            goal_msg.seed_point.header.stamp = self.get_clock().now().to_msg()
            goal_msg.seed_point.point.x = x
            goal_msg.seed_point.point.y = y

            # call the action server
            self._send_goal_future = self._action_client.send_goal_async(goal_msg)
            self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future) -> None:
        """ """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            with self.waiting_for_goal_lock:
                self.waiting_for_goal = False
            return
        self.get_logger().info("Goal accepted")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future: Future) -> None:
        """ """
        result = future.result().result
        self.get_logger().info("Result: {}".format(result))

        # Get the image that matches the result header
        self.get_logger().info("Result header: {}".format(result.header))
        segmented_image_msg = None
        with self.stored_image_msgs_lock:
            for image_msg in self.stored_image_msgs:
                self.get_logger().info("Image header: {}".format(image_msg.header))
                if image_msg.header.stamp == result.header.stamp:
                    segmented_image_msg = image_msg
                    break
        if segmented_image_msg is None:
            self.get_logger().error("Could not find image matching result header")
        else:
            # Convert the image to a CV image
            img = self.bridge.imgmsg_to_cv2(
                segmented_image_msg, desired_encoding="bgr8"
            )
            with self.segmented_image_lock:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Overlay a circle on the image
                with self.input_point_lock:
                    if self.input_point is not None:
                        x, y = self.input_point
                        img = cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)
                self.segmented_image = img

            # Store the segmentation result
            with self.segmentation_results_lock:
                self.segmentation_results = []
                for mask_raw in result.detected_items:
                    # Get the mask in the size of the image
                    mask_img = np.zeros(img.shape[:2], dtype=bool)
                    x, y, w, h = (
                        mask_raw.roi.x_offset,
                        mask_raw.roi.y_offset,
                        mask_raw.roi.width,
                        mask_raw.roi.height,
                    )
                    mask_img[y : y + h, x : x + w] = cv2.imdecode(
                        np.frombuffer(mask_raw.mask.data, np.uint8),
                        cv2.IMREAD_GRAYSCALE,
                    )

                    # Overlay the mask on the image
                    overlaid = overlay_mask_on_image(
                        img, mask_img, alpha=0.5, color=[0, 255, 0]
                    )
                    self.segmentation_results.append(overlaid)

        # We are no longer waiting for a goal
        with self.waiting_for_goal_lock:
            self.waiting_for_goal = False


def main(args=None):
    rclpy.init(args=args)

    test_segment_from_point = TestSegmentFromPoint()

    executor = MultiThreadedExecutor()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(test_segment_from_point, executor)
    )
    spin_thread.start()

    # The canvas must be rendered in the main thread in order to update matplotlib
    test_segment_from_point.render_canvas()


if __name__ == "__main__":
    main()
