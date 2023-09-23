#!/usr/bin/env python3
"""
This file defines the TestSegmentFromPoint class, which is a node that can be
used to test the SegmentFromPoint action server.
"""
# Standard imports
from asyncio import Future
import os
import threading
import time
from typing import List

# Third-party imports
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image

# Local Imports
from ada_feeding_msgs.action import SegmentFromPoint
from ada_feeding_perception.helpers import overlay_mask_on_image


class TestSegmentFromPoint(Node):
    """
    The TestSegmentFromPoint class is a node that tests the SegmentFromPoint
    action server. It has two modes: online and offline.
      - In online mode, it displays the live image feed, lets users select a
        point by clicking, calls the action server, and displays the resultant
        masks.
      - In offline mode, it calls the action server on a pre-specified set of
        images and points and saves the resultant masks.
    """

    # pylint: disable=too-many-instance-attributes
    # Although we have many more instance attributes (28) than recommended (7),
    # this class contains a lot of functionalities including online and offline
    # testing, which justifies the large number of attributes.

    def __init__(self) -> None:
        """
        Initializes the TestSegmentFromPoint node. This function reads the
        parameters, creates the action client, and configures the mode we are
        running it in (online or offline).
        """
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

        # Convert between ROS and CV images
        self.bridge = CvBridge()

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
                "~/image",
                self.image_callback,
                1,
            )
        else:  # Configure offline mode
            # Create a publisher to publish the images to
            self.image_pub = self.create_publisher(Image, "~/image", 1)

            # Make the directory to save the segmented images in
            os.makedirs(
                os.path.join(self.base_dir.value, self.save_dir.value), exist_ok=True
            )

    def read_params(self) -> None:
        """
        Reads the parameters for the node. See the individual descriptions for
        each parameter for more information.
        """
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
                        description=(
                            "What mode to run the node in. "
                            "Options are 'online' and 'offline'"
                        ),
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
        if self.mode.value == "offline":
            (
                self.base_dir,
                self.save_dir,
                self.sleep_time,
                self.images,
                self.point_xs,
                self.point_ys,
            ) = self.declare_parameters(
                "",
                [
                    (
                        "base_dir",
                        None,
                        ParameterDescriptor(
                            name="base_dir",
                            type=ParameterType.PARAMETER_STRING,
                            description=(
                                "The base directory that all paths "
                                "in offline mode are relative to."
                            ),
                            read_only=True,
                        ),
                    ),
                    (
                        "offline.save_dir",
                        None,
                        ParameterDescriptor(
                            name="save_dir",
                            type=ParameterType.PARAMETER_STRING,
                            description="Where to save the segmented images.",
                            read_only=True,
                        ),
                    ),
                    (
                        "offline.sleep_time",
                        None,
                        ParameterDescriptor(
                            name="sleep_time",
                            type=ParameterType.PARAMETER_DOUBLE,
                            description=(
                                "How long (secs) to sleep after publishing an image "
                                "before sending a goal to the action server."
                            ),
                            read_only=True,
                        ),
                    ),
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
                        "offline.point_xs",
                        None,
                        ParameterDescriptor(
                            name="point_xs",
                            type=ParameterType.PARAMETER_INTEGER_ARRAY,
                            description="The x coordinates of the points to segment.",
                            read_only=True,
                        ),
                    ),
                    (
                        "offline.point_ys",
                        None,
                        ParameterDescriptor(
                            name="point_ys",
                            type=ParameterType.PARAMETER_INTEGER_ARRAY,
                            description="The y coordinates of the points to segment.",
                            read_only=True,
                        ),
                    ),
                ],
            )
        elif self.mode.value != "online":
            raise ValueError(
                "Invalid mode parameter. Must be either 'online' or 'offline'"
            )

    def image_callback(self, msg: Image) -> None:
        """
        Stores the latest image and, if we have called the action server,
        accumulates the image in a list. This function is only used in online
        mode.
        """
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
        """
        Renders the canvas for the online mode. This function renders the live
        image feed, the segmented image, and the segmentation results.
        """
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
        """
        Handles a click on the live video stream image. This function calls the
        action server if there is not currently an active goal. Only used in
        online mode.

        Parameters
        ----------
        event: the mouse event that triggered this function
        """
        # Only register clicks on the first plot
        if event.inaxes != self.axs[0, 0]:
            return

        # Get the x, y of the click
        x, y = event.xdata, event.ydata
        self.get_logger().info(f"Clicked at {x}, {y}")
        with self.latest_img_msg_lock:
            latest_img_msg = self.latest_img_msg
            self.get_logger().info(f"Latest image header: {self.latest_img_msg.header}")

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
        """
        Handles the response from the action server when we send a goal. If the
        goal was accepted, it sets a callback function for the result. This
        function is only used in online mode.

        Parameters
        ----------
        future: the future that contains the response from the action server.
        """
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
        """
        Handles the result from the action server. This function processes and
        stores the response from the action server, to later be rendered on
        screen. This function is only used in online mode.

        Parameters
        ----------
        future: the future that contains the result from the action server.
        """
        result = future.result().result
        self.get_logger().info(f"Result: {result}")

        # Get the image that matches the result header
        self.get_logger().info(f"Result header: {result.header}")
        segmented_image_msg = None
        with self.stored_image_msgs_lock:
            for image_msg in self.stored_image_msgs:
                self.get_logger().info(f"Image header: {image_msg.header}")
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
                        img = self.overlay_point_on_image(x, y, img)
                self.segmented_image = img

            # Store the segmentation result
            with self.segmentation_results_lock:
                self.segmentation_results = self.get_overlaid_images(img, result)

        # We are no longer waiting for a goal
        with self.waiting_for_goal_lock:
            self.waiting_for_goal = False

    def overlay_point_on_image(self, x: int, y: int, img: np.ndarray) -> np.ndarray:
        """
        Overlays a point on an image.

        Parameters
        ----------
        x: the x-coordinate of the point
        y: the y-coordinate of the point
        img: the image to overlay the point on

        Returns
        -------
        overlaid_img: the image with the point overlaid on it
        """
        return cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), -1)

    def get_overlaid_images(
        self,
        img: np.ndarray,
        result: SegmentFromPoint.Result,
        render_confidence: bool = False,
    ) -> List[np.ndarray]:
        """
        Takes in an image and segmentation result and returns a list of images
        with the segmentation masks overlaid on them.

        Parameters
        ----------
        img: the image to overlay the masks on
        result: the segmentation result
        render_confidence: whether to render the confidence of the masks on the image

        Returns
        -------
        overlaid_images: a list of images with the masks overlaid on them
        """
        overlaid_images = []
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

            # Render the confidence of the mask
            if render_confidence:
                # y_offset is an estimate of how tall the text will be, since
                # the coordinates are for the *bottom*-left corner
                y_offset = 24
                cv2.putText(
                    overlaid,
                    f"Conf: {mask_raw.confidence:.3f}",
                    (x, y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            overlaid_images.append(overlaid)
        return overlaid_images

    def run_offline(self) -> None:
        """
        Runs the node in offline mode. In offline mode, the node calls the
        action server on a pre-specified set of images and points and saves the
        resultant masks.
        """
        images = self.images.value
        point_xs = self.point_xs.value
        point_ys = self.point_ys.value

        # For every image, call the action server
        for i in range(min((len(images), len(point_xs), len(point_ys)))):
            # Load the image and publish it
            image_path = os.path.join(self.base_dir.value, images[i])
            image = cv2.imread(image_path)
            self.get_logger().info(f"Loaded image {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.image_pub.publish(image_msg)
            # Sleep briefly so the action server has time to receive the image
            # before we send the goal
            time.sleep(self.sleep_time.value)

            # Create the goal message
            goal_msg = SegmentFromPoint.Goal()
            goal_msg.seed_point.header.stamp = self.get_clock().now().to_msg()
            goal_msg.seed_point.point.x = float(point_xs[i])
            goal_msg.seed_point.point.y = float(point_ys[i])

            # Call the action server
            self.get_logger().info("Waiting for action server")
            self._action_client.wait_for_server()
            self.get_logger().info(f"Calling action server with goal: {goal_msg}")
            result = self._action_client.send_goal(goal_msg)

            # Process the result
            overlaid_images = self.get_overlaid_images(
                self.overlay_point_on_image(point_xs[i], point_ys[i], image),
                result.result,
                render_confidence=True,
            )

            # Save them
            image_filename = os.path.splitext(os.path.split(image_path)[-1])[0]
            for j, overlaid_image in enumerate(overlaid_images):
                save_path = os.path.join(
                    self.base_dir.value,
                    self.save_dir.value,
                    f"image_{i}_{image_filename}_mask_{j}.png",
                )
                cv2.imwrite(save_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))

        self.get_logger().info("Done!")

    def run(self) -> None:
        """
        Runs the node, based on whether its in online or offline mode.
        """
        if self.mode.value == "online":
            # The canvas must be rendered in the main thread in order to update matplotlib
            self.render_canvas()
        else:
            self.run_offline()


def main(args=None):
    """
    Initializes the node and spins.
    """
    rclpy.init(args=args)

    # Create and spin the node
    test_segment_from_point = TestSegmentFromPoint()
    executor = MultiThreadedExecutor()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(test_segment_from_point, executor)
    )
    spin_thread.start()

    # Run the test
    test_segment_from_point.run()

    # Terminate when the spin thread terminates
    spin_thread.join()


if __name__ == "__main__":
    main()
