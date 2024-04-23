"""
This file defines the TableDetectionNode class, which publishes the 3D PoseStamped
location of the table with respect to camera_depth_optical_frame. 
"""

# Standard imports
import math
import threading
from typing import List, Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_srvs.srv import SetBool

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Point,
    Quaternion,
)

# Local imports
from ada_feeding_perception.helpers import (
    cv2_image_to_ros_msg,
    depth_img_to_pointcloud,
    get_img_msg_type,
    ros_msg_to_cv2_image,
)


class TableDetectionNode(Node):
    """
    This node subscribes to the camera info, depth image, and RGB image topics and exposes a
    service to toggle table detection on and off. When on, the node publishes
    a 3D PoseStamped location of the center of the camera image with respect to the camera's
    frame of perspective at a specified rate.
    This node relies on a few assumptions: (1) a plate is in the camera view, (2) the plate
    is circular and is the largest circle in the camera view, and (3) the area within the
    buffer radius of 50 pixels around the plate is mostly obstacle-free.
    """

    # pylint: disable=too-many-instance-attributes
    # Needed for multiple publishers/subscribers, services, and parameters
    def __init__(self):
        """
        Initialize the TableDetectionNode.
        """

        super().__init__("table_detection")

        # Load the parameters
        self.read_params()

        # Keeps track of whether table detection is on or not
        self.is_on = False
        self.is_on_lock = threading.Lock()

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Create the toggle service
        self.srv = self.create_service(
            SetBool,
            "~/toggle_table_detection",
            self.toggle_table_detection_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the publisher
        self.publisher = self.create_publisher(PoseStamped, "~/table_detection", 1)

        # Subscribe to the camera info topic, to get the camera intrinsics
        self.camera_info = None
        self.camera_info_lock = threading.Lock()
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the aligned depth image topic, to store the latest depth image
        self.latest_depth_img_msg = None
        self.latest_depth_img_msg_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(aligned_depth_topic, self)
        except ValueError as err:
            self.get_logger().warn(
                f"Error getting type of depth image topic. Defaulting to Image. {err}"
            )
            aligned_depth_type = CompressedImage
        self.depth_image_subscriber = self.create_subscription(
            aligned_depth_type,
            aligned_depth_topic,
            self.depth_image_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the RGB image topic, to store the latest image
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()
        image_topic = "~/image"
        try:
            image_type = get_img_msg_type(image_topic, self)
        except ValueError as err:
            self.get_logger().warn(
                f"Error getting type of image topic. Defaulting to CompressedImage. {err}"
            )
            image_type = CompressedImage
        self.image_subscriber = self.create_subscription(
            image_type,
            image_topic,
            self.image_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # If the visualization flag is set, create a publisher for the visualized
        # results of plate detection - a process required to fit a plane on the table
        if self.viz:
            self.viz_plate_pub = self.create_publisher(
                Image, "~/plate_detection_img", 1
            )

    def read_params(self) -> None:
        """
        Reads the parameters for the table detection node.

        Returns
        -------
        rate_hz: The rate (Hz) at which to publish the table pose information.
        viz: Whether to publish a visualization of the plate detection results as an RGB image.
        """
        # The rate (Hz) at which to publish the pose information of the detected table
        rate_hz = self.declare_parameter(
            "rate_hz",
            3.0,  # default value
            ParameterDescriptor(
                name="rate_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description=(
                    "The rate (Hz) at which to publish the pose "
                    "information of the detected table."
                ),
                read_only=True,
            ),
        )
        self.rate_hz = rate_hz.value

        # A boolean to determine whether to publish the visualization of the
        # plate detection results as an RGB image.
        viz = self.declare_parameter(
            "viz",
            False,  # default value
            ParameterDescriptor(
                name="viz",
                type=ParameterType.PARAMETER_BOOL,
                description=(
                    "Whether to publish a visualization of the plate "
                    "detection results as an RGB image."
                ),
                read_only=True,
            ),
        )
        self.viz = viz.value

    def toggle_table_detection_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """
        Callback function for the toggle_table_detection service. This function toggles the
        table detection publisher on or off based on the request data.

        Parameters
        ----------
        request: The given request message.
        response: The created response message.

        Returns
        ----------
        response: The updated response message based on the request.
        """
        self.get_logger().info(f"Incoming service request. data: {request.data}")

        response.success = False
        response.message = f"Failed to set is_on to {request.data}"
        with self.is_on_lock:
            self.is_on = request.data
            response.success = True
            response.message = f"Successfully set is_on to {request.data}"
        return response

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Store the latest camera info message.

        Parameters
        ----------
        msg: The camera info message.
        """
        with self.camera_info_lock:
            self.camera_info = msg

    def depth_image_callback(self, msg: Union[Image, CompressedImage]) -> None:
        """
        Store the latest depth image message.

        Parameters
        ----------
        msg: The depth image message.
        """
        with self.latest_depth_img_msg_lock:
            self.latest_depth_img_msg = msg

    def image_callback(self, msg: Union[Image, CompressedImage]) -> None:
        """
        Store the latest image message.

        Parameters
        ----------
        msg: The image message.
        """
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg

    def visualize_plate_detection(
        self, img_cv2: Image, center: List[float], radius: float, table_buffer: int
    ) -> None:
        """
        Visualizes the plate detection results from the Hough circle transform
        by drawing a circle around the detected plate and a concentric circle
        of larger radius, given the buffer value, in the given image.
        This visualization is then published as a ROS message.

        Parameters
        ----------
        img_cv2: The image to visualize the plate detection results on.
        center: The center coordinates of the detected plate.
        radius: The radius of the detected plate.
        table_buffer: The buffer value to add to the radius of the detected plate.
        """
        # Round radius and center coordinates
        plate_center = np.uint16(np.around(center))
        plate_r = np.uint16(np.round(radius))

        # Set circle center in visualization
        cv2.circle(img_cv2, plate_center, 1, (0, 100, 100), 3)

        # Draw circle outline (white) in visualization
        cv2.circle(img_cv2, plate_center, plate_r, (255, 0, 255), 3)
        cv2.circle(img_cv2, plate_center, plate_r + table_buffer, (255, 0, 255), 3)

        # Publish the image
        self.viz_plate_pub.publish(
            cv2_image_to_ros_msg(img_cv2, compress=False, bridge=self.bridge)
        )

    def fit_table(
        self,
        image_msg: Image,
        camera_info: CameraInfo,
        image_depth_msg: CompressedImage,
    ) -> Tuple[int, int, int]:
        """
        Fits a plane to the table based on the given RGB image and depth
        image messages. Returns the coefficients and constant terms of the
        table plane equation (z = a*x + b*y +c). Returns None if no plates
        are detected in the image.

        Parameters
        ----------
        image_msg: The RGB image to detect plates from.
        camera_info: The camera information.
        image_depth_msg: The depth image corresponding to the RGB image.

        Returns
        ----------
        a: The coefficient of the x term in the table plane equation.
        b: The coefficient of the y term in the table plane equation.
        c: The constant term in the table plane equation.
        """
        # pylint: disable=too-many-locals
        # All local variables used here are necessary

        # Hough Circle transform parameters
        # Tuning Hough circle transform parameters for plate detection:
        # https://medium.com/@isinsuarici/hough-circle-transform-parameter-tuning-with-examples-6b63478377c9
        hough_accum = (
            1.5  # Lowering causes false negatives/raising causes false positives
        )
        hough_min_dist = (
            100  # Minimum distance between the centers of circles to detect
        )
        hough_param1 = 100  # Larger is more selective
        hough_param2 = (
            125  # Larger is more selective and decreases chance of false positives
        )
        hough_min = 75  # Minimum radius of circles to detect
        hough_max = 200  # Maximum radius of circles to detect
        table_buffer = 50  # Extra radius around the plate to use for table detection

        # Convert ROS images to CV images
        image = ros_msg_to_cv2_image(image_msg, self.bridge)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_depth = ros_msg_to_cv2_image(image_depth_msg, self.bridge)

        # Detect all circles from the camera image
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            hough_accum,
            hough_min_dist,
            param1=hough_param1,
            param2=hough_param2,
            minRadius=hough_min,
            maxRadius=hough_max,
        )

        # If no circles are detected, return None, None, None
        if circles is None:
            return None, None, None

        # Determine the largest circle from the detected circles as the plate
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        for x, y, r in circles:
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r

        # If the visualization flag is set to True, publish a visualization of the
        # plate detection results
        if self.viz:
            self.visualize_plate_detection(gray, plate_uv, plate_r, table_buffer)

        # Create mask for depth image
        plate_mask = np.zeros(image_depth.shape)
        cv2.circle(plate_mask, plate_uv, plate_r + table_buffer, 1.0, -1)
        cv2.circle(plate_mask, plate_uv, plate_r, 0.0, -1)
        depth_masked = image_depth * (plate_mask).astype("uint16")

        # Noise removal
        kernel = np.ones((6, 6), np.uint8)
        depth_masked = cv2.morphologyEx(
            depth_masked, cv2.MORPH_OPEN, kernel, iterations=3
        )
        depth_masked = cv2.morphologyEx(
            depth_masked, cv2.MORPH_CLOSE, kernel, iterations=3
        )

        # Remove outliers using the interquartile range proximity rule
        quartile_3, quartile_1 = np.percentile(depth_masked[depth_masked > 0], [75, 25])
        depth_masked[depth_masked > quartile_3 + 1.5 * (quartile_3 - quartile_1)] = 0
        depth_masked[depth_masked < quartile_1 - 1.5 * (quartile_3 - quartile_1)] = 0

        # Deproject the depth array to get a pointcloud of 3D points from the table
        pointcloud = depth_img_to_pointcloud(
            depth_masked,
            0,
            0,
            f_x=camera_info.k[0],
            f_y=camera_info.k[4],
            c_x=camera_info.k[2],
            c_y=camera_info.k[5],
        )
        xy_coords = np.concatenate(
            (pointcloud[:, 0, np.newaxis], pointcloud[:, 1, np.newaxis]), axis=1
        )
        z_coords = pointcloud[:, 2]

        # Fit Plane: z = a*x + b*y + c
        coeffs = np.hstack((np.vstack(xy_coords), np.ones((len(xy_coords), 1))))
        a, b, c = np.linalg.lstsq(coeffs, z_coords, rcond=None)[0]

        return a, b, c

    def get_pose(
        self,
        a: float,
        b: float,
        c: float,
        camera_info: CameraInfo,
    ) -> Tuple[List[int], List[int]]:
        """
        Calculates the pose (position and orientation) of the table plane with
        respect to the camera's frame of perspective. The position is determined
        by deprojecting the center of the camera image to 3D points on the table.

        Parameters
        ----------
        a: The coefficient of the x term in the table plane equation.
        b: The coefficient of the y term in the table plane equation.
        c: The constant term in the table plane equation.
        camera_info: The camera information.

        Returns
        ----------
        center: The 3D coordinates of a point on the table
                deprojected from the center of the camera image.
        quat: A quaternion representing the orientation of the table.
        """
        # pylint: disable=too-many-locals
        # More local variables are used for improved readability

        # Store the camera intrinsics
        f_x = camera_info.k[0]  # Focal length of camera in x direction
        f_y = camera_info.k[4]  # Focal length of camera in y direction
        c_x = camera_info.k[2]  # x-coordinate of principal point
        c_y = camera_info.k[5]  # y-coordinate of principal point

        # Deproject the pixel coordinates of the center of the camera image
        # to a 3D point on the table
        max_u = 640  # Max u index in a 640 x 480 image
        max_v = 480  # Max v index in a 640 x 480 image
        center_x = ((max_u / 2) - c_x) / f_x
        center_y = ((max_v / 2) - c_y) / f_y
        center_z = c / (1 - a * center_x - b * center_y)
        center_x *= center_z
        center_y *= center_z
        center = [
            center_x,
            center_y,
            center_z,
        ]

        # Get the normalized direction vectors of the table plane
        # from the camera's frame of perspective
        # Direction vector of x calculated by the subtraction of the center
        # and a coordinate offset from the center in the -x direction
        x_offset = 0.1  # Offset in the -x direction
        offset_center = [
            center[0] - x_offset,
            center[1],
            a * (center[0] - x_offset) + b * center[1] + c,
        ]
        x_dir_vect = [
            center[0] - offset_center[0],
            center[1] - offset_center[1],
            center[2] - offset_center[2],
        ]
        x_dir_vect /= np.linalg.norm(x_dir_vect)

        # Direction vector of z calculated using table plane equation coefficients
        z_dir_vect = [a, b, -1] / np.linalg.norm([a, b, -1])

        # Direction vector of y calculated by finding the vector orthogonal to both
        # the x and z direction vectors
        # Accomplished through a system of linear equations consisting of:
        # the dot product of x and y, dot product of z and y, & normalization of y
        denominator = math.sqrt(
            math.pow(x_dir_vect[2] * z_dir_vect[1] - x_dir_vect[1] * z_dir_vect[2], 2)
            + math.pow(x_dir_vect[0] * z_dir_vect[2] - x_dir_vect[2] * z_dir_vect[0], 2)
            + math.pow(x_dir_vect[1] * z_dir_vect[0] - x_dir_vect[0] * z_dir_vect[1], 2)
        )
        y_dir_vect = [
            (x_dir_vect[2] * z_dir_vect[1] - x_dir_vect[1] * z_dir_vect[2])
            / denominator,
            (x_dir_vect[0] * z_dir_vect[2] - x_dir_vect[2] * z_dir_vect[0])
            / denominator,
            (x_dir_vect[1] * z_dir_vect[0] - x_dir_vect[0] * z_dir_vect[1])
            / denominator,
        ]

        # Construct the rotation matrix from direction vectors
        rot_matrix = [
            [x_dir_vect[0], y_dir_vect[0], z_dir_vect[0]],
            [x_dir_vect[1], y_dir_vect[1], z_dir_vect[1]],
            [x_dir_vect[2], y_dir_vect[2], z_dir_vect[2]],
        ]

        # Derive a quaternion from the rotation matrix
        quat = R.from_matrix(rot_matrix)
        quat = quat.as_quat()

        return center, quat

    def run(self) -> None:
        """
        Run table detection at a specified fixed rate. This function gets the latest RGB
        image and depth image messages, fits a plane to the table, and calculates the
        center and orientation of the table in the camera frame. Then, with the calculated
        center and orientation, it publishes a PoseStamped message. The PoseStamped
        messages will continue to get published at a specified fixed rate as table detection
        continues to run.
        """
        # Create a fixed rate to run the loop on
        rate = self.create_rate(self.rate_hz)

        # Run while the context is not shut down
        while rclpy.ok():
            # Loop at the specified rate
            rate.sleep()

            # Check if table detection is on
            # If not, reiterate
            with self.is_on_lock:
                is_on = self.is_on
            if not is_on:
                continue

            # Get the latest RGB image message
            rgb_msg = None
            with self.latest_img_msg_lock:
                rgb_msg = self.latest_img_msg

            # Get the latest depth image
            depth_img_msg = None
            with self.latest_depth_img_msg_lock:
                depth_img_msg = self.latest_depth_img_msg

            # Get the camera information
            cam_info = None
            with self.camera_info_lock:
                if self.camera_info is not None:
                    cam_info = self.camera_info
                else:
                    self.get_logger().warn(
                        "Camera info not received, not including in result message"
                    )

            # Fit a plane to the table given the camera image and get the coefficients
            # and constant terms of the table plane equation
            a, b, c = self.fit_table(rgb_msg, cam_info, depth_img_msg)

            # If no plate is detected, warn and reiterate
            if a is None or b is None or c is None:
                self.get_logger().warn(
                    "Plate not detected, returning to default table pose."
                )
                continue

            # Get the center and quaternion of the table in the camera frame
            center, quat = self.get_pose(a, b, c, cam_info)

            # Create PoseStamped message
            fit_table_msg = PoseStamped(
                header=depth_img_msg.header,
                pose=Pose(
                    position=Point(
                        x=center[0],
                        y=center[1],
                        z=center[2],
                    ),
                    orientation=Quaternion(
                        x=quat[0],
                        y=quat[1],
                        z=quat[2],
                        w=quat[3],
                    ),
                ),
            )

            # Publish the PoseStamped message
            self.publisher.publish(fit_table_msg)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    table_detection = TableDetectionNode()
    executor = MultiThreadedExecutor(num_threads=2)

    # Spin in the background since detecting the table will block
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(table_detection,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run table detection
    try:
        table_detection.run()
    except KeyboardInterrupt:
        pass

    # Terminate this node and safely shut down
    table_detection.destroy_node()
    rclpy.shutdown()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
