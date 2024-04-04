"""
This file defines the TableDetectionNode class, which publishes the 3D PoseStamped
location of the table with respect to camera_depth_optical_frame.
"""

# Standard imports
import math
import threading
from typing import Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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
    """

    def __init__(self):
        """
        Initialize the TableDetectionNode.
        """

        super().__init__("table_detection")

        # Table plane parameters
        self._hough_accum = 1.5
        self._hough_min_dist = 100
        self._hough_param1 = 100  # Larger is more selective
        self._hough_param2 = (
            125  # Larger is more selective/decreases chance of false positives
        )
        self._hough_min = 75
        self._hough_max = 200
        self._table_buffer = 50  # Extra radius around plate to use for table detection

        # Orientation calculation parameters
        self.deproject_center_coord = 640 * 239 + 320 - 1
        self.dir_constant = 0.1

        # Rate at which table info gets published
        self.update_hz = 3

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
            aligned_depth_type = Image
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

    def fit_table(
        self,
        image_msg: Image,
        image_depth_msg: Image,
    ):
        """
        Fits a plane to the table based on the given RGB image and depth image messages.

        Parameters
        ----------
        image_msg: The RGB image to detect plates from.
        image_depth_msg: The depth image corresponding to the RGB image.

        Returns
        ----------
        table: The depth array of the table.
        """
        # Convert ROS images to CV images
        image = ros_msg_to_cv2_image(image_msg, self.bridge)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_depth = ros_msg_to_cv2_image(image_depth_msg, self.bridge)

        # Detect Largest Circle (Plate)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            self._hough_accum,
            self._hough_min_dist,
            param1=self._hough_param1,
            param2=self._hough_param2,
            minRadius=self._hough_min,
            maxRadius=self._hough_max,
        )
        if circles is None:
            return None, None, None
        circles = np.round(circles[0, :]).astype("int")
        plate_uv = (0, 0)
        plate_r = 0
        for x, y, r in circles:
            if r > plate_r:
                plate_uv = (x, y)
                plate_r = r

        # Testing - Draw out the circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles:
                center = (x, y)
                # circle center
                cv2.circle(gray, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = r
                cv2.circle(gray, center, radius, (255, 0, 255), 3)
                cv2.circle(gray, center, radius + self._table_buffer, (255, 0, 255), 3)
            # The following lines of code are commented to prevent interruption of the program
            # while running. Uncomment to view images of the detected circles.
            # cv2.imshow("detected circles", gray)
            # cv2.waitKey(0)

        # Create Mask for Depth Image
        plate_mask = np.zeros(image_depth.shape)
        cv2.circle(plate_mask, plate_uv, plate_r + self._table_buffer, 1.0, -1)
        cv2.circle(plate_mask, plate_uv, plate_r, 0.0, -1)
        depth_mask = (image_depth * (plate_mask).astype("uint16")).astype(float)

        # Noise removal
        kernel = np.ones((6, 6), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # Remove Outliers
        # TODO: Consider outlier removal using median + iqr for the future
        depth_var = np.abs(depth_mask - np.mean(depth_mask[depth_mask > 0]))
        depth_std = np.std(depth_mask[depth_mask > 0])
        depth_mask[depth_var > 2.0 * depth_std] = 0

        # Fit Plane: depth = a*u + b*v + c
        d_idx = np.where(depth_mask > 0)
        d = depth_mask[d_idx].astype(float)
        coeffs = np.hstack((np.vstack(d_idx).T, np.ones((len(d_idx[0]), 1)))).astype(
            float
        )

        # Coefficients b and a are reversed because of matrix row/col structure and its
        # correspondence to x/y
        b, a, c = np.linalg.lstsq(coeffs, d, rcond=None)[0]

        # Create Table Depth Image
        u = np.linspace(0, depth_mask.shape[1], depth_mask.shape[1], False)
        v = np.linspace(0, depth_mask.shape[0], depth_mask.shape[0], False)
        u_grid, v_grid = np.meshgrid(u, v)
        table = a * u_grid + b * v_grid + c
        table = table.astype("uint16")

        return table

    def get_orientation(
        self,
        camera_info: CameraInfo,
        table_depth: Image,
    ):
        """
        Calculate the orientation of the table plane with respect to the
        camera's frame of perspective.

        Parameters
        ----------
        camera_info: The camera information.
        table_depth: The depth image of the table.

        Returns
        ----------
        center: The center coordinates of the table in the camera frame.
        q: The quaternion representing the orientation of the table plane.
        """
        # Deproject the depth image to get the 3D points in the camera frame.
        pointcloud = depth_img_to_pointcloud(
            table_depth,
            0,
            0,
            f_x=camera_info.k[0],
            f_y=camera_info.k[4],
            c_x=camera_info.k[2],
            c_y=camera_info.k[5],
        )
        xy_d = np.concatenate(
            (pointcloud[:, 0, np.newaxis], pointcloud[:, 1, np.newaxis]), axis=1
        )
        depth_d = pointcloud[:, 2]
        center = [
            pointcloud[self.deproject_center_coord][0],
            pointcloud[self.deproject_center_coord][1],
            pointcloud[self.deproject_center_coord][2],
        ]

        # Fit Plane: z = a*x + b*y + c
        coeffs = np.hstack((np.vstack(xy_d), np.ones((len(xy_d), 1)))).astype(float)
        a, b, c = np.linalg.lstsq(coeffs, depth_d, rcond=None)[0]

        # Modify the z coordinate of the center given the fitted plane (i.e. make the center
        # the origin of the table plane)
        center[2] = a * center[0] + b * center[1] + c

        # Get the direction vectors of the table plane from the camera's frame of perspective
        x_p = [
            center[0] - self.dir_constant,
            center[1],
            a * (center[0] - self.dir_constant) + b * center[1] + c,
        ]
        x_ref = [center[0] - x_p[0], center[1] - x_p[1], center[2] - x_p[2]]
        x_ref /= np.linalg.norm(x_ref)

        z_ref = [a, b, -1] / np.linalg.norm([a, b, -1])

        # Direction vector of y calculated through a system of linear equations consisting of:
        # the dot product of x and y, dot product of z and y, & norm of y = 1
        denom = math.sqrt(
            math.pow(x_ref[2] * z_ref[1] - x_ref[1] * z_ref[2], 2)
            + math.pow(x_ref[0] * z_ref[2] - x_ref[2] * z_ref[0], 2)
            + math.pow(x_ref[1] * z_ref[0] - x_ref[0] * z_ref[1], 2)
        )
        y_ref = [
            (x_ref[2] * z_ref[1] - x_ref[1] * z_ref[2]) / denom,
            (x_ref[0] * z_ref[2] - x_ref[2] * z_ref[0]) / denom,
            (x_ref[1] * z_ref[0] - x_ref[0] * z_ref[1]) / denom,
        ]

        # Construct rotation matrix from direction vectors
        # TODO: Test tf_transformations quaternion_from_matrix function
        rot_matrix = [
            [x_ref[0], y_ref[0], z_ref[0]],
            [x_ref[1], y_ref[1], z_ref[1]],
            [x_ref[2], y_ref[2], z_ref[2]],
        ]

        # Derive quaternion from rotation matrix
        q = R.from_matrix(rot_matrix)
        q = q.as_quat()

        return center, q

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
        # TODO: Replace the rate with correct fixed value or create paremeter for rate
        rate = self.create_rate(self.update_hz)

        # Run while the context is not shut down
        while rclpy.ok():
            # Loop at the specified rate
            rate.sleep()

            # Check if table detection is on
            # if not, reiterate
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

            # Get table depth from camera at the (u, v) coordinate (320, 240)
            table_depth = self.fit_table(rgb_msg, depth_img_msg)

            # Get the camera information
            cam_info = None
            with self.camera_info_lock:
                if self.camera_info is not None:
                    cam_info = self.camera_info
                else:
                    self.get_logger().warn(
                        "Camera info not received, not including in result message"
                    )

            # Get the center and orientation of the table in the camera frame
            center, orientation = self.get_orientation(cam_info, table_depth)

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
                        x=orientation[0],
                        y=orientation[1],
                        z=orientation[2],
                        w=orientation[3],
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

    # Terminate this node and safely shut down - new code
    table_detection.destroy_node()
    rclpy.shutdown()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
