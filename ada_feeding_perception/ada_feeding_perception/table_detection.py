# Standard imports
import os
import threading
from typing import Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
import torch
from std_srvs.srv import Trigger
import math

from tf_transformations import quaternion_from_matrix
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

# Local imports
from ada_feeding_msgs.msg import Mask
from ada_feeding_msgs.srv import GetPoseStamped
from ada_feeding_perception.helpers import (
    bbox_from_mask,
    crop_image_mask_and_point,
    cv2_image_to_ros_msg,
    download_checkpoint,
    get_connected_component,
    get_img_msg_type,
    ros_msg_to_cv2_image,
)


class TableDetectionNode(Node):
    """
    This node fits a table to the received depth image and publishes a 3d PoseStamped location
    of a specific point on the table with respect to the camera frame.
    """

    def __init__(self, pub_result=False):
        """
        Initialize the TableDetectionNode.
        """

        super().__init__("table_detection")
        self.get_logger().info("Entering table detection node!")

        # Table Plane Params
        self._hough_accum = 1.5
        self._hough_min_dist = 100
        self._hough_param1 = 100  # Larger is more selective
        self._hough_param2 = (
            125  # Larger is more selective/decreases chance of false positives
        )
        self._hough_min = 75
        self._hough_max = 200
        self._table_buffer = 50  # Extra radius around plate to use for table

        # Depth Image Deprojection Constants
        self.crop_x_min = 297
        self.crop_y_min = 248
        self.crop_x_max = 425
        self.crop_y_max = 332

        self.pub_result = pub_result

        self.dir_constant = 0.1

        if self.pub_result:
            self.publisher = self.create_publisher(
                PoseStamped, "table_detection", 1
            )

        # Create the service
        self.srv = self.create_service(
            GetPoseStamped,
            "fit_to_table",
            self.fit_to_table_callback,
        )

        self.camera_info = None
        # Subscribe to the camera info topic, to get the camera intrinsics
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
        Parameters
        ----------
        Find table plane.
        image(image matrix): RGB image of plate
        image_depth(image matrix): depth image of plate
        target_u: u coordinate to obtain table depth
        target_v: v coordinate to obtain table depth

        Returns
        ----------
        table: depth array of the table
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
        print(len(circles))
        for x, y, r in circles:
            print("Radius: " + str(r))
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
        # coefficients b and a are reversed because of matrix row/col structure and its correspondence to x/y
        b, a, c = np.linalg.lstsq(coeffs, d, rcond=None)[
            0
        ]  
        self.get_logger().info(f"d_idx, {d_idx}")
        self.get_logger().info(f"d, {d}")
        self.get_logger().info(f"coeffs, {coeffs}, {coeffs.shape}")

        # TODO: add a comment describing flipped coefficients a & b

        # Create Table Depth Image
        u = np.linspace(0, depth_mask.shape[1], depth_mask.shape[1], False)
        v = np.linspace(0, depth_mask.shape[0], depth_mask.shape[0], False)
        U, V = np.meshgrid(u, v)
        table = a * U + b * V + c
        table = table.astype("uint16")

        return table

    def deproject_depth_image(self, depth_img: Image, camera_info: CameraInfo):
        # For every non-zero depth point, calculate the corresponding 3D point
        # in the camera frame
        u = np.tile(
            np.arange(depth_img.shape[1]), (depth_img.shape[0], 1)
        )  # + self.crop_x_min
        self.get_logger().info(f"u, {u}, {u.shape}, {u.min()}, {u.max()}")

        v = np.tile(
            np.arange(depth_img.shape[0]), (depth_img.shape[1], 1)
        ).T  # + self.crop_y_min
        self.get_logger().info(f"v, {v}, {v.shape}, {v.min()}, {v.max()}")
        
        # Mask out any points where depth is 0
        mask = depth_img > 0
        depth_img_masked = depth_img[mask]
        u_masked = u[mask]
        v_masked = v[mask]
        self.get_logger().info(f"u_masked, {u_masked}, {u_masked.shape}")
        self.get_logger().info(f"v_masked, {v_masked}, {v_masked.shape}")

        # Deproject x, y, and z coordinates from u, v coordinates and depth image
        x = np.multiply(
            u_masked - camera_info.k[2],
            np.divide(depth_img_masked, 1000.0 * camera_info.k[0]),
        )
        self.get_logger().info(f"x, {x}, {x.shape}")
        y = np.multiply(
            v_masked - camera_info.k[5],
            np.divide(depth_img_masked, 1000.0 * camera_info.k[4]),
        )
        self.get_logger().info(f"y, {y}, {y.shape}")
        z = np.divide(depth_img_masked, 1000.0)

        # Concatenate the x and y coordinates together
        xy = np.concatenate((x[..., np.newaxis], y[..., np.newaxis]), axis=1)

        # Calculate center
        center = [x[640 * 239 + 320 - 1], y[640 * 239 + 320 - 1], z[640 * 239 + 320 - 1]]
        
        return np.array(xy), np.array(z), center

    def get_orientation(
        self,
        camera_info: CameraInfo,
        table_depth: Image,
        target_u: int,
        target_v: int,
    ):
        # Deproject all pixels from table depth image to 3D coordinates
        # table_deprojected = np.zeros((table_depth.shape[1], table_depth.shape[0], 3))
        # xy_d = []
        # depth_d = []
        # self.get_logger().info(f"table_depth shape, {table_depth.shape}")
        # for v in range(0, len(table_depth)):
        #     for u in range(0, len(table_depth[v])):
        #         table_deprojected[u][v] = pyrealsense2.rs2_deproject_pixel_to_point(
        #             intrinsics, [u, v], table_depth[v][u] / 1000
        #         )
        #         xy_d.append([table_deprojected[u][v][0], table_deprojected[u][v][1]])
        #         depth_d.append(table_deprojected[u][v][2])
        xy_d, depth_d, center = self.deproject_depth_image(table_depth, camera_info)
        self.get_logger().info(f"xy_d, {np.vstack(xy_d)}")
        self.get_logger().info(f"xy_d shape, {xy_d.shape[0]} x {xy_d.shape[1]}")
        self.get_logger().info(f"depth_d, {np.vstack(depth_d)}")
        self.get_logger().info(f"depth_d shape, {depth_d.shape}")
        self.get_logger().info(f"table center, {center}")

        # Fit Plane: z = a*x + b*y + c
        coeffs = np.hstack((np.vstack(xy_d), np.ones((len(xy_d), 1)))).astype(float)
        self.get_logger().info(f"get orientation coeffs, {coeffs}")
        a, b, c = np.linalg.lstsq(coeffs, depth_d, rcond=None)[0]
        center[2] = (
            a * center[0] + b * center[1] + c
        )  # Modify the z coordinate of the center given the fitted plane (i.e. make the center the origin of the table plane)
        self.get_logger().info(f"coefficients: a = {a}, b = {b}, c = {c}")
        self.get_logger().info(
            f"{center}, {(center[0], center[1], a*center[0]+ b*center[1] + c)}"
        )

        # Get the direction vectors of the table plane from the camera's frame of perspective
        x_p = [
            center[0] - self.dir_constant,
            center[1],
            a * (center[0] - self.dir_constant) + b * center[1] + c,
        ]
        x_ref = [center[0] - x_p[0], center[1] - x_p[1], center[2] - x_p[2]]
        x_ref /= np.linalg.norm(x_ref)
        self.get_logger().info(
            f"magnitude of normalized +x vector = {np.linalg.norm(x_ref)}"
        )
        
        z_ref = [a, b, -1] / np.linalg.norm([a, b, -1])
        self.get_logger().info(
            f"magnitude of normalized +z vector = {np.linalg.norm(z_ref)}"
        )

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
        self.get_logger().info(
            f"magnitude of normalized +y vector = {np.linalg.norm(y_ref)}"
        )

        # Original direction vector of y calculation: 
        # y_p = [center[0], center[1] - 0.1, a*center[0] + b*(center[1] - 0.1) + c]
        # y_ref = [center[0] - y_p[0], center[1] - y_p[1], center[2] - y_p[2]]
        # y_ref /= np.linalg.norm(y_ref)

        self.get_logger().info(
            f"dot product of +x and +y vector = {np.dot(y_ref, x_ref)}"
        )
        self.get_logger().info(
            f"dot product of +x and +z vector = {np.dot(x_ref, z_ref)}"
        )
        self.get_logger().info(
            f"dot product of +y and +z vector = {np.dot(y_ref, z_ref)}"
        )
        self.get_logger().info(
            f"x_ref, y_ref, z_ref coordinates: x_ref = {x_ref}, y_ref = {y_ref}, z_ref = {z_ref}"
        )

        # Construct rotation matrix from direction vectors
        rot_matrix = [
            [x_ref[0], y_ref[0], z_ref[0]],
            [x_ref[1], y_ref[1], z_ref[1]],
            [x_ref[2], y_ref[2], z_ref[2]],
        ]
        self.get_logger().info(f"rotation matrix = {R}")

        # Derive quaternion from rotation matrix
        q = R.from_matrix(rot_matrix)
        q = q.as_quat()
        # q = quaternion_from_matrix(R)
        self.get_logger().info(f"quaternion = {q}")

        return center, q

    def fit_to_table_callback(
        self, request: GetPoseStamped.Request, response: GetPoseStamped.Response
    ):
        self.get_logger().info("Entering fit_table callback!")
        # Get the latest RGB image message
        rgb_msg = None
        with self.latest_img_msg_lock:
            rgb_msg = self.latest_img_msg

        # Get the latest depth image
        depth_img_msg = None
        with self.latest_depth_img_msg_lock:
            depth_img_msg = self.latest_depth_img_msg

        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Get table depth from camera at the (u, v) coordinate (320, 240)
        table_depth = self.fit_table(rgb_msg, depth_img_msg)

        self.get_logger().info(f"table depth, {table_depth}, {type(table_depth)}")

        # Get the camera matrix
        cam_info = None
        with self.camera_info_lock:
            if self.camera_info is not None:
                cam_info = self.camera_info
            else:
                self.get_logger().warn(
                    "Camera info not received, not including in result message"
                )

        center, orientation = self.get_orientation(cam_info, table_depth, 320, 240)

        self.get_logger().info(f"orientation, {orientation}, {type(orientation)}")

        # To be replaced with PoseStamped message
        # response.success = True
        # response.message = f"To be replaced with PoseStamped message"

        response.pose_stamped = PoseStamped(
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

        if self.pub_result:
            self.publisher.publish(response.pose_stamped)

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


def main(args=None):
    """
    Launch the ROS node and spin.
    """

    rclpy.init()

    table_detection = TableDetectionNode(pub_result=True)

    rclpy.spin(table_detection)


if __name__ == "__main__":
    main()
