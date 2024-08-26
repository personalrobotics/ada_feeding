#!/usr/bin/env python3

"""
This file defines the FaceDetection class, which publishes the 3d PointStamped locations
of the largest detected mouth with respect to camera_depth_optical_frame.
"""

# Standard Imports
import collections
from enum import Enum
import os
import threading
from typing import List, Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from skspatial.objects import Plane, Points
from std_srvs.srv import SetBool


# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding_perception.helpers import (
    download_checkpoint,
    get_img_msg_type,
    cv2_image_to_ros_msg,
    ros_msg_to_cv2_image,
)


class DepthComputationMethod(Enum):
    """
    Enum for the different methods of computing the depth of the mouth.
    """

    MEDIAN_MOUTH_POINTS = "median_mouth_points"
    FACIAL_PLANE = "facial_plane"


class FaceDetectionNode(Node):
    """
    This node publishes a 3d PointStamped location
    of the largest detected face with respect to camera_depth_optical_frame.

    TODO: Eventually this node should return all detected faces, and then
    let the client decide which face to use.
    """

    # pylint: disable=duplicate-code
    # Much of the logic of this node mirrors FoodOnForkDetection. This is fine.
    # pylint: disable=too-many-instance-attributes
    # Needed for multiple model loads, publisher, subscribers, and shared variables
    def __init__(
        self,
    ):
        """
        Initializes the FaceDetection node. This node exposes a toggle_face_detection
        service that can be used to toggle the face detection on or off and
        publishes information about detected faces to the /face_detection
        topic when face detection is on.
        """
        super().__init__("face_detection")

        # Read the parameters
        # NOTE: These parameters are only read once. Any changes after the node
        # is initialized will not be reflected.
        (
            face_model_name,
            face_model_base_url,
            landmark_model_name,
            landmark_model_base_url,
            model_dir,
            depth_buffer_size,
            rate_hz,
        ) = self.read_params()
        face_model_name = face_model_name.value
        landmark_model_name = landmark_model_name.value
        self.rate_hz = rate_hz.value

        # Download the checkpoints if they don't exist
        self.face_model_path = os.path.join(model_dir.value, face_model_name)
        if not os.path.isfile(self.face_model_path):
            self.get_logger().info(
                "Face detection model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(
                face_model_name, model_dir.value, face_model_base_url.value
            )
            self.get_logger().info(
                f"Model checkpoint downloaded {self.face_model_path}."
            )
        self.landmark_model_path = os.path.join(model_dir.value, landmark_model_name)
        if not os.path.isfile(self.landmark_model_path):
            self.get_logger().info(
                "Facial landmark model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(
                landmark_model_name, model_dir.value, landmark_model_base_url.value
            )
            self.get_logger().info(
                f"Model checkpoint downloaded {self.landmark_model_path}."
            )

        self.bridge = CvBridge()

        # Keeps track of whether face detection is on or not
        self.is_on = False
        self.is_on_lock = threading.Lock()

        # Create the service
        self.srv = self.create_service(
            SetBool,
            "~/toggle_face_detection",
            self.toggle_face_detection_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the publishers
        self.publisher_results = self.create_publisher(
            FaceDetection, "~/face_detection", 1
        )
        # Currently, RVIZ2 doesn't support visualization of CompressedImage
        # (other than as part of a Camera). Hence, for vizualization purposes
        # this must be an Image. https://github.com/ros2/rviz/issues/738
        self.publisher_image = self.create_publisher(
            CompressedImage, "~/face_detection_img/compressed", 1
        )

        # Create an instance of the Face Detection Cascade Classifier
        self.detector = cv2.CascadeClassifier(self.face_model_path)

        # Create an instance of the Facial landmark Detector with the model
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(self.landmark_model_path)

        # Subscribe to the camera feed
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()
        image_topic = "~/image"
        try:
            image_type = get_img_msg_type(image_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of image topic. Defaulting to CompressedImage. {err}"
            )
            image_type = CompressedImage
        self.img_subscription = self.create_subscription(
            image_type,
            image_topic,
            self.camera_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        depth_buffer_size = depth_buffer_size.value
        self.depth_buffer = collections.deque(maxlen=depth_buffer_size)
        self.depth_buffer_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(aligned_depth_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to Image. {err}"
            )
            aligned_depth_type = Image
        # Subscribe to the depth image
        self.depth_subscription = self.create_subscription(
            aligned_depth_type,
            aligned_depth_topic,
            self.depth_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Subscribe to the camera info
        self.camera_info_lock = threading.Lock()
        # Approximate values of current camera intrinsics matrix
        # (updated with subscription)
        self.camera_matrix = [614, 0, 312, 0, 614, 223, 0, 0, 1]
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def read_params(
        self,
    ) -> Tuple[Parameter, Parameter, Parameter, Parameter, Parameter, Parameter]:
        """
        Read the parameters for this node.

        Returns
        -------
        face_model_name: The name of the face detection model checkpoint to use
        face_model_base_url: The URL to download the model checkpoint from if it is not
        already downloaded
        landmark_model_name: The name of the facial landmark detection model checkpoint to use
        landmark_model_base_url: The URL to download the model checkpoint from if it is not
        already downloaded
        model_dir: The location of the directory where the model checkpoint is / should be stored
        depth_buffer_size: The desired length of the depth image buffer
        """
        return self.declare_parameters(
            "",
            [
                (
                    "face_model_name",
                    None,
                    ParameterDescriptor(
                        name="face_model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the face detection model checkpoint to use",
                        read_only=True,
                    ),
                ),
                (
                    "face_model_base_url",
                    None,
                    ParameterDescriptor(
                        name="face_model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The URL to download the model checkpoint from "
                            "if it is not already downloaded"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "landmark_model_name",
                    None,
                    ParameterDescriptor(
                        name="landmark_model_name",
                        type=ParameterType.PARAMETER_STRING,
                        description="The name of the face detection model checkpoint to use",
                        read_only=True,
                    ),
                ),
                (
                    "landmark_model_base_url",
                    None,
                    ParameterDescriptor(
                        name="landmark_model_base_url",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The URL to download the model checkpoint from "
                            "if it is not already downloaded"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "model_dir",
                    None,
                    ParameterDescriptor(
                        name="model_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description=(
                            "The location of the directory where the model "
                            "checkpoint is / should be stored"
                        ),
                        read_only=True,
                    ),
                ),
                (
                    "depth_buffer_size",
                    30,
                    ParameterDescriptor(
                        name="depth_buffer_size",
                        type=ParameterType.PARAMETER_INTEGER,
                        description=("The desired length of the depth image buffer"),
                        read_only=True,
                    ),
                ),
                (
                    "rate_hz",
                    15,
                    ParameterDescriptor(
                        name="rate_hz",
                        type=ParameterType.PARAMETER_DOUBLE,
                        description=(
                            "The rate at which to run the face detection node"
                        ),
                        read_only=True,
                    ),
                ),
            ],
        )

    def toggle_face_detection_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """
        Callback function for the toggle_face_detection service. Safely toggles
        the face detection on or off depending on the request.
        """

        self.get_logger().info(f"Incoming service request. data: {request.data}")
        response.success = False
        response.message = f"Failed to set is_on to {request.data}"
        with self.is_on_lock:
            self.is_on = request.data
            response.success = True
            response.message = f"Successfully set is_on to {request.data}"
        return response

    def camera_info_callback(self, msg: CameraInfo):
        """
        Callback function for the toggle_face_detection service. Safely toggles
        the face detection on or off depending on the request.

        TODO: We technically only need to read one message from CameraInfo,
        since the intrinsics don't change. If `rclpy` adds a `wait_for_message`
        function, this subscription/callback should be replaced with that.
        """
        with self.camera_info_lock:
            self.camera_matrix = msg.k

    def depth_callback(self, msg: Image):
        """
        Callback function for depth images. If face_detection is on, this
        function publishes the 3d location of the mouth on the /face_detection
        topic in the camera_depth_optical_frame
        """
        # Collect a buffer of depth images (depth_buffer is a collections.deque
        # of max length depth_buffer_size)
        with self.depth_buffer_lock:
            self.depth_buffer.append(msg)

    def camera_callback(self, msg: Union[CompressedImage, Image]):
        """
        Callback function for the camera feed. If face detection is on, this
        function will detect faces in the image and store the location of the
        center of the mouth of the largest detected face.
        """
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg

    def detect_largest_face(
        self, image_bgr: npt.NDArray
    ) -> Tuple[bool, Tuple[int, int], npt.NDArray, Tuple[float, float, float, float]]:
        """
        Detect the largest face in an RGB image.

        Parameters
        ----------
        image_bgr: The OpenCV image to detect faces in.

        Returns
        -------
        is_face_detected: Whether a face was detected in the image.
        mouth_center: The (u,v) coordinates of the somion of the largest face
            detected in the image.
        face_points: A list of (u,v) coordinates of the landmarks of the face.
        face_bbox: The bounding box of the largest face detected in the image,
            in the form (x, y, w, h).
        """

        # pylint: disable=too-many-locals
        # This function is not too complex, but it does have a lot of local variables.

        # Decode the image
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the grayscale image
        # NOTE: This detector will launch multiple threads to detect faces in
        # parallel.
        faces = self.detector.detectMultiScale(image_gray)
        is_face_detected = len(faces) > 0
        img_mouth_center = (0, 0)
        img_face_points = []
        face_bbox = (0, 0, 0, 0)

        if is_face_detected:
            # Detect landmarks (a 3d list) on image
            # Relevant dimensions are 1: faces and 3: individual landmarks 0-68
            _, landmarks = self.landmark_detector.fit(image_gray, faces)

            # Add face markers to the image and find largest face
            largest_face = [0, 0]  # [area (px^2), index]
            for i, face in enumerate(faces):
                (_, _, w, h) = face

                # Update the largest face
                if w * h > largest_face[0]:
                    largest_face = [w * h, i]

            # Get the largest face
            face_bbox = faces[largest_face[1]]

            # Find stomion (mouth center) in image. See below for the landmark indices:
            # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
            img_mouth_center = landmarks[largest_face[1]][0][66]
            img_face_points = landmarks[largest_face[1]][0]

        return is_face_detected, img_mouth_center, img_face_points, face_bbox

    def depth_method_median_mouth_points(
        self,
        mouth_points: npt.NDArray,
        image_depth: npt.NDArray,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Compute the depth of the mouth stomion by taking the median depth of all
        viable mouth points.

        Parameters
        ----------
        mouth_points: A list of (u,v) coordinates of all facial landmarks.
        image_depth: The depth image corresponding to the RGB image that the
            face was detected in.
        threshold: The minimum fraction of mouth points that must be valid for
            the depth to be computed.

        Returns
        -------
        detected: whether the mouth was detected in the depth image.
        depth_mm: The depth of the mouth in mm.
        """
        # pylint: disable=too-many-locals
        # This function is not too complex, but it does have a lot of local variables.

        # Retrieve the depth values over all viable mouth coordinates
        depth_mms = []
        for point in mouth_points:
            u, v = int(point[0]), int(point[1])
            # Ensure that point is contained within the depth image frame
            if 0 <= u < image_depth.shape[1] and 0 <= v < image_depth.shape[0]:
                if image_depth[v][u] != 0:
                    depth_mms.append(image_depth[v][u])
        if len(depth_mms) >= threshold * len(mouth_points):
            # If at least the threshold mouth points are valid, use the median
            # depth of the mouth points
            depth_mm = np.percentile(depth_mms, 50)
            return True, depth_mm

        self.get_logger().warn(
            "The majority of mouth points were invalid (outside the frame or depth not detected). "
            "Unable to compute mouth depth with the median of mouth points."
        )
        return False, 0

    def depth_method_facial_plane(
        self, face_points: npt.NDArray, image_depth: npt.NDArray, threshold: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Compute the depth of the mouth stomion by fitting a plane to the
        internal points of the face and calculating the depth of the stomion
        from the plane.

        Parameters
        ----------
        face_points: A list of (u,v) coordinates of all facial landmarks.
        image_depth: The depth image corresponding to the RGB image that the
            face was detected in.
        threshold: The minimum fraction of face points that must be valid for
            the plane to be fit.

        Returns
        -------
        detected: whether the mouth was detected in the depth image.
        depth_mm: The depth of the mouth in mm.
        """
        # pylint: disable=too-many-locals
        # This function is not too complex, but it does have a lot of local variables.

        # Ignore points along the face outline, as their depth may be the depth
        # of the background
        internal_face_points = face_points[17:]
        face_points_3d = []
        for point in internal_face_points:
            u, v = int(point[0]), int(point[1])
            # Ensure that point is contained within the depth image frame
            if 0 <= u < image_depth.shape[1] and 0 <= v < image_depth.shape[0]:
                if image_depth[v][u] != 0:
                    z = image_depth[v][u]
                    face_points_3d.append([u, v, z])
        if len(face_points_3d) == 0:
            self.get_logger().warn(
                "All internal face points are out of the depth FOV. Ignoring face."
            )
            return False, 0
        face_points_3d = np.array(face_points_3d)

        # Remove points with an outlier depth
        depths = face_points_3d[:, 2]
        outlier_thresh = 1.5
        depth_q1 = np.percentile(depths, 25)
        depth_q3 = np.percentile(depths, 75)
        depth_iqr = depth_q3 - depth_q1
        non_outliers = np.logical_and(
            depths >= (depth_q1 - outlier_thresh * depth_iqr),
            depths <= (depth_q3 + outlier_thresh * depth_iqr),
        )
        if np.sum(non_outliers) < face_points_3d.shape[0]:
            outlier_depths = face_points_3d[np.logical_not(non_outliers)][:, 2]
            self.get_logger().debug(
                "Facial plane face detection method: removing outliers depths "
                f"{outlier_depths}"
            )
        face_points_3d = face_points_3d[non_outliers]

        if face_points_3d.shape[0] >= threshold * len(internal_face_points):
            # Fit a plane to detected face points
            plane = Plane.best_fit(Points(face_points_3d))
            a, b, c, d = plane.cartesian()

            # Locate u,v of stomion. See below for the landmark indices:
            # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
            stomion_u, stomion_v = face_points[66]
            # Calculate depth of stomion u,v from fit plane
            depth_mm = (d + (a * stomion_u) + (b * stomion_v)) / (-c)
            return True, depth_mm
        self.get_logger().warn(
            "The majority of internal face points were invalid (outside the frame or depth not detected). "
            "Unable to compute mouth depth from facial plane."
        )
        return False, 0

    def get_mouth_depth(
        self,
        rgb_msg: Union[CompressedImage, Image],
        face_points: npt.NDArray,
        methods: List[DepthComputationMethod] = [
            DepthComputationMethod.MEDIAN_MOUTH_POINTS,
            DepthComputationMethod.FACIAL_PLANE,
        ],
    ) -> Tuple[bool, float]:
        """
        Get the depth of the mouth of the largest detected face in the image.

        Parameters
        ----------
        rgb_msg: The RGB image that the face was detected in.
        face_points: A list of (u,v) coordinates of all facial landmarks.

        Returns
        -------
        detected: whether the mouth was detected in the depth image.
        depth_mm: The depth of the mouth in mm.
        """

        # pylint: disable=too-many-locals, dangerous-default-value
        # This function is not too complex, but it does have a lot of local variables.
        # In this case, a list as the default value is okay since we don't change it.

        # Pull out a list of (u,v) coordinates of all facial landmarks that can be
        # used to approximate the mouth center
        mouth_points = face_points[48:68]

        # Find depth image closest in time to RGB image that face was in
        with self.depth_buffer_lock:
            min_time_diff = float("inf")
            closest_depth_msg = None
            for depth_msg in self.depth_buffer:
                depth_time = depth_msg.header.stamp.sec + (
                    depth_msg.header.stamp.nanosec / (10**9)
                )
                img_time = rgb_msg.header.stamp.sec + (
                    rgb_msg.header.stamp.nanosec / (10**9)
                )
                time_diff = abs(depth_time - img_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_depth_msg = depth_msg
            if closest_depth_msg is None:
                self.get_logger().warn(
                    "No depth image message received.", throttle_duration_sec=1
                )
                return False, 0
            if (
                np.abs(closest_depth_msg.header.stamp.sec - rgb_msg.header.stamp.sec)
                >= 1.0
            ):
                self.get_logger().warn(
                    "Incosistent messages. Depth image message received at "
                    f"{closest_depth_msg.header.stamp}. RGB image message received "
                    f"at {rgb_msg.header.stamp}. Time difference: {min_time_diff} secs."
                )
        image_depth = ros_msg_to_cv2_image(closest_depth_msg, self.bridge)

        # Compute the depth of the mouth. Use the first method that works.
        for method in methods:
            if method == DepthComputationMethod.MEDIAN_MOUTH_POINTS:
                detected, depth_mm = self.depth_method_median_mouth_points(
                    mouth_points, image_depth
                )
                self.get_logger().debug(
                    f"Face detection with MEDIAN_MOUTH_POINTS: {detected}. {depth_mm}."
                )
            elif method == DepthComputationMethod.FACIAL_PLANE:
                detected, depth_mm = self.depth_method_facial_plane(
                    face_points, image_depth
                )
                self.get_logger().debug(
                    f"Face detection with FACIAL_PLANE: {detected}. {depth_mm}."
                )
            else:
                self.get_logger().warn(
                    f"Invalid depth computation method: {method}. Skipping."
                )
            if detected:
                return detected, depth_mm

        return False, 0

    def get_stomion_point(self, u: int, v: int, depth_mm: float) -> Point:
        """
        Get the 3d location of the mouth. This function assumes the color
        and depth images have the same frame, and returns the point in
        that frame.

        Parameters
        ----------
        u: The u coordinate of the mouth in the RGB image.
        v: The v coordinate of the mouth in the RGB image.
        depth_mm: The depth of the mouth in mm.

        Returns
        -------
        mouth_point: The 3d location of the mouth in the camera frame.
        """
        # Get the camera matrix
        with self.camera_info_lock:
            camera_matrix = self.camera_matrix

        # Compute the point. See https://www.youtube.com/watch?v=qByYk6JggQU&t=443s
        # for the derivation of the formulae.
        mouth_point = Point()
        mouth_point.x = (float(depth_mm) * (float(u) - camera_matrix[2])) / (
            1000.0 * camera_matrix[0]
        )
        mouth_point.y = (float(depth_mm) * (float(v) - camera_matrix[5])) / (
            1000.0 * camera_matrix[4]
        )
        mouth_point.z = float(depth_mm) / 1000.0

        return mouth_point

    def run(self) -> None:
        """
        Run face detection at the specified rate. Specifically, this function
        gets the latest RGB image message, gets the depth image message closest
        to it in time, runs face detection on the RGB image, and then determines
        the depth of the detected face.
        """
        rate = self.create_rate(self.rate_hz)
        while rclpy.ok():
            # Loop at the specified rate
            rate.sleep()

            # Check if face detection is on
            with self.is_on_lock:
                is_on = self.is_on
            if not is_on:
                continue

            # Create the FaceDetection message
            face_detection_msg = FaceDetection()

            # Get the latest RGB image message
            with self.latest_img_msg_lock:
                rgb_msg = self.latest_img_msg
            if rgb_msg is None:
                self.get_logger().warn(
                    "No RGB image message received.", throttle_duration_sec=1
                )
                continue

            # Detect the largest face in the RGB image
            image_bgr = ros_msg_to_cv2_image(rgb_msg, self.bridge)
            (
                is_face_detected,
                img_mouth_center,
                img_face_points,
                face_bbox,
            ) = self.detect_largest_face(image_bgr)

            if is_face_detected:
                # Get the depth of the mouth
                is_face_detected_depth, depth_mm = self.get_mouth_depth(
                    rgb_msg, img_face_points
                )
                if is_face_detected_depth:
                    # Get the 3d location of the mouth
                    face_detection_msg.detected_mouth_center.header = rgb_msg.header
                    face_detection_msg.detected_mouth_center.point = (
                        self.get_stomion_point(
                            img_mouth_center[0], img_mouth_center[1], depth_mm
                        )
                    )
                else:
                    is_face_detected = False

            # Annotate the image with the face
            if is_face_detected:
                self.annotate_image(image_bgr, img_face_points, face_bbox)

            # Publish the face detection image
            self.publisher_image.publish(
                cv2_image_to_ros_msg(image_bgr, compress=True, encoding="bgr8")
            )

            # Publish 3d point
            face_detection_msg.is_face_detected = is_face_detected
            self.publisher_results.publish(face_detection_msg)

    def annotate_image(
        self,
        image_bgr: npt.NDArray,
        img_face_points: npt.NDArray,
        face_bbox: Tuple[int, int, int, int],
    ) -> None:
        """
        Annotate the image with the face and facial landmarks center.

        Parameters
        ----------
        image_bgr: The OpenCV image to annotate.
        img_face_points: A list of (u,v) coordinates of the facial landmarks.
        face_bbox: The bounding box of the largest face detected in the image,
            in the form (x, y, w, h).
        """
        # Annotate the image with a red rectangle around largest face
        x, y, w, h = face_bbox
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Annotate the image with the face.
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (255, 255, 255), 2)
        for landmark_x, landmark_y in img_face_points:
            cv2.circle(
                image_bgr,
                (int(landmark_x), int(landmark_y)),
                1,
                (0, 255, 0),
                5,
            )


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    face_detection = FaceDetectionNode()
    executor = MultiThreadedExecutor(num_threads=4)

    # Spin in the background since detecting faces will block
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(face_detection,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run face detection
    try:
        face_detection.run()
    except KeyboardInterrupt:
        pass

    # Terminate this node
    face_detection.destroy_node()
    rclpy.shutdown()
    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
