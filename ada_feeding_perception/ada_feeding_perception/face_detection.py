#!/usr/bin/env python3

"""
This file defines the FaceDetection class, which publishes the 3d PointStamped locations
of the largest detected mouth with respect to camera_depth_optical_frame.
"""

# Standard Imports
import collections
import os
import threading
from typing import Tuple, Union

# Third-party imports
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from std_srvs.srv import SetBool

# Local imports
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding_perception.helpers import (
    download_checkpoint,
    get_img_msg_type,
    cv2_image_to_ros_msg,
)


class FaceDetectionNode(Node):
    """
    This node publishes a 3d PointStamped location
    of the largest detected face with respect to camera_depth_optical_frame.

    TODO: Eventually this node should return all detected faces, and then
    let the client decide which face to use.
    """

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
        self.publisher_image = self.create_publisher(Image, "~/face_detection_img", 1)

        # Create an instance of the Face Detection Cascade Classifier
        self.detector = cv2.CascadeClassifier(self.face_model_path)

        # Create an instance of the Facial landmark Detector with the model
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(self.landmark_model_path)

        # Subscribe to the camera feed
        self.latest_img_msg = None
        self.latest_img_msg_lock = threading.Lock()
        image_topic = "~/image"
        self.img_subscription = self.create_subscription(
            get_img_msg_type(image_topic, self),
            image_topic,
            self.camera_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        depth_buffer_size = depth_buffer_size.value
        self.depth_buffer = collections.deque(maxlen=depth_buffer_size)
        self.depth_buffer_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        # Subscribe to the depth image
        self.depth_subscription = self.create_subscription(
            get_img_msg_type(aligned_depth_topic, self),
            aligned_depth_topic,
            self.depth_callback,
            1,
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
            1,
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

        # pylint: disable=duplicate-code
        # We follow similar logic in any service to toggle a node
        # (e.g., face detection)

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
        self, img_msg: Union[CompressedImage, Image], publish_annotated_img: bool = True
    ) -> Tuple[bool, Tuple[int, int], np.ndarray]:
        """
        Detect the largest face in an RGB image.

        Parameters
        ----------
        img_msg: The RGB image to detect faces in.
        publish_annotated_img: Whether to publish an annotated image with the
            detected faces and mouth points.

        Returns
        -------
        is_face_detected: Whether a face was detected in the image.
        mouth_center: The (u,v) coordinates of the somion of the largest face
            detected in the image.
        mouth_points: A list of (u,v) coordinates of the facial landmarks of the mouth.
        """

        # pylint: disable=too-many-locals
        # This function is not too complex, but it does have a lot of local variables.

        # Decode the image
        image_rgb = cv2.imdecode(
            np.frombuffer(img_msg.data, np.uint8), cv2.IMREAD_COLOR
        )
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the grayscale image
        # NOTE: This detector will launch multiple threads to detect faces in
        # parallel.
        faces = self.detector.detectMultiScale(image_gray)
        is_face_detected = len(faces) > 0
        img_mouth_center = (0, 0)
        img_mouth_points = []

        if is_face_detected:
            # Detect landmarks (a 3d list) on image
            # Relevant dimensions are 1: faces and 3: individual landmarks 0-68
            _, landmarks = self.landmark_detector.fit(image_gray, faces)

            # Add face markers to the image and find largest face
            largest_face = [0, 0]  # [area (px^2), index]
            for i, face in enumerate(faces):
                (x, y, w, h) = face

                # Update the largest face
                if w * h > largest_face[0]:
                    largest_face = [w * h, i]

                # Annotate the image with the face. See below for the landmark indices:
                # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                if publish_annotated_img:
                    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    for j in range(48, 68):
                        landmark_x, landmark_y = landmarks[i][0][j]
                        cv2.circle(
                            image_rgb,
                            (int(landmark_x), int(landmark_y)),
                            1,
                            (0, 255, 0),
                            5,
                        )

            # Annotate the image with a red rectangle around largest face
            (x, y, w, h) = faces[largest_face[1]]
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Find stomion (mouth center) in image
            img_mouth_center = landmarks[largest_face[1]][0][66]
            img_mouth_points = landmarks[largest_face[1]][0][48:68]

        # Publish annotated image
        if publish_annotated_img:
            annotated_msg = cv2_image_to_ros_msg(image_rgb, compress=False)
            self.publisher_image.publish(annotated_msg)

        return is_face_detected, img_mouth_center, img_mouth_points

    def get_mouth_depth(
        self, rgb_msg: Union[CompressedImage, Image], face_points: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Get the depth of the mouth of the largest detected face in the image.

        Parameters
        ----------
        rgb_msg: The RGB image that the face was detected in.
        face_points: A list of (u,v) coordinates of the facial landmarks whose depth
            should be averaged to get the predicted depth of the mouth.

        Returns
        -------
        detected: whether the mouth was detected in the depth image.
        depth_mm: The depth of the mouth in mm.
        """

        # pylint: disable=too-many-locals
        # This function is not too complex, but it does have a lot of local variables.

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
            self.get_logger().info(
                f"Closest depth image message received at {closest_depth_msg.header.stamp}. "
                f"Corresponding RGB image message received at {rgb_msg.header.stamp}. "
                f"Time difference: {min_time_diff} seconds."
            )
        image_depth = self.bridge.imgmsg_to_cv2(
            closest_depth_msg,
            desired_encoding="passthrough",
        )

        # Retrieve the depth value averaged over all viable mouth coordinates
        depth_sum = 0
        num_points_in_frame = 0
        num_points_at_depth = 0
        for point in face_points:
            x, y = int(point[0]), int(point[1])
            # Ensure that point is contained within the depth image frame
            if 0 <= x < image_depth.shape[1] and 0 <= y < image_depth.shape[0]:
                num_points_in_frame += 1
            # Ensure that the point is farther away from the camera than the fork tip (~450 mm)
            if image_depth[y][x] > 450:
                num_points_at_depth += 1
                depth_sum += image_depth[y][x]
        if num_points_in_frame < 0.5 * len(face_points):
            self.get_logger().warn(
                "Detected face in the RGB image, but majority of mouth points "
                "were outside the frame of the depth image. Ignoring this face."
            )
            return False, 0
        if num_point_at_depth == 0:
            self.get_logger().warn(
                "Detected face in the RGB image, but no mouth points were "
                "detected farther away than the fork tip. Ignoring this face."
            )
            return False, 0
        depth_mm = depth_sum / float(num_points_in_frame)
        return True, depth_mm

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
            (
                is_face_detected,
                img_mouth_center,
                img_mouth_points,
            ) = self.detect_largest_face(rgb_msg)

            if is_face_detected:
                # Get the depth of the mouth
                is_face_detected_depth, depth_mm = self.get_mouth_depth(
                    rgb_msg, img_mouth_points
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

            # Publish 3d point
            face_detection_msg.is_face_detected = is_face_detected
            self.publisher_results.publish(face_detection_msg)


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
