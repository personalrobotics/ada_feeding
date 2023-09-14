#!/usr/bin/env python3

"""
This file defines the FaceDetection class, which publishes the 3d PointStamped locations
of the largest detected mouth with respect to camera_depth_optical_frame.
"""

# Standard Imports
import collections
import os
import threading
from typing import Tuple

# Third-party imports
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
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

        self._default_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

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
            get_img_msg_type(image_topic, self), image_topic, self.camera_callback, 1
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
        )

        # Subscribe to the camera info
        self.camera_info_lock = threading.Lock()
        # Approximate values of current camera intrinsics matrix
        # (updated with subscription)
        self.camera_matrix = [614, 0, 312, 0, 614, 223, 0, 0, 1]
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "~/camera_info", self.camera_info_callback, 1
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

    def depth_callback(self, msg: CompressedImage):
        """
        Callback function for depth images. If face_detection is on, this
        function publishes the 3d location of the mouth on the /face_detection
        topic in the camera_depth_optical_frame
        """
        # Collect a buffer of depth images (depth_buffer is a collections.deque
        # of max length depth_buffer_size)
        with self.depth_buffer_lock:
            self.depth_buffer.append(msg)

    def camera_callback(self, msg: CompressedImage):
        """
        Callback function for the camera feed. If face detection is on, this
        function will detect faces in the image and store the location of the
        center of the mouth of the largest detected face.
        """
        with self.latest_img_msg_lock:
            self.latest_img_msg = msg

    def run(self) -> None:
        """
        Run face detection at the specified rate. Specifically, this function
        gets the latest RGB image message, gets the depth image message closest
        to it in time, runs face detection on the RGB image, and then determines
        the depth of the detected face.
        """

        # pylint: disable=too-many-locals
        # TODO: Split the RGB part of face detection and the depth part of face
        # detection into separate functions.

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

            # Decode the image
            image_rgb = cv2.imdecode(
                np.frombuffer(rgb_msg.data, np.uint8), cv2.IMREAD_COLOR
            )
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

            # Detect faces using the haarcascade classifier on the grayscale image
            faces = self.detector.detectMultiScale(image_gray)
            is_face_detected = len(faces) > 0

            if is_face_detected:
                # Detect landmarks (a 3d list) on image
                # Relevant dimensions are 1: faces and 3: individual landmarks 0-68
                _, landmarks = self.landmark_detector.fit(image_gray, faces)

                # Add face markers to the image and find largest face
                largest_face = [0, 0]  # [area (px^2), index]
                for i, face in enumerate(faces):
                    (x, y, w, h) = face
                    # Annotate the image with the face. See below for the landmark indices:
                    # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                    cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    for j in range(48, 68):
                        x, y = landmarks[i][0][j]
                        cv2.circle(image_rgb, (int(x), int(y)), 1, (0, 255, 0), 5)

                    # Update the largest face
                    if w * h > largest_face[0]:
                        largest_face = [w * h, i]

                # Annotate the image with a red rectangle around largest face
                (x, y, w, h) = faces[largest_face[1]]
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Find stomion (mouth center) in image
                img_mouth_center = landmarks[largest_face[1]][0][66]
                u, v = int(round(img_mouth_center[0])), int(round(img_mouth_center[1]))
                img_mouth_points = landmarks[largest_face[1]][0][48:68]

                # Find depth image closest in time to saved color image
                time_diffs = []
                with self.depth_buffer_lock:
                    for i, depth_msg in enumerate(self.depth_buffer):
                        depth_time = depth_msg.header.stamp.sec + (
                            depth_msg.header.stamp.nanosec / (10**9)
                        )
                        img_time = rgb_msg.header.stamp.sec + (
                            rgb_msg.header.stamp.nanosec / (10**9)
                        )
                        time_diffs.append(abs(depth_time - img_time))
                    closest_depth_msg = self.depth_buffer[np.argmin(time_diffs)]
                image_depth = self.bridge.imgmsg_to_cv2(
                    closest_depth_msg,
                    desired_encoding="passthrough",
                )

                # Retrieve the depth value averaged over all mouth coordinates
                depth_sum = 0
                for point in img_mouth_points:
                    depth_sum += image_depth[int(point[1])][int(point[0])]
                depth = depth_sum / float(len(img_mouth_points))

                # Get the camera matrix
                with self.camera_info_lock:
                    camera_matrix = self.camera_matrix

                # Create target 3d point, with mm measurements converted to m
                # Equations and explanation can be found at https://youtu.be/qByYk6JggQU
                mouth_location = PointStamped()
                mouth_location.header = rgb_msg.header
                mouth_location.point.x = (
                    float(depth) * (float(u) - camera_matrix[2])
                ) / (1000.0 * camera_matrix[0])
                mouth_location.point.y = (
                    float(depth) * (float(v) - camera_matrix[5])
                ) / (1000.0 * camera_matrix[4])
                mouth_location.point.z = float(depth) / 1000.0

                face_detection_msg.detected_mouth_center = mouth_location

            # Publish 3d point
            face_detection_msg.is_face_detected = is_face_detected
            self.publisher_results.publish(face_detection_msg)

            # Publish annotated image with face and mouth landmarks
            annotated_msg = cv2_image_to_ros_msg(image_rgb, compress=False)
            self.publisher_image.publish(annotated_msg)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    face_detection = FaceDetectionNode()
    executor = MultiThreadedExecutor()

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
