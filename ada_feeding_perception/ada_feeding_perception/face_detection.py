#!/usr/bin/env python3

"""
This file defines the FaceDetection class, which publishes the 3d PointStamped locations
of the largest detected mouth with respect to camera_depth_optical_frame.
"""
from ada_feeding_msgs.msg import FaceDetection
from std_srvs.srv import SetBool
import cv2
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import os
from threading import Lock
from typing import Tuple
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import collections


from ada_feeding_perception.helpers import (
    download_checkpoint,
)


class FaceDetectionNode(Node):
    """
    This node publishes a 3d PointStamped location
    of the largest detected mouth with respect to camera_depth_optical_frame.
    """
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
        ) = self.read_params()
        self.face_model_name = face_model_name.value
        self.landmark_model_name = landmark_model_name.value

        # Download the checkpoints if they don't exist
        self.face_model_path = os.path.join(model_dir.value, self.face_model_name)
        if not os.path.isfile(self.face_model_path):
            self.get_logger().info(
                "Face detection model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(
                self.face_model_name, model_dir.value, face_model_base_url.value
            )
            self.get_logger().info(
                "Model checkpoint downloaded %s."
                % os.path.join(model_dir.value, self.face_model_name)
            )
        self.landmark_model_path = os.path.join(
            model_dir.value, self.landmark_model_name
        )
        if not os.path.isfile(self.landmark_model_path):
            self.get_logger().info(
                "Facial landmark model checkpoint does not exist. Downloading..."
            )
            download_checkpoint(
                self.landmark_model_name, model_dir.value, landmark_model_base_url.value
            )
            self.get_logger().info(
                "Model checkpoint downloaded %s."
                % os.path.join(model_dir.value, self.landmark_model_name)
            )

        # Keeps track of whether face detection is on or not

     
        self.is_on = False
        self.is_on_lock = Lock()
        self.bridge = CvBridge()
        self.is_face_detected = False
        self.depth_buffer = collections.deque(maxlen=10)
        self.img_timestamp = None

        # Approximate values of current camera info (updated with subscription)
        self.camera_matrix = [614, 0, 312, 0, 614, 223, 0, 0, 1]

        # Keeps track of the detected mouth points
        self.img_mouth_center = []
        self.img_mouth_points = []

        # Create the service
        self.srv = self.create_service(
            SetBool,
            "toggle_face_detection",
            self.toggle_face_detection_callback,
        )

        # Subscribe to the camera feed
        self.img_subscription = self.create_subscription(
            CompressedImage, "camera/color/image_raw/compressed", self.camera_callback, 1
        )

        # Subscribe to the depth image
        self.depth_subscription = self.create_subscription(
            Image, "/camera/depth/image_rect_raw", self.depth_callback, 1
        )

        # Subscribe to the camera info
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 1
        )

        # Create the publishers
        self.publisher_results = self.create_publisher(
            FaceDetection, "face_detection", 1
        )
        self.publisher_image = self.create_publisher(Image, "face_detection_img", 1)

    def read_params(
        self,
    ) -> Tuple[Parameter, Parameter, Parameter, Parameter, Parameter]:
        """
        Read the parameters for this node.

        Returns
        -------
        face_model_name: The name of the face detection model checkpoint to use
        face_model_base_url: The URL to download the model checkpoint from if it is not already downloaded
        landmark_model_name: The name of the facial landmark detection model checkpoint to use
        landmark_model_base_url: The URL to download the model checkpoint from if it is not already downloaded
        model_dir: The location of the directory where the model checkpoint is / should be stored
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
                        description="The URL to download the model checkpoint from if it is not already downloaded",
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
                        description="The URL to download the model checkpoint from if it is not already downloaded",
                        read_only=True,
                    ),
                ),
                (
                    "model_dir",
                    None,
                    ParameterDescriptor(
                        name="model_dir",
                        type=ParameterType.PARAMETER_STRING,
                        description="The location of the directory where the model checkpoint is / should be stored",
                        read_only=True,
                    ),
                ),
            ],
        )

    def toggle_face_detection_callback(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        """
        Callback function for the toggle_face_detection service. Safely toggles
        the face detection on or off depending on the request.
        """
        self.get_logger().info(
            "Incoming service request. data: %s" % (request.data)
        )
        with self.is_on_lock:
            self.is_on = request.data
        response.success = request.data
        return response
    
    def camera_info_callback(self, msg: CameraInfo):
        """
        Callback function for the toggle_face_detection service. Safely toggles
        the face detection on or off depending on the request.
        """
        self.camera_matrix = msg.k

    def depth_callback(self, msg: CompressedImage):
        """
        Callback function for depth images. If face_detection is on, this 
        function publishes the 3d location of the mouth on the /face_detection 
        topic in the camera_depth_optical_frame
        """

        # Collect last ten depth images
        self.depth_buffer.append(msg)
        with self.is_on_lock:
            is_on = self.is_on

        # Check if face detection is on
        if is_on:
            face_detection_msg = FaceDetection()
            if self.is_face_detected:
                # Retrieve the 2d location of the mouth center
                u = int(self.img_mouth_center[0])
                v = int(self.img_mouth_center[1])

                # Find depth image closest in time to saved color image
                difference_array = np.array([])
                for depth_img in self.depth_buffer:
                    depth_time = depth_img.header.stamp.sec + (depth_img.header.stamp.nanosec/(10**9))
                    img_time = self.img_timestamp.sec + (self.img_timestamp.nanosec/(10**9))
                    difference_array = np.append(difference_array, abs(depth_time - img_time))

                closest_index = difference_array.argmin()

                closest_depth = self.depth_buffer[closest_index]
                
                # Retrieve the depth value averaged over all mouth coordinates 
                width = msg.width
                depth_sum = 0
                for point in self.img_mouth_points:
                    depth_sum += closest_depth.data[(int(point[0])) + (int(point[1]) * width)]
                depth = depth_sum/float(len(self.img_mouth_points))

                # Create target 3d point, with mm measurements converted to m
                mouth_location = PointStamped()
                mouth_location.header = msg.header
                mouth_location.point.x = (float(depth)*(float(u)-self.camera_matrix[2]))/(1000*self.camera_matrix[0])
                mouth_location.point.y = (float(depth)*(float(v)-self.camera_matrix[5]))/(1000*self.camera_matrix[4])
                mouth_location.point.z = float(depth)/1000.0

                face_detection_msg.detected_mouth_center = mouth_location

            # Publish 3d point
            face_detection_msg.is_face_detected = self.is_face_detected
            self.publisher_results.publish(face_detection_msg)

    def camera_callback(self, msg: CompressedImage):
        """
        Callback function for the camera feed. If face detection is on, this
        function will detect faces in the image and store the location of
        the largest detected face/mouth.
        """
        image_rgb = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        with self.is_on_lock:
            is_on = self.is_on

        if is_on:
            self.is_face_detected = False
            
            # Convert image to Grayscale
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

            # Create an instance of the Face Detection Cascade Classifier
            detector = cv2.CascadeClassifier(self.face_model_path)
            # Detect faces using the haarcascade classifier on the grayscale image
            faces = detector.detectMultiScale(image_gray)

            # Check if a face has been detected
            if len(faces) > 0:
                self.is_face_detected = True


            if self.is_face_detected:
                # Create an instance of the Facial landmark Detector with the model
                landmark_detector = cv2.face.createFacemarkLBF()
                landmark_detector.loadModel(self.landmark_model_path)

                # Detect landmarks (a 3d list) on image
                # Relevant dimensions are 1: faces and 3: individual landmarks 0-68
                _, landmarks = landmark_detector.fit(image_gray, faces)

                # Add face markers to the image and find largest face
                largest_face = 0
                largest_face_index = 0
                for i in range(len(faces)):
                    # Draw a white rectangle around each face
                    (x, y, w, d) = faces[i]
                    largest_face = max(largest_face, w*d)
                    largest_face_index = i
                    cv2.rectangle(
                        image_rgb, (x, y), (x + w, y + d), (255, 255, 255), 2
                    )
                    # Display mouth landmarks (48-67, as explained in the below link)
                    # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                    for j in range(48, 67):
                        x, y = landmarks[i][0][j]
                        cv2.circle(image_rgb, (int(x), int(y)), 1, (0, 255, 0), 5)

            


                # Find stomion (mouth center) in image
                self.img_mouth_center = landmarks[largest_face_index][0][66]
                self.img_mouth_points = landmarks[largest_face_index][0][48:67]

                self.img_timestamp = msg.header.stamp


            annotated_msg = self.bridge.cv2_to_imgmsg(image_rgb, "rgb8")


            # Publish annotated image with face and mouth landmarks
            self.publisher_image.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)

    face_detection = FaceDetectionNode()

    rclpy.spin(face_detection)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
