#!/usr/bin/env python3

"""
This file defines the FaceDetection class, which ...
"""
from ada_feeding_msgs.msg import FaceDetection
from ada_feeding_msgs.srv import ToggleFaceDetection
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os
from threading import Lock
from typing import Tuple
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point

from ada_feeding_perception.helpers import (
    download_checkpoint,
)


class FaceDetectionNode(Node):
    def __init__(
        self,
    ):
        """
        Initializes the FaceDetection node, which exposes a ToggleFaceDetection
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

        # Download the checkpoint if it doesn't exist
        self.face_model_path = os.path.join(model_dir.value, self.face_model_name)
        if not os.path.isfile(self.face_model_path):
            self.get_logger().info("Face detection model checkpoint does not exist. Downloading...")
            download_checkpoint(self.face_model_name, model_dir.value, face_model_base_url.value)
            self.get_logger().info(
                "Model checkpoint downloaded %s."
                % os.path.join(model_dir.value, self.face_model_name)
            )
        self.landmark_model_path = os.path.join(model_dir.value, self.landmark_model_name)
        if not os.path.isfile(self.landmark_model_path):
            self.get_logger().info("Facial landmark model checkpoint does not exist. Downloading...")
            download_checkpoint(self.landmark_model_name, model_dir.value, landmark_model_base_url.value)
            self.get_logger().info(
                "Model checkpoint downloaded %s."
                % os.path.join(model_dir.value, self.landmark_model_name)
            )
  
        # Convert between ROS and CV images
        self.bridge = CvBridge()

        # Keeps track of whether face detection is on or not
        self.is_on = False
        self.is_on_lock = Lock()

        # Keeps track of the detected mouth center
        self.img_mouth_center = []
        self.is_face_detected = False


        # Create the service
        self.srv = self.create_service(
            ToggleFaceDetection,
            "ToggleFaceDetection",
            self.toggle_face_detection_callback,
        )

        # Subscribe to the camera feed
        self.img_subscription = self.create_subscription(
            Image, "camera/color/image_raw", self.camera_callback, 1
        )
        self.img_subscription  # prevent unused variable warning

        #subscribe to the depth image
        self.depth_subscription = self.create_subscription(
            Image, "camera/aligned_depth_to_color/image_raw", self.depth_callback, 1
        )
        self.depth_subscription  # prevent unused variable warning

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

    def toggle_face_detection_callback(self, request, response):
        """
        Callback function for the ToggleFaceDetection service. Safely toggles
        the face detection on or off depending on the request.
        """
        self.get_logger().info(
            "Incoming service request. turn_on: %s" % (request.turn_on)
        )
        if request.turn_on:
            # Turn on face detection
            self.is_on_lock.acquire()
            self.is_on = True
            self.is_on_lock.release()
            response.face_detection_is_on = True
        else:
            self.is_on_lock.acquire()
            self.is_on = False
            self.is_on_lock.release()
            response.face_detection_is_on = False
        return response
    
    

    def depth_callback(self, msg):
        print("depth callback")
        if len(self.img_mouth_center) == 2:
            u = int(self.img_mouth_center[0])
            v = int(self.img_mouth_center[1])
            width = msg.width
            print("Made it")
            depth = msg.data[(u%width) + (v*width)]
            #point_source = self.pixel_to_3d_point(msg, self.img_mouth_center[0], self.img_mouth_center[1])

            depth_point = Point()
            depth_point.x = float(u)
            depth_point.y = float(v)
            depth_point.z = float(depth)
            point_source = PointStamped()
            point_source.header = msg.header
            point_source.point = depth_point
            color_frame = 'camera_color_optical_frame'
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer,self)

            
            #point_source = []
            # get the transformation from source_frame to target_frame.
            try:
                point_target = tf_buffer.transform(point_source, color_frame)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                self.get_logger().info('Unable to find the transformation to %s'
                            % target_frame)


            # Publish the face detection information
            face_detection_msg = FaceDetection()
            face_detection_msg.is_face_detected = self.is_face_detected

            face_detection_msg.detected_mouth_center = point_target
            
            self.publisher_results.publish(face_detection_msg)
            
    

    def camera_callback(self, msg):
        """
        Callback function for the camera feed. If face detection is on, this
        function will detect faces in the image and publish information about
        them to the /face_detection topic.
        """
        self.is_on_lock.acquire()
        is_on = self.is_on
        self.is_on_lock.release()
        if is_on:
            self.is_face_detected = False
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # convert image to RGB colour
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # convert image to Grayscale
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            
            # create an instance of the Face Detection Cascade Classifier
            detector = cv2.CascadeClassifier(self.face_model_path)
            # Detect faces using the haarcascade classifier on the "grayscale image"
            faces = detector.detectMultiScale(image_gray)
            if len(faces) > 0:
                self.is_face_detected = True


            
            if self.is_face_detected:

                # Detect mouth center

                # create an instance of the Facial landmark Detector with the model
                landmark_detector  = cv2.face.createFacemarkLBF()
                landmark_detector.loadModel(self.landmark_model_path)

                # Detect landmarks on "image_gray"
                _, landmarks = landmark_detector.fit(image_gray, faces)
                for landmark in landmarks:
                    count = 0
                    # Add a dummy face marker to the sensor_msgs/Image
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    for face in faces:
                        #save the coordinates in x, y, w, d variables
                        (x,y,w,d) = face
                        # Draw a white coloured rectangle around each face using the face's coordinates
                        # on the "image_template" with the thickness of 2 
                        cv2.rectangle(cv_image,(x,y),(x+w, y+d),(255, 255, 255), 2)

                for x,y in landmark[0]:
                    # display landmarks on "image_rgb"
                    # with white colour in BGR and thickness 1
                    if count >= 48:
                        cv2.circle(cv_image, (int(x),int(y)), 1, (0, 255, 0), 5)
                    count += 1
                annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                annotated_img = annotated_msg
                # Find stomion in image
   
                self.img_mouth_center = landmark[0][66]
                # Publish the detected mouth center
                #face_detection_msg.detected_mouth_center = PointStamped()
                #face_detection_msg.detected_mouth_center.header = msg.header
                #face_detection_msg.detected_mouth_center.point.x = 
                #face_detection_msg.detected_mouth_center.point.y = 
                #face_detection_msg.detected_mouth_center.point.z = 
            else:
                annotated_img = msg
                #self.img_mouth_center = []
            self.publisher_image.publish(annotated_img)


def main(args=None):
    rclpy.init(args=args)

    face_detection = FaceDetectionNode()

    rclpy.spin(face_detection)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
