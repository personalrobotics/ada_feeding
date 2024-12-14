#!/usr/bin/env python3
# Copyright (c) 2024, Personal Robotics Laboratory
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from ada_feeding_msgs.msg import FoodOnForkDetection
from std_srvs.srv import SetBool
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from threading import Lock


class FoodOnForkDetectionNode(Node):
    def __init__(
        self,
        food_on_fork_detection_interval=90,
        num_images_with_food=90,
    ):
        """
        Initializes the FoodOnForkDetection node, which exposes a SetBool
        service that can be used to toggle the food on fork detection on or off and
        publishes information  to the /food_on_fork_detection topic when food-on-fork
        detection is on.

        After food_on_fork_detection_interval images without food, this dummy function
        detects food for num_images_with_food frames.

        Parameters:
        ----------
        food_on_fork_detection_interval: The number of frames between each food detection.
        num_images_with_food: The number of frames that must have a food in them.
        """
        super().__init__("food_on_fork_detection")

        # Internal variables to track when food should be detected
        self.food_on_fork_detection_interval = food_on_fork_detection_interval
        self.num_images_with_food = num_images_with_food
        self.num_consecutive_images_without_food = (
            self.food_on_fork_detection_interval
        )  # Start predicting FoF
        self.num_consecutive_images_with_food = 0

        # Keeps track of whether food on fork detection is on or not
        self.is_on = False
        self.is_on_lock = Lock()

        # Create the service
        self.srv = self.create_service(
            SetBool,
            "toggle_food_on_fork_detection",
            self.toggle_food_on_fork_detection_callback,
        )

        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            CompressedImage,
            "camera/color/image_raw/compressed",
            self.camera_callback,
            1,
        )

        # Create the publishers
        self.publisher_results = self.create_publisher(
            FoodOnForkDetection, "food_on_fork_detection", 1
        )

    def toggle_food_on_fork_detection_callback(self, request, response):
        """
        Callback function for the SetBool service. Safely toggles
        the food on fork detection on or off depending on the request.
        """
        self.get_logger().info("Incoming service request. turn_on: %s" % (request.data))
        if request.data:
            # Reset counters
            self.num_consecutive_images_without_food = (
                self.food_on_fork_detection_interval
            )  # Start predicting FoF
            self.num_consecutive_images_with_food = 0
            # Turn on food-on-fork detection
            self.is_on_lock.acquire()
            self.is_on = True
            self.is_on_lock.release()
            response.success = True
            response.message = "Successfully turned food-on-fork detection on"
        else:
            self.is_on_lock.acquire()
            self.is_on = False
            self.is_on_lock.release()
            response.success = True
            response.message = "Successfully turned food-on-fork detection off"
        return response

    def camera_callback(self, msg):
        """
        Callback function for the camera feed. If food-on-fork detection is on, this
        function will detect food in the image and publish information about
        them to the /food_on_fork_detection topic.
        """
        self.get_logger().debug("Received image")
        self.is_on_lock.acquire()
        is_on = self.is_on
        self.is_on_lock.release()
        if is_on:
            # Update the number of consecutive images with/without a food
            is_food_detected = False
            if self.num_consecutive_images_with_food == self.num_images_with_food:
                self.num_consecutive_images_without_food = 0
                self.num_consecutive_images_with_food = 0
            if (
                self.num_consecutive_images_without_food
                == self.food_on_fork_detection_interval
            ):
                # Detect food on the fork
                self.num_consecutive_images_with_food += 1
                is_food_detected = True
            else:
                # Don't detect food
                self.num_consecutive_images_without_food += 1

            # Publish the food-on-fork detection information
            food_on_fork_detection_msg = FoodOnForkDetection()
            food_on_fork_detection_msg.header = msg.header
            food_on_fork_detection_msg.probability = 1.0 if is_food_detected else 0.0
            food_on_fork_detection_msg.status = food_on_fork_detection_msg.SUCCESS
            food_on_fork_detection_msg.message = (
                "Food detected" if is_food_detected else "No food detected"
            )
            self.publisher_results.publish(food_on_fork_detection_msg)


def main(args=None):
    rclpy.init(args=args)

    food_on_fork_detection = FoodOnForkDetectionNode()

    rclpy.spin(food_on_fork_detection)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
