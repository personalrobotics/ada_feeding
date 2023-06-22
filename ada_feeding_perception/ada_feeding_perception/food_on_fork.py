#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class FoodOnForkSubscriber(Node):
    """
    Class for FoodOnForkSubscriber the inherits from the Node class
    """
    def __init__(self):
        """
        Constructor for FoodOnForkSubscriber
        """
        super().__init__('food_on_fork_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,  # when the subscriber gets something, it calls the callback function
            1)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.c = 272

    def listener_callback(self, msg):
        """
        Callback function for the subscriber

        Parameters:
            msg: the message from the subscriber
        """
        img = None
        # Convert from ROS image to OpenCV format using cv_bridge
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if img is not None:
            # Convert img to numpy array
            img = np.array(img, dtype=np.uint8)
            # (0, 0, 255) is red cause BGR

            # Fork tines
            centers = [(359, 275), (367, 274), (375, 273), (381, 272)]
            for center in centers:
                cv.circle(img=img, center=center, radius=3, color=(0, 0, 255), thickness=1, lineType=8, shift=0)

            # Bounding rectangle
            cv.rectangle(img=img, pt1=())

            cv.imshow("img", img)
            cv.waitKey(0)
            print(self.c)
            self.c += 1


class FoodOnForkPublisher(Node):
    def __init__(self):
        super().__init__('food_on_fork_publisher')
        self.publisher = self.create_publisher(
            Int8MultiArray,
            '/food_on_fork',
            1
        )
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Int8MultiArray()
        msg.data = -1
        self.publisher_.publis(msg)
        self.get_logger().info('Publishing: "%s' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    food_on_fork = FoodOnForkSubscriber()

    rclpy.spin(food_on_fork)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    food_on_fork.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
