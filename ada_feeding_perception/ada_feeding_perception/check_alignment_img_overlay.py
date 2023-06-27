#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class CheckAlignmentImgOverlay(Node):
    def __init__(self):
        super().__init__('food_on_fork')
        self.most_recent_color_img = None
        # color topic subscription
        self.subscription_color = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback_color,  # when the subscriber gets something, it calls the callback function
            1
        )
        self.subscription_color

        # depth topic subscription
        self.subscription_depth = self.create_subscription(
            Image,
            'camera/aligned_depth_to_color/image_raw',
            self.listener_callback_depth,  # when the subscriber gets something, it calls the callback function
            1
        )
        self.subscription_depth

        # cv bridge
        self.bridge = CvBridge()

    def listener_callback_color(self, color_img_msg):
        color_img = None
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if color_img is not None:
            self.most_recent_color_img = color_img

    def listener_callback_depth(self, depth_img_msg):
        depth_img = None
        try:
            # passthrough is essentially rendering the image as it sees
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        if depth_img is not None:
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")  # converts image message and keeps
            # all the values as is because "passthrough" is used
            depth_img = (depth_img - np.max(depth_img)) / (
                        np.max(depth_img) - np.min(depth_img)) * 255  # scales the image to 0-255
            cv.imshow("depth_after_scaling", depth_img)
            depth_img = cv.cvtColor(depth_img, cv.COLOR_GRAY2BGR)
            result = cv.bitwise_and(self.most_recent_color_img, depth_img)
            cv.imshow("result", result)
            cv.waitKey(0)
            # print(np.max(depth_img))
            # depth_img = (depth_img - np.max(depth_img)) / (np.max(depth_img) - np.min(depth_img)) * 255
            # print(np.max(depth_img))
            # cv.imshow("depth_bgr62", depth_img)
            # cv.waitKey(0)
            # # depth_normalized = cv.normalize(depth_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
            # # # depth_colormap = cv.applyColorMap(depth_normalized, cv.COLORMA)
            # # depth_bgr = cv.cvtColor(depth_normalized, cv.COLOR_GRAY2BGR)
            # # print(np.max(depth_bgr))
            # # cv.imshow("depth_bgr", depth_bgr)
            # # cv.waitKey(0)
            # result = cv.bitwise_and(self.most_recent_color_img, depth_img)
            # cv.imshow("Result", result)
            # cv.waitKey(0)
        else:
            print("no most recent color img")




            # # cv.imshow("depth_59", depth_img)
            # # depth_img = np.array(depth_img, dtype=np.uint16)
            # cv.imshow("depth_60", depth_img)
            # depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) * 255
            # cv.imshow("depth_63", depth_img)
            # print("64", np.max(depth_img))
            # depth_img = np.array(depth_img, dtype=np.uint8)
            # # depth_img = depth_img.astype(np.uint8)
            # cv.imshow("depth_66", depth_img)
            # if self.most_recent_color_img is not None:
            #     depth_img = cv.cvtColor(depth_img, cv.COLOR_GRAY2BGR)
            #     cv.imshow("most_recent_color_img", self.most_recent_color_img)
            #     cv.imshow("depth_img", depth_img)
            #     print(self.most_recent_color_img.shape, self.most_recent_color_img.dtype)
            #     print(depth_img.shape, depth_img.dtype)
            #     result = cv.bitwise_and(self.most_recent_color_img, depth_img)
            #     cv.imshow("Result", result)
            #     cv.waitKey(0)
            # else:
            #     print("no most recent color img")


def main(args=None):
    print("Running CheckAlignmentImgOverlay")
    rclpy.init(args=args)
    food_on_fork = CheckAlignmentImgOverlay()
    executor = MultiThreadedExecutor()
    rclpy.spin(food_on_fork, executor=executor)

    food_on_fork.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
