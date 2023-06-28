#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class FoodOnFork(Node):
    """
    This class is the second version of the Food on Fork node written. There is an error in
    the process used to visualize the depth image. The overall concept is valid, however.
    """
    def __init__(self):
        super().__init__('food_on_fork')

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
            # Fork tines
            centers = [(359, 275), (367, 274), (375, 273), (381, 272)]
            for center in centers:
                # (0, 0, 255) is red cause BGR
                cv.circle(img=color_img, center=center, radius=3, color=(0, 0, 255), thickness=1,
                          lineType=8, shift=0)

            # Bounding rectangle
            tine1_x, tine1_y = centers[0]
            pt1 = (tine1_x - 50, tine1_y - 75)  # left-top corner of the rectangle
            pt2 = (tine1_x + 75, tine1_y + 30)  # right-bottom corner of the rectangle
            cv.rectangle(img=color_img, pt1=pt1, pt2=pt2, color=(0, 0, 255))

            cv.imshow("after adding markings on color", color_img)
            cv.waitKey(1)

    def listener_callback_depth(self, depth_img_msg):
        unchanged_depth_img = None
        manipulated_depth_img = None
        try:
            # passthrough is essentially rendering the image as it sees
            unchanged_depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
            manipulated_depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        if unchanged_depth_img is not None and manipulated_depth_img is not None:
            # convert to np array
            unchanged_depth_img = np.array(unchanged_depth_img, dtype=np.uint32)
            manipulated_depth_img = np.array(manipulated_depth_img, dtype=np.float32)

            # scale/normalize the manipulated depth image to be within 0 and 255 for rendering
            manipulated_depth_img = cv.normalize(src=manipulated_depth_img, dst=None, alpha=0, beta=255,
                                                 norm_type=cv.NORM_MINMAX)
            # print("Min: ", np.min(manipulated_depth_img))
            # print("Max: ", np.max(manipulated_depth_img))
            cv.imshow("after normalizing", manipulated_depth_img)

            cv.circle(img=manipulated_depth_img, center=(400, 100), radius=3, color=(0, 255, 255), thickness=1,
                      lineType=8, shift=0)

            # Fork tines (Note: center = (col, row) format)
            centers = [(359, 275), (367, 274), (375, 273), (381, 272)]
            for center in centers:
                # (0, 0, 255) is red cause BGR
                cv.circle(img=manipulated_depth_img, center=center, radius=3, color=(0, 0, 255), thickness=1,
                          lineType=8, shift=0)

            # Bounding rectangle
            tine1_col, tine1_row = centers[0]
            pt1 = (tine1_col - 50, tine1_row - 55)  # left-top corner of the rectangle
            pt2 = (tine1_col + 75, tine1_row + 30)  # right-bottom corner of the rectangle
            cv.rectangle(img=manipulated_depth_img, pt1=pt1, pt2=pt2, color=(0, 0, 255))

            cv.imshow("after adding markings on depth", manipulated_depth_img)

            # 1. Distance threshold (Note: to index into cv img, we need (row, col) format
            #        -> https://stackoverflow.com/questions/9623435/image-processing-mainly-opencv-indexing-conventions)
            min_dist = unchanged_depth_img[(tine1_row, tine1_col)] - 20  # distance @ left-most tine - 2 cm
            max_dist = unchanged_depth_img[(tine1_row, tine1_col)] + 40  # distance @ left-most tine + 4 cm

            # 2. Consider only the part of the image with the rectangle
            pt1_col, pt1_row = pt1
            pt2_col, pt2_row = pt2

            # create a mask that has all zeros and then change the values between the rectangle to be true
            mask_img = np.zeros_like(unchanged_depth_img, dtype=bool)
            mask_img[pt1_row:pt2_row, pt1_col:pt2_col] = True
            result = unchanged_depth_img
            result[~mask_img] = 0  # anything not in the mask area should be 0

            # result > min_dist -> if true: keeps the result otherwise makes it 0
            result = result & (result > min_dist)
            # result < min_dist -> if true: keeps the result otherwise makes it 0
            result = result & (result < max_dist)

            print("number: ", np.count_nonzero(result))
            cv.waitKey(1)


def main(args=None):
    print("Running food_on_fork2")
    rclpy.init(args=args)
    food_on_fork = FoodOnFork()

    rclpy.spin(food_on_fork)

    food_on_fork.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
