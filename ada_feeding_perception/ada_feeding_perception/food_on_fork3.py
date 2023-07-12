#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import csv
from datetime import datetime
import time


class FoodOnFork(Node):
    """
    This class is updated last on 6/29/2023! It has code that colors the pixels that
    are in the range of where the food is
    """
    def __init__(self):
        super().__init__('food_on_fork')
        self.most_recent_color_img = None
        self.data = [("Timestamp", "Number of Pixels in Range")]

        # color topic subscription
        self.subscription_color = self.create_subscription(
            Image,
            '/camera2/color/image_raw',
            self.listener_callback_color,  # when the subscriber gets something, it calls the callback function
            1
        )
        self.subscription_color

        # depth topic subscription
        self.subscription_depth = self.create_subscription(
            Image,
            # 'camera/aligned_depth_to_color/image_raw',
            'camera2/depth/image_rect_raw',
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
            changed_depth_img = np.copy(depth_img)
        except CvBridgeError as e:
            print(e)

        if depth_img is not None:
            # convert to np array
            changed_depth_img = np.asarray(changed_depth_img)

            # Remove anything that is greater than 1500 mm
            changed_depth_img = np.where(changed_depth_img > 1500, 1500, changed_depth_img)

            # normalize the image to be between 0 and 255, where 0 (black) is close and
            # 255 (white) is farther away
            depth_img_normalized = cv.normalize(src=changed_depth_img, dst=None, alpha=0, beta=255,
                                                norm_type=cv.NORM_MINMAX)

            # convert to uint8 type for display purposes (puts it into range of 0 and 255 as well)
            depth_img_normalized = depth_img_normalized.astype(np.uint8)

            # invert the image so that 0 (black) is far and 255 (white) is close
            depth_img_normalized = 255 - depth_img_normalized

            # convert to three channel image because you want to use that to "and" that with the original
            # image to be able to overlay the two images correctly
            depth_img_color_img = cv.cvtColor(depth_img_normalized, cv.COLOR_GRAY2BGR)

            # Fork tines (Note: center = (col, row) format)
            centers = [(359, 275), (367, 274), (375, 273), (381, 272)]

            # consider only the points within the rectangle
            tine1_col, tine1_row = centers[0]
            # pt1 = (tine1_col - 50, tine1_row - 55)  # left-top corner of the rectangle
            # pt2 = (tine1_col + 75, tine1_row + 30)  # right-bottom corner of the rectangle
            pt1 = (297, 248)
            pt2 = (422, 332)
            pt1_col, pt1_row = pt1
            pt2_col, pt2_row = pt2

            # consider the distances between min and max depth
            min_dist = 330 - 20  # approx fork tine location distance - 2 cm
            max_dist = 330 + 40  # approx fork tine location distance + 4 cm

            # create mask that satisfies the rectangle and distance conditions
            mask_img = np.zeros_like(depth_img, dtype=bool)
            # print(mask_img.shape)
            mask_img[pt1_row:pt2_row, pt1_col:pt2_col] = True
            mask_img[np.logical_not((min_dist < depth_img) & (depth_img < max_dist))] = False

            # print(depth_img_color_img.shape)
            # print(depth_img_color_img)
            depth_img_color_img[mask_img] = (0, 0, 255)

            # append to the data list
            self.data.append([datetime.now(), np.count_nonzero(mask_img)])
            print(self.data)

            with open(r'/home/atharva2/atharvak_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/data/Test_7-7-23.csv', 'w', newline='') as f:
                f_write = csv.writer(f)

                for v in self.data:
                    f_write.writerow(v)

            cv.imshow("changed depth image after depth", depth_img_color_img)

            if self.most_recent_color_img is not None:
                cv.imshow("color img", self.most_recent_color_img)
            cv.waitKey(1)


def main(args=None):
    print("Running CheckAlignmentImgOverlay")
    rclpy.init(args=args)
    food_on_fork = FoodOnFork()
    executor = MultiThreadedExecutor()
    rclpy.spin(food_on_fork, executor=executor)

    food_on_fork.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
