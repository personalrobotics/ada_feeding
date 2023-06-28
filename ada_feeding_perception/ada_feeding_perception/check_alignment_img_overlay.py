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
    """
    This class checks if the depth image and the rgb image are aligned. It also
    checks if the output depth values for the fork tines are in-sync with the
    actual values

    Attributes:
        - most_recent_color_img: Color image that was most recently received from the color topic
        - overlay_img: boolean value that tells whether or not to produce an image overlaid on depth image
        - find_dist_at_tine: boolean value that tells whether or not to produce a distance to the fork tines

    There are also callback functions defined to be used when listening to the color topic
    and aligned depth topic
    """
    def __init__(self):
        super().__init__('food_on_fork')
        self.most_recent_color_img = None
        self.overlay_img = True
        self.find_dist_at_tine = True
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
            changed_depth_img = np.copy(depth_img)
        except CvBridgeError as e:
            print(e)

        if depth_img is not None:
            cv.imshow("before anything", changed_depth_img)
            changed_depth_img = np.asarray(changed_depth_img)  # convert to np array

            # The fork is located about 30cm away from the camera.
            # When I performed np.max(changed_depth_img), I got like 65000+, which is mm and
            # when converted to cm, that is 6500 cm, which is completely unecessary. We only care
            # about 1.5 meters away. So, I decided to cut anything greater than 1500 mm (1.5 m) to 1500 mm!
            # Helps with visualization!
            changed_depth_img = np.where(changed_depth_img > 1500, 1500, changed_depth_img)

            # print("max dist: ", np.max(changed_depth_img)) # should be 1500!

            cv.imshow("after making 1500 plus 1500",
                      changed_depth_img)  # expected to be black because not in the correct type

            # convert the depth image to display something between 0 and 255 for rendering purposes
            # Note that 0 (black) is close and 255 (white) is farther away!
            depth_img_normalized = cv.normalize(src=changed_depth_img, dst=None, alpha=0, beta=255,
                                                norm_type=cv.NORM_MINMAX)

            # convert to uint8, which has bounds between 0 and 255!!
            depth_img_normalized = depth_img_normalized.astype(np.uint8)
            cv.imshow("Normalized depth image", depth_img_normalized)

            # invert the image for purposes of using it a as a mask for color image on top of this one!
            depth_img_normalized = 255 - depth_img_normalized
            cv.imshow("After inversion", depth_img_normalized)

            # convert to three channel image because you want to use that to "and" that with the original
            # image to be able to overlay the two images correctly
            depth_img_color_img = cv.cvtColor(depth_img_normalized, cv.COLOR_GRAY2BGR)
            cv.imshow("After converting to color", depth_img_color_img)

            # overlay the color image on top of this converted depth image
            if self.overlay_img:
                if self.most_recent_color_img is not None:
                    overlaid_img = cv.bitwise_and(self.most_recent_color_img, depth_img_color_img)
                    cv.imshow("Overlaid Image", overlaid_img)
                else:
                    print("most recent image is none :(")

            # find the distance for a hardcoded tine position
            if self.find_dist_at_tine:
                depth_color_img_changed = np.copy(depth_img_color_img)
                # Fork tines (Note: center = (col, row) format)
                centers = [(359, 275), (367, 274), (375, 273), (381, 272)]
                for center in centers:
                    # (0, 0, 255) is red cause BGR
                    cv.circle(img=depth_color_img_changed, center=center, radius=3, color=(0, 0, 255), thickness=1,
                              lineType=8, shift=0)

                # Bounding rectangle
                tine1_col, tine1_row = centers[0]
                pt1 = (tine1_col - 50, tine1_row - 55)  # left-top corner of the rectangle
                pt2 = (tine1_col + 75, tine1_row + 30)  # right-bottom corner of the rectangle
                cv.rectangle(img=depth_color_img_changed, pt1=pt1, pt2=pt2, color=(0, 0, 255))

                # position
                for i in range(len(centers)):
                    tine_col, tine_row = centers[i]
                    print(i + 1, ": ", depth_img[tine_row, tine_col])
                cv.imshow("after adding markings on depth", depth_color_img_changed)

            cv.waitKey(1)


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
