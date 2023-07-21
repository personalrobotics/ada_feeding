#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import joblib


class FoodOnFork(Node):
    """
    Class FoodOnFork inherits from the Node class. This is a node that subscribes to the
    raw depth image and publishes to a topic `food_on_fork`. The published messages are of type
    Float32. The published messages return the probability of the presence of food on the fork.
    """
    def __init__(self):
        """
        Initialize FoodOnFork node
        """
        super().__init__('food_on_fork')

        # depth topic subscription
        self.subscription_depth = self.create_subscription(
            Image,
            'camera/depth/image_rect_raw',
            self.listener_callback_depth,
            1
        )
        self.subscription_depth

        # publisher for FoF vs. no FoF
        self.publisher_depth = self.create_publisher(
            Float32,
            'food_on_fork',
            1
        )

        # cv bridge
        self.bridge = CvBridge()

        # initialize the model
        self.model = joblib.load("/home/atharva2/atharvak_node_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/logistic_reg_model.pkl")

    def listener_callback_depth(self, depth_img_msg):
        """
        Calculates and publishes probability indicating whether the provided depth image has food on fork

        Parameters:
            depth_img_msg: depth image message from the subscriber
        """
        depth_img = None
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        if depth_img is not None:
            num_pixels = self.food_on_fork_num_pixels(depth_img)
            # print("num pixels: ", num_pixels)
            prediction = self.predict_food_on_fork(num_pixels)
            # print("prediction: ", prediction)
            float32msg = Float32()
            float32msg.data = prediction
            self.publisher_depth.publish(float32msg)

    def food_on_fork_num_pixels(self, depth_img,
                                left_top_corner=(297, 248),
                                right_bottom_corner=(422, 332),
                                min_dist=(330 - 20),
                                max_dist=(330 + 40)) -> int:
        """
        Calculates the number of pixels in the provided depth image (through a depth image message)

        Parameters:
            depth_img: depth image
            left_top_corner: Tuple(int, int): Top-left point of the bounding box rectangle
            right_bottom_corner: Tuple(int, int): Bottom-right point of the bounding box of the rectangle
            min_dist: int: minimum depth to consider (note that 330 is approx distance to the fork tine)
            max_dist: int: maximum depth to consider (note that 330 is approx distance to the fork tine)

        Returns:
            number of pixels within the specified parameter range
        """

        # consider the points for the rectangle
        pt1_col, pt1_row = left_top_corner
        pt2_col, pt2_row = right_bottom_corner

        # create mask that satisfies the rectangle and distance conditions
        mask_img = np.zeros_like(depth_img, dtype=bool)
        mask_img[pt1_row:pt2_row, pt1_col:pt2_col] = True
        mask_img[np.logical_not((min_dist < depth_img) & (depth_img < max_dist))] = False

        return np.count_nonzero(mask_img)

    def predict_food_on_fork(self, num_pixels) -> float:
        """
        Calculates the probability of the presence of food on fork

        Parameters:
            num_pixels: number of pixels detected based on which the probability is output

        Returns:
            probability of the presence of food on fork
        """
        # print("num pixels in method: ", num_pixels)
        num_pixels = np.array([[num_pixels]])
        num_pixels_reshape = num_pixels.reshape(-1, 1)
        prediction_prob = self.model.predict_proba(num_pixels_reshape)
        # print("Pred_prob", prediction_prob[0][1])

        # prediction_prob is a matrix that looks like: [[percentage1, percentage2]]
        # Note that percentage1 is the percent probability that the predicted value is 0 (no food on fork)
        # and percentage2 is the percent probability that the predicted value is 1 (food on fork)
        # we care about the probability that there is food on fork. As such, [0][1] makes sense!
        return float(prediction_prob[0][1])


def main(args=None):
    print("Running Food On Fork Node")
    rclpy.init(args=args)
    food_on_fork = FoodOnFork()
    executor = MultiThreadedExecutor()
    rclpy.spin(food_on_fork, executor=executor)

    food_on_fork.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
