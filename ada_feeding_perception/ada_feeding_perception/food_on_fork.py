#!/usr/bin/env python3
"""
This file defines the FoodOnFork node class, which listens to a topic, /camera/depth/image_rect_raw
and uses the depth image recieved there to calculate the probability of Food on Fork. It then launches
a topic, /food_on_fork to which the probability of Food on Fork is published.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import joblib
from typing import Tuple


class FoodOnFork(Node):
    """
    This file defines the FoodOnFork node class, which listens to a topic, /camera/depth/image_rect_raw
    and uses the depth image recieved there to calculate the probability of Food on Fork. It then launches
    a topic, /food_on_fork to which the probability of Food on Fork is published.
    """

    def __init__(self):
        """
        Initialize FoodOnFork node
        """
        super().__init__("food_on_fork")

        # read all the parameters
        (
            left_top_corner_x,
            left_top_corner_y,
            right_bottom_corner_x,
            right_bottom_corner_y,
            min_dist,
            max_dist,
            model_loc,
            test,
        ) = self.read_params()

        # set the read in parameters
        self.left_top_corner = (left_top_corner_x.value, left_top_corner_y.value)
        self.right_bottom_corner = (
            right_bottom_corner_x.value,
            right_bottom_corner_y.value,
        )
        self.min_dist = min_dist.value
        self.max_dist = max_dist.value

        self.test = test.value
        self.get_logger().info(str(self.test))
        # self.get_logger().info(str(self.left_top_corner))
        # self.get_logger().info(str(self.right_bottom_corner))
        # self.get_logger().info(str(self.min_dist))
        # self.get_logger().info(str(self.max_dist))

        # depth topic subscription
        self.subscription_depth = self.create_subscription(
            Image, "camera/depth/image_rect_raw", self.listener_callback_depth, 1
        )
        self.subscription_depth

        # publisher for FoF vs. no FoF
        self.publisher_depth = self.create_publisher(Float32, "food_on_fork", 1)

        # cv bridge
        self.bridge = CvBridge()

        # initialize the model
        self.model = joblib.load(model_loc.value)

    def read_params(
        self,
    ) -> Tuple[
        Parameter,
        Parameter,
        Parameter,
        Parameter,
        Parameter,
        Parameter,
        Parameter,
        Parameter,
    ]:
        """
        Read the parameters for this node.

        Returns
        -------
        a tuple of:
        left_top_corner_x: x-value of Top-left point of the bounding box rectangle
        left_top_corner_y: y-value of Top-left point of the bounding box rectangle
        right_bottom_corner_x: x-value of Bottom-right point of the bounding box of the rectangle
        right_bottom_corner_y: y-value of Bottom-right point of the bounding box of the rectangle
        min_dist: minimum depth to consider (note that 330 is approx distance to the fork tine)
        max_dist: maximum depth to consider (note that 330 is approx distance to the fork tine)
        model_loc: location of the model
        test: boolean value representing whether this node is testing
        """
        return self.declare_parameters(
            "",
            [
                (
                    "left_top_corner_x",
                    None,
                    ParameterDescriptor(
                        name="left_top_corner_x",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="x-value of Left top corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "left_top_corner_y",
                    None,
                    ParameterDescriptor(
                        name="left_top_corner_y",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="y-value of Left top corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "right_bottom_corner_x",
                    None,
                    ParameterDescriptor(
                        name="right_bottom_corner_x",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="x-value of Right bottom corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "right_bottom_corner_y",
                    None,
                    ParameterDescriptor(
                        name="right_bottom_corner_y",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="y-value of Right bottom corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "min_dist",
                    None,
                    ParameterDescriptor(
                        name="min_dist",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="minimum depth to consider (note that 330 is approx distance to the fork tine)",
                        read_only=True,
                    ),
                ),
                (
                    "max_dist",
                    None,
                    ParameterDescriptor(
                        name="max_dist",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="maximum depth to consider (note that 330 is approx distance to the fork tine)",
                        read_only=True,
                    ),
                ),
                (
                    "model_location",
                    None,
                    ParameterDescriptor(
                        name="model_location",
                        type=ParameterType.PARAMETER_STRING,
                        description="model location",
                        read_only=True,
                    ),
                ),
                (
                    "test",
                    None,
                    ParameterDescriptor(
                        name="test",
                        type=ParameterType.PARAMETER_BOOL,
                        description="testing",
                        read_only=True,
                    ),
                ),
            ],
        )

    def listener_callback_depth(self, depth_img_msg: Image) -> None:
        """
        Calculates and publishes probability indicating whether the provided depth image has food on fork

        Parameters:
        ----------
        depth_img_msg: depth image message from the subscriber
        """
        depth_img = None
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except CvBridgeError as e:
            print(e)

        if depth_img is not None:
            print(depth_img.dtype)
            num_pixels = self.food_on_fork_num_pixels(depth_img)
            if self.test:
                self.get_logger().info("num pixels: " + str(num_pixels))
            prediction = self.predict_food_on_fork(num_pixels)
            if self.test:
                self.get_logger().info("Prediction Probability: " + str(prediction))
            float32msg = Float32()
            float32msg.data = prediction
            self.publisher_depth.publish(float32msg)

    def food_on_fork_num_pixels(self, depth_img: np.ndarray) -> int:
        """
        Calculates the number of pixels in the provided depth image (through a depth image message)

        Parameters:
        ----------
        depth_img: depth image
        left_top_corner: Tuple(int, int): Top-left point of the bounding box rectangle
        right_bottom_corner: Tuple(int, int): Bottom-right point of the bounding box of the rectangle
        min_dist: int: minimum depth to consider (note that 330 is approx distance to the fork tine)
        max_dist: int: maximum depth to consider (note that 330 is approx distance to the fork tine)

        Returns:
        ----------
        number of pixels within the specified parameter range
        """

        # consider the points for the rectangle
        pt1_col, pt1_row = self.left_top_corner
        pt2_col, pt2_row = self.right_bottom_corner

        # create mask that satisfies the rectangle and distance conditions

        # For instance, take a mask = [[F, F, F, F], [F, F, F, F], [F, F, F, F]]
        mask_img = np.zeros_like(depth_img, dtype=bool)
        # The pixels within the rectangular range will be true,
        # resulting in mask = [[F, T, T, F], [F, T, T, F], [F, T, T, F]]
        mask_img[pt1_row:pt2_row, pt1_col:pt2_col] = True
        # The pixels within the depth range will be true,
        # resulting in mask = [[F, F, F, F], [F, T, T, F], [F, F, F, F]]
        mask_img[
            np.logical_not((self.min_dist < depth_img) & (depth_img < self.max_dist))
        ] = False

        return np.count_nonzero(mask_img)

    def predict_food_on_fork(self, num_pixels: int) -> float:
        """
        Calculates the probability of the presence of food on fork

        Parameters:
        ----------
        num_pixels: number of pixels detected based on which the probability is output

        Returns:
        ----------
        probability of the presence of food on fork
        """
        print("num pixels in method: ", num_pixels)
        num_pixels = np.array([[num_pixels]])
        prediction_prob = self.model.predict_proba(num_pixels)
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


if __name__ == "__main__":
    main()
