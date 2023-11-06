#!/usr/bin/env python3
"""
This file defines the FoodOnFork node class, which listens to a topic, /camera/depth/image_rect_raw
and uses the depth image recieved there to calculate the probability of Food on Fork. It then
launches a topic, /food_on_fork to which the probability of Food on Fork is published.
"""

# Standard imports
import cv2 as cv  # needed sometimes
from cv_bridge import CvBridge, CvBridgeError
import joblib
import numpy as np
import numpy.typing as npt
from typing import Optional
from typing import Tuple

# Third-Party imports
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from rclpy.node import Node
from rclpy.parameter import Parameter

# Local imports
import helpers


class FoodOnFork(Node):
    """
    This file defines the FoodOnFork node class, which listens to a topic,
    /camera/depth/image_rect_raw and uses the depth image recieved there to calculate the
    probability of Food on Fork. It then launches a topic, /food_on_fork to which the probability
    of Food on Fork is published.
    """

    def __init__(self):
        """
        Initialize FoodOnFork node
        """
        super().__init__("food_on_fork")

        # read all the parameters
        (
            top_left_corner_x,
            top_left_corner_y,
            bottom_right_corner_x,
            bottom_right_corner_y,
            min_depth,
            max_depth,
            model_loc,
            single_feature,
            test,
        ) = self.read_params()

        # set the read in parameters
        self.left_top_corner = (top_left_corner_x.value, top_left_corner_y.value)
        self.right_bottom_corner = (
            bottom_right_corner_x.value,
            bottom_right_corner_y.value,
        )
        self.min_depth = min_depth.value
        self.max_depth = max_depth.value

        self.test = test.value
        self.get_logger().info(str(self.test))

        # depth topic subscription
        # self.subscription_depth = self.create_subscription(
        #     Image, "/camera/depth/image_rect_raw", self.listener_callback_depth, 1
        # )
        self.subscription_depth = self.create_subscription(
            Image,
            "~/aligned_depth",
            self.listener_callback_depth,
            1,
        )
        # self.subscription_depth = self.create_subscription(
        #     Image, "~/image", self.listener_callback_depth, 1
        # )

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
        top_left_corner_x: x-value of Top-left point of the bounding box rectangle
        top_left_corner_y: y-value of Top-left point of the bounding box rectangle
        bottom_right_corner_x: x-value of Bottom-right point of the bounding box of the rectangle
        bottom_right_corner_y: y-value of Bottom-right point of the bounding box of the rectangle
        min_depth: minimum depth to consider (note that 330 is approx distance to the fork tine)
        max_depth: maximum depth to consider (note that 330 is approx distance to the fork tine)
        model_loc: location of the model
        single_feature: boolean value representing the usage of single feature Logistic Reg vs. multi-feature NB
        test: boolean value representing whether this node is testing
        """
        return self.declare_parameters(
            "",
            [
                (
                    "top_left_corner_x",
                    None,
                    ParameterDescriptor(
                        name="top_left_corner_x",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="x-value of top-left corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "top_left_corner_y",
                    None,
                    ParameterDescriptor(
                        name="top_left_corner_y",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="y-value of top-left corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "bottom_right_corner_x",
                    None,
                    ParameterDescriptor(
                        name="bottom_right_corner_x",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="x-value of bottom-right corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "bottom_right_corner_y",
                    None,
                    ParameterDescriptor(
                        name="bottom_right_corner_y",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="y-value of bottom-right corner of the rectangle",
                        read_only=True,
                    ),
                ),
                (
                    "min_depth",
                    None,
                    ParameterDescriptor(
                        name="min_depth",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="minimum depth to consider (note that 330 is approx distance "
                        "to the fork tine)",
                        read_only=True,
                    ),
                ),
                (
                    "max_depth",
                    None,
                    ParameterDescriptor(
                        name="max_depth",
                        type=ParameterType.PARAMETER_INTEGER,
                        description="maximum depth to consider (note that 330 is approx distance "
                        "to the fork tine)",
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
                    "single_feature",
                    None,
                    ParameterDescriptor(
                        name="single_feature",
                        type=ParameterType.PARAMETER_BOOL,
                        description="whether or not the we are using a model with a single feature",
                        read_only=True,
                    ),
                ),
                (
                    "test",
                    None,
                    ParameterDescriptor(
                        name="test",
                        type=ParameterType.PARAMETER_BOOL,
                        description="to indicate whether the node is running under test mode",
                        read_only=True,
                    ),
                ),
            ],
        )

    def listener_callback_depth(self, depth_img_msg: Image) -> None:
        """
        Calculates and publishes probability indicating whether the provided depth image has food
        on fork

        Parameters:
        ----------
        depth_img_msg: depth image message from the subscriber
        """
        depth_img = None
        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        except CvBridgeError as e:
            print(e)
            return

        if self.single_feature:
            # Only number of pixels is the input (single feature):
            num_pixels = helpers.food_on_fork_featurizer_num_pixels(
                depth_img,
                self.left_top_corner,
                self.right_bottom_corner,
                self.min_depth,
                self.max_depth,
            )
            if self.test:
                self.get_logger().info("num pixels: " + str(num_pixels))
            prediction = self.predict_food_on_fork(num_pixels)
            if self.test:
                self.get_logger().info("Prediction Probability: " + str(prediction))
        else:
            # when pixels are individually considered as features

            # Tuning the rectangle
            # depth_img_copy = np.copy(depth_img)
            # cv.rectangle(depth_img_copy, (317, 238), (445, 322), (255, 0, 0))
            # cv.imshow("aligned_depthImg", depth_img_copy)

            cropped_img_np = helpers.food_on_fork_featurizer_all_pixels(
                depth_img,
                self.left_top_corner,
                self.right_bottom_corner,
                self.min_depth,
                self.max_depth,
            )
            x_test = cropped_img_np.flatten()

            # Get the prediction, which is a float value
            prediction = self.predict_food_on_fork(x_test=x_test)

            if self.test:
                self.get_logger().info(str(prediction))

            # cv.imshow("cropped_img_preds", helpers.normalize_to_uint8(cropped_img_np))
            # cv.waitKey(1)

        float32msg = Float32()
        float32msg.data = prediction
        self.publisher_depth.publish(float32msg)

    def predict_food_on_fork(
        self, num_pixels: Optional[int] = None, x_test: Optional[npt.NDArray] = None
    ) -> float:
        """
        Calculates the probability of the presence of food on fork based on the provided parameters.
        If num_pixels is None, then it predicts a probability based on the Categorical NB approach.
        If

        Parameters:
        ----------
        num_pixels: number of pixels detected based on which the probability is output
        x_test: Flattened array with whether each pixel is in the range

        Returns:
        ----------
        probability of the presence of food on fork; if both parameters are None or not passed in, then there will be a
        default value of -5.0 returned, indicating an error
        """
        if num_pixels is not None:
            # Logistic Regression Approach
            print("num pixels in method: ", num_pixels)
            num_pixels = np.array([[num_pixels]])
            prediction_prob = self.model.predict_proba(num_pixels)

            # prediction_prob is a matrix that looks like: [[percentage1, percentage2]] Note that
            # percentage1 is the percent probability that the predicted value is 0 (no food on
            # fork) and percentage2 is the percent probability that the predicted value is 1 (
            # food on fork) we care about the probability that there is food on fork. As such,
            # [0][1] makes sense!
            return float(prediction_prob[0][1])
        if x_test is not None:
            # Categorical NB approach
            x_test_reshape = x_test.reshape(
                1, -1
            )  # such that we have (num_images, num_features)
            prediction_prob = self.model.predict_proba(x_test_reshape)
            return float(prediction_prob[0][1])

        return -5.0


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
