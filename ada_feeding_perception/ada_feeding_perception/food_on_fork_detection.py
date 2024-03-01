"""
This module contains a ROS2 node that: (a) takes in parameters specifying a FoodOnFork
class to use and kwargs for the class's constructor; (b) exposes a ROS2 service to
toggle the perception algorithm on and off; and (c) when the perception algorithm is
on, subscribes to the depth image topic and publishes the confidence that there is food
on the fork.
"""
# Standard imports
import collections.abc
import os
import threading
from typing import Any, Dict, Tuple

# Third-party imports
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_feeding.helpers import import_from_string
from ada_feeding_msgs.msg import FoodOnForkDetection
from ada_feeding_perception.food_on_fork_detectors import FoodOnForkDetector
from ada_feeding_perception.helpers import (
    get_img_msg_type,
    ros_msg_to_cv2_image,
)


class FoodOnForkDetectionNode(Node):
    """
    A ROS2 node that takes in parameters specifying a FoodOnForkDetector class to use and
    kwargs for the class's constructor, exposes a ROS2 service to toggle the perception
    algorithm on and off, and when the perception algorithm is on, subscribes to the
    depth image topic and publishes the confidence that there is food on the fork.
    """

    # pylint: disable=duplicate-code
    # Much of the logic of this node mirrors FaceDetection. This is fine.
    # pylint: disable=too-many-instance-attributes
    # Needed for multiple publishers/subscribers, model parameters, etc.
    def __init__(self):
        """
        Initializes the FoodOnForkDetection.
        """
        super().__init__("food_on_fork_detection")

        # Load the parameters
        (
            model_class,
            model_path,
            model_dir,
            model_kwargs,
            self.rate_hz,
            self.crop_top_left,
            self.crop_bottom_right,
            self.depth_min_mm,
            self.depth_max_mm,
        ) = self.read_params()

        # Construct the FoodOnForkDetector model
        food_on_fork_class = import_from_string(model_class)
        assert issubclass(
            food_on_fork_class, FoodOnForkDetector
        ), f"Model {model_class} must subclass FoodOnForkDetector"
        self.model = food_on_fork_class(**model_kwargs)
        self.model.crop_top_left = self.crop_top_left
        self.model.crop_bottom_right = self.crop_bottom_right
        if len(model_path) > 0:
            self.model.load(os.path.join(model_dir, model_path))

        # Create the TF buffer, in case the perception algorithm needs it
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.model.tf_buffer = self.tf_buffer

        # Create the service to toggle the perception algorithm on and off
        self.is_on = False
        self.is_on_lock = threading.Lock()
        self.srv = self.create_service(
            SetBool,
            "~/toggle_food_on_fork_detection",
            self.toggle_food_on_fork_detection,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the publisher
        self.pub = self.create_publisher(
            FoodOnForkDetection, "~/food_on_fork_detection", 1
        )

        # Create the CameraInfo subscribers
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "~/camera_info",
            self.camera_info_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the depth image subscriber
        self.cv_bridge = CvBridge()
        self.depth_img = None
        self.depth_img_lock = threading.Lock()
        aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(aligned_depth_topic, self)
        except ValueError as err:
            self.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to Image. {err}"
            )
            aligned_depth_type = Image
        # Subscribe to the depth image
        self.depth_subscription = self.create_subscription(
            aligned_depth_type,
            aligned_depth_topic,
            self.depth_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def read_params(
        self,
    ) -> Tuple[
        str, str, str, Dict[str, Any], float, Tuple[int, int], Tuple[int, int], int, int
    ]:
        """
        Reads the parameters for the FoodOnForkDetection.

        Returns
        -------
        model_class: The FoodOnFork class to use. Must be a subclass of FoodOnFork.
        model_path: The path to the model file. This must be relative to the model_dir
            parameter. Ignored if the empty string.
        model_dir: The directory to load the model from.
        model_kwargs: The keywords to pass to the FoodOnFork class's constructor.
        rate_hz: The rate (Hz) at which to publish.
        crop_top_left: The top left corner of the crop box.
        crop_bottom_right: The bottom right corner of the crop box.
        depth_min_mm: The minimum depth (mm) to consider.
        depth_max_mm: The maximum depth (mm) to consider.
        """
        # Read the model_class
        model_class = self.declare_parameter(
            "model_class",
            descriptor=ParameterDescriptor(
                name="model_class",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The FoodOnFork class to use. Must be a subclass of FoodOnFork."
                ),
                read_only=True,
            ),
        )
        model_class = model_class.value

        # Read the model_path
        model_path = self.declare_parameter(
            "model_path",
            descriptor=ParameterDescriptor(
                name="model_path",
                type=ParameterType.PARAMETER_STRING,
                description=(
                    "The path to the model file. This must be relative to the "
                    "model_dir parameter. Ignored if the empty string."
                ),
                read_only=True,
            ),
        )
        model_path = model_path.value

        # Read the model_dir
        model_dir = self.declare_parameter(
            "model_dir",
            descriptor=ParameterDescriptor(
                name="model_dir",
                type=ParameterType.PARAMETER_STRING,
                description=("The directory to load the model from."),
                read_only=True,
            ),
        )
        model_dir = model_dir.value

        # Read the model_kwargs
        model_kwargs = {}
        model_kws = self.declare_parameter(
            "model_kws",
            descriptor=ParameterDescriptor(
                name="model_kws",
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description=(
                    "The keywords to pass to the FoodOnFork class's constructor."
                ),
                read_only=True,
            ),
        )
        for kw in model_kws.value:
            full_name = f"model_kwargs.{kw}"
            arg = self.declare_parameter(
                full_name,
                descriptor=ParameterDescriptor(
                    name=kw,
                    description="Custom keyword argument for the model.",
                    dynamic_typing=True,
                    read_only=True,
                ),
            )
            if isinstance(arg, collections.abc.Sequence):
                arg = list(arg.value)
            else:
                arg = arg.value
            model_kwargs[kw] = arg

        # Get the rate at which to operate
        rate_hz = self.declare_parameter(
            "rate_hz",
            10.0,
            descriptor=ParameterDescriptor(
                name="rate_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The rate (Hz) at which to publish.",
                read_only=True,
            ),
        )
        rate_hz = rate_hz.value

        # Get the crop box
        crop_top_left = self.declare_parameter(
            "crop_top_left",
            (0, 0),
            descriptor=ParameterDescriptor(
                name="crop_top_left",
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description="The top left corner of the crop box.",
                read_only=True,
            ),
        )
        crop_top_left = crop_top_left.value
        crop_bottom_right = self.declare_parameter(
            "crop_bottom_right",
            (0, 0),
            descriptor=ParameterDescriptor(
                name="crop_bottom_right",
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description="The bottom right corner of the crop box.",
                read_only=True,
            ),
        )
        crop_bottom_right = crop_bottom_right.value

        # Get the depth range
        depth_min_mm = self.declare_parameter(
            "depth_min_mm",
            0,
            descriptor=ParameterDescriptor(
                name="depth_min_mm",
                type=ParameterType.PARAMETER_INTEGER,
                description="The minimum depth (mm) to consider.",
                read_only=True,
            ),
        )
        depth_min_mm = depth_min_mm.value
        depth_max_mm = self.declare_parameter(
            "depth_max_mm",
            20000,
            descriptor=ParameterDescriptor(
                name="depth_max_mm",
                type=ParameterType.PARAMETER_INTEGER,
                description="The maximum depth (mm) to consider.",
                read_only=True,
            ),
        )
        depth_max_mm = depth_max_mm.value

        return (
            model_class,
            model_path,
            model_dir,
            model_kwargs,
            rate_hz,
            crop_top_left,
            crop_bottom_right,
            depth_min_mm,
            depth_max_mm,
        )

    def toggle_food_on_fork_detection(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """
        Toggles the perception algorithm on and off.

        Parameters
        ----------
        request: The request to toggle the perception algorithm on and off.
        response: The response to toggle the perception algorithm on and off.

        Returns
        -------
        response: The response to toggle the perception algorithm on and off.
        """

        self.get_logger().info(f"Incoming service request. data: {request.data}")
        response.success = False
        response.message = f"Failed to set is_on to {request.data}"
        with self.is_on_lock:
            self.is_on = request.data
            response.success = True
            response.message = f"Successfully set is_on to {request.data}"
        return response

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Callback for the camera info. Note that we assume CameraInfo never
        changes, and therefore destroy the subscriber after the first message.

        Parameters
        ----------
        msg: The camera info message.
        """
        if self.model.camera_info is None:
            self.model.camera_info = msg
        self.destroy_subscription(self.camera_info_sub)

    def depth_callback(self, msg: Image) -> None:
        """
        Callback for the depth image.

        Parameters
        ----------
        msg: The depth image message.
        """
        with self.depth_img_lock:
            self.depth_img = msg

    def run(self) -> None:
        """
        Runs the FoodOnForkDetection.
        """
        rate = self.create_rate(self.rate_hz)
        while rclpy.ok():
            # Loop at the specified rate
            rate.sleep()

            # Check if food on fork detection is on
            with self.is_on_lock:
                is_on = self.is_on
            if not is_on:
                continue

            # Create the FoodOnForkDetection message
            food_on_fork_detection_msg = FoodOnForkDetection()

            # Get the latest depth image
            with self.depth_img_lock:
                depth_img_msg = self.depth_img
                self.depth_img = None
            if depth_img_msg is None:
                continue
            food_on_fork_detection_msg.header = depth_img_msg.header

            # Convert the depth image to a cv2 image
            depth_img_cv2 = ros_msg_to_cv2_image(depth_img_msg, self.cv_bridge)
            depth_img_cv2 = depth_img_cv2[
                self.crop_top_left[1] : self.crop_bottom_right[1],
                self.crop_top_left[0] : self.crop_bottom_right[0],
            ]
            depth_img_cv2 = np.where(
                (depth_img_cv2 >= self.depth_min_mm)
                & (depth_img_cv2 <= self.depth_max_mm),
                depth_img_cv2,
                0,
            )
            X = np.expand_dims(depth_img_cv2, axis=0)

            # Get the probability that there is food on the fork
            try:
                proba = self.model.predict_proba(X)[0]
                food_on_fork_detection_msg.probability = proba
            # pylint: disable=broad-except
            # This is necessary because we don't know what exceptions the model
            # might raise.
            except Exception as err:
                err_str = f"Error predicting food on fork: {err}"
                self.get_logger().error(err_str)
                food_on_fork_detection_msg.probability = -1.0
                food_on_fork_detection_msg.message = err_str

            # Publish the FoodOnForkDetection message
            self.pub.publish(food_on_fork_detection_msg)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    food_on_fork_detection = FoodOnForkDetectionNode()
    executor = MultiThreadedExecutor(num_threads=4)

    # Spin in the background since detecting faces will block
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(food_on_fork_detection,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run face detection
    try:
        food_on_fork_detection.run()
    except KeyboardInterrupt:
        pass

    # Terminate this node
    food_on_fork_detection.destroy_node()
    rclpy.shutdown()
    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
