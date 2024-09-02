"""
This module contains a ROS2 node that: (a) takes in parameters specifying a FoodOnFork
class to use and kwargs for the class's constructor; (b) exposes a ROS2 service to
toggle the perception algorithm on and off; and (c) when the perception algorithm is
on, subscribes to the depth image topic and publishes the confidence that there is food
on the fork.
"""
# Standard imports
import collections
import os
import threading
from typing import Any, Dict, Tuple

# Third-party imports
from cv_bridge import CvBridge
import cv2
import numpy as np
import numpy.typing as npt
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_srvs.srv import SetBool
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Local imports
from ada_feeding.helpers import import_from_string
from ada_feeding_msgs.msg import FoodOnForkDetection
from ada_feeding_perception.food_on_fork_detectors import FoodOnForkDetector
from ada_feeding_perception.helpers import (
    cv2_image_to_ros_msg,
    get_img_msg_type,
    ros_msg_to_cv2_image,
)
from ada_feeding_perception.ada_feeding_perception_node import ADAFeedingPerceptionNode
from .depth_post_processors import (
    create_spatial_post_processor,
    create_temporal_post_processor,
    post_processor_chain,
)


class FoodOnForkDetectionNode:
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
    def __init__(
        self,
        node: ADAFeedingPerceptionNode,
    ):
        """
        Initializes the FoodOnForkDetection.

        Parameters
        ----------
        node : ADAFeedingPerceptionNode
            The node that contains all functionality to get camera images (RGB and depth)
            and camera info.
        """
        self._node = node

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
            temporal_window_size,
            spatial_num_pixels,
            self.viz,
            self.viz_upper_thresh,
            self.viz_lower_thresh,
            rgb_image_buffer,
            no_fof_points_offset,
        ) = self.read_params()

        # Create the post-processors
        self.cv_bridge = CvBridge()
        self.post_processor = post_processor_chain(
            [create_temporal_post_processor, create_spatial_post_processor],
            [
                {
                    "temporal_window_size": temporal_window_size,
                    "bridge": self.cv_bridge,
                },
                {"spatial_num_pixels": spatial_num_pixels, "bridge": self.cv_bridge},
            ],
            compress=None,
        )

        # Construct the FoodOnForkDetector model
        food_on_fork_class = import_from_string(model_class)
        assert issubclass(
            food_on_fork_class, FoodOnForkDetector
        ), f"Model {model_class} must subclass FoodOnForkDetector"
        self.model = food_on_fork_class(**model_kwargs)
        self.model.crop_top_left = self.crop_top_left
        self.model.crop_bottom_right = self.crop_bottom_right
        if len(model_path) > 0:
            self.model.load(
                os.path.join(model_dir, model_path),
                no_fof_points_offset=no_fof_points_offset,
            )

        # Create the TF buffer, in case the perception algorithm needs it
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self._node)
        self.model.tf_buffer = self.tf_buffer

        # Create the service to toggle the perception algorithm on and off
        self.is_on = False
        self.is_on_lock = threading.Lock()
        self.srv = self._node.create_service(
            SetBool,
            "~/toggle_food_on_fork_detection",
            self.toggle_food_on_fork_detection,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the publisher
        self.pub = self._node.create_publisher(
            FoodOnForkDetection, "~/food_on_fork_detection", 1
        )

        # Create the CameraInfo subscribers
        self.camera_info_topic = "~/camera_info"
        self._node.add_subscription(
            CameraInfo,
            self.camera_info_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Create the depth image subscriber
        self.depth_img_cv2 = None
        self.depth_img_header = None
        self.aligned_depth_topic = "~/aligned_depth"
        try:
            aligned_depth_type = get_img_msg_type(self.aligned_depth_topic, self._node)
        except ValueError as err:
            self._node.get_logger().error(
                f"Error getting type of depth image topic. Defaulting to CompressedImage. {err}"
            )
            aligned_depth_type = CompressedImage
        # Subscribe to the depth image
        self._node.add_subscription(
            aligned_depth_type,
            self.aligned_depth_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # If the visualization flag is set, create a subscriber to the RGB image
        # and publisher for the RGB visualization
        if self.viz:
            self.rgb_pub = self._node.create_publisher(
                Image, "~/food_on_fork_detection_img", 1
            )
            self.rgb_image_topic = "~/image"
            try:
                image_type = get_img_msg_type(self.rgb_image_topic, self._node)
            except ValueError as err:
                self._node.get_logger().error(
                    f"Error getting type of image topic. Defaulting to CompressedImage. {err}"
                )
                image_type = CompressedImage
            self._node.add_subscription(
                image_type,
                self.rgb_image_topic,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
                callback_group=MutuallyExclusiveCallbackGroup(),
                num_msgs=rgb_image_buffer,
            )

    def read_params(
        self,
    ) -> Tuple[
        str,
        str,
        str,
        Dict[str, Any],
        float,
        Tuple[int, int],
        Tuple[int, int],
        int,
        int,
        int,
        int,
        bool,
        float,
        float,
        int,
        Tuple[float, float, float],
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
        temporal_window_size: The size of the temporal window for post-processing.
            Disabled by default.
        spatial_num_pixels: The number of pixels for the spatial post-processor.
            Disabled by default.
        viz: Whether to publish a visualization of the result as an RGB image.
        viz_upper_thresh: The upper threshold for declaring FoF in the viz.
        viz_lower_thresh: The lower threshold for declaring FoF in the viz.
        rgb_image_buffer: The number of RGB images to store at a time for visualization.
        no_fof_points_offset: The offset to add to the "no food-on-fork" points. Only used
            in the FoodOnForkDistanceToNoFOFDetector model_class.
        """
        # pylint: disable=too-many-locals
        # There are many parameters to load.

        # Read the model_class
        model_class = self._node.declare_parameter(
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
        model_path = self._node.declare_parameter(
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
        model_dir = self._node.declare_parameter(
            "food_on_fork_detection_model_dir",
            descriptor=ParameterDescriptor(
                name="food_on_fork_detection_model_dir",
                type=ParameterType.PARAMETER_STRING,
                description=("The directory to load the model from."),
                read_only=True,
            ),
        )
        model_dir = model_dir.value

        # Read the model_kwargs
        model_kwargs = {}
        model_kws = self._node.declare_parameter(
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
            arg = self._node.declare_parameter(
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
        rate_hz = self._node.declare_parameter(
            "food_on_fork_detection_rate_hz",
            10.0,
            descriptor=ParameterDescriptor(
                name="food_on_fork_detection_rate_hz",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The rate (Hz) at which to publish.",
                read_only=True,
            ),
        )
        rate_hz = rate_hz.value

        # Get the crop box
        crop_top_left = self._node.declare_parameter(
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
        crop_bottom_right = self._node.declare_parameter(
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
        depth_min_mm = self._node.declare_parameter(
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
        depth_max_mm = self._node.declare_parameter(
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

        # Configure the post-processors
        temporal_window_size = self._node.declare_parameter(
            "temporal_window_size",
            1,
            descriptor=ParameterDescriptor(
                name="temporal_window_size",
                type=ParameterType.PARAMETER_INTEGER,
                description="The size of the temporal window for post-processing. Disabled by default.",
                read_only=True,
            ),
        )
        temporal_window_size = temporal_window_size.value
        spatial_num_pixels = self._node.declare_parameter(
            "spatial_num_pixels",
            1,
            descriptor=ParameterDescriptor(
                name="spatial_num_pixels",
                type=ParameterType.PARAMETER_INTEGER,
                description="The number of pixels for the spatial post-processor. Disabled by default.",
                read_only=True,
            ),
        )
        spatial_num_pixels = spatial_num_pixels.value

        # Get the visualization parameters
        viz = self._node.declare_parameter(
            "food_on_fork_detection_viz",
            False,
            descriptor=ParameterDescriptor(
                name="food_on_fork_detection_viz",
                type=ParameterType.PARAMETER_BOOL,
                description="Whether to publish a visualization of the result as an RGB image.",
                read_only=True,
            ),
        )
        viz = viz.value
        viz_upper_thresh = self._node.declare_parameter(
            "viz_upper_thresh",
            0.5,
            descriptor=ParameterDescriptor(
                name="viz_upper_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The upper threshold for declaring FoF in the viz.",
                read_only=True,
            ),
        )
        viz_upper_thresh = viz_upper_thresh.value
        viz_lower_thresh = self._node.declare_parameter(
            "viz_lower_thresh",
            0.5,
            descriptor=ParameterDescriptor(
                name="viz_lower_thresh",
                type=ParameterType.PARAMETER_DOUBLE,
                description="The lower threshold for declaring FoF in the viz.",
                read_only=True,
            ),
        )
        viz_lower_thresh = viz_lower_thresh.value
        rgb_image_buffer = self._node.declare_parameter(
            "rgb_image_buffer",
            30,
            descriptor=ParameterDescriptor(
                name="rgb_image_buffer",
                type=ParameterType.PARAMETER_INTEGER,
                description=(
                    "The number of RGB images to store at a time for visualization. Default: 30"
                ),
                read_only=True,
            ),
        )
        rgb_image_buffer = rgb_image_buffer.value

        # Get the no_fof_points_offset
        no_fof_points_offset = self._node.declare_parameter(
            "no_fof_points_offset",
            [0.0, 0.0, 0.0],
            descriptor=ParameterDescriptor(
                name="no_fof_points_offset",
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description=(
                    "The offset to add to the 'no food-on-fork' points. Only used in the "
                    "FoodOnForkDistanceToNoFOFDetector model_class."
                ),
                read_only=True,
            ),
        )
        no_fof_points_offset = tuple(no_fof_points_offset.value)

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
            temporal_window_size,
            spatial_num_pixels,
            viz,
            viz_upper_thresh,
            viz_lower_thresh,
            rgb_image_buffer,
            no_fof_points_offset,
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

        self._node.get_logger().info(
            f"Incoming service request for food-on-fork detection. data: {request.data}"
        )
        response.success = False
        response.message = f"Failed to set is_on to {request.data}"
        with self.is_on_lock:
            self.is_on = request.data
            response.success = True
            response.message = f"Successfully set is_on to {request.data}"
        return response

    def visualize_result(
        self, result: FoodOnForkDetection, t: npt.NDArray, debug: bool = True
    ) -> None:
        """
        Annotates the nearest RGB image message with the result and publishes it.

        Parameters
        ----------
        result: The result of the food on fork detection.
        t: The transform(s) used in the detection. Size (N, 4, 4) where N is the
            number of transforms.
        debug: Whether to overlay additional debug information on the image.
        """
        # Get the RGB image with timestamp closest to the depth image
        img_buffer = self._node.get_all_msgs(self.rgb_image_topic)
        img_msg = None
        # At the end of this for loop, img_message will be the most
        # recent image that is older than the depth image, or the
        # oldest image if there are no images older than the depth
        # image.
        for i, img_msg in enumerate(img_buffer):
            img_msg_stamp = (
                img_msg.header.stamp.sec + img_msg.header.stamp.nanosec * 1e-9
            )
            result_stamp = result.header.stamp.sec + result.header.stamp.nanosec * 1e-9
            if img_msg_stamp > result_stamp:
                if i > 0:
                    img_msg = img_buffer[i - 1]
                break
        # If img_msg is None, that means we haven't received an RGB image yet
        if img_msg is None:
            return

        # Convert the RGB image to a cv2 image
        img_cv2 = ros_msg_to_cv2_image(img_msg, self.cv_bridge)

        # Allow the model to overlay additional debug information on the image
        if debug:
            img_cv2 = self.model.overlay_debug_info(img_cv2, t)

        # Get the message to write on the image
        proba = result.probability
        status = result.status
        if proba > self.viz_upper_thresh:
            pred = "Food on Fork"
            color = (0, 255, 0)
        elif (
            proba <= self.viz_lower_thresh
            or status == FoodOnForkDetection.ERROR_TOO_FEW_POINTS
        ):
            pred = "No Food on Fork"
            color = (0, 0, 255)
        elif status == FoodOnForkDetection.SUCCESS:
            pred = "Uncertain (Ask User)"
            color = (255, 0, 0)
        else:
            pred = "Unknown Error"
            color = (255, 255, 255)
        msg = f"{pred}: {proba:.2f}"

        # Write the message on the top-left corner of the image
        cv2.putText(
            img_cv2,
            msg,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv2.LINE_AA,
        )

        # Add a rectangular border around the image in the specified color
        cv2.rectangle(
            img_cv2,
            (0, 0),
            (img_cv2.shape[1], img_cv2.shape[0]),
            color,
            10,
        )

        # Also add a rectangle around the crop box
        cv2.rectangle(
            img_cv2,
            self.crop_top_left,
            self.crop_bottom_right,
            color,
            2,
        )

        # Publish the image
        self.rgb_pub.publish(
            cv2_image_to_ros_msg(img_cv2, compress=False, bridge=self.cv_bridge)
        )

    def run(self) -> None:
        """
        Runs the FoodOnForkDetection.
        """
        rate = self._node.create_rate(self.rate_hz)
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

            # Get the latest camera info msg
            if self.model.camera_info is None:
                camera_info = self._node.get_latest_msg(self.camera_info_topic)
                if camera_info is not None:
                    self.model.camera_info = camera_info

            # Get the latest depth image
            depth_msg = self._node.get_latest_msg(self.aligned_depth_topic)
            depth_img_cv2, depth_img_header = self.post_processor(depth_msg)
            if depth_img_cv2 is None:
                continue
            food_on_fork_detection_msg.header = depth_img_header

            # Convert the depth image to a cv2 image, crop it, and remove depth
            # values outside the range of interest
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

            # Get the desired transform(s)
            transforms = FoodOnForkDetector.get_transforms(
                self.model.transform_frames,
                self.tf_buffer,
            )
            if np.count_nonzero(transforms) == 0:
                self._node.get_logger().warning(
                    (
                        f"Failed to get transform {self.model.transform_frames}. "
                        "Food-on-fork detection model will not work."
                    ),
                    throttle_duration_sec=1.0,
                )
                continue
            t = np.expand_dims(transforms, 0)

            # Get the probability that there is food on the fork
            try:
                proba, status = self.model.predict_proba(X, t)
                proba = proba[0]
                status = int(status[0])
                food_on_fork_detection_msg.probability = proba
                food_on_fork_detection_msg.status = status
                if status == FoodOnForkDetection.SUCCESS:
                    food_on_fork_detection_msg.message = "No errors."
                elif status == FoodOnForkDetection.ERROR_TOO_FEW_POINTS:
                    food_on_fork_detection_msg.message = (
                        "Error: Too few detected points. This typically means there is "
                        "no food on the fork."
                    )
                elif status == FoodOnForkDetection.ERROR_NO_TRANSFORM:
                    food_on_fork_detection_msg.message = (
                        "Error: Could not get requested transform(s)."
                    )
            # pylint: disable=broad-except
            # This is necessary because we don't know what exceptions the model
            # might raise.
            except Exception as err:
                err_str = f"Error predicting food on fork: {err}"
                self._node.get_logger().error(err_str)
                food_on_fork_detection_msg.probability = np.nan
                food_on_fork_detection_msg.status = FoodOnForkDetection.UNKNOWN_ERROR
                food_on_fork_detection_msg.message = err_str

            # Visualize the results
            if self.viz:
                # # NOTE: This is commented by default since it can block the main thread.
                # # However, the 3D plot it shows is incredibly useful for adjusting `no_fof_points_offset`.
                # self.model.visualize_img(depth_img_cv2, t[0])
                self.visualize_result(food_on_fork_detection_msg, t[0])

            # Publish the FoodOnForkDetection message
            self.pub.publish(food_on_fork_detection_msg)


def main(args=None):
    """
    Launch the ROS node and spin.
    """
    rclpy.init(args=args)

    node = ADAFeedingPerceptionNode("food_on_fork_detection")
    food_on_fork_detection = FoodOnForkDetectionNode(node)
    executor = MultiThreadedExecutor(num_threads=4)

    # Spin in the background since detecting faces will block
    # the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
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
    node.destroy_node()
    rclpy.shutdown()
    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == "__main__":
    main()
