"""
This script takes in a variety of command line arguments and then trains and test a
FoodOnForkDetector as configured by the arguments.
"""

# Standard Imports
import argparse
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from sensor_msgs.msg import CameraInfo
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Local imports
from ada_feeding.helpers import import_from_string
from ada_feeding_perception.food_on_fork_detectors import (
    FoodOnForkDetector,
    FoodOnForkLabel,
)
from ada_feeding_perception.helpers import ros_msg_to_cv2_image


def show_normalized_depth_img(img, wait=True, window_name="img"):
    """
    Show the normalized depth image. Useful for debugging.

    Parameters
    ----------
    img: npt.NDArray
        The depth image to show.
    wait: bool, optional
        If True, wait for a key press before closing the window.
    window_name: str, optional
        The name of the window to show the image in.
    """
    # Show the normalized depth image
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype("uint8")
    cv2.imshow(window_name, img_normalized)
    cv2.waitKey(0 if wait else 1)


def read_args() -> argparse.Namespace:
    """
    Read the command line arguments.

    Returns
    -------
    args: argparse.Namespace
        The command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and test one or more FoodOnForkDetectors on an offline dataset."
        )
    )

    # Configure the models
    parser.add_argument(
        "--model-classes",
        help=(
            "A JSON-encoded string where keys are an arbitrary model ID and "
            "values are the class names to use for that model. e.g., "
            '{"dummy_detector": "ada_feeding_perception.food_on_fork_detectors.FoodOnForkDummyDetector", '
            '"t_test_detector": "ada_feeding_perception.food_on_fork_detectors.FoodOnForkTTestDetector"}'
        ),
        required=True,
    )
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help=(
            "A JSON-encoded string where keys are the model ID and values are "
            "a dictionary of keyword arguments to pass to the model's constructor. e.g., "
            '{"dummy_detector": {"proba": 0.1}}'
        ),
    )

    # Configure the cropping/masking of depth images. These should exactly match
    # the cropping/masking done in the real-time detector (in config/food_on_fork_detection.yaml).
    parser.add_argument(
        "--crop-top-left",
        default=(0, 0),
        type=int,
        nargs="+",
        help=("The top-left corner of the crop rectangle. The format is (x, y)."),
    )
    parser.add_argument(
        "--crop-bottom-right",
        default=(640, 480),
        type=int,
        nargs="+",
        help=("The bottom-right corner of the crop rectangle. The format is (x, y)."),
    )
    parser.add_argument(
        "--depth-min-mm",
        default=0,
        type=int,
        help=("The minimum depth value to consider in the depth images."),
    )
    parser.add_argument(
        "--depth-max-mm",
        default=20000,
        type=int,
        help=("The maximum depth value to consider in the depth images."),
    )

    # Configure the dataset
    parser.add_argument(
        "--data-dir",
        default="../data/food_on_fork",
        help=(
            "The directory containing the training and testing data. This path should "
            "have a file called `bags_metadata.csv` that contains the labels for bagfiles, "
            "and one folder per bagfile referred to in the CSV. This path should be "
            "relative to **this file's** location."
        ),
    )
    parser.add_argument(
        "--depth-topic",
        default="/local/camera/aligned_depth_to_color/image_raw",
        help=("The topic to use for depth images."),
    )
    parser.add_argument(
        "--color-topic",
        default="/local/camera/color/image_raw/compressed",
        help=("The topic to use for color images. Used for debugging."),
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/local/camera/color/camera_info",
        help=("The topic to use for camera info."),
    )
    parser.add_argument(
        "--include-motion",
        default=False,
        action="store_true",
        help=("If set, include images when the robot arm is moving in the dataset."),
    )
    parser.add_argument(
        "--rosbags-select",
        default=[],
        type=str,
        nargs="+",
        help="If set, only rosbags listed here will be included",
    )
    parser.add_argument(
        "--rosbags-skip",
        default=[],
        type=str,
        nargs="+",
        help="If set, rosbags listed here will be excluded",
    )

    # Configure the training and testing operations
    parser.add_argument(
        "--no-train",
        default=False,
        action="store_true",
        help="If set, do not train the models and instead only test them.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help=(
            "The random seed to use for the train-test split and in the detector. "
            "If unspecified, the seed will be the current time."
        ),
    )
    parser.add_argument(
        "--train-set-size",
        default=0.8,
        type=float,
        help="The fraction of the dataset to use for training",
    )
    parser.add_argument(
        "--model-dir",
        default="../model",
        help=(
            "The directory to save and load the trained model to/from. The path should be "
            "relative to **this file's** location. "
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="../model/results",
        help=(
            "The directory to save the train and test results to. The path should be "
            "relative to **this file's** location. "
        ),
    )
    parser.add_argument(
        "--lower-thresh",
        default=0.5,
        type=float,
        help=(
            "If the predicted probability of food on fork is <= this value, the "
            "detector will predict that there is no food on the fork."
        ),
    )
    parser.add_argument(
        "--upper-thresh",
        default=0.5,
        type=float,
        help=(
            "If the predicted probability of food on fork is > this value, the "
            "detector will predict that there is food on the fork."
        ),
    )
    parser.add_argument(
        "--max-eval-n",
        default=None,
        type=int,
        help=(
            "The maximum number of evaluations to perform. If None, all evaluations "
            "will be performed. Typically used when debugging a detector"
        ),
    )
    parser.add_argument(
        "--viz",
        default=False,
        action="store_true",
        help="If set, visualize images during the process",
    )

    return parser.parse_args()


def load_data(
    data_dir: str,
    depth_topic: str,
    color_topic: str,
    camera_info_topic: str,
    crop_top_left: Tuple[int, int],
    crop_bottom_right: Tuple[int, int],
    depth_min_mm: int,
    depth_max_mm: int,
    include_motion: bool,
    rosbags_select: List[str] = [],
    rosbags_skip: List[str] = [],
    viz: bool = False,
) -> Tuple[npt.NDArray, npt.NDArray[int], CameraInfo]:
    """
    Load the data specified in the command line arguments.

    Parameters
    ----------
    data_dir: str
        The directory containing the training and testing data. The path should be
        relative to **this file's** location. This directory should have two
        subdirectories: 'food' and 'no_food', each containing either .png files
        corresponding to depth images or ROS bag files with the topics specified
        in command line arguments.
    depth_topic: str
        The topic to use for depth images.
    color_topic: str
        The topic to use for color images. Used for debugging.
    camera_info_topic: str
        The topic to use for camera info.
    crop_top_left, crop_bottom_right: Tuple[int, int]
        The top-left and bottom-right corners of the crop rectangle.
    depth_min_mm, depth_max_mm: int
        The minimum and maximum depth values to consider in the depth images.
    include_motion: bool
        If True, include images when the robot arm is moving in the dataset.
    rosbags_select: List[str], optional
        If set, only rosbags in this list will be included
    rosbags_skip: List[str], optional
        If set, rosbags in this list will be excluded
    viz: bool, optional
        If True, visualize the depth images as they are loaded.

    Returns
    -------
    X: npt.NDArray
        The depth images to predict on.
    y: npt.NDArray[int]
        The labels for whether there is food on the fork.
    camera_info: CameraInfo
        The camera info for the depth images. We assume it is static across all
        depth images.
    """
    # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, too-many-statements
    # Okay since we want to make it a flexible method.

    absolute_data_dir = os.path.join(os.path.dirname(__file__), data_dir)

    w = crop_bottom_right[0] - crop_top_left[0]
    h = crop_bottom_right[1] - crop_top_left[1]
    X = np.zeros((0, h, w), dtype=np.uint16)
    y = np.zeros(0, dtype=int)

    # Load the metadata
    metadata = pd.read_csv(os.path.join(absolute_data_dir, "bags_metadata.csv"))
    bagname_to_annotations = {}
    for _, row in metadata.iterrows():
        rosbag_name = row["rosbag_name"]
        time_from_start = row["time_from_start"]
        food_on_fork = row["food_on_fork"]
        arm_moving = row["arm_moving"]
        if rosbag_name not in bagname_to_annotations:
            bagname_to_annotations[rosbag_name] = []
        bagname_to_annotations[rosbag_name].append(
            (time_from_start, food_on_fork, arm_moving)
        )

    # Load the data
    bridge = CvBridge()
    camera_info = None
    for rosbag_name, annotations in bagname_to_annotations.items():
        if (
            (len(rosbags_select) > 0 and rosbag_name not in rosbags_select) or 
            (len(rosbags_skip) > 0 and rosbag_name in rosbags_skip)
        ):
            print(f"Skipping rosbag {rosbag_name}")
            continue
        annotations.sort()
        i = 0
        num_images_no_points = 0
        with Reader(os.path.join(absolute_data_dir, rosbag_name)) as reader:
            # Get the depth message count
            for connection in reader.connections:
                if connection.topic == depth_topic:
                    depth_msg_count = connection.msgcount
                    break
            # Extend X and y by depth_msg_count
            j = y.shape[0]
            X = np.concatenate((X, np.zeros((depth_msg_count, h, w), dtype=np.uint16)))
            y = np.concatenate((y, np.zeros(depth_msg_count, dtype=int)))

            start_time = None
            for connection, timestamp, rawdata in reader.messages():
                if start_time is None:
                    start_time = timestamp
                # Depth Image
                if connection.topic == depth_topic:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    elapsed_time = (timestamp - start_time) / 10.0**9
                    while (
                        i < len(annotations) - 1
                        and elapsed_time > annotations[i + 1][0]
                    ):
                        i += 1
                    arm_moving = annotations[i][2]
                    if not include_motion and arm_moving:
                        # Skip images when the robot arm is moving
                        continue
                    if annotations[i][1] == FoodOnForkLabel.FOOD.value:
                        label = 1
                    elif annotations[i][1] == FoodOnForkLabel.NO_FOOD.value:
                        label = 0
                    else:
                        # Skip images with unknown label
                        continue
                    img = ros_msg_to_cv2_image(msg, bridge)
                    img = img[
                        crop_top_left[1] : crop_bottom_right[1],
                        crop_top_left[0] : crop_bottom_right[0],
                    ]
                    img = np.where(
                        (img >= depth_min_mm) & (img <= depth_max_mm), img, 0
                    )
                    if np.all(img == 0):
                        num_images_no_points += 1
                    X[j, :, :] = img
                    y[j] = label
                    j += 1
                # Camera Info
                elif connection.topic == camera_info_topic and camera_info is None:
                    camera_info = deserialize_cdr(rawdata, connection.msgtype)
                # RGB Image
                elif connection.topic == color_topic and viz:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    print((timestamp - start_time) / 10.0**9)
                    img = ros_msg_to_cv2_image(msg, bridge)
                    # A box around the forktip
                    x0, y0 = 308, 242
                    w, h = 128, 128
                    fof_color = (0, 255, 0)
                    no_fof_color = (255, 0, 0)
                    color = fof_color if len(y) == 0 or y[-1] == 1 else no_fof_color
                    img = cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), color, 2)
                    img = cv2.circle(img, (x0 + w // 2, y0 + h // 2), 5, color, -1)
                    cv2.imshow("RGB Image", img)
                    cv2.waitKey(1)

    # Truncate extra all-zero rows on the end of Z and y
    print(f"Proportion of img with no pixels: {num_images_no_points/j}")
    X = X[:j]
    y = y[:j]

    return X, y, camera_info


def load_models(
    model_classes: str,
    model_kwargs: str,
    seed: int,
    crop_top_left: Tuple[int, int],
    crop_bottom_right: Tuple[int, int],
) -> Dict[str, FoodOnForkDetector]:
    """
    Load the models specified in the command line arguments.

    Parameters
    ----------
    model_classes: str
        A JSON-encoded dictionary where keys are an arbitrary model ID and values
        are the class names to use for that model.
    model_kwargs: str
        A JSON-encoded dictionary where keys are the model ID and values are a
        dictionary of keyword arguments to pass to the model's constructor.
    seed: int
        The random seed to use in the detector.
    crop_top_left, crop_bottom_right: Tuple[int, int]
        The top-left and bottom-right corners of the crop rectangle.

    Returns
    -------
    models: dict
        A dictionary where keys are the model ID and values are the model instances.
    """
    # Parse the JSON strings
    model_classes = json.loads(model_classes)
    model_kwargs = json.loads(model_kwargs)

    models = {}
    for model_id, model_class in model_classes.items():
        # Load the class
        model_class = import_from_string(model_class)

        # Get the kwargs
        kwargs = model_kwargs.get(model_id, {})

        # Create the model
        models[model_id] = model_class(**kwargs)
        models[model_id].seed = seed
        models[model_id].crop_top_left = crop_top_left
        models[model_id].crop_bottom_right = crop_bottom_right

    return models


def train_models(
    models: Dict[str, Any], X: npt.NDArray, y: npt.NDArray, model_dir: str
) -> None:
    """
    Train the models on the training data.

    Parameters
    ----------
    models: dict
        A dictionary where keys are the model ID and values are the model instances.
    X: npt.NDArray
        The depth images to train on.
    y: npt.NDArray
        The labels for the depth images.
    model_dir: str
        The directory to save the trained model to. The path should be
        relative to **this file's** location.
    """
    absolute_model_dir = os.path.join(os.path.dirname(__file__), model_dir)

    for model_id, model in models.items():
        print(f"Training model {model_id}...", end="")
        model.fit(X, y)
        save_path = model.save(os.path.join(absolute_model_dir, model_id))
        print(f"Done. Saved to '{save_path}'.")


def evaluate_models(
    models: Dict[str, Any],
    train_X: npt.NDArray,
    test_X: npt.NDArray,
    train_y: npt.NDArray,
    test_y: npt.NDArray,
    model_dir: str,
    output_dir: str,
    lower_thresh: float,
    upper_thresh: float,
    max_eval_n: Optional[int] = None,
    viz: bool = False,
) -> None:
    """
    Test the models on the testing data.

    Parameters
    ----------
    models: dict
        A dictionary where keys are the model ID and values are the model instances.
    train_X, test_X: npt.NDArray
        The depth images to test on.
    train_y, test_Y: npt.NDArray
        The labels for the depth images.
    model_dir: str
        The directory to load the trained model from. The path should be
        relative to **this file's** location.
    output_dir: str
        The directory to save the train and test results to. The path should be
        relative to **this file's** location.
    lower_thresh: float
        If the predicted probability of food on fork is <= this value, the
        detector will predict that there is no food on the fork.
    upper_thresh: float
        If the predicted probability of food on fork is > this value, the
        detector will predict that there is food on the fork.
    max_eval_n: int, optional
        The maximum number of evaluations to perform. If None, all evaluations
        will be performed. Typically used when debugging a detector.
    viz: bool, optional
        If True, visualize the depth images as they are evaluated.
    """
    # pylint: disable=too-many-locals, too-many-arguments, too-many-nested-blocks
    # This function is meant to be flexible.

    absolute_model_dir = os.path.join(os.path.dirname(__file__), model_dir)
    absolute_output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    results_df = []
    results_df_columns = [
        "model_id",
        "y_true",
        "y_pred_proba",
        "y_pred",
        "seed",
        "dataset",
    ]
    results_txt = ""
    for model_id, model in models.items():
        # First, load the model
        load_path = os.path.join(absolute_model_dir, model_id)
        print(f"Loading model {model_id} from {load_path}...", end="")
        model.load(load_path)
        print("Done.")
        results_txt += f"Model {model_id} from {load_path}:\n"

        for label, (X, y) in [
            ("train", (train_X, train_y)),
            ("test", (test_X, test_y)),
        ]:
            if max_eval_n is not None:
                X = X[:max_eval_n]
                y = y[:max_eval_n]
            print(f"Evaluating model {model_id} on {label} dataset...", end="")
            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X, lower_thresh, upper_thresh, y_pred_proba)
            for i in range(y_pred_proba.shape[0]):
                results_df.append(
                    [model_id, y[i], y_pred_proba[i], y_pred[i], model.seed, label]
                )
            print("Done.")

            if viz:
                # Visualize all images where the model was wrong
                last_0_0 = None
                for i in range(y_pred_proba.shape[0]):
                    if y[i] != y_pred[i]:
                        print(f"y_true: {y[i]}, y_pred: {y_pred[i]}")
                        show_normalized_depth_img(X[i], wait=False, window_name="misclassified_img")
                        if last_0_0 is not None:
                            show_normalized_depth_img(X[last_0_0], wait=False, window_name="last_correct_no_fof_img")
                    if y[i] == 0 and y_pred[i] == 0:
                        last_0_0 = i

            # Compute the summary statistics
            txt = textwrap.indent(f"Results on {label} dataset:\n", " " * 4)
            results_txt += txt
            print(txt, end="")
            for metric in [
                accuracy_score,
                confusion_matrix,
            ]:
                txt = textwrap.indent(f"{metric.__name__}:\n", " " * 8)
                results_txt += txt
                print(txt, end="")
                val = metric(y, y_pred)
                txt = textwrap.indent(f"{val}\n", " " * 12)
                results_txt += txt
                print(txt, end="")

        results_txt += "\n"

    # Save the results
    results_df = pd.DataFrame(results_df, columns=results_df_columns)
    results_df.to_csv(
        os.path.join(
            absolute_output_dir, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}_results.csv"
        )
    )
    with open(
        os.path.join(
            absolute_output_dir, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}_results.txt"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(results_txt)


def main() -> None:
    """
    Train and test a FoodOnForkDetector as configured by the command line arguments.
    """
    # Load the arguments
    args = read_args()

    # Load the dataset
    X, y, camera_info = load_data(
        args.data_dir,
        args.depth_topic,
        args.color_topic,
        args.camera_info_topic,
        args.crop_top_left,
        args.crop_bottom_right,
        args.depth_min_mm,
        args.depth_max_mm,
        args.include_motion,
        args.rosbags_select,
        args.rosbags_skip,
        # viz=args.viz,
    )
    print("X.shape", X.shape, "y.shape", y.shape)

    # Do a train-test split of the dataset
    if args.seed is None:
        seed = int(time.time())
    else:
        seed = args.seed
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=args.train_set_size, random_state=seed
    )

    # Load the models
    models = load_models(
        args.model_classes,
        args.model_kwargs,
        seed,
        args.crop_top_left,
        args.crop_bottom_right,
    )
    for _, model in models.items():
        model.camera_info = camera_info

    # Train the model
    if not args.no_train:
        train_models(models, train_X, train_y, args.model_dir)

    # Evaluate the model
    evaluate_models(
        models,
        train_X,
        test_X,
        train_y,
        test_y,
        args.model_dir,
        args.output_dir,
        args.lower_thresh,
        args.upper_thresh,
        args.max_eval_n,
        viz=args.viz,
    )


if __name__ == "__main__":
    main()
