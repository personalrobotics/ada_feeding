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
from typing import Any, Dict, Tuple

# Third-party imports
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Local imports
from ada_feeding.helpers import import_from_string
from ada_feeding_perception.food_on_fork_detectors import FoodOnForkDetector


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

    # Configure the dataset
    parser.add_argument(
        "--data-dir",
        default="../data/food_on_fork",
        help=(
            "The directory containing the training and testing data. The path should be "
            "relative to **this file's** location. This directory should have two "
            "subdirectories: 'food' and 'no_food', each containing either .png files "
            "corresponding to depth images or ROS bag files with the topics specified "
            "in command line arguments."
        ),
    )
    parser.add_argument(
        "--data-type",
        help=(
            "The type of data to use. Either 'bag' or 'png'. If 'bag', the data "
            "subdirectoryies should contain ROS bag files. If 'png', the data "
            "subdirectories should contain .png files."
        ),
        choices=["bag", "png"],
        required=True,
    )
    parser.add_argument(
        "--depth-topic",
        default="/local/camera/aligned_depth_to_color/image_raw",
        help=(
            "The topic to use for depth images. This is only used if --data-type is "
            "'bag'."
        ),
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/local/camera/color/camera_info",
        help=(
            "The topic to use for camera info. This is only used if --data-type is "
            "'bag'."
        ),
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

    return parser.parse_args()


def load_data(
    data_dir: str, data_type: str, depth_topic: str, camera_info_topic: str
) -> Tuple[npt.NDArray, npt.NDArray]:
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
    data_type: str
        The type of data to use. Either 'bag' or 'png'. If 'bag', the data
        subdirectoryies should contain ROS bag files. If 'png', the data
        subdirectories should contain .png files.
    depth_topic: str
        The topic to use for depth images. This is only used if --data-type is
        'bag'.
    camera_info_topic: str
        The topic to use for camera info. This is only used if --data-type is
        'bag'.

    Returns
    -------
    X: npt.NDArray
        The depth images to predict on.
    y: npt.NDArray
        The labels for whether there is food on the fork.
    """
    absolute_data_dir = os.path.join(os.path.dirname(__file__), data_dir)

    X = []
    y = []
    if data_type == "bag":
        raise NotImplementedError("Bag file loading not implemented yet.")
    elif data_type == "png":
        food_dir = os.path.join(absolute_data_dir, "food")
        no_food_dir = os.path.join(absolute_data_dir, "no_food")
        for data_path, label in [(food_dir, 1), (no_food_dir, 0)]:
            print(f"Loading data from {data_path} with label {label}...", end="")
            n = 0
            for filename in os.listdir(data_path):
                if filename.endswith(".png"):
                    # Load the image
                    image = cv2.imread(
                        os.path.join(data_path, filename), cv2.IMREAD_UNCHANGED
                    )

                    # Add the image and label to the dataset
                    X.append(image)
                    y.append(label)
                    n += 1
            print(f"Loaded {n} images.")
    else:
        raise ValueError(f"Invalid data type: {data_type}. Must be 'bag' or 'png'.")

    return np.array(X), np.array(y)


def load_models(
    model_classes: str, model_kwargs: str, seed: int
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
    """
    # pylint: disable=too-many-locals, too-many-arguments
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
            print(f"Evaluating model {model_id} on {label} dataset...", end="")
            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X, lower_thresh, upper_thresh, y_pred_proba)
            for i in range(y_pred_proba.shape[0]):
                results_df.append(
                    [model_id, y[i], y_pred_proba[i], y_pred[i], model.seed, label]
                )
            print("Done.")

            # Compute the summary statistics
            txt = textwrap.indent(f"Results on {label} dataset:\n", " " * 4)
            results_txt += txt
            print(txt, end="")
            for metric in [
                accuracy_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
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
    X, y = load_data(
        args.data_dir, args.data_type, args.depth_topic, args.camera_info_topic
    )

    # Do a train-test split of the dataset
    if args.seed is None:
        seed = int(time.time())
    else:
        seed = args.seed
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=args.train_set_size, random_state=seed
    )

    # Load the models
    models = load_models(args.model_classes, args.model_kwargs, seed)

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
    )


if __name__ == "__main__":
    main()
