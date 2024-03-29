"""
This file contains an abstract class, FoodOnForkDetector, that takes in a single depth
image and returns a confidence in [0,1] that there is food on the fork.
"""
# Standard imports
from abc import ABC, abstractmethod
from enum import Enum
import os
import time
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from overrides import override
import rclpy
from sensor_msgs.msg import CameraInfo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
import tf2_ros
from tf2_ros.buffer import Buffer
from transforms3d._gohlketransforms import quaternion_matrix

# Local imports
from ada_feeding_msgs.msg import FoodOnForkDetection
from ada_feeding_perception.helpers import (
    depth_img_to_pointcloud,
    show_3d_scatterplot,
    show_normalized_depth_img,
)


class FoodOnForkLabel(Enum):
    """
    An enumeration of possible labels for food on the fork.
    """

    NO_FOOD = 0
    FOOD = 1
    UNSURE = 2


class FoodOnForkDetector(ABC):
    """
    An abstract class for any perception algorithm that takes in a single depth
    image and returns a confidence in [0,1] that there is food on the fork.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initializes the perception algorithm.

        Parameters
        ----------
        verbose: Whether to print debug messages.
        """
        self.__camera_info = None
        self.__crop_top_left = (0, 0)
        self.__crop_bottom_right = (640, 480)
        self.__seed = int(time.time() * 1000)
        self.verbose = verbose

    @property
    def camera_info(self) -> Optional[CameraInfo]:
        """
        The camera info for the depth image.

        Returns
        -------
        camera_info: The camera info for the depth image, or None if not set.
        """
        return self.__camera_info

    @camera_info.setter
    def camera_info(self, camera_info: CameraInfo) -> None:
        """
        Sets the camera info for the depth image.

        Parameters
        ----------
        camera_info: The camera info for the depth image.
        """
        self.__camera_info = camera_info

    @property
    def crop_top_left(self) -> Tuple[int, int]:
        """
        The top left corner of the region of interest in the depth image.

        Returns
        -------
        crop_top_left: The top left corner of the region of interest in the depth
            image.
        """
        return self.__crop_top_left

    @crop_top_left.setter
    def crop_top_left(self, crop_top_left: Tuple[int, int]) -> None:
        """
        Sets the top left corner of the region of interest in the depth image.

        Parameters
        ----------
        crop_top_left: The top left corner of the region of interest in the depth
            image.
        """
        self.__crop_top_left = crop_top_left

    @property
    def crop_bottom_right(self) -> Tuple[int, int]:
        """
        The bottom right corner of the region of interest in the depth image.

        Returns
        -------
        crop_bottom_right: The bottom right corner of the region of interest in
            the depth image.
        """
        return self.__crop_bottom_right

    @crop_bottom_right.setter
    def crop_bottom_right(self, crop_bottom_right: Tuple[int, int]) -> None:
        """
        Sets the bottom right corner of the region of interest in the depth image.

        Parameters
        ----------
        crop_bottom_right: The bottom right corner of the region of interest in
            the depth image.
        """
        self.__crop_bottom_right = crop_bottom_right

    @property
    def seed(self) -> int:
        """
        The random seed to use in the detector.

        Returns
        -------
        seed: The random seed to use in the detector.
        """
        return self.__seed

    @seed.setter
    def seed(self, seed: int) -> None:
        """
        Sets the random seed to use in the detector.

        Parameters
        ----------
        seed: The random seed to use in the detector.
        """
        self.__seed = seed

    @property
    def transform_frames(self) -> List[Tuple[str, str]]:
        """
        Gets the parent and child frame for every transform that this classifier
        wants to use.

        Returns
        -------
        frames: A list of (parent_frame_id, child_frame_id) tuples.
        """
        return []

    @staticmethod
    def get_transforms(frames: List[Tuple[str, str]], tf_buffer: Buffer) -> npt.NDArray:
        """
        Gets the most recent transforms that are necessary for this classifier.
        These are then passed into fit, predict_proba, and predict.

        Parameters
        ----------
        frames: A list of (parent_frame_id, child_frame_id) tuples to get transforms
            for. Size: (num_transforms, 2).
        tf_buffer: The tf buffer that stores the transforms.

        Returns
        -------
        transforms: The transforms (homogenous coordinates) that are necessary
            for this classifier. Size (num_transforms, 4, 4). Note that if the
            transform is not found, it will be a zero matrix.
        """
        transforms = []
        for parent_frame_id, child_frame_id in frames:
            try:
                transform = tf_buffer.lookup_transform(
                    parent_frame_id,
                    child_frame_id,
                    rclpy.time.Time(),
                )
                # Convert the transform into a matrix
                M = quaternion_matrix(
                    [
                        transform.transform.rotation.w,
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                    ],
                )
                M[:3, 3] = [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
                transforms.append(M)
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as err:
                print(
                    f"Error getting transform from {parent_frame_id} to {child_frame_id}: {err}"
                )
                transforms.append(np.zeros((4, 4), dtype=float))

        return np.array(transforms)

    @abstractmethod
    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray[int],
        t: npt.NDArray[float],
        viz_save_dir: Optional[str] = None,
    ) -> None:
        """
        Trains the perception algorithm on a dataset of depth images and
        corresponding labels.

        Parameters
        ----------
        X: The depth images to train on. Size (num_images, height, width).
        y: The labels for the depth images. Size (num_images,). Must be one of the
            values enumerated in FoodOnForkLabel.
        t: The transforms (homogenous coordinates) that are necessary for this
            classifier. Size (num_images, num_transforms, 4, 4). Should be outputted
            by `get_transforms`.
        viz_save_dir: The directory to save visualizations to. If None, no
            visualizations will be saved.
        """

    @abstractmethod
    def save(self, path: str) -> str:
        """
        Saves the model to a file.

        Parameters
        ----------
        path: The path to save the perception algorithm to. This file should not
            have an extension; this function will add the appropriate extension.

        Returns
        -------
        save_path: The path that the model was saved to.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the model a file.

        Parameters
        ----------
        path: The path to load the perception algorithm from. If the path does
            not have an extension, this function will add the appropriate
            extension.
        """

    @abstractmethod
    def predict_proba(
        self,
        X: npt.NDArray,
        t: npt.NDArray[float],
    ) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        """
        Predicts the probability that there is food on the fork for a set of
        depth images.

        Parameters
        ----------
        X: The depth images to predict on.
        t: The transforms (homogenous coordinates) that are necessary for this
            classifier. Size (num_images, num_transforms, 4, 4). Should be outputted
            by `get_transforms`.

        Returns
        -------
        y: The predicted probabilities that there is food on the fork.
        statuses: The status of each prediction. Must be one of the const values
            declared in the FoodOnForkDetection message.
        """

    def predict(
        self,
        X: npt.NDArray,
        t: npt.NDArray[float],
        lower_thresh: float,
        upper_thresh: float,
        proba: Optional[npt.NDArray] = None,
        statuses: Optional[npt.NDArray[int]] = None,
    ) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
        """
        Predicts whether there is food on the fork for a set of depth images.

        Parameters
        ----------
        X: The depth images to predict on.
        t: The transforms (homogenous coordinates) that are necessary for this
            classifier. Size (num_images, num_transforms, 4, 4). Should be outputted
            by `get_transforms`.
        lower_thresh: The lower threshold for food on the fork.
        upper_thresh: The upper threshold for food on the fork.
        proba: The predicted probabilities that there is food on the fork. If either
            proba or statuses is None, this function will call predict_proba to get
            the proba and statuses.
        statuses: The status of each prediction. Must be one of the const values
            declared in the FoodOnForkDetection message. If either proba or statuses
            is None, this function will call predict_proba to get the proba and
            statuses.

        Returns
        -------
        y: The predicted labels for whether there is food on the fork. Must be one
            of the values enumerated in FoodOnForkLabel.
        statuses: The status of each prediction. Must be one of the const values
            declared in the FoodOnForkDetection message.
        """
        # pylint: disable=too-many-arguments
        # These many are fine.
        if proba is None or statuses is None:
            proba, statuses = self.predict_proba(X, t)
        return (
            np.where(
                proba < lower_thresh,
                FoodOnForkLabel.NO_FOOD.value,
                np.where(
                    proba > upper_thresh,
                    FoodOnForkLabel.FOOD.value,
                    FoodOnForkLabel.UNSURE.value,
                ),
            ),
            statuses,
        )

    def overlay_debug_info(self, img: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
        """
        Overlays debug information onto a depth image.

        Parameters
        ----------
        img: The depth image to overlay debug information onto.
        t: The closest transforms (homogenous coordinates) to this image's timestamp.
            Size (num_transforms, 4, 4). Should be outputted by `get_transforms`.

        Returns
        -------
        img_with_debug_info: The depth image with debug information overlayed.
        """
        # pylint: disable=unused-argument
        return img

    def visualize_img(self, img: npt.NDArray, t: npt.NDArray) -> None:
        """
        Visualizes a depth image. This function is used for debugging, so it helps
        to not only visualize the img, but also subclass-specific information that
        can help explain why the img would result in a particular prediction.

        It is acceptable for this function to block until the user closes a window.

        Parameters
        ----------
        img: The depth image to visualize.
        t: The closest transforms (homogenous coordinates) to this image's timestamp.
            Size (num_transforms, 4, 4). Should be outputted by `get_transforms`.
        """
        # pylint: disable=unused-argument
        show_normalized_depth_img(img, wait=True, window_name="img")


class FoodOnForkDummyDetector(FoodOnForkDetector):
    """
    A dummy perception algorithm that always predicts the same probability.
    """

    def __init__(self, proba: float, verbose: bool = False) -> None:
        """
        Initializes the dummy perception algorithm.

        Parameters
        ----------
        proba: The probability that the dummy algorithm should always predict.
        verbose: Whether to print debug messages.
        """
        super().__init__(verbose)
        self.proba = proba

    @override
    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray[int],
        t: npt.NDArray[float],
        viz_save_dir: Optional[str] = None,
    ) -> None:
        pass

    @override
    def save(self, path: str) -> str:
        return ""

    @override
    def load(self, path: str) -> None:
        pass

    @override
    def predict_proba(
        self,
        X: npt.NDArray,
        t: npt.NDArray[float],
    ) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        return (
            np.full(X.shape[0], self.proba),
            np.full(X.shape[0], FoodOnForkDetection.SUCCESS),
        )


class FoodOnForkDistanceToNoFOFDetector(FoodOnForkDetector):
    """
    A perception algorithm that stores a representative subset of "no FoF" points.
    It then calculates the average distance between each test point and the nearest
    no FoF point, and uses a classifier to predict the probability of a
    test point being FoF based on that distance.
    """

    # pylint: disable=too-many-instance-attributes
    # These many are fine.

    AGGREGATORS = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
        "25p": lambda x: np.percentile(x, 25),
        "75p": lambda x: np.percentile(x, 75),
        "90p": lambda x: np.percentile(x, 90),
        "95p": lambda x: np.percentile(x, 95),
    }

    def __init__(
        self,
        camera_matrix: npt.NDArray,
        prop_no_fof_points_to_store: float = 0.5,
        min_points: int = 40,
        min_distance: float = 0.001,
        aggregator_name: Optional[str] = "90p",
        verbose: bool = False,
    ) -> None:
        """
        Initializes the algorithm.

        Parameters
        ----------
        camera_matrix: The camera intrinsic matrix (K).
        prop_no_fof_points_to_store: The proportion of no FoF pointclouds in
            the train set to set aside for storing no FoF points. Note that not
            all points in these pointclouds are stored; only those that are >=
            min_distance m away from the currently stored points.
        min_points: The minimum number of points in a pointcloud to consider it
            for comparison. If a pointcloud has fewer points than this, it will
            return a probability of nan (prediction of UNSURE).
        min_distance: The minimum distance (m) between stored no FoF points.
        aggregator_name: The name of the aggregator to use to aggregate the
            distances between the test point and the stored no FoF points. If None,
            all aggregators are used. This is typically only useful to compare
            the performance of different aggregators.
        verbose: Whether to print debug messages.
        """
        # pylint: disable=too-many-arguments
        # These many are fine.

        super().__init__(verbose)
        self.camera_matrix = camera_matrix
        self.prop_no_fof_points_to_store = prop_no_fof_points_to_store
        self.min_points = min_points
        self.min_distance = min_distance
        self.aggregator_name = aggregator_name

        # The attributes that are stored/loaded by the model
        self.no_fof_points = None
        self.clf = None
        self.best_aggregator_name = None

    @property
    @override
    def transform_frames(self) -> List[Tuple[str, str]]:
        return [("forkTip", "camera_color_optical_frame")]

    @staticmethod
    def distances_between_pointclouds(
        pointcloud1: npt.NDArray,
        pointcloud2: npt.NDArray,
    ) -> npt.NDArray:
        """
        For every point in pointcloud1, gets the minimum distance to points in
        pointcloud2. Note that this is not
        symmetric; the order of the pointclouds matters.

        Parameters
        ----------
        pointcloud1: The test pointcloud. Size (n, k).
        pointcloud2: The training pointcloud. Size (m, k).

        Returns
        -------
        distances: The minimum distance from each point in pointcloud1 to points
            in pointcloud2. Size (n,).
        """
        return np.min(pairwise_distances(pointcloud1, pointcloud2), axis=1)

    def get_representative_points(self, pointcloud: npt.NDArray) -> npt.NDArray:
        """
        Returns a subset of points from the pointcloud such that every point in
        pointcloud is no more than min_distance away from one of the representative
        points.

        Parameters
        ----------
        pointcloud: The pointcloud to get representative points from. Size (n, k).

        Returns
        -------
        representative_points: The subset of points from the pointcloud. Size (m, k).
        """
        if self.verbose:
            print("Getting a subset of representative points")

        # Include the first point
        representative_points = pointcloud[0:1, :]

        # Add points that are >= min_distance m away from the stored points
        for i in range(1, len(pointcloud)):
            if self.verbose:
                print(f"Point {i}/{len(pointcloud)}")
            contender_point = pointcloud[i]
            # Get the distance between the contender point and the representative points
            distance = np.min(
                np.linalg.norm(representative_points - contender_point, axis=1)
            )
            if distance >= self.min_distance:
                representative_points = np.vstack(
                    [representative_points, contender_point]
                )
        return representative_points

    @override
    def fit(
        self,
        X: npt.NDArray,
        y: npt.NDArray[int],
        t: npt.NDArray[float],
        viz_save_dir: Optional[str] = None,
    ) -> None:
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # This is the main logic of the algorithm, so it's okay to have a lot of
        # local variables.

        # Only keep the datapoints where the transform is not 0
        i_to_keep = np.where(np.logical_not(np.all(t == 0, axis=(2, 3))))[0]
        X = X[i_to_keep]
        y = y[i_to_keep]
        t = t[i_to_keep]

        # Get the most up-to-date camera matrix
        if self.camera_info is not None:
            self.camera_matrix = np.array(self.camera_info.k)

        # Convert all images to pointclouds, excluding those with too few points
        pointclouds = []
        y_pointclouds = []
        for i, img in enumerate(X):
            pointcloud = depth_img_to_pointcloud(
                img,
                *self.crop_top_left,
                f_x=self.camera_matrix[0],
                f_y=self.camera_matrix[4],
                c_x=self.camera_matrix[2],
                c_y=self.camera_matrix[5],
                transform=t[i, 0, :, :],
            )
            if len(pointcloud) >= self.min_points:
                pointclouds.append(pointcloud)
                y_pointclouds.append(y[i])

        # Convert to np arrays
        # Pointclouds must be dtype object to store arrays of different lengths
        pointclouds = np.array(pointclouds, dtype=object)
        y_pointclouds = np.array(y_pointclouds)
        if self.verbose:
            print(
                f"Converted {X.shape[0]} depth images to {pointclouds.shape[0]} pointclouds"
            )

        # Split the no FoF and FoF pointclouds. The "train set" consists of only
        # no FoF pointclouds, and is used to find a representative subset of points
        # to store. The "val set" consists of both no FoF and FoF pointclouds, and
        # is used to train the classifier.
        no_fof_pointclouds = pointclouds[y_pointclouds == FoodOnForkLabel.NO_FOOD.value]
        fof_pointclouds = pointclouds[y_pointclouds == FoodOnForkLabel.FOOD.value]
        no_fof_pointclouds_train, no_fof_pointclouds_val = train_test_split(
            no_fof_pointclouds,
            train_size=self.prop_no_fof_points_to_store,
            random_state=self.seed,
        )
        val_pointclouds = np.concatenate([no_fof_pointclouds_val, fof_pointclouds])
        val_labels = np.concatenate(
            [
                np.zeros((no_fof_pointclouds_val.shape[0],)),
                np.ones((fof_pointclouds.shape[0],)),
            ]
        )
        if self.verbose:
            print("Split the no FoF pointclouds into train and val")

        # Store a representative subset of the points
        all_no_fof_pointclouds_train = np.concatenate(no_fof_pointclouds_train)
        self.no_fof_points = self.get_representative_points(
            all_no_fof_pointclouds_train
        )
        if self.verbose:
            print(
                f"Stored a representative subset of {self.no_fof_points.shape[0]}/"
                f"{all_no_fof_pointclouds_train.shape[0]} no FoF pointclouds"
            )

        # Get the aggregators
        if self.aggregator_name is None:
            aggregator_names = list(self.AGGREGATORS.keys())
        else:
            aggregator_names = [self.aggregator_name]

        # Get the distances from each point in the "val set" to the stored points
        val_distances = {name: [] for name in aggregator_names}
        for i, pointcloud in enumerate(val_pointclouds):
            if self.verbose:
                print(
                    "Computing distance to stored points for val point "
                    f"{i}/{val_pointclouds.shape[0]}"
                )
            point_distances = (
                FoodOnForkDistanceToNoFOFDetector.distances_between_pointclouds(
                    pointcloud, self.no_fof_points
                )
            )
            for name in aggregator_names:
                aggregator = self.AGGREGATORS[name]
                distance = aggregator(point_distances)
                val_distances[name].append(distance)
        val_distances = {
            name: np.array(val_distances[name]) for name in aggregator_names
        }

        # Split the validation set into train and val. This is to pick the best
        # aggregator to use.
        val_train_i, val_val_i = train_test_split(
            np.arange(val_labels.shape[0]),
            train_size=0.8,
            random_state=self.seed,
            stratify=val_labels,
        )

        # Train the classifier(s)
        f1_scores = {}
        clfs = {}
        for name in aggregator_names:
            # Train the classifier
            if self.verbose:
                print(f"Training the classifier for aggregator {name}")
            clf = LogisticRegression(random_state=self.seed, penalty=None)
            clf.fit(
                val_distances[name].reshape(-1, 1)[val_train_i, :],
                val_labels[val_train_i],
            )
            clfs[name] = clf
            if self.verbose:
                print(
                    f"Trained the classifier for aggregator {name}, got coeff "
                    f"{clf.coef_} and intercept {clf.intercept_}"
                )

            # Get the f1 score
            y_pred = clf.predict(val_distances[name].reshape(-1, 1)[val_val_i])
            y_true = val_labels[val_val_i]
            f1_scores[name] = f1_score(y_true, y_pred)
            if self.verbose:
                print(f"F1 score for aggregator {name}: {f1_scores[name]}")

        # Save a visualization of the classifier
        if viz_save_dir is not None:
            max_aggregated_distance = max(
                np.max(val_distances[name][val_val_i]) for name in aggregator_names
            )
            for name in aggregator_names:
                # Create a scatterplot where the x-axis is the distance in val_val
                # and the y-axis is the label.  Add some y-jitter to make it easier
                # to see the points.
                y_true = val_labels[val_val_i] + np.random.normal(
                    0, 0.1, val_val_i.shape[0]
                )
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(
                    val_distances[name][val_val_i], y_true, label="True", alpha=0.5
                )
                ax.set_xlim(0, max_aggregated_distance)

                # Add a line for the probability predictions of num_points over the range
                # of distances
                num_points = 100
                distances = np.linspace(0.0, max_aggregated_distance, num_points)
                probas = clfs[name].predict_proba(distances.reshape(-1, 1))[:, 1]
                ax.plot(distances, probas, label="Classifier Probabilities")

                # Add a title
                ax.set_title(
                    f"Classifier for Aggregator {name}. F1 Score: {f1_scores[name]}"
                )

                # Save the figure
                fig.savefig(
                    os.path.join(
                        viz_save_dir,
                        f"classifier_{clf.__class__.__name__}_aggregator_{name}.png",
                    )
                )

        # Pick the best aggregator
        self.best_aggregator_name, best_f1_score = max(
            f1_scores.items(), key=lambda x: x[1]
        )
        if self.verbose:
            print(
                f"Best aggregator: {self.best_aggregator_name} with f1 score {best_f1_score}"
            )
        self.clf = clfs[self.best_aggregator_name]

    @override
    def save(self, path: str) -> str:
        if (
            self.no_fof_points is None
            or self.clf is None
            or self.best_aggregator_name is None
        ):
            raise ValueError(
                "The model has not been trained yet. Call fit before saving."
            )
        # If the path has an extension, remove it.
        path = os.path.splitext(path)[0]
        np.savez_compressed(
            path,
            no_fof_points=self.no_fof_points,
            clf=np.array([self.clf], dtype=object),
            best_aggregator_name=self.best_aggregator_name,
        )
        return path + ".npz"

    @override
    def load(self, path: str) -> None:
        ext = os.path.splitext(path)[1]
        if len(ext) == 0:
            path = path + ".npz"
        params = np.load(path, allow_pickle=True)
        self.no_fof_points = params["no_fof_points"]
        self.clf = params["clf"][0]
        self.best_aggregator_name = str(params["best_aggregator_name"])
        if self.verbose:
            print(
                f"Loaded model with intercept {self.clf.intercept_} and coef {self.clf.coef_} "
                f"and best aggregator {self.best_aggregator_name} and num stored points "
                f"{self.no_fof_points.shape[0]}"
            )

    @override
    def predict_proba(
        self,
        X: npt.NDArray,
        t: npt.NDArray[float],
    ) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        probas = []
        statuses = []

        # Get the prediction per image.
        if self.verbose:
            inference_times = []
        for i, img in enumerate(X):
            if self.verbose:
                start_time = time.time()

            # If all elements of the transform are 0, set the proba to nan
            if np.all(np.isclose(t[i, 0, :, :], 0.0)):
                probas.append(np.nan)
                statuses.append(FoodOnForkDetection.ERROR_NO_TRANSFORM)
                continue

            # Convert the image to a pointcloud
            pointcloud = depth_img_to_pointcloud(
                img,
                *self.crop_top_left,
                f_x=self.camera_matrix[0],
                f_y=self.camera_matrix[4],
                c_x=self.camera_matrix[2],
                c_y=self.camera_matrix[5],
                transform=t[i, 0, :, :],
            )

            # If there are too few points, set the proba to nan
            if len(pointcloud) < self.min_points:
                probas.append(np.nan)
                statuses.append(FoodOnForkDetection.ERROR_TOO_FEW_POINTS)
                continue

            # If there are enough points, use the classifier to predict the probability
            # of food on the fork. Else, return an error status
            distances = FoodOnForkDistanceToNoFOFDetector.distances_between_pointclouds(
                pointcloud, self.no_fof_points
            )
            distance = self.AGGREGATORS[self.best_aggregator_name](distances)
            proba = self.clf.predict_proba(np.array([[distance]]))[0, 1]
            probas.append(proba)
            statuses.append(FoodOnForkDetection.SUCCESS)
            if self.verbose:
                inference_times.append(time.time() - start_time)
        if self.verbose:
            print(
                f"Inference Time: min {np.min(inference_times)}, max {np.max(inference_times)}, "
                f"mean {np.mean(inference_times)}, 25th percentile {np.percentile(inference_times, 25)}, "
                f"50th percentile {np.percentile(inference_times, 50)}, "
                f"75th percentile {np.percentile(inference_times, 75)}."
            )

        return np.array(probas), np.array(statuses, dtype=int)

    @override
    def predict(
        self,
        X: npt.NDArray,
        t: npt.NDArray[float],
        lower_thresh: float,
        upper_thresh: float,
        proba: Optional[npt.NDArray] = None,
        statuses: Optional[npt.NDArray[int]] = None,
    ) -> Tuple[npt.NDArray[int], npt.NDArray[int]]:
        # pylint: disable=too-many-arguments
        # These many are fine.
        if proba is None or statuses is None:
            proba, statuses = self.predict_proba(X, t)
        return (
            np.where(
                (proba < lower_thresh)
                | (statuses == FoodOnForkDetection.ERROR_TOO_FEW_POINTS),
                FoodOnForkLabel.NO_FOOD.value,
                np.where(
                    proba > upper_thresh,
                    FoodOnForkLabel.FOOD.value,
                    FoodOnForkLabel.UNSURE.value,
                ),
            ),
            statuses,
        )

    @override
    def overlay_debug_info(self, img: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
        # pylint: disable=too-many-locals
        # This is done to make it clear what the camera matrix values are.

        # First, convert all no_fof_points back to the camera frame by applying
        # the inverse of the homogenous transform t[0, :, :]
        no_fof_points_homogenous = np.hstack(
            [self.no_fof_points, np.ones((self.no_fof_points.shape[0], 1))]
        )
        no_fof_points_camera = np.dot(
            np.linalg.inv(t[0, :, :]), no_fof_points_homogenous.T
        ).T[:, :3]

        # For every point in the no_fof_points, convert them back into (u,v) pixel
        # coordinates.
        no_fof_points_mm = (no_fof_points_camera * 1000).astype(int)
        f_x = self.camera_matrix[0]
        f_y = self.camera_matrix[4]
        c_x = self.camera_matrix[2]
        c_y = self.camera_matrix[5]
        us = (f_x * no_fof_points_mm[:, 0] / no_fof_points_mm[:, 2] + c_x).astype(int)
        vs = (f_y * no_fof_points_mm[:, 1] / no_fof_points_mm[:, 2] + c_y).astype(int)

        # For every point, draw a circle around that point in the image
        color = (0, 0, 0)
        alpha = 0.75
        radius = 5
        img_with_debug_info = img.copy()
        for u, v in zip(us, vs):
            cv2.circle(img_with_debug_info, (u, v), radius, color, -1)
        return cv2.addWeighted(img_with_debug_info, alpha, img.copy(), 1 - alpha, 0)

    @override
    def visualize_img(self, img: npt.NDArray, t: npt.NDArray) -> None:
        # Convert the image to a pointcloud
        pointclouds = [
            depth_img_to_pointcloud(
                img,
                *self.crop_top_left,
                f_x=self.camera_matrix[0],
                f_y=self.camera_matrix[4],
                c_x=self.camera_matrix[2],
                c_y=self.camera_matrix[5],
                transform=t[0, :, :],
            )
        ]
        colors = [[0, 0, 1]]
        sizes = [5]
        markerstyles = ["o"]
        labels = ["Test"]

        if self.no_fof_points is not None:
            print(f"Visualizing the {self.no_fof_points.shape[0]} stored no FoF points")
            pointclouds.append(self.no_fof_points)
            colors.append([1, 0, 0])
            sizes.append(5)
            markerstyles.append("^")
            labels.append("Train")

        show_3d_scatterplot(
            pointclouds,
            colors,
            sizes,
            markerstyles,
            labels,
            title="Img vs. Stored No FoF Points",
        )
