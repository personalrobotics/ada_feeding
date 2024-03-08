"""
This file contains an abstract class, FoodOnForkDetector, that takes in a single depth
image and returns a confidence in [0,1] that there is food on the fork.
"""
# Standard imports
from abc import ABC, abstractmethod
from enum import Enum
import os
import time
from typing import Callable, Optional, Tuple

# Third-party imports
import numpy as np
import numpy.typing as npt
from overrides import override
from sensor_msgs.msg import CameraInfo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from tf2_ros.buffer import Buffer

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
        self.__tf_buffer = None
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
    def tf_buffer(self) -> Optional[Buffer]:
        """
        The tf buffer for the depth image.

        Returns
        -------
        tf_buffer: The tf buffer for the depth image, or None if not set.
        """
        return self.__tf_buffer

    @tf_buffer.setter
    def tf_buffer(self, tf_buffer: Buffer) -> None:
        """
        Sets the tf buffer for the depth image.

        Parameters
        ----------
        tf_buffer: The tf buffer for the depth image.
        """
        self.__tf_buffer = tf_buffer

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

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray[int]) -> None:
        """
        Trains the perception algorithm on a dataset of depth images and
        corresponding labels.

        Parameters
        ----------
        X: The depth images to train on.
        y: The labels for the depth images.
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
        self, X: npt.NDArray
    ) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        """
        Predicts the probability that there is food on the fork for a set of
        depth images.

        Parameters
        ----------
        X: The depth images to predict on.

        Returns
        -------
        y: The predicted probabilities that there is food on the fork.
        statuses: The status of each prediction. Must be one of the const values
            declared in the FoodOnForkDetection message.
        """

    def predict(
        self,
        X: npt.NDArray,
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
            proba, statuses = self.predict_proba(X)
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

    def visualize_img(self, img: npt.NDArray) -> None:
        """
        Visualizes a depth image. This function is used for debugging, so it helps
        to not only visualize the img, but also subclass-specific information that
        can help explain why the img would result in a particular prediction.

        It is acceptable for this function to block until the user closes a window.

        Parameters
        ----------
        img: The depth image to visualize.
        """
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
    def fit(self, X: npt.NDArray, y: npt.NDArray[int]) -> None:
        pass

    @override
    def save(self, path: str) -> str:
        return ""

    @override
    def load(self, path: str) -> None:
        pass

    @override
    def predict_proba(
        self, X: npt.NDArray
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

    def __init__(
        self,
        camera_matrix: npt.NDArray,
        prop_no_fof_points_to_store: float = 0.5,
        min_points: int = 40,
        min_distance: float = 0.001,
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
        verbose: Whether to print debug messages.
        """
        # pylint: disable=too-many-arguments
        # These many are fine.

        super().__init__(verbose)
        self.camera_matrix = camera_matrix
        self.prop_no_fof_points_to_store = prop_no_fof_points_to_store
        self.min_points = min_points
        self.min_distance = min_distance

        # The attributes that are stored/loaded by the model
        self.no_fof_points = None
        self.clf = None

    @staticmethod
    def distance_between_pointclouds(
        pointcloud1: npt.NDArray,
        pointcloud2: npt.NDArray,
        aggregator: Callable[npt.NDArray, float] = np.mean,
    ) -> npt.NDArray:
        """
        For every point in pointcloud1, gets the minimum distance to points in
        pointcloud2, and then aggregates those distances. Note that this is not
        symmetric; the order of the pointclouds matters.

        Parameters
        ----------
        pointcloud1: The test pointcloud. Size (n, k).
        pointcloud2: The training pointcloud. Size (m, k).
        aggregator: The function to use to aggregate the distances. Should take
            in a size (n,) np array and output a float. Default is np.mean.

        Returns
        -------
        distance: The aggregate of the minimum distances from each point in the test
            pointcloud to the nearest point in the training pointcloud.
        """
        return aggregator(np.min(pairwise_distances(pointcloud1, pointcloud2), axis=1))

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
    def fit(self, X: npt.NDArray, y: npt.NDArray[int]) -> None:
        # pylint: disable=too-many-locals
        # This is the main logic of the algorithm, so it's okay to have a lot of
        # local variables.

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

        # Get the distances from each point in the "val set" to the stored points
        val_distances = []
        for i, pointcloud in enumerate(val_pointclouds):
            if self.verbose:
                print(
                    "Computing distance to stored points for val point "
                    f"{i}/{val_pointclouds.shape[0]}"
                )
            val_distances.append(
                FoodOnForkDistanceToNoFOFDetector.distance_between_pointclouds(
                    pointcloud, self.no_fof_points
                )
            )
        val_distances = np.array(val_distances)

        # Train the classifier
        if self.verbose:
            print("Training the classifier")
        self.clf = LogisticRegression(random_state=self.seed, penalty=None)
        self.clf.fit(val_distances.reshape(-1, 1), val_labels)
        if self.verbose:
            print(
                f"Trained the classifier, with coeff {self.clf.coef_} and "
                f"intercept {self.clf.intercept_}"
            )

    @override
    def save(self, path: str) -> str:
        if self.no_fof_points is None or self.clf is None:
            raise ValueError(
                "The model has not been trained yet. Call fit before saving."
            )
        # If the path has an extension, remove it.
        path = os.path.splitext(path)[0]
        np.savez_compressed(
            path,
            no_fof_points=self.no_fof_points,
            clf=np.array([self.clf], dtype=object),
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

    @override
    def predict_proba(
        self, X: npt.NDArray
    ) -> Tuple[npt.NDArray[float], npt.NDArray[int]]:
        probas = []
        statuses = []

        # Get the prediction per image.
        print_every_n = 50
        for i, img in enumerate(X):
            if self.verbose and i % print_every_n == 0:
                start_time = time.time()

            # Convert the image to a pointcloud
            pointcloud = depth_img_to_pointcloud(
                img,
                *self.crop_top_left,
                f_x=self.camera_matrix[0],
                f_y=self.camera_matrix[4],
                c_x=self.camera_matrix[2],
                c_y=self.camera_matrix[5],
            )

            # If there are enough points, use the classifier to predict the probability
            # of food on the fork. Else, return an error status
            if len(pointcloud) >= self.min_points:
                distance = (
                    FoodOnForkDistanceToNoFOFDetector.distance_between_pointclouds(
                        pointcloud, self.no_fof_points
                    )
                )
                proba = self.clf.predict_proba(np.array([[distance]]))[0, 1]
                probas.append(proba)
                statuses.append(FoodOnForkDetection.SUCCESS)
                if self.verbose and i % print_every_n == 0:
                    print(
                        f"Predicted on pointcloud {i}/{X.shape[0]} in "
                        f"{time.time() - start_time} seconds"
                    )
            else:
                probas.append(np.nan)
                statuses.append(FoodOnForkDetection.ERROR_TOO_FEW_POINTS)

        return np.array(probas), np.array(statuses)

    @override
    def visualize_img(self, img: npt.NDArray) -> None:
        # Convert the image to a pointcloud
        pointclouds = [
            depth_img_to_pointcloud(
                img,
                *self.crop_top_left,
                f_x=self.camera_matrix[0],
                f_y=self.camera_matrix[4],
                c_x=self.camera_matrix[2],
                c_y=self.camera_matrix[5],
            )
        ]
        colors = [[0, 0, 1]]
        sizes = [5]
        markerstyles = ["o"]
        labels = ["Test"]

        if self.no_fof_points is not None:
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
