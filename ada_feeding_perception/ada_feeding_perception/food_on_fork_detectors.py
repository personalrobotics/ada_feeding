"""
This file contains an abstract class, FoodOnForkDetector, that takes in a single depth
image and returns a confidence in [0,1] that there is food on the fork.
"""
# Standard imports
from abc import ABC, abstractmethod
from enum import Enum
import os
import time
from typing import Optional, Tuple

# Third-party imports
import numpy as np
import numpy.typing as npt
from overrides import override
import scipy
from sensor_msgs.msg import CameraInfo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from tf2_ros.buffer import Buffer

# Local imports
from ada_feeding_perception.helpers import (
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
    TOO_FEW_POINTS = 3


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
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        """
        Predicts the probability that there is food on the fork for a set of
        depth images.

        Parameters
        ----------
        X: The depth images to predict on.

        Returns
        -------
        y: The predicted probabilities that there is food on the fork.
        """

    def predict(
        self,
        X: npt.NDArray,
        lower_thresh: float,
        upper_thresh: float,
        proba: Optional[npt.NDArray] = None,
    ) -> npt.NDArray[int]:
        """
        Predicts whether there is food on the fork for a set of depth images.

        Parameters
        ----------
        X: The depth images to predict on.
        lower_thresh: The lower threshold for food on the fork.
        upper_thresh: The upper threshold for food on the fork.
        proba: The predicted probabilities that there is food on the fork. If
            None, this function will call predict_proba to get the probabilities.

        Returns
        -------
        y: The predicted labels for whether there is food on the fork.
        """
        if proba is None:
            proba = self.predict_proba(X)
        return np.where(
            proba < lower_thresh,
            FoodOnForkLabel.NO_FOOD.value,
            np.where(
                proba > upper_thresh,
                FoodOnForkLabel.FOOD.value,
                FoodOnForkLabel.UNSURE.value,
            ),
        )


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
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        return np.full(X.shape[0], self.proba)


class FoodOnForkPointCloudTTestDetector(FoodOnForkDetector):
    """
    A food-on-fork detection algorithm that determines the probability that a
    test image and the most similar no-FoF image from the training set are from
    the same underlying distribution. The algorithm reasons about images as 3D
    point clouds and uses a t-test to compare the distributions of the two
    images.
    """

    # pylint: disable=too-many-instance-attributes
    # Necessary for this class.

    def __init__(
        self,
        camera_matrix: npt.NDArray,
        min_points: int = 40,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the food-on-fork detection algorithm.

        Parameters
        ----------
        camera_matrix: The camera intrinsic matrix (K).
        min_points: The minimum number of points in a pointcloud to consider it
            for comparison. If a pointcloud has fewer points than this, it will
            return a probability of nan (prediction of UNSURE).
        verbose: Whether to print debug messages.
        """
        # pylint: disable=too-many-arguments
        # Necessary for this class.
        super().__init__(verbose)
        self.camera_matrix = camera_matrix
        self.min_points = min_points

        self.no_fof_means = None
        self.no_fof_covs = None
        self.no_fof_ns = None

    def depth_to_pointcloud(self, depth_image: npt.NDArray) -> npt.NDArray:
        """
        Converts a depth image to a point cloud.

        Parameters
        ----------
        depth_image: The depth image to convert to a point cloud.

        Returns
        -------
        pointcloud: The point cloud representation of the depth image.
        """
        # Get the pixel coordinates
        pixel_coords = np.mgrid[
            int(self.crop_top_left[1]) : int(self.crop_bottom_right[1]),
            int(self.crop_top_left[0]) : int(self.crop_bottom_right[0]),
        ]
        # Mask out values outside the depth range
        mask = depth_image > 0
        depth_values = depth_image[mask]
        pixel_coords = pixel_coords[:, mask]
        # Convert mm to m
        depth_values = np.divide(depth_values, 1000.0)
        # Extract the values from the camera matrix
        f_x = self.camera_matrix[0]
        f_y = self.camera_matrix[4]
        c_x = self.camera_matrix[2]
        c_y = self.camera_matrix[5]
        # Convert to 3D coordinates
        pointcloud = np.zeros((depth_values.shape[0], 3))
        pointcloud[:, 0] = np.multiply(
            pixel_coords[1] - c_x, np.divide(depth_values, f_x)
        )
        pointcloud[:, 1] = np.multiply(
            pixel_coords[0] - c_y, np.divide(depth_values, f_y)
        )
        pointcloud[:, 2] = depth_values
        return pointcloud

    def fit(self, X: npt.NDArray, y: npt.NDArray[int]) -> None:
        """
        Converts all the no-FoF images to pointclouds. Gets the mean, covariance,
        and num points within each pointcloud and stores them.

        Parameters
        ----------
        X: The depth images to train on.
        y: The labels for the depth images.
        """
        # Get the most up-to-date camera info
        if self.camera_info is not None:
            self.camera_matrix = np.array(self.camera_info.k)
        no_fof_imgs = []
        no_fof_pointclouds = []
        for i, img in enumerate(X):
            if y[i] != FoodOnForkLabel.NO_FOOD.value:
                continue
            pointcloud = self.depth_to_pointcloud(img)
            if len(pointcloud) >= self.min_points:
                no_fof_imgs.append(img)
                no_fof_pointclouds.append(pointcloud)
        self.no_fof_means = np.array([np.mean(pc, axis=0) for pc in no_fof_pointclouds])
        self.no_fof_covs = np.array(
            [np.cov(pc, rowvar=False, bias=False) for pc in no_fof_pointclouds]
        )
        self.no_fof_ns = np.array([pc.shape[0] for pc in no_fof_pointclouds])
        self.no_fof_X = np.array(no_fof_imgs)

    @override
    def save(self, path: str) -> str:
        if (
            self.no_fof_means is None
            or self.no_fof_covs is None
            or self.no_fof_ns is None
        ):
            raise ValueError(
                "The model has not been trained yet. Call fit before saving."
            )
        # If the path has an extension, remove it.
        path = os.path.splitext(path)[0]
        np.savez_compressed(
            path,
            no_fof_means=self.no_fof_means,
            no_fof_covs=self.no_fof_covs,
            no_fof_ns=self.no_fof_ns,
            no_fof_X=self.no_fof_X,
        )
        return path + ".npz"

    @override
    def load(self, path: str) -> None:
        ext = os.path.splitext(path)[1]
        if len(ext) == 0:
            path = path + ".npz"
        params = np.load(path)
        self.no_fof_means = params["no_fof_means"]
        self.no_fof_covs = params["no_fof_covs"]
        self.no_fof_ns = params["no_fof_ns"]
        self.no_fof_X = params["no_fof_X"]

    @staticmethod
    def hotellings_t_test(
        samp1_means: npt.NDArray,
        samp1_covs: npt.NDArray,
        samp1_ns: npt.NDArray,
        samp2_mean: npt.NDArray,
        samp2_cov: npt.NDArray,
        samp2_n: int,
    ) -> npt.NDArray:
        """
        Performs a Hotelling's T^2 test to compare the distributions of pairwise
        samples where the underlying populations are assumed to be multivariate
        Gaussian distributions with unequal covariances. Based on:
        https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Hotellings_Two-Sample_T2.pdf

        Samp2 is expected to be a simple sample. Samp1 can be a list of m samples.
        This function computes the p-value pairwise between samp2 and each sample
        in samp1.

        Parameters
        ----------
        samp1_means: The means of the m samples to compare samp2 with. Shape (m, k)
        samp1_covs: The covariances of the m samples to compare samp2 with. Shape (m, k, k)
        samp1_ns: The number of points in each of the m samples to compare samp2 with. Shape (m,)
        samp2_mean: The mean of the sample to pairwise compare with each sample in samp1. Shape (k,)
        samp2_cov: The covariance of the sample to pairwise compare with each sample in samp1. Shape (k, k)
        samp2_n: The number of points in the sample to pairwise compare with each sample in samp1.

        Returns
        -------
        ps: The p-values of the pairwise tests between samp1 and every sample in samp2.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        # Necessary for this function.

        # Get sizes
        m, k = samp1_means.shape

        # Calculate the S Matrix, of size (m,k,k)
        samp1_covs_div_ns = samp1_covs / np.repeat(
            samp1_ns, [k**2] * m, axis=0
        ).reshape((m, k, k))
        samp2_cov_div_n = samp2_cov / samp2_n
        S = samp1_covs_div_ns + samp2_cov_div_n

        # Calculate the T^2 statistic, of size (m,)
        means_diff = samp1_means - samp2_mean  # (m,k)
        S_inv = np.linalg.inv(S)  # (m,k,k)
        t_sq = np.einsum(
            "ij,ij->i", means_diff, np.einsum("ijk,ik->ij", S_inv, means_diff)
        )  # (m,)

        # Define custom ot product and trace functions for this matrix shape
        def dot_mkk_mkk(a: npt.NDArray, b: npt.NDArray):
            return np.einsum("ijk,ikl->ijl", a, b)

        def trace_mkk(a: npt.NDArray):
            return np.einsum("ijj->i", a)

        # Calculate the degrees of freedom, of size (m,)
        df1 = np.repeat(k, m)
        df2 = np.divide(
            trace_mkk(dot_mkk_mkk(S, S)) + trace_mkk(S) ** 2.0,
            (
                (
                    trace_mkk(dot_mkk_mkk(samp1_covs_div_ns, samp1_covs_div_ns))
                    + trace_mkk(samp1_covs_div_ns) ** 2.0
                )
                / (samp1_ns - 1)
            )
            + (
                (
                    np.trace(np.dot(samp2_cov_div_n, samp2_cov_div_n))
                    + np.trace(samp2_cov_div_n) ** 2.0
                )
                / (samp2_n - 1)
            ),
        )

        # Calculate the corresponding F value
        f_vals = np.multiply(np.divide(df2 - df1 + 1, df1 * df2), t_sq)

        # Calculate the p values
        ps = 1 - scipy.stats.f.cdf(f_vals, df1, df2 - df1 + 1)

        return ps

    @staticmethod
    def test_hotelling_t_test() -> None:
        """
        Tests the Hotelling's T^2 test function.
        """
        # pylint: disable=too-many-locals
        # Necessary for this function.
        print("Testing Hotelling's T^2 test function...")

        # Verify that two samples with the same mean and covariance have a p-value
        # of 1.0
        m = 100
        k = 2
        n = 100
        samp1_means = np.zeros((m, k))
        cov_magnitude = 1.0
        samp1_covs = np.repeat(np.eye(k).reshape((1, k, k)), m, axis=0) * cov_magnitude
        samp1_ns = np.ones(m) * n
        samp2_mean = samp1_means[0]
        samp2_cov = samp1_covs[0]
        samp2_n = samp1_ns[0]
        ps = FoodOnForkPointCloudTTestDetector.hotellings_t_test(
            samp1_means, samp1_covs, samp1_ns, samp2_mean, samp2_cov, samp2_n
        )
        assert np.allclose(ps, 1.0)

        # Verify that two samples with small/large differences in mean relative to the
        # covariance have a p-value close to 1.0/0.0
        for div, target_p in [(100, 1.0), (1, 0.0)]:
            diff = cov_magnitude / (div)
            samp2_mean = np.array([samp1_means[0, 0] - diff, samp1_means[0, 1] + diff])
            ps = FoodOnForkPointCloudTTestDetector.hotellings_t_test(
                samp1_means, samp1_covs, samp1_ns, samp2_mean, samp2_cov, samp2_n
            )
            assert np.allclose(ps, target_p, atol=0.01)

        # Test with realistic values from pointclouds
        samp1_means = np.array(
            [
                [0.03519467, 0.05854858, 0.32684903],
            ]
        )
        samp1_covs = np.array(
            [
                [
                    [3.10068497e-05, -3.51810972e-06, 7.46688582e-06],
                    [-3.51810972e-06, 4.12632792e-05, -4.49831451e-05],
                    [7.46688582e-06, -4.49831451e-05, 5.09933589e-05],
                ],
            ]
        )
        samp1_ns = np.array([1795])
        samp2_mean = np.array([0.03419327, 0.05895837, 0.32637653])
        samp2_cov = np.array(
            [
                [7.00616784e-06, 4.17580593e-06, -4.68206042e-06],
                [4.17580593e-06, 2.97615922e-05, -3.61209300e-05],
                [-4.68206042e-06, -3.61209300e-05, 4.50266659e-05],
            ]
        )
        samp2_n = 1960
        ps = FoodOnForkPointCloudTTestDetector.hotellings_t_test(
            samp1_means, samp1_covs, samp1_ns, samp2_mean, samp2_cov, samp2_n
        )
        # Get the mean difference and the determinant of the cov matrix
        mean_diff = np.linalg.norm(samp1_means[0] - samp2_mean)
        cov_magnitude = np.linalg.norm(samp1_covs[0])
        print(ps)
        print(mean_diff, cov_magnitude)

        print("Hotelling's T^2 test function passed!")

    @override
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        if (
            self.no_fof_means is None
            or self.no_fof_covs is None
            or self.no_fof_ns is None
        ):
            raise ValueError(
                "The model has not been trained yet. Call fit before predicting."
            )
        pointclouds = [self.depth_to_pointcloud(img) for img in X]
        probas = []
        n = len(pointclouds)
        for i, pointcloud in enumerate(pointclouds):
            if self.verbose:
                print(f"Predicting on pointcloud {i+1}/{n}")
            m = pointcloud.shape[0]
            if m < self.min_points:
                # probas.append(np.nan)
                probas.append(0.0)
                continue
            # Calculate the T^2 statistic and p-value
            pointcloud_mean = np.mean(pointcloud, axis=0)
            pointcloud_cov = np.cov(pointcloud, rowvar=False, bias=False)
            pointcloud_n = pointcloud.shape[0]
            ps = FoodOnForkPointCloudTTestDetector.hotellings_t_test(
                self.no_fof_means,
                self.no_fof_covs,
                self.no_fof_ns,
                pointcloud_mean,
                pointcloud_cov,
                pointcloud_n,
            )
            closest_train_img_i = np.argmax(ps)
            print(f"pointcloud_mean {pointcloud_mean}")
            print(f"pointcloud_cov {pointcloud_cov}")
            print(f"pointcloud_n {pointcloud_n}")
            print(f"closest_train_img_mean {self.no_fof_means[closest_train_img_i]}")
            print(f"closest_train_img_cov {self.no_fof_covs[closest_train_img_i]}")
            print(f"closest_train_img_n {self.no_fof_ns[closest_train_img_i]}")
            p = ps[closest_train_img_i]
            print(f"p {p}")
            show_normalized_depth_img(X[i], wait=False, window_name="test_img")
            show_normalized_depth_img(
                self.no_fof_X[closest_train_img_i], wait=False, window_name="train_img"
            )

            # Sample from the mean and cov of each image to see how well the
            # distributional assumptions match.
            sampled_pointcloud = np.random.multivariate_normal(
                pointcloud_mean, pointcloud_cov, pointcloud_n
            )
            closest_train_img_sampled_pointcloud = np.random.multivariate_normal(
                self.no_fof_means[closest_train_img_i],
                self.no_fof_covs[closest_train_img_i],
                self.no_fof_ns[closest_train_img_i],
            )
            print("ASDF")
            print(pointcloud.tolist())
            print("GHJK")
            print(self.depth_to_pointcloud(self.no_fof_X[closest_train_img_i]).tolist())
            show_3d_scatterplot(
                [
                    # pointcloud,
                    sampled_pointcloud,
                    # self.depth_to_pointcloud(self.no_fof_X[closest_train_img_i]),
                    closest_train_img_sampled_pointcloud,
                ],
                colors=[
                    # [1, 0, 0],
                    [0, 1, 0],
                    # [0, 0, 1],
                    [0, 0, 0],
                ],
                sizes=[5, 5],  # ,5,5],
                markerstyles=["o", "x"],  # , "o", "x"],
                labels=["Test", "Train"],  # , "Test Sampled", "Train Sampled"],
                title="Test vs Train",
                mean_colors=[[0, 0, 1], [0, 0, 0]],  # , [0, 0, 1], [0, 0, 0]],
                mean_sizes=[20, 20],  # , 20, 20],
                mean_markerstyles=["^", "^"],  # , "^", "^"],
            )

            # p is the probability that the null hypothesis is true, i.e. the
            # probability that the pointcloud is from the same distribution as
            # the no-FoF pointclouds. Hence, we take 1 - p.
            probas.append(1.0 - p)

        return np.array(probas)


class FoodOnForkDistanceToNoFOFDetector(FoodOnForkDetector):
    """
    A perception algorithm that stores many no FoF points. It then calculates the
    average distance between each test point and the nearest no FoF point. It
    then trains a classifier to predict the probability of a test point being
    FoF based on that distance.
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
        prop_no_fof_points_to_store: The proportion of no FoF points to store.
        min_points: The minimum number of points in a pointcloud to consider it
            for comparison. If a pointcloud has fewer points than this, it will
            return a probability of nan (prediction of UNSURE).
        min_distance: The minimum distance between stored no FoF points.
        verbose: Whether to print debug messages.
        """
        super().__init__(verbose)
        self.camera_matrix = camera_matrix
        self.prop_no_fof_points_to_store = prop_no_fof_points_to_store
        self.min_points = min_points
        self.min_distance = min_distance

        self.no_fof_points = None
        self.clf = None

    def depth_to_pointcloud(self, depth_image: npt.NDArray) -> npt.NDArray:
        """
        Converts a depth image to a point cloud.

        Parameters
        ----------
        depth_image: The depth image to convert to a point cloud.

        Returns
        -------
        pointcloud: The point cloud representation of the depth image.
        """
        # Get the pixel coordinates
        pixel_coords = np.mgrid[
            int(self.crop_top_left[1]) : int(self.crop_bottom_right[1]),
            int(self.crop_top_left[0]) : int(self.crop_bottom_right[0]),
        ]
        # Mask out values outside the depth range
        mask = depth_image > 0
        depth_values = depth_image[mask]
        pixel_coords = pixel_coords[:, mask]
        # Convert mm to m
        depth_values = np.divide(depth_values, 1000.0)
        # Extract the values from the camera matrix
        f_x = self.camera_matrix[0]
        f_y = self.camera_matrix[4]
        c_x = self.camera_matrix[2]
        c_y = self.camera_matrix[5]
        # Convert to 3D coordinates
        pointcloud = np.zeros((depth_values.shape[0], 3))
        pointcloud[:, 0] = np.multiply(
            pixel_coords[1] - c_x, np.divide(depth_values, f_x)
        )
        pointcloud[:, 1] = np.multiply(
            pixel_coords[0] - c_y, np.divide(depth_values, f_y)
        )
        pointcloud[:, 2] = depth_values
        return pointcloud

    def get_distance_from_train_points(self, pointcloud: npt.NDArray) -> npt.NDArray:
        """
        Gets the average of the minimum distances from each point in the test
        pointcloud to the nearest no FoF point in the training set.

        Parameters
        ----------
        pointcloud: The test pointcloud. Size (n, 3).

        Returns
        -------
        distance: The average of the minimum distances from each point in the test
            pointcloud to the nearest no FoF point in the training set.
        """
        return np.mean(
            np.min(pairwise_distances(pointcloud, self.no_fof_points), axis=1)
        )

    @override
    def fit(self, X: npt.NDArray, y: npt.NDArray[int]) -> None:
        # Convert all images to pointclouds, removing those with too few points
        pointclouds = []
        y_pointclouds = []
        for i, img in enumerate(X):
            pointcloud = self.depth_to_pointcloud(img)
            if len(pointcloud) >= self.min_points:
                pointclouds.append(pointcloud)
                y_pointclouds.append(y[i])
        pointclouds = np.array(pointclouds, dtype=object)
        y_pointclouds = np.array(y_pointclouds)
        print(
            f"Got pointclouds, {len(pointclouds)} total, {len(pointclouds[y_pointclouds == FoodOnForkLabel.NO_FOOD.value])} no FoF, {len(pointclouds[y_pointclouds == FoodOnForkLabel.FOOD.value])} FoF"
        )

        # Split the no FoF and FoF pointclouds
        no_fof_pointclouds = pointclouds[y_pointclouds == FoodOnForkLabel.NO_FOOD.value]
        fof_pointclouds = pointclouds[y_pointclouds == FoodOnForkLabel.FOOD.value]

        # Randomly split the no FoF points
        no_fof_pointclouds_train, no_fof_pointclouds_val = train_test_split(
            no_fof_pointclouds,
            train_size=self.prop_no_fof_points_to_store,
            random_state=self.seed,
        )

        # Store the no FoF points
        all_no_fof_points = np.concatenate(no_fof_pointclouds_train)
        no_fof_points_to_store = all_no_fof_points[0:1, :]
        # Add points that are >= min_distance m away from the stored points
        for i in range(1, len(all_no_fof_points)):
            print(f"{i}/{len(all_no_fof_points)}")
            contender_point = all_no_fof_points[i]
            # Get the distance between the contender point and th stored points
            distance = np.min(
                pairwise_distances([contender_point], no_fof_points_to_store), axis=1
            )[0]
            if distance >= self.min_distance:
                no_fof_points_to_store = np.vstack(
                    [no_fof_points_to_store, contender_point]
                )
        self.no_fof_points = no_fof_points_to_store

        print("Split data, self.no_fof_points.shape", self.no_fof_points.shape)

        # Get the distances from each non-stored point and the stored points
        no_fof_distances = []
        for i, pc in enumerate(no_fof_pointclouds_val):
            print(f"{i}/{len(no_fof_pointclouds_val)}")
            no_fof_distances.append(self.get_distance_from_train_points(pc))
        no_fof_distances = np.array(no_fof_distances)
        print(f"Got no_fof_distances, {no_fof_distances}, {no_fof_distances.shape}")
        fof_distances = np.array(
            [self.get_distance_from_train_points(pc) for pc in fof_pointclouds]
        )
        print(f"Got fof_distances {fof_distances}, {fof_distances.shape}")

        # Train the classifier
        classifier_X = np.concatenate([no_fof_distances, fof_distances])
        classifier_y = np.concatenate(
            [np.zeros((no_fof_distances.shape[0],)), np.ones((fof_distances.shape[0],))]
        )
        print(
            f"Training classifier with {classifier_X} {classifier_X.shape} {classifier_y} {classifier_y.shape}"
        )
        self.clf = LogisticRegression(random_state=self.seed, penalty=None)
        self.clf.fit(classifier_X.reshape(-1, 1), classifier_y)
        print("Trained classifier")
        print(self.clf.coef_, self.clf.intercept_)

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
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray:
        probas = []

        # show_3d_scatterplot(
        #     [self.no_fof_points],
        #     colors=[[0, 0, 1]],
        #     sizes=[5],
        #     markerstyles=["o"],
        #     labels=["Train"],
        #     title="Train Points",
        # )
        # raise Exception()

        # Convert all images to pointclouds, removing those with too few points
        num_images_no_points = 0
        for i, img in enumerate(X):
            start_time = time.time()
            pointcloud = self.depth_to_pointcloud(img)
            if len(pointcloud) >= self.min_points:
                distance = self.get_distance_from_train_points(pointcloud)
                # print("distance", distance)
                proba = self.clf.predict_proba(np.array([[distance]]))[0, 1]
                # print(f"proba {proba}")
                probas.append(proba)
                if i % 50 == 0:
                    print(
                        f"Predicted on pointcloud {i+1}/{len(X)} in {time.time() - start_time} seconds"
                    )
            else:
                probas.append(np.nan)
                num_images_no_points += 1

        print(f"num_images_no_points {num_images_no_points}")
        return np.array(probas)

    @override
    def predict(
        self,
        X: npt.NDArray,
        lower_thresh: float,
        upper_thresh: float,
        proba: Optional[npt.NDArray] = None,
    ) -> npt.NDArray[int]:
        if proba is None:
            proba = self.predict_proba(X)
        return np.where(
            np.isnan(proba),
            FoodOnForkLabel.TOO_FEW_POINTS.value,
            np.where(
                proba <= lower_thresh,
                FoodOnForkLabel.NO_FOOD.value,
                np.where(
                    proba > upper_thresh,
                    FoodOnForkLabel.FOOD.value,
                    FoodOnForkLabel.UNSURE.value,
                ),
            ),
        )


if __name__ == "__main__":
    FoodOnForkPointCloudTTestDetector.test_hotelling_t_test()
