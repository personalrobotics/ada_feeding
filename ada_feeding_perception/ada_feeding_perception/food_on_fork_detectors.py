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
from tf2_ros.buffer import Buffer


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
        crop_top_left: Tuple[float, float] = (297, 248),
        crop_bottom_right: Tuple[float, float] = (425, 332),
        depth_min_mm: int = 310,
        depth_max_mm: int = 370,
        min_points: int = 40,
        verbose: bool = False,
    ) -> None:
        """
        Initializes the food-on-fork detection algorithm.

        Parameters
        ----------
        camera_matrix: The camera intrinsic matrix (K).
        crop_top_left, crop_bottom_right: Specifies the subset of the depth image
            to convert to a pointcloud. This is a rectanglar region around the fork.
        depth_min_mm, depth_max_mm: The minimum and maximum depth values to
            consider for the pointcloud. Points outside this range will be
            ignored.
        min_points: The minimum number of points in a pointcloud to consider it
            for comparison. If a pointcloud has fewer points than this, it will
            return a probability of nan (prediction of UNSURE).
        verbose: Whether to print debug messages.
        """
        # pylint: disable=too-many-arguments
        # Necessary for this class.
        super().__init__(verbose)
        self.camera_matrix = camera_matrix
        self.crop_top_left = crop_top_left
        self.crop_bottom_right = crop_bottom_right
        self.depth_min_mm = depth_min_mm
        self.depth_max_mm = depth_max_mm
        self.min_points = min_points

        self.no_fof_means = None
        self.no_fof_covs = None
        self.no_fof_ns = None

    def depth_to_pointcloud(
        self, depth_image: npt.NDArray, is_cropped: bool = False
    ) -> npt.NDArray:
        """
        Converts a depth image to a point cloud.

        Parameters
        ----------
        depth_image: The depth image to convert to a point cloud.
        is_cropped: Whether the depth image has already been cropped to the
            region of interest.

        Returns
        -------
        pointcloud: The point cloud representation of the depth image.
        """
        # Get the depth values
        if is_cropped:
            depth_values = depth_image
        else:
            depth_values = depth_image[
                int(self.crop_top_left[1]) : int(self.crop_bottom_right[1]),
                int(self.crop_top_left[0]) : int(self.crop_bottom_right[0]),
            ]
        # Get the pixel coordinates
        pixel_coords = np.mgrid[
            int(self.crop_top_left[1]) : int(self.crop_bottom_right[1]),
            int(self.crop_top_left[0]) : int(self.crop_bottom_right[0]),
        ]
        # Mask out values outside the depth range
        mask = (depth_values > self.depth_min_mm) & (depth_values < self.depth_max_mm)
        depth_values = depth_values[mask]
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
            self.camera_matrix = np.array(self.camera_info.K)
        no_fof_imgs = X[y == FoodOnForkLabel.NO_FOOD.value]
        # TODO: remove the `is_cropped` from below once we move to ROS bags as
        # the training set.
        no_fof_pointclouds = [
            self.depth_to_pointcloud(img, is_cropped=True) for img in no_fof_imgs
        ]
        self.no_fof_means = np.array([np.mean(pc, axis=0) for pc in no_fof_pointclouds])
        self.no_fof_covs = np.array(
            [np.cov(pc, rowvar=False, bias=False) for pc in no_fof_pointclouds]
        )
        self.no_fof_ns = np.array([pc.shape[0] for pc in no_fof_pointclouds])

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
        )
        return path + ".npz"

    @override
    def load(self, path: str) -> None:
        prefix, ext = os.path.splitext(path)
        if len(ext) == 0:
            path = path + ".npz"
        params = np.load(path)
        self.no_fof_means = params["no_fof_means"]
        self.no_fof_covs = params["no_fof_covs"]
        self.no_fof_ns = params["no_fof_ns"]

    # pylint: disable=too--many-arguments, too-many-locals
    # Necessary for this function.
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

        # Calculate the p value
        p = 1 - scipy.stats.f.cdf(f_vals, df1, df2 - df1 + 1)

        return p

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
        # TODO: remove the `is_cropped` from below once we move to ROS bags as
        # the training set.
        pointclouds = [self.depth_to_pointcloud(img, is_cropped=True) for img in X]
        probas = []
        n = len(pointclouds)
        for i in range(len(pointclouds)):
            if self.verbose:
                print(f"Predicting on pointcloud {i+1}/{n}")
            pointcloud = pointclouds[i]
            m = pointcloud.shape[0]
            if m < self.min_points:
                probas.append(np.nan)
                continue
            # Calculate the T^2 statistic and p-value
            ps = FoodOnForkPointCloudTTestDetector.hotellings_t_test(
                self.no_fof_means,
                self.no_fof_covs,
                self.no_fof_ns,
                np.mean(pointcloud, axis=0),
                np.cov(pointcloud, rowvar=False, bias=False),
                pointcloud.shape[0],
            )
            p = np.max(ps)
            # p is the probability that the null hypothesis is true, i.e. the
            # probability that the pointcloud is from the same distribution as
            # the no-FoF pointclouds. Hence, we take 1 - p.
            probas.append(1.0 - p)

        return np.array(probas)
