"""
This file contains an abstract class, FoodOnForkDetector, that takes in a single depth
image and returns a confidence in [0,1] that there is food on the fork.
"""
# Standard imports
from abc import ABC, abstractmethod
from enum import Enum

# Third-party imports
import numpy as np
import numpy.typing as npt
from sensor_msgs.msg import CameraInfo
from tf2_ros.buffer import Buffer


class FoodOnForkLabel(Enum):
    """
    An enumeration of possible labels for food on the fork.
    """

    UNSURE = -1
    NO_FOOD = 0
    FOOD = 1


class FoodOnForkDetector(ABC):
    """
    An abstract class for any perception algorithm that takes in a single depth
    image and returns a confidence in [0,1] that there is food on the fork.
    """

    def __init__(self) -> None:
        """
        Initializes the perception algorithm.
        """
        self.__camera_info = None
        self.__tf_buffer = None

    @property
    def camera_info(self) -> CameraInfo:
        """
        The camera info for the depth image.

        Returns
        -------
        camera_info: The camera info for the depth image.
        """
        assert self.__camera_info is not None, "Camera Info has not been set."
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
    def tf_buffer(self) -> Buffer:
        """
        The tf buffer for the depth image.

        Returns
        -------
        tf_buffer: The tf buffer for the depth image.
        """
        assert self.__tf_buffer is not None, "TF Buffer has not been set."
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

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray[FoodOnForkLabel]) -> None:
        """
        Trains the perception algorithm on a dataset of depth images and
        corresponding labels.

        Parameters
        ----------
        X: The depth images to train on.
        y: The labels for the depth images.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the perception algorithm to a file.

        Parameters
        ----------
        path: The path to save the perception algorithm to.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the perception algorithm from a file.

        Parameters
        ----------
        path: The path to load the perception algorithm from.
        """

    @abstractmethod
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray[FoodOnForkLabel]:
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
        self, X: npt.NDArray, lower_thresh: float, upper_thresh: float
    ) -> npt.NDArray[FoodOnForkLabel]:
        """
        Predicts whether there is food on the fork for a set of depth images.

        Parameters
        ----------
        X: The depth images to predict on.
        lower_thresh: The lower threshold for food on the fork.
        upper_thresh: The upper threshold for food on the fork.

        Returns
        -------
        y: The predicted labels for whether there is food on the fork.
        """
        proba = self.predict_proba(X)
        return np.where(
            proba < lower_thresh,
            FoodOnForkLabel.NO_FOOD,
            np.where(
                proba > upper_thresh, FoodOnForkLabel.FOOD, FoodOnForkLabel.UNSURE
            ),
        )


class FoodOnForkDummyDetector(FoodOnForkDetector):
    """
    A dummy perception algorithm that always predicts the same probability.
    """

    def __init__(self, proba: float) -> None:
        """
        Initializes the dummy perception algorithm.

        Parameters
        ----------
        proba: The probability that the dummy algorithm should always predict.
        """
        super().__init__()
        self.proba = proba

    def fit(self, X: npt.NDArray, y: npt.NDArray[FoodOnForkLabel]) -> None:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def predict_proba(self, X: npt.NDArray) -> npt.NDArray[FoodOnForkLabel]:
        return np.full(X.shape[0], self.proba)
