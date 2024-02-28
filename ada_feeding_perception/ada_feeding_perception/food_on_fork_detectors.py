"""
This file contains an abstract class, FoodOnForkDetector, that takes in a single depth
image and returns a confidence in [0,1] that there is food on the fork.
"""
# Standard imports
from abc import ABC, abstractmethod
from enum import Enum
import time
from typing import Optional

# Third-party imports
import numpy as np
import numpy.typing as npt
from overrides import override
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
        self.__seed = int(time.time() * 1000)

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
        self,
        X: npt.NDArray,
        lower_thresh: float,
        upper_thresh: float,
        proba: Optional[npt.NDArray] = None,
    ) -> npt.NDArray[FoodOnForkLabel]:
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

    def __init__(self, proba: float) -> None:
        """
        Initializes the dummy perception algorithm.

        Parameters
        ----------
        proba: The probability that the dummy algorithm should always predict.
        """
        super().__init__()
        self.proba = proba

    @override
    def fit(self, X: npt.NDArray, y: npt.NDArray[FoodOnForkLabel]) -> None:
        pass

    @override
    def save(self, path: str) -> str:
        return ""

    @override
    def load(self, path: str) -> None:
        pass

    @override
    def predict_proba(self, X: npt.NDArray) -> npt.NDArray[FoodOnForkLabel]:
        return np.full(X.shape[0], self.proba)
