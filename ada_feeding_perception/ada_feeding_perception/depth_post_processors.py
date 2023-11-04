#!/usr/bin/env python3
"""
This module contains a series of post-processors for depth images.
"""

# Standard imports
from typing import Callable

# Third-party imports
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
from sensor_msgs.msg import Image


def create_mask_post_processor(
    mask_img: npt.NDArray[np.uint8], bridge: CvBridge
) -> Callable[[Image], Image]:
    """
    Creates the mask post-processor function, which applies a fixed mask to the image.

    Parameters
    ----------
    mask_img : npt.NDArray[np.uint8]
        The mask to apply.
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    Callable[[Image], Image]
        The post-processor function.
    """

    def mask_post_processor(msg: Image) -> Image:
        """
        Applies a fixed mask to the image. Scales the mask to the image.

        Parameters
        ----------
        msg : Image
            The image to mask.

        Returns
        -------
        Image
            The masked image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = bridge.imgmsg_to_cv2(msg)

        # Scale the mask to be the size of the img
        mask = cv.resize(mask_img, img.shape[:2][::-1])

        # Apply the mask to the img
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # Get the new img message
        masked_msg = bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg

    return mask_post_processor


def create_temporal_post_processor(
    temporal_window_size: int, bridge: CvBridge
) -> Callable[[Image], Image]:
    """
    Creates the temporal post-processor function, with a dedicated window.
    This post-processor only keeps pixels that are consistently non-zero across
    the temporal window.

    Parameters
    ----------
    temporal_window_size : int
        The size of the temporal window (num frames).
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    Callable[[Image], Image]
        The post-processor function.
    """

    temporal_window = []

    def temporal_post_processor(msg: Image) -> Image:
        """
        The temporal post-processor stores the last `temporal_window_size` images.
        It returns the most recent image, but only the pixels in that image that
        are non-zero across all images in the window.

        Parameters
        ----------
        msg : Image
            The image to process.

        Returns
        -------
        Image
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = bridge.imgmsg_to_cv2(msg)

        # Add it to the window
        temporal_window.append(img)

        # If the window is full, remove the oldest image
        if len(temporal_window) > temporal_window_size:
            temporal_window.pop(0)

        # Get the mask
        mask = (img > 0).astype(np.uint8)
        for i in range(0, len(temporal_window) - 1):
            mask = np.bitwise_and(mask, (temporal_window[i] > 0).astype(np.uint8))

        # Mask the latest image
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # Get the new img message
        masked_msg = bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg

    return temporal_post_processor


def create_spatial_post_processor(
    spatial_num_pixels: int, bridge: CvBridge
) -> Callable[[Image], Image]:
    """
    Creates the spatial post-processor function, which applies the `opening` morpholical
    transformation to the image.

    Parameters
    ----------
    spatial_num_pixels : int
        The size of the square opening kernel.
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    Callable[[Image], Image]
        The post-processor function.
    """

    def spatial_post_processor(msg: Image) -> Image:
        """
        Applies the `opening` morpholical transformation to the image.

        Parameters
        ----------
        msg : Image
            The image to process.

        Returns
        -------
        Image
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = bridge.imgmsg_to_cv2(msg)

        # Apply the opening morphological transformation
        mask = (img > 0).astype(np.uint8)
        kernel = np.ones((spatial_num_pixels, spatial_num_pixels), np.uint8)
        opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        masked_img = cv.bitwise_and(img, img, mask=opened_mask)

        # Get the new img message
        masked_msg = bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg

    return spatial_post_processor


def create_threshold_post_processor(
    threshold_min: int, threshold_max: int, bridge: CvBridge
) -> Callable[[Image], Image]:
    """
    Creates the threshold post-processor function, which only keeps pixels within a
    min and max value.

    Parameters
    ----------
    threshold_min : int
        The minimum threshold.
    threshold_max : int
        The maximum threshold.
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    Callable[[Image], Image]
        The post-processor function.
    """

    def threshold_post_processor(msg: Image) -> Image:
        """
        Applies a threshold to the image.

        Parameters
        ----------
        msg : Image
            The image to process.
        threshold_min : int
            The minimum threshold.
        threshold_max : int
            The maximum threshold.
        bridge : CvBridge
            The CvBridge object to use.

        Returns
        -------
        Image
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img = bridge.imgmsg_to_cv2(msg)

        # Apply the threshold
        mask = cv.inRange(img, threshold_min, threshold_max)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # Get the new img message
        masked_msg = bridge.cv2_to_imgmsg(masked_img)
        masked_msg.header = msg.header

        return masked_msg

    return threshold_post_processor
