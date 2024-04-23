#!/usr/bin/env python3
"""
This module contains a series of post-processors for depth images.
"""

# Standard imports
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

# Local imports
from .helpers import ros_msg_to_cv2_image, cv2_image_to_ros_msg

_CV2ImageMsg = Tuple[npt.NDArray, Header]
_ImageMsgTypes = Union[Image, CompressedImage, _CV2ImageMsg]


def post_processor_chain(
    fns: List[Callable[[Any], Callable[[_ImageMsgTypes], _CV2ImageMsg]]],
    kwargs: List[Dict[str, Any]],
    compress: Optional[bool] = None,
) -> Callable[[Union[Image, CompressedImage]], _ImageMsgTypes]:
    """
    Chains together multiple post-processors into a single function, where
    the first post-processor takes as input a ROS2 message, the final one
    outputs a ROS2 message if output_msg else a _CV2ImageMsg, and the intermediate
    ones output a _CV2ImageMsg.

    The main purpose of this function is to prevent unnecessary conversions
    back-and-forth between ROS2 and CV2 image types.

    Parameters
    ----------
    fns: List[Callable[[Any], Callable[[_ImageMsgTypes], _ImageMsgTypes]]]
        A list of post-processor functions from this file.
    kwargs: List[Dict[str, Any]]
        The parameters to pass those functions.
    compress: Optional[bool]
        If None, the return message will be a _CV2ImageMsg. Else, it will be either
        a CompressedImage or Image, as specified by the bool.

    Returns
    -------
    Callable[[Union[Image, CompressedImage]], _ImageMsgTypes]
        A function that runs the post-processors and returns the desired image type.
    """
    assert len(fns) == len(
        kwargs
    ), "The length of the functions and kwargs in a post-processor chain must be equal"
    n = len(fns)
    post_processors = []
    bridge = None
    for i in range(n):
        post_processors.append(fns[i](**kwargs[i]))
        if "bridge" in kwargs[i]:
            bridge = kwargs[i]["bridge"]

    def chain(msg: Union[Image, CompressedImage]) -> _ImageMsgTypes:
        """
        Post-process the image through all the specified post-processors.

        Parameters
        ----------
        msg : Union[Image, CompressedImage]
            The image to mask.

        Returns
        -------
        _ImageMsgTypes
            The masked image. All other attributes of the message remain the same.
        """
        # Run the post-processors
        for post_processor in post_processors:
            msg = post_processor(msg)

        # Convert to the return type
        if compress is None:
            return msg
        masked_img, header = msg
        out_msg = cv2_image_to_ros_msg(masked_img, compress, bridge=bridge)
        out_msg.header = header
        return out_msg

    return chain


def __in_msg(msg: _ImageMsgTypes, bridge: CvBridge) -> _CV2ImageMsg:
    """
    A shared function across all post-processors to process the input image.

    Parameters
    ----------
    msg : _ImageMsgTypes
        The image to mask.
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    _CV2ImageMsg
        The CV2 image and its associated header.
    """
    if isinstance(msg, (CompressedImage, Image)):
        return (
            ros_msg_to_cv2_image(msg, bridge),
            msg.header,
        )
    return msg


def create_identity_post_processor(
    bridge: CvBridge,
) -> Callable[[_ImageMsgTypes], _ImageMsgTypes]:
    """
    A dummy identity post-processer, that just returns the same message.

    Parameters
    ----------
    bridge : CvBridge
        The CvBridge object to use.

    Returns
    -------
    Callable[[_ImageMsgTypes], _CV2ImageMsg]
        The post-processor function.
    """

    def identity_post_processor(msg: _ImageMsgTypes) -> _CV2ImageMsg:
        """
        An identity post-processor that merely returns the same image.

        Parameters
        ----------
        msg : _ImageMsgTypes
            The image to mask.

        Returns
        -------
        _CV2ImageMsg
            The masked image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        return __in_msg(msg, bridge)

    return identity_post_processor


def create_mask_post_processor(
    mask_img: npt.NDArray[np.uint8],
    bridge: CvBridge,
) -> Callable[[_ImageMsgTypes], _CV2ImageMsg]:
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
    Callable[[_ImageMsgTypes], _CV2ImageMsg]
        The post-processor function.
    """

    def mask_post_processor(msg: _ImageMsgTypes) -> _CV2ImageMsg:
        """
        Applies a fixed mask to the image. Scales the mask to the image.

        Parameters
        ----------
        msg : _ImageMsgTypes
            The image to mask.

        Returns
        -------
        _CV2ImageMsg
            The masked image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img, header = __in_msg(msg, bridge)

        # Scale the mask to be the size of the img
        mask = cv.resize(mask_img, img.shape[:2][::-1])

        # Apply the mask to the img
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # Return the new img message
        return masked_img, header

    return mask_post_processor


def create_temporal_post_processor(
    temporal_window_size: int,
    bridge: CvBridge,
) -> Callable[[_ImageMsgTypes], _ImageMsgTypes]:
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
    Callable[[_ImageMsgTypes], _CV2ImageMsg]
        The post-processor function.
    """

    temporal_window = []

    def temporal_post_processor(msg: _ImageMsgTypes) -> _CV2ImageMsg:
        """
        The temporal post-processor stores the last `temporal_window_size` images.
        It returns the most recent image, but only the pixels in that image that
        are non-zero across all images in the window.

        Parameters
        ----------
        msg : _ImageMsgTypes
            The image to process.

        Returns
        -------
        _CV2ImageMsg
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img, header = __in_msg(msg, bridge)

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
        return masked_img, header

    return temporal_post_processor


def create_spatial_post_processor(
    spatial_num_pixels: int,
    bridge: CvBridge,
) -> Callable[[_ImageMsgTypes], _ImageMsgTypes]:
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
    Callable[[_ImageMsgTypes], _CV2ImageMsg]
        The post-processor function.
    """

    def spatial_post_processor(msg: _ImageMsgTypes) -> _CV2ImageMsg:
        """
        Applies the `opening` morpholical transformation to the image.

        Parameters
        ----------
        msg : _ImageMsgTypes
            The image to process.

        Returns
        -------
        _CV2ImageMsg
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img, header = __in_msg(msg, bridge)

        # Apply the opening morphological transformation
        mask = (img > 0).astype(np.uint8)
        kernel = np.ones((spatial_num_pixels, spatial_num_pixels), np.uint8)
        opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        masked_img = cv.bitwise_and(img, img, mask=opened_mask)

        # Get the new img message
        return masked_img, header

    return spatial_post_processor


def create_threshold_post_processor(
    threshold_min: int,
    threshold_max: int,
    bridge: CvBridge,
) -> Callable[[_ImageMsgTypes], _ImageMsgTypes]:
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
    Callable[[_ImageMsgTypes], _CV2ImageMsg]
        The post-processor function.
    """

    def threshold_post_processor(msg: _ImageMsgTypes) -> _CV2ImageMsg:
        """
        Applies a threshold to the image.

        Parameters
        ----------
        msg : _ImageMsgTypes
            The image to process.
        threshold_min : int
            The minimum threshold.
        threshold_max : int
            The maximum threshold.
        bridge : CvBridge
            The CvBridge object to use.

        Returns
        -------
        _CV2ImageMsg
            The processed image. All other attributes of the message remain the same.
        """
        # Read the ROS msg as a CV image
        img, header = __in_msg(msg, bridge)

        # Apply the threshold
        mask = cv.inRange(img, threshold_min, threshold_max)
        masked_img = cv.bitwise_and(img, img, mask=mask)

        # Get the new img message
        return masked_img, header

    return threshold_post_processor
