"""
This file contains helper functions for the ada_feeding_perception package.
"""
# Standard imports
import os
import pprint
from typing import Optional, Tuple, Union
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Third-party imports
import cv2
from cv_bridge import CvBridge
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
from rosbags.typesys.types import (
    sensor_msgs__msg__CompressedImage as rCompressedImage,
    sensor_msgs__msg__Image as rImage,
)
from sensor_msgs.msg import CompressedImage, Image
from skimage.morphology import flood_fill


def ros_msg_to_cv2_image(
    msg: Union[Image, rImage, CompressedImage, rCompressedImage],
    bridge: Optional[CvBridge] = None,
) -> npt.NDArray:
    """
    Convert a ROS Image or CompressedImage message to a cv2 image. By default,
    this will maintain the depth of the image (e.g., 16-bit depth for depth
    images) and maintain the format. Any conversions should be done outside
    of this function.

    NOTE: This has been tested with color messages that are Image and CompressedImage
    and depth messages that are Image. It has not been tested with depth messages
    that are CompressedImage.

    Parameters
    ----------
    msg: the ROS Image or CompressedImage message to convert
    bridge: the CvBridge to use for the conversion. This is only used if `msg`
        is a ROS Image message. If `bridge` is None, a new CvBridge will be
        created.
    """
    if isinstance(msg, (Image, rImage)):
        if bridge is None:
            bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if isinstance(msg, (CompressedImage, rCompressedImage)):
        # TODO: This should use bridge as well
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
    raise ValueError("msg must be a ROS Image or CompressedImage")


def cv2_image_to_ros_msg(
    image: npt.NDArray,
    compress: bool,
    bridge: Optional[CvBridge] = None,
    encoding: str = "passthrough",
) -> Union[Image, CompressedImage]:
    """
    Convert a cv2 image to a ROS Image or CompressedImage message. Note that this
    does not set the header of the message; that must be done outside of this
    function.

    NOTE: This has been tested with converting an 8-bit greyscale image to a
    CompressedImage message. It has not been tested in any other circumstance.

    Parameters
    ----------
    image: the cv2 image to convert
    compress: whether or not to compress the image. If True, a CompressedImage
        message will be returned. If False, an Image message will be returned.
    bridge: the CvBridge to use for the conversion. This is only used if `msg`
        is a ROS Image message. If `bridge` is None, a new CvBridge will be
        created.
    encoding: the encoding to use for the ROS Image message. This is only used
        if `compress` is False.
    """
    if bridge is None:
        bridge = CvBridge()
    if compress:
        success, compressed_image = cv2.imencode(".jpg", image)
        if success:
            return CompressedImage(
                format="jpeg",
                data=compressed_image.tostring(),
            )
        raise RuntimeError("Failed to compress image")
    # If we get here, we're not compressing the image
    return bridge.cv2_to_imgmsg(image, encoding=encoding)


def get_img_msg_type(
    topic: str, node: Node, timeout_sec: Optional[float] = 1.0
) -> type:
    """
    Get the type of the image message on the given topic.

    Parameters
    ----------
    topic: the topic to get the image message type for
    node: the node to use to get the topic type
    timeout_sec: the timeout to use when getting the topic type. If None, this
        will wait forever. If 0.0, this is non-blocking.

    Returns
    -------
    the type of the image message on the given topic, either Image or CompressedImage
    """
    # Spin the node once to get the publishers list
    rclpy.spin_once(node, timeout_sec=timeout_sec)

    # Resolve the topic name (e.g., handle remappings)
    final_topic = node.resolve_topic_name(topic)

    # Get the publishers on the topic
    topic_endpoints = node.get_publishers_info_by_topic(final_topic)

    # Return the type of the first publisher on this topic that publishes
    # an image message
    for endpoint in topic_endpoints:
        if endpoint.topic_type == "sensor_msgs/msg/CompressedImage":
            return CompressedImage
        if endpoint.topic_type == "sensor_msgs/msg/Image":
            return Image
    raise ValueError(
        f"No publisher found with img type for topic {final_topic}. "
        "Publishers: {[str(endpoint) for endpoint in topic_endpoints]}"
    )


def download_checkpoint(
    checkpoint_name: str, model_dir: str, checkpoint_base_url: str
) -> None:
    """
    Download the model checkpoint and save it in `model_dir`sss

    Parameters
    ----------
    checkpoint_name: the name of the checkpoint file to download, e.g.,
        sam_vit_b_01ec64.pth
    model_dir: the directory to save the model checkpoint to
    checkpoint_base_url: the base url to download the model checkpoint from

    Returns
    -------
    None
    """
    # Create the models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Download the model checkpoint
    save_path = os.path.join(model_dir, checkpoint_name)
    url = urljoin(checkpoint_base_url, checkpoint_name)
    urlretrieve(url, save_path)


class BoundingBox:
    """
    A class representing a bounding box on an image
    """

    # pylint: disable=too-few-public-methods
    # This is a data class

    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        """
        Parameters
        ----------
        xmin: the x-coordinate of the top-left corner of the bounding box
        ymin: the y-coordinate of the top-left corner of the bounding box
        xmax: the x-coordinate of the bottom-right corner of the bounding box
        ymax: the y-coordinate of the bottom-right corner of the bounding box
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def bbox_from_mask(mask: npt.NDArray[np.bool_]) -> BoundingBox:
    """
    Takes in a binary mask and returns the smallest axis-aligned bounding box
    that contains all the True pixels in the mask.

    Parameters
    ----------
    mask: a binary mask

    Returns
    -------
    ymin, xmin, ymax, xmax: the coordinates of top-left and bottom-right
        corners of the bounding box
    """
    # Find the bounding box coordinates from the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return BoundingBox(xmin, ymin, xmax, ymax)


def crop_image_mask_and_point(
    image: npt.NDArray,
    mask: npt.NDArray[np.bool_],
    point: Tuple[int, int],
    bbox: BoundingBox,
) -> Tuple[npt.NDArray, npt.NDArray[np.bool_], Tuple[int, int]]:
    """
    Crop the image and mask to the bounding box.

    Parameters
    ----------
    image: the image to crop
    mask: the mask to crop. This must have the same rows and columns as `image `.
    point: the point to crop (x,y)
    bbox: the bounding box to crop to

    Returns
    -------
    cropped_image: the cropped image. Has the same information as the original,
        but only in the bounding box.
    cropped_mask: the cropped mask. Has the same information as the original,
        but only in the bounding box.
    cropped_point: the cropped point.
    """
    cropped_image = image[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax]
    cropped_mask = mask[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax]
    # Update the point to be in the cropped image
    cropped_point = (point[0] - bbox.xmin, point[1] - bbox.ymin)
    return cropped_image, cropped_mask, cropped_point


def overlay_mask_on_image(
    image: npt.NDArray,
    mask: npt.NDArray[np.bool_],
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
):
    """
    Overlay a mask on an image.

    Parameters
    ----------
    image: the image to overlay the mask on
    mask: the mask to overlay on the image
    alpha: the alpha value to use for blending the image and mask
    color: the color to use for the mask

    Returns
    -------
    blended: the blended image
    """
    # Assuming mask is a binary mask, we convert to 3 channel, and apply a color (Green here)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask == 1] = color

    # Blend the image and the mask
    blended = cv2.addWeighted(image, alpha, mask_rgb, 1 - alpha, 0)
    return blended


def get_connected_component(
    mask: npt.NDArray[np.bool_], point: Tuple[int, int]
) -> npt.NDArray[np.bool_]:
    """
    Takes in a binary mask and returns a new mask that has only the connected
    component that contains the given point.

    This function considers two pixels connected if they share an edge (i.e., it
    excludes diagonals).

    Parameters
    ----------
    mask: a binary mask. This should be a 2D array of booleans.
    point: a point in the mask (x,y)

    Returns
    -------
    connected_component: a binary mask containing only the connected component
        that contains the given point

    Raises
    ------
    IndexError: if the given point is not in the mask
    ValueError: if the mask is not a 2D array
    """
    # Check that the point and mask satisfy the constraints
    if len(mask.shape) != 2:
        raise ValueError(f"Mask must be a 2D array, it instead has shape {mask.shape}")
    if (
        point[1] < 0
        or point[1] >= mask.shape[0]
        or point[0] < 0
        or point[0] >= mask.shape[1]
    ):
        raise IndexError(f"Point {point} is not in mask of shape {mask.shape}")
    # Convert mask to uint8
    mask_uint = mask.astype(np.uint8)
    # Flood fill the mask with 3. Swap x and y because flood_fill expects
    # (row, col) instead of (x, y).
    mask_filled = flood_fill(mask_uint, (point[1], point[0]), 3, connectivity=1)
    # Return the new mask
    return mask_filled == 3


def test_get_connected_component() -> None:
    """
    Test the get_connected_component function.
    """
    # Test that the function works for a simple case
    mask = np.array(
        [[0, 1, 0, 0, 1], [1, 1, 0, 0, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype=bool
    )
    point = (1, 1)
    ans = np.array(
        [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=bool
    )
    ret = get_connected_component(mask, point)
    print("ret")
    pprint.pprint(ret)
    print("ans")
    pprint.pprint(ans)
    assert np.all(ret == ans)

    # Test that the function doesn't count diagonals as connected
    mask = np.array(
        [[0, 1, 0, 0, 1], [1, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]], dtype=bool
    )
    point = (1, 1)
    ans = np.array(
        [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=bool
    )
    ret = get_connected_component(mask, point)
    print("ret")
    pprint.pprint(ret)
    print("ans")
    pprint.pprint(ans)
    assert np.all(ret == ans)

    # Test that the function treats x and y correctly
    mask = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=bool
    )
    point = (1, 3)
    ans = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=bool
    )
    ret = get_connected_component(mask, point)
    print("ret")
    pprint.pprint(ret)
    print("ans")
    pprint.pprint(ans)
    assert np.all(ret == ans)

    # Test that the function tests the bounds of the seed point correctly
    mask = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=bool
    )
    point = (3, 4)
    ans = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1]], dtype=bool
    )
    ret = get_connected_component(mask, point)
    print("ret")
    pprint.pprint(ret)
    print("ans")
    pprint.pprint(ans)
    assert np.all(ret == ans)


if __name__ == "__main__":
    test_get_connected_component()
