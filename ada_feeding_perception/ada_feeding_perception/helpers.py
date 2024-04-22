"""
This file contains helper functions for the ada_feeding_perception package.
"""
# Standard imports
import os
import parse
import pprint
from typing import List, Optional, Tuple, Union
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Third-party imports
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node

try:
    from rosbags.typesys.types import (
        sensor_msgs__msg__CompressedImage as rCompressedImage,
        sensor_msgs__msg__Image as rImage,
    )
except (TypeError, ModuleNotFoundError) as err:
    rclpy.logging.get_logger("ada_feeding_perception_helpers").warn(
        "rosbags is not installed, or a wrong version is installed (needs 0.9.19). "
        f"Typechecking against rosbag types will not work. Error: {err}"
    )
from sensor_msgs.msg import CompressedImage, Image
from skimage.morphology import flood_fill


def show_normalized_depth_img(img, wait=True, window_name="img"):
    """
    Show the normalized depth image. Useful for debugging.

    Parameters
    ----------
    img: npt.NDArray
        The depth image to show.
    wait: bool, optional
        If True, wait for a key press before closing the window.
    window_name: str, optional
        The name of the window to show the image in.
    """
    # Show the normalized depth image
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype("uint8")
    cv2.imshow(window_name, img_normalized)
    cv2.waitKey(0 if wait else 1)


def show_3d_scatterplot(
    pointclouds: List[npt.NDArray],
    colors: List[npt.NDArray],
    sizes: List[int],
    markerstyles: List[str],
    labels: List[str],
    title: str,
    mean_colors: Optional[List[npt.NDArray]] = None,
    mean_sizes: Optional[List[int]] = None,
    mean_markerstyles: Optional[List[str]] = None,
):
    """
    Show a 3D scatterplot of the given point clouds.

    Parameters
    ----------
    pointclouds: List[npt.NDArray]
        The point clouds to show. Each point cloud should be a Nx3 array of
        points.
    colors: List[npt.NDArray]
        The colors to use for the points. Each color should be a size 3 array of
        colors RGB colos in range [0,1].
    sizes: List[int]
        The sizes to use for the points.
    markerstyles: List[str]
        The marker styles to use for the point clouds.
    labels: List[str]
        The labels to use for the point clouds.
    title: str
        The title of the plot.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    # This is meant to be a flexible function to help with debugging.

    # Check that the inputs are valid
    assert (
        len(pointclouds)
        == len(colors)
        == len(sizes)
        == len(markerstyles)
        == len(labels)
    )
    if mean_colors is not None:
        assert mean_sizes is not None
        assert mean_markerstyles is not None
        assert len(mean_colors) == len(mean_sizes) == len(mean_markerstyles)

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each point cloud
    configs = [pointclouds, colors, sizes, markerstyles, labels]
    if mean_colors is not None:
        configs += [mean_colors, mean_sizes, mean_markerstyles]
    for config in zip(*configs):
        pointcloud = config[0]
        color = config[1]
        size = config[2]
        markerstyle = config[3]
        label = config[4]
        ax.scatter(
            pointcloud[:, 0],
            pointcloud[:, 1],
            pointcloud[:, 2],
            color=color,
            s=size,
            label=label,
            marker=markerstyle,
        )
        if len(config) > 5:
            mean_color = config[5]
            mean_size = config[6]
            mean_markerstyle = config[7]
            mean = pointcloud.mean(axis=0)
            ax.scatter(
                mean[0].reshape((1, 1)),
                mean[1].reshape((1, 1)),
                mean[2].reshape((1, 1)),
                color=mean_color,
                s=mean_size,
                label=label + " mean",
                marker=mean_markerstyle,
            )

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Show the plot
    plt.show()


def depth_img_to_pointcloud(
    depth_image: npt.NDArray,
    u_offset: int,
    v_offset: int,
    f_x: float,
    f_y: float,
    c_x: float,
    c_y: float,
    unit_conversion: float = 1000.0,
    transform: Optional[npt.NDArray] = None,
) -> npt.NDArray:
    """
    Converts a depth image to a point cloud.

    Parameters
    ----------
    depth_image: The depth image to convert to a point cloud.
    u_offset: An offset to add to the column index of every pixel in the depth
        image. This is useful if the depth image was cropped.
    v_offset: An offset to add to the row index of every pixel in the depth
        image. This is useful if the depth image was cropped.
    f_x: The focal length of the camera in the x direction, using the pinhole
        camera model.
    f_y: The focal length of the camera in the y direction, using the pinhole
        camera model.
    c_x: The x-coordinate of the principal point of the camera, using the pinhole
        camera model.
    c_y: The y-coordinate of the principal point of the camera, using the pinhole
        camera model.
    unit_conversion: The depth values are divided by this constant. Defaults to 1000,
        as RealSense returns depth in mm, but we want the pointcloud in m.
    transform: An optional transform to apply to the point cloud. If set, this should
        be a 4x4 matrix.

    Returns
    -------
    pointcloud: The point cloud representation of the depth image.
    """
    # pylint: disable=too-many-arguments
    # Although we could reduce it by passing in a camera matrix, I prefer to
    # keep the arguments explicit.

    # Get the pixel coordinates
    pixel_coords = np.mgrid[: depth_image.shape[0], : depth_image.shape[1]]
    pixel_coords[0] += v_offset
    pixel_coords[1] += u_offset

    # Mask out values outside the depth range
    mask = depth_image > 0
    depth_values = depth_image[mask]
    pixel_coords = pixel_coords[:, mask]

    # Convert units (e.g., mm to m)
    depth_values = np.divide(depth_values, unit_conversion)

    # Convert to 3D coordinates
    pointcloud = np.zeros((depth_values.shape[0], 3))
    pointcloud[:, 0] = np.multiply(pixel_coords[1] - c_x, np.divide(depth_values, f_x))
    pointcloud[:, 1] = np.multiply(pixel_coords[0] - c_y, np.divide(depth_values, f_y))
    pointcloud[:, 2] = depth_values

    # Apply the transform if it exists
    if transform is not None:
        pointcloud = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
        pointcloud = np.dot(transform, pointcloud.T).T[:, :3]

    return pointcloud


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
    image_types = [Image]
    compressed_image_types = [CompressedImage]
    try:
        image_types.append(rImage)
        compressed_image_types.append(rCompressedImage)
    except NameError as _:
        # This only happens if rosbags wasn't imported, which is logged above.
        pass
    if bridge is None:
        bridge = CvBridge()
    if isinstance(msg, tuple(image_types)):
        return bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if isinstance(msg, tuple(compressed_image_types)):
        if ";" in msg.format: # compressed depth image
            encoding, _, fmt = parse.parse("{:s}; {:s} ({:s})", msg.format)
            if encoding.lower() != "16uc1" or fmt.lower() != "png":
                raise NotImplementedError(
                    f"Encoding ({encoding}) and format ({fmt}) not yet supported"
                )
            DEPTH_HEADER_SIZE = 12
            return cv2.imdecode(np.frombuffer(msg.data[DEPTH_HEADER_SIZE:], np.uint8), cv2.IMREAD_UNCHANGED)
        return bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
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
        return bridge.cv2_to_compressed_imgmsg(image, dst_format="jpeg")
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
