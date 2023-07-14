"""
This file contains helper functions for the ada_feeding_perception package.
"""
# Standard imports
import os
import pprint
from typing import Tuple
from urllib.parse import urljoin
from urllib.request import urlretrieve

# Third-party imports
import cv2
import numpy as np
import numpy.typing as npt
from skimage.morphology import flood_fill


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


def bbox_from_mask(mask: npt.NDArray[bool]) -> BoundingBox:
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
    image: np.ndarray,
    mask: npt.NDArray[bool],
    point: Tuple[int, int],
    bbox: BoundingBox,
) -> Tuple[np.ndarray, npt.NDArray[bool], Tuple[int, int]]:
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
    image: np.ndarray,
    mask: npt.NDArray[bool],
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
    mask: npt.NDArray[bool], point: Tuple[int, int]
) -> npt.NDArray[bool]:
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
        raise ValueError(
            "Mask must be a 2D array, it instead has shape {}".format(mask.shape)
        )
    if (
        point[1] < 0
        or point[1] >= mask.shape[0]
        or point[0] < 0
        or point[0] >= mask.shape[1]
    ):
        raise IndexError("Point %s is not in mask of shape %s" % (point, mask.shape))
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
