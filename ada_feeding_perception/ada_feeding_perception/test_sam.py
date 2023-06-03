"""
This script demonstrates how to use the Segment Anything Model (SAM) to
segment an object from an image given a point inside the object. Unlike the
other scripts in this package, this script does not use ROS. Therefore, it can
be useful as an initial check to ensure Segment Anything works on your computer.
"""
# Standard imports
import argparse
import json
import os

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch

# Local imports
from ada_feeding_perception.helpers import (
    bbox_from_mask,
    crop_image_mask_and_point,
    download_checkpoint,
    get_connected_component,
    overlay_mask_on_image,
)

# Configure the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_image", type=str, required=True, help="path to input image"
)
parser.add_argument(
    "--input_point", type=str, required=True, help="path to input point file"
)
args = parser.parse_args()

# Read the command-line arguments
image_path = args.input_image
input_point = np.array(json.loads(args.input_point))

# Model parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_b"
model_name = "sam_vit_b_01ec64.pth"
checkpoint_base_url = "https://dl.fbaipublicfiles.com/segment_anything/"
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "model"))
sam_checkpoint = os.path.join(model_dir, model_name)
# if the model can't be found download it
if not os.path.isfile(sam_checkpoint):
    print("SAM model checkpoint not found. Downloading...")
    download_checkpoint(model_name, model_dir, checkpoint_base_url)
    print("Download complete! Model saved at {}".format(sam_checkpoint))

# Read the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image shape: {}".format(image.shape))

# Load the model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

# Segment the image with that input point
masks, scores, logits = predictor.predict(
    point_coords=np.array([input_point]),
    point_labels=np.array([1]),
    multimask_output=True,
    # When False, it will return a single mask
)

# Sort the masks from highest to lowest score
scored_masks = list(zip(scores, masks))
scored_masks_sorted = sorted(scored_masks, key=lambda x: x[0], reverse=True)


def show_points(coords, labels, ax, marker_size=128):
    """
    Render the points in `coords` on `ax` with the corresponding `labels`.

    Parameters
    ----------
    coords: the coordinates of the points to render
    labels: the labels of the points to render, 1 is green and 0 is red.
    ax: the matplotlib axis to render the points on
    marker_size: the size of the markers to use
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for points, color in [(pos_points, "green"), (neg_points, "red")]:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1,
        )


number_output = 3
# After getting the masks
for i, (score, mask) in enumerate(scored_masks_sorted):
    if i >= number_output:
        break
    # Clean the mask to only contain the connected component containing
    # the seed point
    cleaned_mask = get_connected_component(mask, input_point)
    # Compute the bounding box
    bbox = bbox_from_mask(cleaned_mask)
    # Crop the image and the mask
    cropped_image, cropped_mask, _ = crop_image_mask_and_point(
        image, cleaned_mask, input_point, bbox
    )
    # Convert the mask to an image
    mask_img = np.where(cropped_mask, 255, 0).astype(np.uint8)

    # Overlay the mask on the image
    overlaid_image = overlay_mask_on_image(cropped_image, cropped_mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.gca().scatter(
        [input_point[0]],
        [input_point[1]],
        color="green",
        marker="*",
        s=128,
        edgecolor="white",
        linewidth=1,
    )
    plt.axis("on")
    plt.show()

    # Create subplots to display the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the cropped image
    axes[0].imshow(cropped_image)
    axes[0].axis("off")
    axes[0].set_title("Cropped Image {}".format(i))

    # Show the overlaid image
    axes[1].imshow(overlaid_image)
    axes[1].axis("off")
    axes[1].set_title("Cropped Image {} with Mask, Conf {:.3f}".format(i, score))

    plt.tight_layout()
    plt.show()
