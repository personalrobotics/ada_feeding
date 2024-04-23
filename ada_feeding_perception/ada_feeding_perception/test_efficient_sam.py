"""
This script demonstrates how to use the Segment Anything Model (SAM) to
segment an object from an image given a point inside the object. Unlike the
other scripts in this package, this script does not use ROS. Therefore, it can
be useful as an initial check to ensure Segment Anything works on your computer.
"""
# pylint: disable=duplicate-code
# Since this script is intentionally not a ROS node, it will have overlap with
# the corresponding ROS node.

# Standard imports
import argparse
import json
import os
import time

# Third-party imports
import cv2
from efficient_sam.efficient_sam import build_efficient_sam
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

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

# Constant model parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "efficient_sam_vitt.pt"
CHECKPOINT_BASE_URL = (
    "https://raw.githubusercontent.com/yformer/EfficientSAM/main/weights/"
)

# Read the command-line arguments
image_path = args.input_image
input_point = np.array(json.loads(args.input_point))
input_points = torch.tensor(input_point.reshape((1, 1, 1, 2))).to(device=DEVICE)
input_labels = torch.tensor([[[1]]]).to(device=DEVICE)

# Read the image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f"Image shape: {image.shape}")
image_tensor = transforms.ToTensor()(image).to(device=DEVICE)

# Download the model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "model"))
efficient_sam_checkpoint = os.path.join(model_dir, MODEL_NAME)
# if the model can't be found download it
if not os.path.isfile(efficient_sam_checkpoint):
    print("EfficientSAM model checkpoint not found. Downloading...")
    download_checkpoint(MODEL_NAME, model_dir, CHECKPOINT_BASE_URL)
    print(f"Download complete! Model saved at {efficient_sam_checkpoint}")

# Load the model
efficient_sam = build_efficient_sam(
    encoder_patch_embed_dim=192,
    encoder_num_heads=3,
    checkpoint=os.path.join(model_dir, MODEL_NAME),
).eval()
efficient_sam.to(device=DEVICE)
time_start = time.time()
predicted_logits, predicted_iou = efficient_sam(
    image_tensor[None, ...],
    input_points,
    input_labels,
)
sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
predicted_logits = torch.take_along_dim(
    predicted_logits, sorted_ids[..., None, None], dim=2
)
# The masks are already sorted by their predicted IOUs.
# The first dimension is the batch size (we have a single image. so it is 1).
# The second dimension is the number of masks we want to generate (in this case, it is only 1)
# The third dimension is the number of candidate masks output by the model.
# For this demo we use the first mask.
masks = torch.ge(predicted_logits[0, 0, :, :, :], 0).cpu().detach().numpy()
scores = predicted_iou[0, 0, :].cpu().detach().numpy()
scored_masks = list(zip(scores, masks))
scored_masks_sorted = sorted(scored_masks, key=lambda x: x[0], reverse=True)
time_end = time.time()
print(f"Time taken to segment the image: {time_end - time_start:.3f} seconds")


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


NUMBER_OUTPUT = 3
# After getting the masks
for i, (score, mask) in enumerate(scored_masks_sorted):
    if i >= NUMBER_OUTPUT:
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
    axes[0].set_title(f"Cropped Image {i}")

    # Show the overlaid image
    axes[1].imshow(overlaid_image)
    axes[1].axis("off")
    axes[1].set_title(f"Cropped Image {i} with Mask, Conf {score:.3f}")

    plt.tight_layout()
    plt.show()
