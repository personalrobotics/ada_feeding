import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import json
from segment_anything import sam_model_registry, SamPredictor
import argparse
from skimage.measure import regionprops

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

parser.add_argument("--input_image", type=str, required=True, help="path to input image")
parser.add_argument("--input_point", type=str, required=True, help="path to input point file")

args = parser.parse_args()

# read input_image and input_point from the topic

# Access the values of the arguments
image_path = args.input_image
input_point = np.array(eval(args.input_point))

# Model parameters
device = 'cuda'
model_type = "vit_b"
sam_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'sam_vit_b_01ec64.pth'))
# if the model can't be found download it
if not os.path.isfile(sam_checkpoint):
    import urllib.request
    print("SAM model checkpoint not found. Downloading...")
    if not os.path.exists(os.path.dirname(sam_checkpoint)):
        os.makedirs(os.path.dirname(sam_checkpoint))
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    urllib.request.urlretrieve(url, sam_checkpoint)
    print("Download complete! Model saved at {}".format(sam_checkpoint))

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)
# Multiple points can be input
# labels 1 (foreground point) or 0 (background point)
input_point = np.array([input_point])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    # When False, it will return a single mask
)

# create a list of tuples containing the score and corresponding mask
scored_masks = [(score, mask) for score, mask in zip(scores, masks)]

# sort the list of tuples in descending order based on the score
scored_masks_sorted = sorted(scored_masks, key=lambda x: x[0], reverse=True)

def bbox_from_mask(mask):
    # Find the bounding box coordinates from the mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

def crop_image_and_mask(image, mask, bbox):
    minr, minc, maxr, maxc = bbox
    cropped_image = image[minr:maxr, minc:maxc]
    cropped_mask = mask[minr:maxr, minc:maxc]
    return cropped_image, cropped_mask

def show_mask_on_image(image, mask, alpha=0.5):
    # Assuming mask is a binary mask, we convert to 3 channel, and apply a color (Green here)
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask == 1] = [0, 255, 0]
    
    # Blend the image and the mask
    blended = cv2.addWeighted(image, alpha, mask_rgb, 1 - alpha, 0)
    return blended

# After getting the masks
for i, (score, mask) in enumerate(scored_masks_sorted):
    # compute the bounding box from the mask
    bbox = bbox_from_mask(mask)
    # crop the image and the mask
    cropped_image, cropped_mask = crop_image_and_mask(image, mask, bbox)
    # save the cropped mask as an image
    mask_image = np.where(cropped_mask, 255, 0).astype(np.uint8)

    # Overlay the mask on the image
    overlaid_image = show_mask_on_image(cropped_image, cropped_mask)

    # Create subplots to display the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show the cropped image
    axes[0].imshow(cropped_image)
    axes[0].axis('off')
    axes[0].set_title('Candidate {} Cropped Image'.format(i))

    # Show the overlaid image
    axes[1].imshow(overlaid_image)
    axes[1].axis('off')
    axes[1].set_title('Candidate {} Cropped Image with Mask'.format(i))

    plt.tight_layout()
    plt.show()