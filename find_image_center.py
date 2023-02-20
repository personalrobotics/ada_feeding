import cv2
import numpy as np
import math

# This will work because it DOES pick out blue color and uses that grayscale or black-and-white image in mask

def get_image_center_from_partial_plate_view():
    # load an image using 'imread'
    original_image = cv2.imread("/Users/raidakarim/Downloads/half_blue_plate.png")
    (h, w) = original_image.shape[:2]
    # where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
    centerX = w // 2
    centerY = h // 2
    print('x:', centerX)
    print('y:', centerY)

if __name__ == '__main__':
    get_image_center_from_partial_plate_view()