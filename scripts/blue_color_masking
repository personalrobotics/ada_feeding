import cv2
import numpy as np

def color_detect_mask():
    # load an image using 'imread'
    original_image = cv2.imread("/Users/raidakarim/Downloads/half_blue_plate.png")
    # convert images from BGR (Blue, Green, Red) to HSV (Hue-- color, Saturation-- density, Value-- lightness)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
   
   
    ### blue color detection and masking ###
    lower_blue = np.array([94,  80, 2])
    upper_blue = np.array([126, 255, 255])
    # define range of blue color in HSV here to create a mask of blue colored object
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    # The 'bitwise_and' of the image and mask is done
    # so that only the blue color objects are highlighted and stored in res
    res = cv2.bitwise_and(original_image, original_image, mask=mask)

    ### display image, mask, res in 3 separate windows ###
    cv2.imshow('original_image', original_image)
    cv2.imshow('mask', mask)
    # we show only the object(s) with the blue color.
    cv2.imshow('res', res)

    k = cv2.waitKey(20000) # images stay for 20 seconds

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    color_detect_mask()
