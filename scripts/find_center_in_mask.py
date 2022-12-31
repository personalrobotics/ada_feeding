import cv2
import numpy as np

# this will work because it DOES pick out blue color and uses that grayscale image in mask

def find_center():
    # load an image using 'imread'
    # original_image = cv2.imread("/Users/raidakarim/Downloads/half_blue_plate.png")
    original_image = cv2.imread("/Users/raidakarim/Downloads/full_blue_plate.png")
    
    # convert images from BGR (Blue, Green, Red) to HSV (Hue-- color, Saturation-- density, Value-- lightness)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    ### blue color detection and masking ###
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    
    # define range of blue color in HSV here to create a mask of blue colored object
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # convert the mask image to binary image
    # The method returns two outputs.
    # The first is the threshold that was used and the second output is the thresholded image.
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    
    # calculate moments of binary image
    m = cv2.moments(thresh)
    
    # calculate x,y coordinates of center
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])
    
    # highlight the center and put text
    cv2.circle(original_image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(original_image, "centroid", (cX-25, cY-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # display the image
    cv2.imshow('original_image', original_image)

    cv2.waitKey(20000)  # images stay for 20 seconds

    # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    find_center()
