# Python program for Detection of blue color using OpenCV with Python
#!/usr/bin/python
import get_ros_image
import cv2
import numpy as np

def color_detection():

    # get current image from robot camera
    #original_image = cv2.imread("/Users/raidakarim/Downloads/blue_color_plate.png")
    original_image = get_ros_image

    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])

    # Here we are defining range of bluecolor in HSV
    # This creates a mask of blue coloured
    # objects found in the frame.
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # if blue color range present in the masked image, there will be nonzero; 
    # otherwise there will be no nonzero
    ones = cv2.countNonZero(mask)

    # The bitwise and of the frame and mask is done so
    # that only the blue coloured objects are highlighted
    # and stored in res
    #res = cv2.bitwise_and(original_image,original_image, mask= mask)
    #cv2.imshow('original_image', original_image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('res', res)
    #cv2.waitKey(15000)

    # send boolean true (if blue color detected) / false (if blue color not detected)
    detected = bool(ones > 0)
    print("detected", detected)
    return detected

if __name__ == '__main__':
    color_detection()