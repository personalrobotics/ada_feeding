#!/usr/bin/python
import get_ros_image
import cv2
import numpy as np
import math
import rospy
from std_msgs.msg import Float64MultiArray
import tf

# This will work because it DOES pick out blue color and uses that grayscale or black-and-white image in mask

def get_distance():
    # load an image using 'imread'
    # original_image = cv2.imread("/Users/raidakarim/Downloads/half_blue_plate.png")
    original_image = get_ros_image
    (h, w) = original_image.shape[:2]
    # where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
    centerX = w // 2
    centerY = h // 2]

    # convert images from BGR (Blue, Green, Red) to HSV (Hue-- color, Saturation-- density, Value-- lightness)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    ### blue color detection and masking ###
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    # define range of blue color in HSV here to create a mask of blue colored object
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # calculate moments of binary image
    m = cv2.moments(mask)
    #print(m)

    # calculate x,y coordinates of center
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])

    center = [centerX, centerY]
    currPoint = [cX, cY]

    # print center coordinates
    #print('center', center)
    #print('currPoint', currPoint)

    # Calculate Euclidean distance
    distance = math.dist(center, currPoint)
    distance = str(round(distance, 2))
    #print('distance', distance)

    cv2.circle(original_image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(original_image, "centroid", (cX-25, cY-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(original_image, (centerX, centerY), 5, (255, 255, 255), -1)
    cv2.putText(original_image, "centroid", (centerX - 25, centerY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # display the image
    cv2.imshow('original_image', original_image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(20000)  # images stay for 20 seconds

    deltaX = cX - centerX
    deltaY = cY - centerY

    # get angle in randians
    rad_angle =  = math.atan2(deltaX, deltaY)
    vector_x = math.cos(rad_angle)
    vector_y = math.sin(rad_angle)

    # convert angle to vector
    full_vector = [vector_x, vector_y, 0.0]
    vector_array = np.asarray(full_vector)

    pub = rospy.Publisher('vector_topic', Float64MultiArray, queue_size=10)
    rospy.init_node(NodeHandle::subscribe)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
       pub.publish(vector_array)
       r.sleep()

if __name__ == '__main__':
    get_distance()