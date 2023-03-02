#!/usr/bin/env python
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Bool
# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError
# math tools
import numpy as np
import math
# service
from ada_feeding.srv import PlateService, PlateServiceResponse

# declare variables
cv_image = None
vector_array = None
detected = False
bridge = CvBridge()

# communicates with xml and js files
def service_callback(req):
    # implement color detector
    # Converts images from BGR to HSV
    if cv_image != None:
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([94, 80, 2])
        upper_blue = np.array([126, 255, 255])
        # Here we are defining range of bluecolor in HSV
        # This creates a mask of blue coloured
        # objects found in the frame.
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # if blue color range present in the masked image, there will be nonzeros; 
        # otherwise there will be no nonzero
        ones = cv2.countNonZero(mask)
        #cv2.imshow('original_image', original_image)
        #cv2.imshow('mask', mask)
        #cv2.waitKey(15000)

        # boolean true (if blue color detected) / false (if blue color not detected)
        detected = bool(ones > 0)
        req.alert = detected
        print("plate detected", detected)

        if detected == True:
            # get height and width
            (h, w) = cv_image.shape[:2]
            # where w//2, h//2 are the required frame/image centeroid's XYcoordinates.
            centerX = w // 2
            centerY = h // 2

            # calculate moments of binary image
            m = cv2.moments(mask)
            #print(m)

            # calculate x,y coordinates of center
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])

            center = [centerX, centerY]
            currPoint = [cX, cY]
            
            #display the image
            #cv2.circle(cv_image, (cX, cY), 5, (255, 255, 255), -1)
            #cv2.putText(cv_image, "centroid", (cX-25, cY-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.circle(cv_image, (centerX, centerY), 5, (255, 255, 255), -1)
            #cv2.putText(cv_image, "centroid", (centerX - 25, centerY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.imshow('cv_image', original_image)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(20000)  # images stay for 20 seconds
            
            deltaX = cX - centerX
            deltaY = cY - centerY

            if deltaX > 5 or deltaY > 5:
                # get angle in randians
                rad_angle = math.atan2(deltaX, deltaY)
                vector_x = math.cos(rad_angle)
                vector_y = math.sin(rad_angle)

                # convert angle to vector
                full_vector = [vector_x, vector_y, 0.0]
                vector_array = np.asarray(full_vector)
                req.vector = vector_array
            
        # return plate detect boolean and distance offset
        return PlateServiceResponse(req.alert, req.vector)
    else: 
        print("CV image could not be found.")

#  processes image
def subscriber_callback(img_msg):
    # implement:
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    # Convert the ROS Image message to a CV2 Image
    cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    # Show the converted image
    # show_image(cv_image)

# main function
# start ros service and subscriber
def main():
    # We declare our node using init_node()
    rospy.init_node('starter', anonymous=True)

    # communicates with camera topic
    # This declares that our node subscribes to the "/camera/rgb/image_raw" topic which is of type Image. 
    # When new ros image messages are received, subscriber_callback is invoked with the message as the first argument.
    rospy.Subscriber("/camera/color/image_raw/image", Image, subscriber_callback)

    # This declares a new service named 'alert_and_offset' with the PlateService service type. 
    # All requests are passed to service_callback function.
    s = rospy.Service('alert_and_offset', PlateService, service_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()