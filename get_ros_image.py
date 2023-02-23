#!/usr/bin/env python
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image
# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError

# Create a subscriber for an Image topic, and define a callback function
# Use CTRL+C to stop the program

# Define a callback for the Image message
def image_callback(img_msg):
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    # Show the converted image
    # show_image(cv_image)

# Initalize a subscriber to the "/camera/rgb/image_raw" topic with the function "image_callback" as a callback
def image_listener():
    # Initialize the ROS Node named 'opencv_example', allow multiple nodes to be run with this name
    rospy.init_node('listener', anonymous=True)
    # Initialize the CvBridge class
    bridge = CvBridge()
    sub_image = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
    # Loop to keep the program from shutting down unless ROS is shut down, or CTRL+C is pressed
    while not rospy.is_shutdown():
        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    return sub_image

if __name__ == '__main__':
    image_listener()