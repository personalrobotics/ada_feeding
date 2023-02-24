#!/usr/bin/env python
# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import Image
# Import OpenCV libraries and tools
import cv2
from cv_bridge import CvBridge, CvBridgeError

image = None
bridge = CvBridge()

# communicates with xml and js files
def service_callback():
    # implement

    # return plate detect boolean and distance offset

# communicates with camera topic to process image
def subscriber_callback(img_msg):
    # implement
    # log some info about the image topic
    rospy.loginfo(img_msg.header)
    # Try to convert the ROS Image message to a CV2 Image
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    # Show the converted image
    # show_image(cv_image)

# main function
def main():
    rospy.init_node('starter', anonymous=True)

    # start ros service and subscriber
    s = rospy.Service('process_image', ?, service_callback)
    print("Ready to process camera image.")
    # This declares that our node subscribes to the "/camera/rgb/image_raw" topic which is of type Image. 
    # When new image messages are received, subscriber_callback is invoked with the message as the first argument.
    rospy.Subscriber("/camera/rgb/image_raw", Image, subscriber_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()