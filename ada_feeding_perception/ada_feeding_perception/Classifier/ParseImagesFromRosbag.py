#!/usr/bin/env python3

import cv2 as cv
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os



def parse_images_from_rosbag(rosbag_dir_path, rosbag_name, image_dir_path, bridge):
    """
    Takes the rosbag name, a directory to the rosbag and a directory to the images
    Then, extracts the images from each frame of the rosbag and saves them into a new folder
    with the name of the rosbag and (_depth and _color) respective to the color
    Parameters:
        rosbag_dir_path: String: path to the overall rosbag folder
        rosbag_name: String: name of the particular rosbag
        image_dir_path: String: path to the overall image folder
    """
    # access the particular rosbag
    rosbag_access_folder = rosbag_dir_path + rosbag_name

    # loop through and save the images (use both raw image for debugging purposes)
    # topics = ["/camera/depth/image_rect_raw", "/camera/color/image_raw"]
    topics = ["/camera/depth/image_rect_raw"]

    # reader instance for reading
    with Reader(rosbag_access_folder) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == topics[0]:
                food_on_fork(depth_img_msg=msg)

                # only for viewing purposes (use changed_depth_image)
                # changed_depth_img = np.asarray(image_to_save)
                # cv.imshow("depth image", changed_depth_img)
            elif connection.topic == topics[1]:
                print("header", msg.header)
                image_to_save = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                image_to_save = cv.cvtColor(image_to_save, cv.COLOR_RGB2BGR)

                # convert the seconds into nanoseconds (1e9 is the conversion factor!)

                # cv.imshow("raw color image", image_to_save)

            # cv.waitKey(5)


def food_on_fork(depth_img_msg, fork_tine_centers=[(359, 275), (367, 274), (375, 273), (381, 272)]):
    depth_img = None
    try:
        # passthrough is essentially rendering the image as it sees
        depth_img = bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")
        changed_depth_img = np.copy(depth_img)
    except CvBridgeError as e:
        print(e)

