import csv
import os
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


def parse_images_from_rosbags(rosbag_dir_path, rosbag_save_dir_path, rosbag_name, left_top_corner, right_bottom_corner):
    rosbag_access_folder = rosbag_dir_path + rosbag_name
    # we only care about the rgb image topic
    topics = ["/camera/color/image_raw"]
    bridge = CvBridge()

    if not os.path.exists(os.path.join(rosbag_save_dir_path, rosbag_name)):
        os.makedirs(os.path.join(rosbag_save_dir_path, rosbag_name))

    with Reader(rosbag_access_folder) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == topics[0]:
                rgb_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                rgb_img_copy = np.copy(rgb_img)
                cv.rectangle(rgb_img_copy, left_top_corner, right_bottom_corner, (0, 0, 255))
                savepath = os.path.join(rosbag_save_dir_path, rosbag_name) + "/" + str(msg.header.stamp.sec) + "_" + str(msg.header.stamp.nanosec)
                cv.imwrite(savepath + ".png", rgb_img_copy)


if __name__ == "__main__":
    rosbag_dir_path = "/home/atharva2/atharvak_ws/src/rosbags/"
    rosbag_save_dir_path = "/home/atharva2/atharvak_ws/src/rosbag_rgb/"
    left_top_corner = (297, 248)
    right_bottom_corner = (425, 332)

    rosbag_names = []
    for p in os.listdir(rosbag_dir_path):
        rosbag_names.append(p)

    for rosbag_name in rosbag_names:
        parse_images_from_rosbags(rosbag_dir_path, rosbag_save_dir_path, rosbag_name, left_top_corner, right_bottom_corner)