#!/usr/bin/env python3
import csv

import cv2 as cv
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os


if __name__ == "__main__":
    rosbag_access_folder = "/home/atharva2/atharvak_ws/src/rosbags/"
    image_save_folder = "/home/atharva2/atharvak_ws/src/images_8-30-23/"
    csv_to_write = "/home/atharva2/atharvak_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/Master_with_labels_8-30-23.csv"
    bridge = CvBridge()

    list_of_rosbags = os.listdir(rosbag_access_folder)
    list_of_valid_rosbags = []

    for rosbag in list_of_rosbags:
        rosbag_arr = rosbag.split("_")
        if rosbag_arr[1] == "8-30-23":
            list_of_valid_rosbags.append(rosbag)

    depth_topic = "/camera/depth/image_rect_raw"
    fields = ["rosbag_path", "timestamp_sec", "timestamp_nanosec", "label", "binary_label"]
    rows = []
    for rosbag in list_of_valid_rosbags:
        rosbag_loc = os.path.join(rosbag_access_folder, rosbag)
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)
        with Reader(rosbag_loc) as reader:
            for connection, timestamp, rawdata in reader.messages():
                msg = deserialize_cdr(rawdata, connection.msgtype)
                each_row = []
                if connection.topic == depth_topic:
                    each_row.append(rosbag_loc)
                    each_row.append(msg.header.stamp.sec)
                    each_row.append(msg.header.stamp.nanosec)
                    rows.append(each_row)
                    depth_img = bridge.imgmsg_to_cv2(msg, "passthrough")
                    loc = image_save_folder + rosbag + "_" + str(msg.header.stamp.sec) + "_" + str(msg.header.stamp.nanosec)
                    cv.imwrite(loc + ".png", depth_img)
        print("done: ", rosbag)

    print(rows)
    with open(csv_to_write, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
