#!/usr/bin/env python3
import csv
import os

import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from cv_bridge import CvBridge, CvBridgeError


def parse_images_from_rosbags(f_write, rosbag_dir_path, rosbag_name,
                              left_top_corner=(297, 248), right_bottom_corner=(422, 332),
                              min_dist=(330 - 20), max_dist=(330 + 40)):
    rosbag_access_folder = rosbag_dir_path + rosbag_name
    topics = ["/camera/depth/image_rect_raw"]

    with Reader(rosbag_access_folder) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == topics[0]:
                # get the number of pixels from food_on_fork_num_pixels function
                num_pixels = food_on_fork_num_pixels(depth_img_msg=msg,
                                                     left_top_corner=left_top_corner,
                                                     right_bottom_corner=right_bottom_corner,
                                                     min_dist=min_dist, max_dist=max_dist)
                # write the number of pixels into csv!
                timestamp = str((msg.header.stamp.sec * 1000000000) + msg.header.stamp.nanosec)
                f_write.writerow([rosbag_access_folder, msg.header.stamp.sec, msg.header.stamp.nanosec, num_pixels])


def food_on_fork_num_pixels(depth_img_msg,
                            left_top_corner=(297, 248),
                            right_bottom_corner=(422, 332),
                            min_dist=(330 - 20),
                            max_dist=(330 + 40)) -> int:
    """
    Returns the number of pixels in the provided depth image (through a depth image message)

    Parameters:
        depth_img_msg: depth image message
        left_top_corner: Tuple(int, int): Top-left point of the bounding box rectangle
        right_bottom_corner: Tuple(int, int): Bottom-right point of the bounding box of the rectangle
        min_dist: int: minimum depth to consider (note that 330 is approx distance to the fork tine)
        max_dist: int: maximum depth to consider (note that 330 is approx distance to the fork tine)
    """

    bridge = CvBridge()
    depth_img = bridge.imgmsg_to_cv2(depth_img_msg, "passthrough")

    # consider the points for the rectangle
    pt1_col, pt1_row = left_top_corner
    pt2_col, pt2_row = right_bottom_corner

    # create mask that satisfies the rectangle and distance conditions
    mask_img = np.zeros_like(depth_img, dtype=bool)
    mask_img[pt1_row:pt2_row, pt1_col:pt2_col] = True
    mask_img[np.logical_not((min_dist < depth_img) & (depth_img < max_dist))] = False

    return np.count_nonzero(mask_img)


def main():
    rosbag_dir_path = "/home/atharva2/atharvak_ws/src/rosbags/"
    rosbag_names = []
    for p in os.listdir(rosbag_dir_path):
        rosbag_names.append(p)

    trials = {
        "Ross": {
            "left_top_corner": (297, 248),
            "right_bottom_corner": (422, 332),
            "min_dist": (330 - 20),
            "max_dist": (330 + 40)
        }
    }

    name = "Ross"
    date = "7-11-23"
    left_top_corner = trials[name]["left_top_corner"]
    right_bottom_corner = trials[name]["right_bottom_corner"]
    min_dist = trials[name]["min_dist"]
    max_dist = trials[name]["max_dist"]

    with open(r'/home/atharva2/atharvak_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/Classifier/' +
              name + '_' + date + '.csv', 'a', newline='') as f:
        f_write = csv.writer(f)
        f_write.writerow(["rosbag_path", "timestamp_sec", "timestamp_nanosec", "num_pixels"])
        for rosbag_name in rosbag_names:
            parse_images_from_rosbags(f_write, rosbag_dir_path, rosbag_name, left_top_corner, right_bottom_corner,
                                      min_dist, max_dist)
            print("done with: ", rosbag_name)


if __name__ == "__main__":
    main()
