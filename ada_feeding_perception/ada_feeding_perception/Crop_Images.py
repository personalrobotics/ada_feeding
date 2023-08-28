"""
This script will go through the rosbags and crop the depth and rgb images to the specified frustum.
The images will be used to create a dataset that can be used to train a CNN
"""
import os
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import pandas as pd


def crop_img(imageToSave, rectangle_left_top=(297, 248), rectangle_right_bottom=(422+3, 332)):
    cropped_img = np.copy(imageToSave)
    left_top_x, left_top_y = rectangle_left_top
    right_bottom_x, right_bottom_y = rectangle_right_bottom
    cropped_img = cropped_img[left_top_y:right_bottom_y, left_top_x:right_bottom_x]
    return cropped_img


if __name__ == "__main__":
    rosbags_abs_location = "/home/atharva2/atharvak_ws/src/rosbags/"
    cropped_img_storage_abs_location = "/home/atharva2/atharvak_ws/src/with_real_labels/cropped_images/"
    non_cropped_img_storage_abs_location = "/home/atharva2/atharvak_ws/src/with_real_labels/non_cropped_images/"
    master_labels = "../ada_feeding_perception/Master_with_labels_7-11-23.csv"
    depth_img_folder = "depth_img/"

    # read the csv
    df = pd.read_csv(master_labels)

    rectangle_left_top = (297, 248)
    rectangle_right_bottom = (422, 332)

    bridge = CvBridge()

    # check if the saved location exists; if doesn't exist, create one
    if not os.path.exists(cropped_img_storage_abs_location + depth_img_folder + "/food"):
        os.makedirs(cropped_img_storage_abs_location + depth_img_folder + "/food")

    if not os.path.exists(cropped_img_storage_abs_location + depth_img_folder + "/no_food"):
        os.makedirs(cropped_img_storage_abs_location + depth_img_folder + "/no_food")

    if not os.path.exists(cropped_img_storage_abs_location + depth_img_folder + "/hand"):
        os.makedirs(cropped_img_storage_abs_location + depth_img_folder + "/hand")

    if not os.path.exists(non_cropped_img_storage_abs_location + depth_img_folder + "/food"):
        os.makedirs(non_cropped_img_storage_abs_location + depth_img_folder + "/food")

    if not os.path.exists(non_cropped_img_storage_abs_location + depth_img_folder + "/no_food"):
        os.makedirs(non_cropped_img_storage_abs_location + depth_img_folder + "/no_food")

    if not os.path.exists(non_cropped_img_storage_abs_location + depth_img_folder + "/hand"):
        os.makedirs(non_cropped_img_storage_abs_location + depth_img_folder + "/hand")

    # go through the list of rosbags present in the dir_list
    rosbags_dir_list = os.listdir(rosbags_abs_location)

    count = 0
    for rosbag in rosbags_dir_list:
        with Reader(rosbags_abs_location + rosbag) as reader:
            for connection, timestamp, rawdata in reader.messages():
                msg = deserialize_cdr(rawdata, connection.msgtype)
                imageToSave = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if connection.topic == '/camera/depth/image_rect_raw':
                    timestamp_sec = msg.header.stamp.sec
                    timestamp_nsec = msg.header.stamp.nanosec

                    # depth image
                    # filter the right df row to get the correct label
                    filtered_df = df.loc[(df['rosbag_path'] == rosbags_abs_location + rosbag) & (
                                df['timestamp_sec'] == timestamp_sec) & (df['timestamp_nanosec'] == timestamp_nsec)]
                    label = filtered_df['label'].iloc[0]

                    # if label == 'hand':
                    #     label = 'food'
                    storage_val = depth_img_folder + label + "/" + rosbag + "_" + str(timestamp_sec) + "_" + str(
                        timestamp_nsec) + ".png"

                    # store the original depth image
                    storageloc = non_cropped_img_storage_abs_location + storage_val
                    cv.imwrite(storageloc, imageToSave)

                    # store the cropped depth image
                    cropped_img = crop_img(imageToSave)
                    storageloc = cropped_img_storage_abs_location + storage_val
                    cv.imwrite(storageloc, cropped_img)
                    count += 1

        print("done with: ", rosbag)
    print("Count: ", count)