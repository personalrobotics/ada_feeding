"""
This file runs the training for Categorical Naive Bayes. To run the training, follow steps outlined in the Perception
Readme. To run this file, you will need to have the data in a zip file (don't unzip it; code will do it at runtime).
Additionally, the data should already be split such that there is <main folder>, which has a <train> and <test>
subfolders. Under each of these subfolders, it will need to have all the depth (raw or aligned_depth) corresponding
to it labels.

If you just want to train and test the model on the test set, you can perform set the --use_entire_dataset_bool to
"False". Otherwise, if you want to train the model on the entire dataset and then test the model on the unseen data,
make sure to set --use_entire_dataset_bool to "True". Then, you want to specify a --data_file_zip with a zipfile of
the data. If you are training on the entire dataset, it is also necessary to specify a --model_save_file to save the
model into.
"""

# Standard imports
import argparse
import joblib
import os
import shutil
import tempfile
from typing import Tuple
import zipfile

# Third-party imports
import cv2 as cv
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB

# Local imports
import helpers


def unzip_file(zip_path, extract_dir):
    """
    Unzips the file provided in the zip_path and puts it into the extract_dir directory

    Parameters:
    ----------
    zip_path: Zip file to read and extract images from
    extract_dir: Extracted folder to store all the images in
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def get_labels_folders(extracted_folders) -> Tuple[dict, dict, str, str]:
    """
    gets the train and test path for each image in the train and test folders, which are present in the
    extracted_folders

    Parameters:
    ----------
    extracted_folders: the folder that essentially contains two sub-folders (train/test) and this function extracts
    the paths to all the images from these folders

    Returns:
    ----------
    A tuple of
    train_labels_folder: paths to each of the images in the train set (dict)
    test_labels_folder: paths to each of the images in the test set (dict)
    train_folder_path: path to get to the train folder (String)
    test_folder_path: path to get to the test folder (String)
    """
    train_test_data_folder_path = extracted_folders[0]  # train_test_data
    train_test_data_folder = os.listdir(
        os.path.join(temp_dir, train_test_data_folder_path)
    )

    train_folder_path = train_test_data_folder[0]  # train
    train_folder = os.listdir(
        os.path.join(temp_dir, train_test_data_folder_path, train_folder_path)
    )

    test_folder_path = train_test_data_folder[1]  # test
    test_folder = os.listdir(
        os.path.join(temp_dir, train_test_data_folder_path, test_folder_path)
    )

    train_labels_folder = {}
    test_labels_folder = {}

    train_folder_path = os.path.join(
        temp_dir, train_test_data_folder_path, train_folder_path
    )
    test_folder_path = os.path.join(
        temp_dir, train_test_data_folder_path, test_folder_path
    )
    for labels_path in train_folder:
        train_labels_folder[labels_path] = os.listdir(
            os.path.join(train_folder_path, labels_path)
        )

    for labels_path in test_folder:
        test_labels_folder[labels_path] = os.listdir(
            os.path.join(test_folder_path, labels_path)
        )

    return train_labels_folder, test_labels_folder, train_folder_path, test_folder_path


def plt_show_depth_img(img, show=True, title=""):
    """
    Plot the provided image

    Parameters:
    ----------
    img: np.array of voxels
    show: boolean as to whether or not to show the graphed image
    title: a title for the graphed image
    """
    fig, ax = plt.subplots()
    for i in range(0, 128, 8):
        for j in range(0, 84, 4):
            ax.add_patch(Rectangle((i, j), 8, 4, edgecolor="r", linewidth=1, fill=None))
    plt.imshow(img)

    plt.title(title)
    plt.colorbar()
    if show:
        plt.show()


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Train CategoricalNB")

    # get the command line arguments
    parser.add_argument(
        "--use_entire_dataset_bool",
        type=str,
        help="Whether or not to use the entire dataset",
        required=True,
    )
    parser.add_argument(
        "--data_file_zip", type=str, help="zip file location", required=True
    )
    parser.add_argument(
        "--model_save_file", type=str, help="path to save the model", required=False
    )
    args = parser.parse_args()
    print(args)

    # depth variables
    min_dist = 310
    max_dist = 370

    # Variable to decide whether or not to use the entire dataset for training
    use_entire_dataset = args.use_entire_dataset_bool.lower() in (
        "true",
        "1",
        "yes",
        "t",
        "y",
    )
    print(use_entire_dataset)

    # unzip the files into a temporary folder
    temp_dir = tempfile.mkdtemp()
    # file to unzip (make sure you load it into the folder to actually train -> this file is not committed because of
    # size)
    zip_file_path = args.data_file_zip
    unzip_file(zip_file_path, temp_dir)

    # use the extracted folders for operations!
    extracted_folders = os.listdir(temp_dir)
    train_labels_folder, test_labels_folder, train_path, test_path = get_labels_folders(
        extracted_folders
    )

    X_train = []
    y_train = []
    indexToFilePath = {}
    index = 0
    for label in train_labels_folder:
        print(label)
        for filename in train_labels_folder[label]:
            filepath = os.path.join(train_path, label, filename)
            indexToFilePath[index] = filepath
            img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
            shape = img.shape
            # print(shape)
            # cv.imshow("image", helpers.normalize_to_uint8(img))
            # cv.waitKey(10)

            # convert the image such that anything outside the depth bounds specified above will be 0
            img_converted = np.where(
                np.logical_or(img < min_dist, img > max_dist), 0, 1
            ).astype("uint8")
            # plt_show_depth_img(img_converted, title=filepath + "-" + label)
            # if label == 'food':
            #     plt_show_depth_img(img_converted, title=filepath + "-" + label)

            X_train.append(img_converted.flatten())
            if label == "no_food":
                y_train.append(0)
            else:
                y_train.append(1)
            index += 1

    if not use_entire_dataset:
        X_test = []
        y_test = []
        for label in test_labels_folder:
            for filename in test_labels_folder[label]:
                filepath = os.path.join(test_path, label, filename)
                img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
                shape = img.shape
                # cv.imshow("image", helpers.normalize_to_uint8(img))
                # cv.waitKey(10)

                # convert the image such that anything outside the depth bounds specified above will be 0
                img_converted = np.where(
                    np.logical_or(img < min_dist, img > max_dist), 0, 1
                ).astype("uint8")
                # plt_show_depth_img(img_converted, title=filepath + "-" + label)

                X_test.append(img_converted.flatten())
                if label == "no_food":
                    y_test.append(0)
                else:
                    y_test.append(1)

        X_test = np.array(X_test)
        y_test = np.array(y_test)
    else:
        for label in test_labels_folder:
            for filename in test_labels_folder[label]:
                filepath = os.path.join(test_path, label, filename)
                img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
                shape = img.shape
                # cv.imshow("image", helpers.normalize_to_uint8(img))
                # cv.waitKey(10)

                # convert the image such that anything outside the depth bounds specified above will be 0
                img_converted = np.where(
                    np.logical_or(img < min_dist, img > max_dist), 0, 1
                ).astype("uint8")
                # plt_show_depth_img(img_converted, title=filepath + "-" + label)

                X_train.append(img_converted.flatten())
                if label == "no_food":
                    y_train.append(0)
                else:
                    y_train.append(1)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(X_train.shape, y_train.shape)
    if not use_entire_dataset:
        print(X_test.shape, y_test.shape)

    # Train Naive Bayes Classifier
    clf = CategoricalNB(min_categories=2)
    clf.fit(X_train, y_train)
    for val in [0, 1]:
        prob = clf.predict_proba(val * np.ones((1, 128 * 84), dtype=np.uint8))
        print(f"All {val}'s proba is {prob}")

    # Get the predictions on the train set
    print("X_train", X_train.shape)
    y_pred_train = clf.predict(X_train)
    y_pred_train_proba = clf.predict_proba(X_train)
    print(y_pred_train_proba)

    # Get the predictions on the test set
    if not use_entire_dataset:
        print("X_test", X_test.shape)
        y_pred_test = clf.predict(X_test)
        y_pred_test_proba = clf.predict_proba(X_test)
        print(y_pred_test_proba)

    # Get the train accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    print("Train accuracy", acc_train)
    print("Train confusion matrix\n", confusion_matrix(y_train, y_pred_train))

    # Get the test accuracy
    if not use_entire_dataset:
        acc_test = accuracy_score(y_test, y_pred_test)
        print("Test accuracy", acc_test)
        print("Test confusion matrix\n", confusion_matrix(y_test, y_pred_test))

        # # Visualize what was learnt
        # print("clf.feature_log_prob_", [m.shape for m in clf.feature_log_prob_])
        # print("clf.feature_log_prob_", [np.sum(np.e**m, axis=1) for m in clf.feature_log_prob_])
        # print("clf.feature_log_prob_", clf.feature_log_prob_, len(clf.feature_log_prob_), clf.feature_log_prob_[-1].shape)
        # print("clf.class_count_", clf.class_count_, len(clf.class_count_), clf.class_count_[0].shape)

        conditional_probabilities = [np.e**m for m in clf.feature_log_prob_]
        prob_that_pixel_is_in_range_given_fof = []
        prob_that_pixel_is_in_range_given_no_fof = []
        for i in range(len(conditional_probabilities)):
            try:
                prob_that_pixel_is_in_range_given_fof.append(
                    conditional_probabilities[i][1, 1]
                )
            except (
                IndexError
            ):  # If column 1 doesn't exist, that means the pixel was never in range
                prob_that_pixel_is_in_range_given_fof.append(0)
            try:
                prob_that_pixel_is_in_range_given_no_fof.append(
                    conditional_probabilities[i][0, 1]
                )
            except (
                IndexError
            ):  # If column 1 doesn't exist, that means the pixel was never in range
                prob_that_pixel_is_in_range_given_no_fof.append(0)
        prob_that_pixel_is_in_range_given_fof = np.array(
            prob_that_pixel_is_in_range_given_fof
        ).reshape(shape)
        prob_that_pixel_is_in_range_given_no_fof = np.array(
            prob_that_pixel_is_in_range_given_no_fof
        ).reshape(shape)
        plt_show_depth_img(
            prob_that_pixel_is_in_range_given_no_fof,
            title="Probability that a pixel is in range given no food on the fork",
        )
        plt_show_depth_img(
            prob_that_pixel_is_in_range_given_fof,
            title="Probability that a pixel is in range given food on the fork",
        )
        plt_show_depth_img(
            prob_that_pixel_is_in_range_given_fof
            - prob_that_pixel_is_in_range_given_no_fof,
            title="P(px in range | FoF) - P(px in range | no FoF)",
        )

    # save model
    if use_entire_dataset:
        # make sure to type the right name to save the model
        model_save_filename = args.model_save_file
        joblib.dump(clf, model_save_filename)
        print("model saved!!")

    # delete the folder with the unzipped files
    shutil.rmtree(temp_dir)
