import cv2 as cv
import zipfile
import tempfile
import shutil
import os
import numpy as np


def unzip_file(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)


def get_labels_folders(extracted_folders):
    train_test_data_folder_path = extracted_folders[0]  # train_test_data
    train_test_data_folder = os.listdir(os.path.join(temp_dir, train_test_data_folder_path))

    train_folder_path = train_test_data_folder[0]  # train
    train_folder = os.listdir(os.path.join(temp_dir, train_test_data_folder_path, train_folder_path))

    test_folder_path = train_test_data_folder[1]  # test
    test_folder = os.listdir(os.path.join(temp_dir, train_test_data_folder_path, test_folder_path))

    train_labels_folder = {}
    test_labels_folder = {}

    train_folder_path = os.path.join(temp_dir, train_test_data_folder_path, train_folder_path)
    test_folder_path = os.path.join(temp_dir, train_test_data_folder_path, test_folder_path)
    for labels_path in train_folder:
        train_labels_folder[labels_path] = os.listdir(os.path.join(train_folder_path, labels_path))

    for labels_path in test_folder:
        test_labels_folder[labels_path] = os.listdir(os.path.join(test_folder_path, labels_path))

    return train_labels_folder, test_labels_folder, train_folder_path, test_folder_path


if "__main__" == __name__:
    min_dist = 310
    max_dist = 370

    temp_dir = tempfile.mkdtemp()
    zip_file_path = "./train_test_data_no_hand.zip"
    unzip_file(zip_file_path, temp_dir)

    extracted_folders = os.listdir(temp_dir)
    train_labels_folder, test_labels_folder, train_path, test_path = get_labels_folders(extracted_folders)

    for label in train_labels_folder:
        print(label)
        for filename in train_labels_folder[label]:
            filepath = os.path.join(train_path, label, filename)
            img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
            np_img = np.array(img)
            np_img_converted = np.where(np.logical_or(img < min_dist, img > max_dist), 0, 255).astype('uint8')
            # cv.imshow("pic", np_img_converted)
            # cv.waitKey(0)
            cv.imwrite("/home/atharva2/atharvak_ws/src/ada_feeding/ada_feeding_perception/ada_feeding_perception/CategoricalNaiveBayes/train/" + label + "/" + filename, np_img_converted)


    # delete the folder with the unzipped files
    shutil.rmtree(temp_dir)
