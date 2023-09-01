import zipfile
import tempfile
import shutil
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB
import joblib
from matplotlib.patches import Rectangle
from skimage.measure import block_reduce


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


def normalize_to_uint8(img):
    # Normalize the image to 0-255
    img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')
    return img_normalized


def plt_show_depth_img(img, show=True, title=""):
    # Plot the depth image
    fig, ax = plt.subplots()
    for i in range(0, 128, 8):
        for j in range(0, 84, 4):
            ax.add_patch(Rectangle((i, j), 8, 4, edgecolor="r", linewidth=1, fill=None))
    plt.imshow(img)

    plt.title(title)
    plt.colorbar()
    if show: plt.show()


def find_reduced_features(img):
    return block_reduce(img, block_size=(4, 8), func=np.max)


if "__main__" == __name__:
    # depth variables
    min_dist = 310
    max_dist = 370

    # Variable to decide whether or not to use the entire dataset for training
    use_entire_dataset = False

    # unzip the files into a temporary folder
    temp_dir = tempfile.mkdtemp()
    zip_file_path = "./train_test_data_no_hand_8-30-23.zip"
    unzip_file(zip_file_path, temp_dir)

    # use the extracted folders for operations!
    extracted_folders = os.listdir(temp_dir)
    train_labels_folder, test_labels_folder, train_path, test_path = get_labels_folders(extracted_folders)

    X_train = []
    y_train = []
    for label in train_labels_folder:
        print(label)
        for filename in train_labels_folder[label]:
            filepath = os.path.join(train_path, label, filename)
            img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
            # cv.imshow("image", normalize_to_uint8(img))
            # cv.waitKey(10)

            # convert the image such that anything outside the depth bounds specified above will be 0
            img_converted = np.where(np.logical_or(img < min_dist, img > max_dist), 0, 1).astype('uint8')
            # plt_show_depth_img(img_converted, title=filepath + "-" + label)
            img_converted = find_reduced_features(img_converted)
            shape = img_converted.shape
            # if label == 'food':
            #     plt_show_depth_img(img_converted, title=filepath + "-" + label)

            X_train.append(img_converted.flatten())
            if label == 'no_food':
                y_train.append(0)
            else:
                y_train.append(1)

    if not use_entire_dataset:
        X_test = []
        y_test = []
        for label in test_labels_folder:
            for filename in test_labels_folder[label]:
                filepath = os.path.join(test_path, label, filename)
                img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
                # cv.imshow("image", normalize_to_uint8(img))
                # cv.waitKey(10)

                # convert the image such that anything outside the depth bounds specified above will be 0
                img_converted = np.where(np.logical_or(img < min_dist, img > max_dist), 0, 1).astype('uint8')
                # plt_show_depth_img(img_converted, title=filepath + "-" + label)
                img_converted = find_reduced_features(img_converted)
                shape = img_converted.shape

                X_test.append(img_converted.flatten())
                if label == 'no_food':
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
                # cv.imshow("image", normalize_to_uint8(img))
                # cv.waitKey(10)

                # convert the image such that anything outside the depth bounds specified above will be 0
                img_converted = np.where(np.logical_or(img < min_dist, img > max_dist), 0, 1).astype('uint8')
                # plt_show_depth_img(img_converted, title=filepath + "-" + label)
                img_converted = find_reduced_features(img_converted)
                shape = img_converted.shape

                X_train.append(img_converted.flatten())
                if label == 'no_food':
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

    # Get the predictions on the train set
    print("X_train", X_train.shape)
    y_pred_train = clf.predict(X_train)
    # print(clf.class_log_prior_, np.exp(clf.class_log_prior_))
    # feature_given_fof = np.array([feature[1, :] for feature in clf.feature_log_prob_])
    # print(feature_given_fof, feature_given_fof.min(), feature_given_fof.max(), feature_given_fof.mean(), feature_given_fof.sum())
    # print(list(np.exp(feature_given_fof)))
    c = 0
    num_inac = 0
    for each_img in X_train:
        y_pred_train_ind = clf.predict(each_img.reshape(1, -1))
        if y_pred_train_ind[0] != y_train[c]:
            y_pred_train_ind_proba = clf.predict_proba(each_img.reshape(1, -1))
            num_inac += 1
            print("not accurate: ", y_pred_train_ind_proba)
        else:
            y_pred_train_ind_proba = clf.predict_proba(each_img.reshape(1, -1))
            print("accurate: ", y_pred_train_ind_proba)
        c += 1
        if (c == 100):
            break
    print("num inac: ", num_inac)
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

        conditional_probabilities = [np.e ** m for m in clf.feature_log_prob_]
        prob_that_pixel_is_in_range_given_fof = []
        prob_that_pixel_is_in_range_given_no_fof = []
        for i in range(len(conditional_probabilities)):
            try:
                prob_that_pixel_is_in_range_given_fof.append(conditional_probabilities[i][1, 1])
            except IndexError:  # If column 1 doesn't exist, that means the pixel was never in range
                prob_that_pixel_is_in_range_given_fof.append(0)
            try:
                prob_that_pixel_is_in_range_given_no_fof.append(conditional_probabilities[i][0, 1])
            except IndexError:  # If column 1 doesn't exist, that means the pixel was never in range
                prob_that_pixel_is_in_range_given_no_fof.append(0)
        prob_that_pixel_is_in_range_given_fof = np.array(prob_that_pixel_is_in_range_given_fof).reshape(shape)
        prob_that_pixel_is_in_range_given_no_fof = np.array(prob_that_pixel_is_in_range_given_no_fof).reshape(shape)
        plt_show_depth_img(prob_that_pixel_is_in_range_given_no_fof,
                           title="Probability that a pixel is in range given no food on the fork")
        plt_show_depth_img(prob_that_pixel_is_in_range_given_fof,
                           title="Probability that a pixel is in range given food on the fork")
        plt_show_depth_img(prob_that_pixel_is_in_range_given_fof - prob_that_pixel_is_in_range_given_no_fof,
                           title="P(px in range | FoF) - P(px in range | no FoF)")

    # save model
    if use_entire_dataset:
        model_save_filename = 'categorical_naive_bayes_without_hand_model.pkl'
        joblib.dump(clf, model_save_filename)
        print("model saved!!")
    # delete the folder with the unzipped files
    shutil.rmtree(temp_dir)
