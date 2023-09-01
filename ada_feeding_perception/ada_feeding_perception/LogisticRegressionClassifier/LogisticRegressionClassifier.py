import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import pprint
import seaborn as sns
import os
import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=2)
import matplotlib.collections as clt
import sys

sys.path.append('../ptitprince/')
import ptitprince as pt
import pprint

import random


def main():
    csv_to_read_from = "Ross_7-11-23.csv"
    csv_to_write_to = "Ross_7-11-23_predictions.csv"

    # get the data frame
    df = pd.read_csv(csv_to_read_from)

    # shuffle all the rows
    df = df.sample(frac=1)

    # split the dataset (60% train/20% validation/20% test)
    # Note: the first split occurs at 60%; the second split occurs at 80% and they are all stored in the variables
    train, validation, test = np.split(df, [int(0.6 * len(df)), int(0.8 * len(df))])

    # use these for confusion matrix below
    train_label, val_label, test_label = train["label"], validation["label"], test["label"]

    X_train_series, y_train_series = train["num_pixels"], train["binary_label"]
    X_val_series, y_val_series = validation["num_pixels"], validation["binary_label"]
    X_test_series, y_test_series = test["num_pixels"], test["binary_label"]

    # convert them to numpy array
    X_train = pd.Series.to_numpy(X_train_series)
    X_train = X_train.reshape(-1, 1)  # reshape to accomodate for fit() function
    y_train = pd.Series.to_numpy(y_train_series)

    X_val = pd.Series.to_numpy(X_val_series)
    X_val = X_val.reshape(-1, 1)  # reshape to accomodate for predict() function
    y_val = pd.Series.to_numpy(y_val_series)

    X_test = pd.Series.to_numpy(X_test_series)
    X_test = X_test.reshape(-1, 1)  # reshape
    y_test = pd.Series.to_numpy(y_test_series)

    # Logistic Regression model
    lgr = LogisticRegression()

    # fit the data
    lgr.fit(X_train, y_train)

    # save model
    model_save_filename = 'logistic_reg_model.pkl'
    joblib.dump(lgr, model_save_filename)
    print("model saved!!")

    # predictions
    probabilities = lgr.predict_proba(X_val)
    # print(probabilities)

    y_train_pred = lgr.predict(X_train)
    y_pred_train_proba = lgr.predict_log_proba(X_train)
    y_val_pred = lgr.predict(X_val)
    y_pred_test_proba = lgr.predict_log_proba(X_test)
    print(y_pred_train_proba)
    print(y_pred_test_proba)

    # compare the accuracy scores
    train_accuracy_score = accuracy_score(y_train, y_train_pred)
    val_accuracy_score = accuracy_score(y_val, y_val_pred)

    print("Train accuracy score: ", train_accuracy_score)
    print("Validation accuracy score: ", val_accuracy_score)
    print("Train Confusion Matrix: ", confusion_matrix(y_train, y_train_pred, normalize='true'))
    print("Val Confusion Matrix: ", confusion_matrix(y_val, y_val_pred, normalize='true'))
    print("Majority Class Accuracy Score", sum(y_val) / len(y_val))

    # temp_x_values = np.arange(0, 9001, 1)
    # temp_x_values = temp_x_values.reshape(-1, 1)
    # prob = lgr.predict_proba(temp_x_values)
    # plus_minus = 0.00015
    # for i in np.arange(0, 1.1, 0.1):
    #     print("prob: ", i)
    #     print("num_pixels: ", np.argwhere((prob[:, 1] <= (i + plus_minus)) & (prob[:, 1] >= (i - plus_minus))))
    #     print("count: ", np.count_nonzero(np.argwhere((prob[:, 1] <= (i + plus_minus)) & (prob[:, 1] >= (i - plus_minus)))))
    #     print("---")
    # pprint.pprint(list(zip(temp_x_values, prob)))
    #
    # plt.plot(temp_x_values, prob[:, 1])
    # df['jittered_label_y'] = df['binary_label'] + [random.uniform(-0.05, 0.05) for _ in range(len(df))]
    # plt.scatter(df["num_pixels"], df["jittered_label_y"], alpha=0.4, color="black")
    # plt.xlabel("Number of pixels")
    #
    # figure = plt.gcf()
    # figure.set_size_inches(18, 9)
    # plt.savefig("logistic_reg_with_num_pixels")
    # plt.show()

    # create_pivot_table(lower_prob_bound=0.1, upper_prob_bound=0.3, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.2, upper_prob_bound=0.4, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.3, upper_prob_bound=0.5, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.4, upper_prob_bound=0.6, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.45, upper_prob_bound=0.65, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.5, upper_prob_bound=0.7, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.6, upper_prob_bound=0.8, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.7, upper_prob_bound=0.9, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.3, upper_prob_bound=0.7, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.2, upper_prob_bound=0.8, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.1, upper_prob_bound=0.9, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.3, upper_prob_bound=0.6, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)
    # create_pivot_table(lower_prob_bound=0.2, upper_prob_bound=0.5, X_val=X_val, probabilities=probabilities,
    #                    val_label=val_label, y_val_pred=y_val_pred)


def create_pivot_table(lower_prob_bound, upper_prob_bound, X_val, probabilities, val_label, y_val_pred):
    # print("before: ", X_val)
    # print(probabilities)
    # print(len(X_val), len(probabilities))
    print(lower_prob_bound, " - ", upper_prob_bound)
    combined_matrix = np.column_stack((X_val, probabilities[:, 1]))
    combined_matrix = np.column_stack((combined_matrix, val_label))
    combined_matrix = np.column_stack((combined_matrix, y_val_pred))
    col_to_consider = combined_matrix[:, 1]
    col_with_ask = np.where((col_to_consider > lower_prob_bound) & (col_to_consider < upper_prob_bound), 0.5,
                            combined_matrix[:, 3])
    combined_matrix = np.column_stack((combined_matrix, col_with_ask))
    # print("after: ", combined_matrix)

    # convert to df
    col_names = ['num_pixels', 'pred_prob', 'label', 'pred_label', 'preds']
    combined_matrix_df = pd.DataFrame(combined_matrix, columns=col_names)
    # print(combined_matrix_df)
    pt = pd.pivot_table(data=combined_matrix_df, values='pred_label', index=['label'], columns=['preds'],
                        aggfunc='count')
    print(pt)
    print("---")

if __name__ == "__main__":
    main()
