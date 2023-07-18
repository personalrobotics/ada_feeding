import csv

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
sns.set(style="whitegrid",font_scale=2)
import matplotlib.collections as clt
import sys
sys.path.append('../ptitprince/')
import ptitprince as pt
import pprint


def raincloud_with_boxplot(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=1)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5)

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
                     showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                     saturation=1, orient=ort)

    plt.title("Overall Data")
    plt.axvline(x=1780, color='r', label='lower bound')
    plt.axvline(x=2417, color='r', label='upper bound')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data.png')
    plt.show()

def raincloud_with_boxplot2(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=1)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5)

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
                     showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                     saturation=1, orient=ort)

    plt.title("Overall Data (all labels)")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_all_labels.png')
    plt.show()


def raincloud_with_boxplot_hand_dropped(df):
    # before dropping hand data
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=3)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="label")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with each label")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_each_label.png')
    plt.show()

    # drop hand data
    print("DF size before: ", len(df))
    df.drop(df[df['label'] == "hand"].index, inplace=True)
    print("DF size before: ", len(df))

    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=3)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="label")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with hand dropped")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_hand_dropped.png')
    plt.show()

def raincloud_with_boxplot_position(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=3)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="position")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with position")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_position.png')
    plt.show()

def raincloud_with_boxplot_position2(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=3)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="position")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with position (all labels)")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_position_all_labels.png')
    plt.show()

def raincloud_with_boxplot_position_divided(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=3)

    ax = sns.boxplot(x=dx, y=dy, data=df, hue="position", color="black", showcaps=True, orient=ort, boxprops={'facecolor': 'none', "zorder": 10})
    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, alpha=0.5, ax=ax, hue="position", dodge=True)

    plt.title("Overall Data with positions divided")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_position_divided.png')
    plt.show()


def raincloud_with_boxplot_food(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=8)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="food")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with food types")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_food_types.png')
    plt.show()

def raincloud_with_boxplot_food2(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=8)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="food")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with food types (all labels)")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_food_types_all_labels.png')
    plt.show()

def raincloud_with_boxplot_food_divided(df):
    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=8)

    ax = sns.boxplot(x=dx, y=dy, data=df, hue="food", orient=ort,
                     boxprops={'facecolor': 'none', "zorder": 10})
    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, alpha=0.6, ax=ax, hue="food", dodge=True)
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:7], labels[0:7])

    plt.title("Overall Data with food divided")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_food_divided.png')
    plt.show()

def raincloud_with_boxplot_food_divided_hand_dropped(df):
    # drop hand data
    print("DF size before: ", len(df))
    df.drop(df[df['label'] == "hand"].index, inplace=True)
    print("DF size before: ", len(df))

    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=8)

    ax = pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort)

    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, ax=ax, alpha=0.5, hue="food")

    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10, showcaps=True,
                     boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10}, saturation=1, orient=ort)

    plt.title("Overall Data with food types (hand dropped)")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_food_types_hand_dropped.png')
    plt.show()


    f, ax = plt.subplots(figsize=(7, 5))
    dy = "binary_label"
    dx = "num_pixels"
    ort = "h"
    pal = sns.color_palette(n_colors=8)

    ax = sns.boxplot(x=dx, y=dy, data=df, hue="food", orient=ort,
                     boxprops={'facecolor': 'none', "zorder": 10})
    ax = sns.stripplot(x=dx, y=dy, data=df, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort, alpha=0.6, ax=ax, hue="food", dodge=True)
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[0:7], labels[0:7])

    plt.title("Overall Data with food divided (hand dropped)")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(18, 9)
    plt.savefig('overall_data_with_food_divided_hand_dropped.png')
    plt.show()


def main():
    csv_to_read_from = "Ross_7-11-23.csv"
    csv_to_write_to = "Ross_7-11-23_predictions.csv"

    # get data frame
    df = pd.read_csv(csv_to_read_from)

    # visualize the dataset
    # df_raincloud_with_boxplot = df.copy(deep=True)
    # raincloud_with_boxplot(df_raincloud_with_boxplot)
    #
    # df_raincloud_with_boxplot_hand = df.copy(deep=True)
    # raincloud_with_boxplot_hand_dropped(df_raincloud_with_boxplot_hand)
    #
    # df_raincloud_with_boxplot_position = df.copy(deep=True)
    # raincloud_with_boxplot_position(df_raincloud_with_boxplot_position)
    #
    # df_raincloud_with_boxplot_position_divided = df.copy(deep=True)
    # raincloud_with_boxplot_position_divided(df_raincloud_with_boxplot_position_divided)

    # df_raincloud_with_boxplot_food = df.copy(deep=True)
    # raincloud_with_boxplot_food(df_raincloud_with_boxplot_food)
    #
    # df_raincloud_with_boxplot_food_divided = df.copy(deep=True)
    # raincloud_with_boxplot_food_divided(df_raincloud_with_boxplot_food_divided)
    #
    # df_raincloud_with_boxplot_food_divided_hand_dropped = df.copy(deep=True)
    # raincloud_with_boxplot_food_divided_hand_dropped(df_raincloud_with_boxplot_food_divided_hand_dropped)

    df_raincloud_with_boxplot2 = df.copy(deep=True)
    raincloud_with_boxplot2(df_raincloud_with_boxplot2)

    df_raincloud_with_boxplot_position2 = df.copy(deep=True)
    raincloud_with_boxplot_position2(df_raincloud_with_boxplot_position2)

    df_raincloud_with_boxplot_food2 = df.copy(deep=True)
    raincloud_with_boxplot_food2(df_raincloud_with_boxplot_food2)

    # shuffle all the rows
    df = df.sample(frac=1)

    # split the dataset (60% train/20% validation/20% test)
    # Note: the first split occurs at 60%; the second split occurs at 80% and they are all stored in the variables
    train, validation, test = np.split(df, [int(0.6 * len(df)), int(0.8 * len(df))])
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

    # Note: There are 5 major types of Naive Bayes that is offered through sklearn
    # * BernoulliNB: this is suitable for discrete data (if the features we have are of binary/boolean values)
    # * CategoricalNB: assumes that each feature has categorical distribution
    # * MultinomialNB: classification of discrete features (ex: word counts for text classification)
    # * ComplementNB: adaptation of MultinomialNB but is suitable for imbalanced datasets
    # * GaussianNB: the likelihood of features is assumed to be normally distributed
    #   * Choosing GaussianNB:
    #   * In MultinomialNB, we are assuming that the features are discrete. This can cause issues because it treats each
    #   * number as a discrete group (for instance, 2205 is different 2206). As such, it would make sense to assume that
    #   * the number of pixels (features) are actually continuous.
    clf = GaussianNB()

    # fit the data
    clf.fit(X_train, y_train)

    # predictions
    # pridict_proba will result in a 2d matrix where the left column is the probability that the label is 0
    # and the right column is the probability that the label is 1
    probabilities = clf.predict_proba(X_val)

    temp_x_values = np.arange(0, 9001, 1)
    temp_x_values = temp_x_values.reshape(-1, 1)
    prob = clf.predict_proba(temp_x_values)
    pprint.pprint(list(zip(temp_x_values, prob)))
    raise Exception()
    # print(np.argwhere((prob[:, 1] > 0.19) & (prob[:, 1] < 0.21)))
    with open("temp_csv.csv", 'w', newline='') as f:
        f_write = csv.writer(f)
        f_write.writerow(["num_pixels", "prob_0", "prob_1"])
        i = 0
        for probs in prob:
            f_write.writerow([i, probs[0], probs[1]])
            i += 1

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # compare using the accuracy score
    train_accuracy_score = accuracy_score(y_train, y_train_pred)
    val_accuracy_score = accuracy_score(y_val, y_val_pred)

    print("Train Accuracy Score: ", train_accuracy_score)
    print("Train Confusion Matrix: ", confusion_matrix(y_train, y_train_pred, normalize='true'))
    print("Majority Class Accuracy Score", sum(y_train) / len(y_train))
    print("Validation Accuracy Score: ", val_accuracy_score)


if __name__ == "__main__":
    main()
