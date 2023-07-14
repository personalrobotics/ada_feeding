import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid",font_scale=2)
import matplotlib.collections as clt
import sys
sys.path.append('../ptitprince/')
import ptitprince as pt
import pprint

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

    # Logistic Regression
    lgr = LogisticRegression()

    # fit the data
    lgr.fit(X_train, y_train)

    # predictions
    probabilities = lgr.predict_proba(X_val)
    # print(probabilities)

    y_train_pred = lgr.predict(X_train)
    y_val_pred = lgr.predict(X_val)

    # compare the accuracy scores
    train_accuracy_score = accuracy_score(y_train, y_train_pred)
    val_accuracy_score = accuracy_score(y_val, y_val_pred)

    print("Train accuracy score: ", train_accuracy_score)
    print("Validation accuracy score: ", val_accuracy_score)
    print("Train Confusion Matrix: ", confusion_matrix(y_train, y_train_pred, normalize='true'))
    print("Majority Class Accuracy Score", sum(y_train) / len(y_train))

    temp_x_values = np.arange(0, 9001, 1)
    temp_x_values = temp_x_values.reshape(-1, 1)
    prob = lgr.predict_proba(temp_x_values)
    plus_minus = 0.00015
    for i in np.arange(0, 1.1, 0.1):
        print("prob: ", i)
        print("num_pixels: ", np.argwhere((prob[:, 1] <= (i + plus_minus)) & (prob[:, 1] >= (i - plus_minus))))
        print("count: ", np.count_nonzero(np.argwhere((prob[:, 1] <= (i + plus_minus)) & (prob[:, 1] >= (i - plus_minus)))))
        print("---")
    # pprint.pprint(list(zip(temp_x_values, prob)))

    plt.subplot(2, 1, 1)
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
    plt.axvline(x=1780, color='r', label='lower bound')
    plt.axvline(x=2417, color='r', label='upper bound')

    plt.subplot(2, 1, 2)
    plt.plot(temp_x_values, prob[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
