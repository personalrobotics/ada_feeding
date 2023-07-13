import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def split_data():
    pass


def main():
    csv_to_read_from = "Ross_7-11-23.csv"
    csv_to_write_to = "Ross_7-11-23_predictions.csv"

    # get data frame
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

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # compare using the accuracy score
    train_accuracy_score = accuracy_score(y_train, y_train_pred)
    val_accuracy_score = accuracy_score(y_val, y_val_pred)

    print("Train Accuracy Score: ", train_accuracy_score)
    print("Train Confusion Matrix: ", confusion_matrix(y_train, y_train_pred, normalize='true'))
    print("Majority Class Accuracy Score", sum(y_train)/len(y_train))
    print("Validation Accuracy Score: ", val_accuracy_score)



if __name__ == "__main__":
    main()
