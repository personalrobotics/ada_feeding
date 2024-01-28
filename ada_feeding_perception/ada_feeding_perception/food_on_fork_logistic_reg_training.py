"""
To perform Logistic Regression training, make sure to place the dataset with num_pixels and binary_label. Note that
num_pixels will be an integer value and binary_label will be a prediction of 0 (no food) or 1 (food).

Note that there are two variables within the main() method that needs to be changed to run the training. Firstly,
make sure to set csv_to_read_from to the correct csv to read from and model_save_filename to an appropriate model name.
"""

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


def main():
    """
    Main method to run Logisitic Regression Training
    """
    csv_to_read_from = "../datasets/Logistic_reg_dataset_7-11-23.csv"

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

    # Logistic Regression model
    lgr = LogisticRegression()

    # fit the data
    lgr.fit(X_train, y_train)

    # save model
    model_save_filename = "../model/logistic_reg_model.pkl"
    joblib.dump(lgr, model_save_filename)
    print("model saved!!")


if __name__ == "__main__":
    main()
