
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features(train_df, validation_df, test_df):
    """
    Fit StandardScaler only on legitimate training transactions.
    Transform validation and test data using the fitted scaler.
    """

    feature_columns = [
        column for column in train_df.columns
        if column != "Class"
    ]

    X_train = train_df.loc[
        train_df["Class"] == 0, feature_columns
    ]

    X_validation = validation_df[feature_columns]
    X_test = test_df[feature_columns]

    y_validation = validation_df["Class"].copy()
    y_test = test_df["Class"].copy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_validation": X_validation_scaled,
        "X_test": X_test_scaled,
        "y_validation": y_validation.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
