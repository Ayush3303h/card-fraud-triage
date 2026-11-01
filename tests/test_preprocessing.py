
import unittest

import numpy as np
import pandas as pd

from src.fraud_detection.data_loader import (
    clean_dataset,
    chronological_split,
)
from src.fraud_detection.preprocessing import prepare_features


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Time": range(100),
            "V1": np.random.default_rng(42).normal(size=100),
            "Amount": np.random.default_rng(43).uniform(1, 100, 100),
            "Class": [0] * 95 + [1] * 5,
        })

    def test_duplicate_removal(self):
        duplicated = pd.concat(
            [self.df, self.df.iloc[[0]]],
            ignore_index=True,
        )

        cleaned = clean_dataset(duplicated)

        self.assertEqual(len(cleaned), 100)
        self.assertFalse(cleaned.duplicated().any())

    def test_chronological_split(self):
        train, validation, test = chronological_split(self.df)

        self.assertEqual(len(train), 70)
        self.assertEqual(len(validation), 15)
        self.assertEqual(len(test), 15)

        self.assertLess(train["Time"].max(), validation["Time"].min())
        self.assertLess(validation["Time"].max(), test["Time"].min())

    def test_scaler_fits_legitimate_training_only(self):
        train = self.df.iloc[:70].copy()
        validation = self.df.iloc[70:85].copy()
        test = self.df.iloc[85:].copy()

        result = prepare_features(train, validation, test)

        legitimate_train = train[train["Class"] == 0][
            ["Time", "V1", "Amount"]
        ]

        expected_mean = legitimate_train.mean().to_numpy()

        np.testing.assert_allclose(
            result["scaler"].mean_,
            expected_mean,
        )

        self.assertEqual(result["X_train"].shape[1], 3)
        self.assertEqual(len(result["y_validation"]), 15)
        self.assertEqual(len(result["y_test"]), 15)


if __name__ == "__main__":
    unittest.main()
