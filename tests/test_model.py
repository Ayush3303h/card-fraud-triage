
import unittest

import numpy as np

from src.fraud_detection.model import (
    train_gmm,
    calculate_anomaly_scores,
    get_model_info,
)


class TestGMMModel(unittest.TestCase):

    def setUp(self):
        self.X = np.random.default_rng(42).normal(size=(500, 5))

    def test_model_training(self):
        model = train_gmm(
            self.X,
            n_components=3,
            random_state=42,
        )

        self.assertTrue(model.converged_)
        self.assertEqual(model.n_components, 3)

    def test_anomaly_score_shape(self):
        model = train_gmm(
            self.X,
            n_components=3,
            random_state=42,
        )

        scores = calculate_anomaly_scores(model, self.X[:10])

        self.assertEqual(scores.shape, (10,))
        self.assertTrue(np.isfinite(scores).all())

    def test_model_information(self):
        model = train_gmm(
            self.X,
            n_components=3,
            random_state=42,
        )

        info = get_model_info(model)

        self.assertEqual(info["components"], 3)
        self.assertEqual(info["covariance_type"], "diag")
        self.assertIn("converged", info)


if __name__ == "__main__":
    unittest.main()
