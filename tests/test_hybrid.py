import unittest
import numpy as np
import torch

from src.fraud_detection.autoencoder import TransactionAutoencoder
from src.fraud_detection.hybrid import HybridFraudDetector


class TestHybrid(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.X_train_legit = np.random.normal(size=(100, 30)).astype(np.float32)
        self.X_test = np.random.normal(size=(20, 30)).astype(np.float32)

        self.ae = TransactionAutoencoder(input_dim=30, hidden_dims=[16], latent_dim=4)

    def test_latent_extraction(self):
        detector = HybridFraudDetector(self.ae, latent_components=2)
        Z = detector.extract_latent_vectors(self.X_test, device="cpu")
        self.assertEqual(Z.shape, (20, 4))

    def test_fit_and_hybrid_scoring(self):
        detector = HybridFraudDetector(self.ae, latent_components=2)
        detector.fit_latent_gmm(self.X_train_legit, device="cpu")

        scores, mse, nll = detector.calculate_hybrid_scores(self.X_test, fit_scalers=True, device="cpu")
        self.assertEqual(scores.shape, (20,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0))


if __name__ == "__main__":
    unittest.main()
