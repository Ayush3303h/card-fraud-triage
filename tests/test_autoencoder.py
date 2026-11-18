import unittest
import numpy as np
import torch

from src.fraud_detection.autoencoder import (
    TransactionAutoencoder,
    train_autoencoder,
    calculate_reconstruction_scores,
)


class TestAutoencoder(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.X_train = np.random.normal(size=(200, 30)).astype(np.float32)
        self.X_val = np.random.normal(size=(50, 30)).astype(np.float32)
        self.X_test = np.random.normal(size=(50, 30)).astype(np.float32)

    def test_model_forward_shape(self):
        model = TransactionAutoencoder(input_dim=30, hidden_dims=[16], latent_dim=4)
        x = torch.randn(10, 30)
        out = model(x)
        self.assertEqual(out.shape, (10, 30))

    def test_train_autoencoder_runs(self):
        model, meta = train_autoencoder(
            self.X_train,
            self.X_val,
            input_dim=30,
            hidden_dims=[16],
            latent_dim=4,
            epochs=3,
            batch_size=64,
            device="cpu",
            verbose=False,
        )
        self.assertIn("best_val_loss", meta)
        self.assertIn("total_parameters", meta)
        self.assertGreater(meta["total_parameters"], 0)

    def test_reconstruction_scores(self):
        model = TransactionAutoencoder(input_dim=30, hidden_dims=[16], latent_dim=4)
        scores = calculate_reconstruction_scores(model, self.X_test, batch_size=25, device="cpu")
        self.assertEqual(scores.shape, (50,))
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(np.isfinite(scores)))


if __name__ == "__main__":
    unittest.main()
