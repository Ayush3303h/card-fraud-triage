import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import torch

from src.fraud_detection.evaluation import (
    select_threshold,
    evaluate_predictions,
    evaluate_at_fpr_targets,
    evaluate_review_capacity,
    format_evaluation_report,
)


class HybridFraudDetector:
    """
    Hybrid Anomaly Detection Model combining Deep Learning & Classical Machine Learning.

    Technical Rationale for Hybrid Innovation:
    1. The Undercomplete Autoencoder maps high-dimensional (30D), noisy, non-linear transaction
       data into a smooth, low-dimensional (6D) latent manifold.
    2. A Gaussian Mixture Model (GMM) fits probability density directly on the 6D latent manifold
       (Latent GMM), which eliminates the high-dimensional covariance instability that GMM suffers from in 30D.
    3. An ensemble score fuses normalized structural reconstruction error (from Autoencoder) with
       normalized latent negative log-likelihood (from GMM):
            s_hybrid(x) = alpha * s_recon_norm(x) + (1 - alpha) * s_latent_nll_norm(x)
       This combines neural manifold boundary reconstruction with calibrated probabilistic likelihood.
    """

    def __init__(self, autoencoder, latent_components=8, alpha=0.6, random_state=42):
        self.autoencoder = autoencoder
        self.latent_components = latent_components
        self.alpha = alpha
        self.random_state = random_state
        self.latent_gmm = GaussianMixture(
            n_components=latent_components,
            covariance_type="diag",
            max_iter=150,
            n_init=3,
            random_state=random_state,
        )
        self.scaler_recon = MinMaxScaler()
        self.scaler_nll = MinMaxScaler()

    def extract_latent_vectors(self, X, device=None):
        """Extract latent bottleneck representations using the trained encoder."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.autoencoder.eval()
        self.autoencoder.to(device)

        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
            latent = self.autoencoder.encoder(tensor_x).cpu().numpy()
        return latent

    def fit_latent_gmm(self, X_train_legit, device=None):
        """Fit GMM on legitimate training latent representations."""
        Z_train_legit = self.extract_latent_vectors(X_train_legit, device=device)
        self.latent_gmm.fit(Z_train_legit)
        return self

    def calculate_hybrid_scores(self, X, fit_scalers=False, device=None):
        """Compute hybrid anomaly scores combining reconstruction error and latent negative log-likelihood."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.autoencoder.eval()
        self.autoencoder.to(device)

        # 1. Reconstruction error
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
            recon = self.autoencoder(tensor_x)
            mse_recon = torch.mean((tensor_x - recon) ** 2, dim=1).cpu().numpy()

        # 2. Latent space negative log-likelihood
        Z = self.extract_latent_vectors(X, device=device)
        nll_latent = -self.latent_gmm.score_samples(Z)

        # Normalize scores to [0, 1] range
        if fit_scalers:
            norm_recon = self.scaler_recon.fit_transform(mse_recon.reshape(-1, 1)).ravel()
            norm_nll = self.scaler_nll.fit_transform(nll_latent.reshape(-1, 1)).ravel()
        else:
            norm_recon = self.scaler_recon.transform(mse_recon.reshape(-1, 1)).ravel()
            norm_nll = self.scaler_nll.transform(nll_latent.reshape(-1, 1)).ravel()

        # Weighted combination
        hybrid_scores = self.alpha * norm_recon + (1.0 - self.alpha) * norm_nll
        return hybrid_scores, mse_recon, nll_latent
