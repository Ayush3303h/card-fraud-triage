import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TransactionAutoencoder(nn.Module):
    """
    Undercomplete Denoising Autoencoder for Credit Card Fraud Detection.
    Learns to reconstruct legitimate transaction vectors through a compressed latent space.
    Anomalous/fraudulent transactions incur higher reconstruction error.
    """

    def __init__(
        self,
        input_dim=30,
        hidden_dims=(20, 12),
        latent_dim=6,
        dropout=0.05,
        noise_factor=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.noise_factor = noise_factor

        # Encoder: compresses 30-dim transaction down to latent_dim
        encoder_layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_d = h_dim
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: reconstructs 30-dim transaction from latent representation
        decoder_layers = []
        in_d = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_d = h_dim
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def set_seed(seed=42):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_autoencoder(
    X_train_legit,
    X_val_legit,
    input_dim=30,
    hidden_dims=(20, 12),
    latent_dim=6,
    dropout=0.05,
    noise_factor=0.02,
    batch_size=256,
    epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=8,
    random_state=42,
    device=None,
    verbose=True,
):
    """
    Train autoencoder exclusively on legitimate training transactions.
    Monitors reconstruction loss on legitimate validation transactions for early stopping.
    """
    set_seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = TransactionAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        noise_factor=noise_factor,
    ).to(device)

    train_tensor = torch.tensor(X_train_legit, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_legit, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_x)

        avg_train_loss = total_train_loss / len(X_train_legit)

        # Evaluate validation reconstruction loss (no noise, eval mode)
        model.eval()
        with torch.no_grad():
            val_batch = val_tensor.to(device)
            val_recon = model(val_batch)
            avg_val_loss = criterion(val_recon, val_batch).item()

        scheduler.step(avg_val_loss)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or patience_counter >= patience):
            print(
                f"  Epoch [{epoch:>2}/{epochs:>2}] | "
                f"Train MSE: {avg_train_loss:.6f} | "
                f"Val MSE: {avg_val_loss:.6f} | "
                f"Best Val: {best_val_loss:.6f} (Ep {best_epoch})"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {best_epoch}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    training_meta = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "total_epochs": epoch,
        "history": history,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, training_meta


def calculate_reconstruction_scores(model, X, batch_size=512, device=None):
    """
    Calculate sample-wise Mean Squared Error (MSE) reconstruction anomaly scores.
    Higher score indicates an unusual pattern that the autoencoder struggled to reconstruct.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model.eval()
    model.to(device)

    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            recon = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=1).cpu().numpy()
            scores.append(mse)

    return np.concatenate(scores)


def explain_reconstruction_error(model, x_single, feature_names=None, device=None):
    """
    Explain why a specific transaction was flagged as anomalous by decomposing
    the total reconstruction error into per-feature squared error contributions.
    Returns a sorted list of features ranked by contribution to the anomaly score.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model.eval()
    model.to(device)

    x_tensor = torch.tensor(np.asarray(x_single, dtype=np.float32).reshape(1, -1)).to(device)
    with torch.no_grad():
        recon = model(x_tensor)
        feature_errors = ((x_tensor - recon) ** 2).squeeze(0).cpu().numpy()

    total_error = float(np.sum(feature_errors))
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(feature_errors))]

    contributions = []
    for name, err in zip(feature_names, feature_errors):
        pct = (err / (total_error + 1e-12)) * 100
        contributions.append({
            "feature": name,
            "error": float(err),
            "percentage": float(pct),
        })

    contributions.sort(key=lambda item: item["error"], reverse=True)
    return contributions

