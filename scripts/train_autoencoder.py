import json
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.data_loader import (
    load_dataset,
    clean_dataset,
    chronological_split,
)
from src.fraud_detection.preprocessing import prepare_features
from src.fraud_detection.autoencoder import (
    train_autoencoder,
    calculate_reconstruction_scores,
)
from src.fraud_detection.evaluation import (
    select_threshold,
    evaluate_predictions,
    evaluate_at_fpr_targets,
    evaluate_review_capacity,
    format_evaluation_report,
)


def main():
    config_path = PROJECT_ROOT / "configs" / "autoencoder_config.json"
    with open(config_path, "r") as file:
        config = json.load(file)

    data_path = PROJECT_ROOT / config["data_path"]
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("CARD FRAUD TRIAGE - PHASE 2: AUTOENCODER TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load and clean data
    print("\n[1/6] Loading and cleaning dataset...")
    df = load_dataset(data_path)
    original_rows = len(df)
    df = clean_dataset(df)
    cleaned_rows = len(df)
    print(f"Original rows: {original_rows:,} | Cleaned rows: {cleaned_rows:,} (Removed: {original_rows - cleaned_rows:,})")

    # 2. Chronological split
    print("\n[2/6] Splitting data chronologically...")
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
    )
    print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,} | Test rows: {len(test_df):,}")

    # 3. Preprocessing (Scaler fitted ONLY on legitimate training samples)
    print("\n[3/6] Preprocessing features with StandardScaler (legitimate train only)...")
    data = prepare_features(train_df, val_df, test_df)

    X_train_legit = data["X_train"]  # already filtered to Class == 0 by prepare_features
    val_legit_mask = data["y_validation"] == 0
    X_val_legit = data["X_validation"][val_legit_mask]

    print(f"Legitimate training samples: {len(X_train_legit):,}")
    print(f"Legitimate validation samples: {len(X_val_legit):,}")
    print(f"Total test samples: {len(data['X_test']):,} (Fraud: {data['y_test'].sum()})")

    # 4. Train Autoencoder
    print("\n[4/6] Training Undercomplete Denoising Autoencoder...")
    model_params = config["model"]
    train_params = config["training"]

    model, meta = train_autoencoder(
        X_train_legit=X_train_legit,
        X_val_legit=X_val_legit,
        input_dim=model_params["input_dim"],
        hidden_dims=model_params["hidden_dims"],
        latent_dim=model_params["latent_dim"],
        dropout=model_params["dropout"],
        noise_factor=model_params["noise_factor"],
        batch_size=train_params["batch_size"],
        epochs=train_params["epochs"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        patience=train_params["patience"],
        random_state=train_params["random_state"],
        verbose=True,
    )
    print(f"Model parameters: {meta['total_parameters']:,} | Best Val MSE: {meta['best_val_loss']:.6f} at epoch {meta['best_epoch']}")

    # 5. Generate reconstruction anomaly scores
    print("\n[5/6] Computing MSE reconstruction anomaly scores...")
    val_scores = calculate_reconstruction_scores(model, data["X_validation"])
    test_scores = calculate_reconstruction_scores(model, data["X_test"])

    # Threshold selection on validation (maximizing F1)
    val_summary = select_threshold(data["y_validation"], val_scores)
    threshold = val_summary["threshold"]
    print(f"Selected F1-optimal threshold from validation set: {threshold:.6f}")

    # 6. Evaluate on holdout test set
    print("\n[6/6] Evaluating on untouched test set...")
    test_metrics = evaluate_predictions(data["y_test"], test_scores, threshold)
    fpr_results = evaluate_at_fpr_targets(
        data["y_validation"],
        val_scores,
        data["y_test"],
        test_scores,
        targets=config["fpr_targets"],
    )
    review_results = evaluate_review_capacity(
        data["y_test"],
        test_scores,
        capacities=config["review_capacities"],
    )

    # Save artifacts
    results = {
        "model": {
            "type": "Undercomplete Denoising Autoencoder",
            "input_dim": model_params["input_dim"],
            "hidden_dims": model_params["hidden_dims"],
            "latent_dim": model_params["latent_dim"],
            "best_epoch": meta["best_epoch"],
            "best_val_loss": meta["best_val_loss"],
            "parameters": meta["total_parameters"],
        },
        "validation": val_summary,
        "test": test_metrics,
        "fpr_analysis": fpr_results,
        "review_capacity": review_results,
        "dataset": {
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "duplicates_removed": original_rows - cleaned_rows,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        },
    }

    results_file = artifacts_dir / "autoencoder_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation metrics to: {results_file}")

    model_file = artifacts_dir / "autoencoder_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": data["scaler"],
            "threshold": threshold,
            "feature_columns": data["feature_columns"],
            "config": config,
        },
        model_file,
    )
    print(f"Saved model checkpoint to: {model_file}")

    # Print summary report
    report = format_evaluation_report(
        model_name="Phase 2: Autoencoder (Reconstruction Anomaly Detection)",
        test_metrics=test_metrics,
        fpr_results=fpr_results,
        review_results=review_results,
        threshold=threshold,
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
