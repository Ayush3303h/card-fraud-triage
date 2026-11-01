
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.data_loader import (
    load_dataset,
    clean_dataset,
    chronological_split,
)
from src.fraud_detection.preprocessing import prepare_features
from src.fraud_detection.model import (
    train_gmm,
    calculate_anomaly_scores,
    get_model_info,
)
from src.fraud_detection.evaluation import (
    select_threshold,
    evaluate_predictions,
    evaluate_at_fpr_targets,
    evaluate_review_capacity,
)


def main():
    config_path = PROJECT_ROOT / "configs" / "gmm_config.json"

    with open(config_path, "r") as file:
        config = json.load(file)

    data_path = PROJECT_ROOT / config["data_path"]

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CARD FRAUD TRIAGE - GMM TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load and clean data
    print("\n[1/6] Loading dataset...")

    df = load_dataset(data_path)
    original_rows = len(df)

    df = clean_dataset(df)

    print(f"Original rows: {original_rows:,}")
    print(f"Rows after duplicate removal: {len(df):,}")
    print(f"Duplicates removed: {original_rows - len(df):,}")

    # 2. Chronological split
    print("\n[2/6] Creating chronological splits...")

    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=config["train_ratio"],
        validation_ratio=config["validation_ratio"],
    )

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")

    # 3. Preprocessing
    print("\n[3/6] Scaling features...")

    data = prepare_features(train_df, val_df, test_df)

    print("Training legitimate samples:", len(data["X_train"]))
    print("Validation samples:", len(data["X_validation"]))
    print("Test samples:", len(data["X_test"]))

    # 4. Train GMM
    print("\n[4/6] Training Gaussian Mixture Model...")

    model = train_gmm(
        data["X_train"],
        **config["model"],
    )

    model_info = get_model_info(model)
    print("Model information:", model_info)

    if not model_info["converged"]:
        raise RuntimeError("GMM did not converge. Check model configuration.")

    # 5. Generate scores and select threshold on validation
    print("\n[5/6] Generating anomaly scores...")

    validation_scores = calculate_anomaly_scores(
        model,
        data["X_validation"],
    )

    test_scores = calculate_anomaly_scores(
        model,
        data["X_test"],
    )

    threshold_results = select_threshold(
        data["y_validation"],
        validation_scores,
    )

    threshold = threshold_results["threshold"]

    print("\nValidation threshold results:")
    print(json.dumps(threshold_results, indent=4))

    # 6. Final test evaluation
    print("\n[6/6] Evaluating on test data...")

    test_results = evaluate_predictions(
        data["y_test"],
        test_scores,
        threshold,
    )

    fpr_results = evaluate_at_fpr_targets(
        data["y_validation"],
        validation_scores,
        data["y_test"],
        test_scores,
        targets=config["fpr_targets"],
    )

    review_results = evaluate_review_capacity(
        data["y_test"],
        test_scores,
        capacities=config["review_capacities"],
    )

    results = {
        "model": model_info,
        "validation": threshold_results,
        "test": test_results,
        "fpr_analysis": fpr_results,
        "review_capacity": review_results,
        "dataset": {
            "original_rows": original_rows,
            "cleaned_rows": len(df),
            "duplicates_removed": original_rows - len(df),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
        },
    }

    # Save model, scaler and metadata together
    bundle = {
        "model": model,
        "scaler": data["scaler"],
        "feature_columns": data["feature_columns"],
        "threshold": threshold,
        "config": config,
    }

    joblib.dump(bundle, artifacts_dir / "gmm_model.joblib")

    with open(artifacts_dir / "gmm_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS - GMM K=38")
    print("=" * 60)

    for metric in [
        "average_precision",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "confusion_matrix",
    ]:
        print(f"{metric}: {test_results[metric]}")

    print("\nFPR analysis:")
    print(json.dumps(fpr_results, indent=4))

    print("\nReview capacity:")
    print(json.dumps(review_results, indent=4))

    print("\nSaved model:", artifacts_dir / "gmm_model.joblib")
    print("Saved results:", artifacts_dir / "gmm_results.json")
    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
