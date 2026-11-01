
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
from src.fraud_detection.model import calculate_anomaly_scores
from src.fraud_detection.evaluation import (
    evaluate_predictions,
    evaluate_at_fpr_targets,
    evaluate_review_capacity,
)


def main():
    artifacts_dir = PROJECT_ROOT / "artifacts"

    model_path = artifacts_dir / "gmm_model.joblib"
    results_path = artifacts_dir / "gmm_results.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model not found. Run scripts/train_gmm.py first."
        )

    with open(PROJECT_ROOT / "configs" / "gmm_config.json") as file:
        config = json.load(file)

    # Load trained model bundle
    bundle = joblib.load(model_path)

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    feature_columns = bundle["feature_columns"]

    # Recreate the same chronological test split
    df = clean_dataset(load_dataset(PROJECT_ROOT / config["data_path"]))

    _, validation_df, test_df = chronological_split(
        df,
        config["train_ratio"],
        config["validation_ratio"],
    )

    X_test = test_df[feature_columns]
    y_test = test_df["Class"].to_numpy()

    X_test_scaled = scaler.transform(X_test)

    # Evaluate saved model without retraining
    test_scores = calculate_anomaly_scores(model, X_test_scaled)

    metrics = evaluate_predictions(
        y_test,
        test_scores,
        threshold,
    )

    print("=" * 55)
    print("SAVED GMM MODEL - TEST EVALUATION")
    print("=" * 55)
    print(f"Components: {model.n_components}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Test transactions: {len(y_test):,}")

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=4))

    # Recalculate operational metrics using validation scores
    X_validation_scaled = scaler.transform(
        validation_df[feature_columns]
    )
    y_validation = validation_df["Class"].to_numpy()

    validation_scores = calculate_anomaly_scores(
        model,
        X_validation_scaled,
    )

    fpr_results = evaluate_at_fpr_targets(
        y_validation,
        validation_scores,
        y_test,
        test_scores,
        targets=config["fpr_targets"],
    )

    review_results = evaluate_review_capacity(
        y_test,
        test_scores,
        capacities=config["review_capacities"],
    )

    print("\nFPR budget analysis:")
    print(json.dumps(fpr_results, indent=4))

    print("\nReview capacity analysis:")
    print(json.dumps(review_results, indent=4))

    # Confirm consistency with the training results
    if results_path.exists():
        with open(results_path) as file:
            original_results = json.load(file)

        original_ap = original_results["test"]["average_precision"]

        print("\nConsistency check:")
        print(f"Training AP: {original_ap:.10f}")
        print(f"Reloaded AP: {metrics['average_precision']:.10f}")

        print(
            "Results match:",
            abs(original_ap - metrics["average_precision"]) < 1e-10,
        )


if __name__ == "__main__":
    main()
