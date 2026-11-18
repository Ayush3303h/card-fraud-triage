import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.evaluation import format_evaluation_report


def display_results_from_artifact(results_path):
    """Display evaluation report from saved artifact JSON."""
    with open(results_path, "r") as file:
        results = json.load(file)

    model_info = results.get("model", {})
    val_info = results.get("validation", {})
    test_metrics = results.get("test", {})
    fpr_results = results.get("fpr_analysis", [])
    review_results = results.get("review_capacity", [])

    model_title = (
        f"Phase 1: Gaussian Mixture Model (K={model_info.get('components', 38)}, "
        f"Cov={model_info.get('covariance_type', 'diag')})"
    )

    report = format_evaluation_report(
        model_name=model_title,
        test_metrics=test_metrics,
        fpr_results=fpr_results,
        review_results=review_results,
        threshold=val_info.get("threshold"),
    )
    print(report)


def evaluate_live(model_path, config_path, results_path):
    """Run live evaluation if trained model joblib and dataset exist."""
    import joblib
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

    with open(config_path) as file:
        config = json.load(file)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    feature_columns = bundle["feature_columns"]

    df = clean_dataset(load_dataset(PROJECT_ROOT / config["data_path"]))
    _, validation_df, test_df = chronological_split(
        df,
        config["train_ratio"],
        config["validation_ratio"],
    )

    X_test_scaled = scaler.transform(test_df[feature_columns])
    y_test = test_df["Class"].to_numpy()
    test_scores = calculate_anomaly_scores(model, X_test_scaled)

    X_val_scaled = scaler.transform(validation_df[feature_columns])
    y_val = validation_df["Class"].to_numpy()
    validation_scores = calculate_anomaly_scores(model, X_val_scaled)

    test_metrics = evaluate_predictions(y_test, test_scores, threshold)
    fpr_results = evaluate_at_fpr_targets(
        y_val, validation_scores, y_test, test_scores, targets=config["fpr_targets"]
    )
    review_results = evaluate_review_capacity(
        y_test, test_scores, capacities=config["review_capacities"]
    )

    model_title = f"Phase 1: Gaussian Mixture Model (K={model.n_components}, Live Evaluated)"
    report = format_evaluation_report(
        model_name=model_title,
        test_metrics=test_metrics,
        fpr_results=fpr_results,
        review_results=review_results,
        threshold=threshold,
    )
    print(report)


def main():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    model_path = artifacts_dir / "gmm_model.joblib"
    results_path = artifacts_dir / "gmm_results.json"
    config_path = PROJECT_ROOT / "configs" / "gmm_config.json"

    # If joblib model and dataset exist, run live; otherwise load saved results
    with open(config_path) as file:
        config = json.load(file)
    data_path = PROJECT_ROOT / config["data_path"]

    if model_path.exists() and data_path.exists():
        evaluate_live(model_path, config_path, results_path)
    elif results_path.exists():
        display_results_from_artifact(results_path)
    else:
        raise FileNotFoundError(
            f"Neither saved model ({model_path}) nor results artifact ({results_path}) was found."
        )


if __name__ == "__main__":
    main()
