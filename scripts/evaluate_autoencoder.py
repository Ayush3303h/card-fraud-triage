import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.evaluation import format_evaluation_report


def display_results_from_artifact(results_path):
    with open(results_path, "r") as file:
        results = json.load(file)

    model_info = results.get("model", {})
    val_info = results.get("validation", {})
    test_metrics = results.get("test", {})
    fpr_results = results.get("fpr_analysis", [])
    review_results = results.get("review_capacity", [])

    model_title = (
        f"Phase 2: {model_info.get('type', 'Undercomplete Autoencoder')} "
        f"(Latent Dim={model_info.get('latent_dim', 6)}, Best Epoch={model_info.get('best_epoch', 28)})"
    )

    report = format_evaluation_report(
        model_name=model_title,
        test_metrics=test_metrics,
        fpr_results=fpr_results,
        review_results=review_results,
        threshold=val_info.get("threshold"),
    )
    print(report)


def main():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_path = artifacts_dir / "autoencoder_results.json"

    if results_path.exists():
        display_results_from_artifact(results_path)
    else:
        raise FileNotFoundError(
            f"Results artifact ({results_path}) was not found. Run scripts/train_autoencoder.py first."
        )


if __name__ == "__main__":
    main()
