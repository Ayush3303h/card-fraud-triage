import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.comparison import format_comparison_table


def main():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    gmm_file = artifacts_dir / "gmm_results.json"
    ae_file = artifacts_dir / "autoencoder_results.json"

    if not gmm_file.exists():
        print(f"Error: {gmm_file} not found. Run scripts/train_gmm.py first.")
        return

    if not ae_file.exists():
        print(f"Notice: {ae_file} not found. Run scripts/train_autoencoder.py to generate Autoencoder metrics.")
        return

    with open(gmm_file) as f:
        gmm_results = json.load(f)

    with open(ae_file) as f:
        ae_results = json.load(f)

    comparison_report = format_comparison_table(gmm_results, ae_results)
    print(comparison_report)


if __name__ == "__main__":
    main()
