import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.ablation import format_ablation_tables, get_ablation_benchmark_results


def main():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    ablation_file = artifacts_dir / "ablation_results.json"

    if ablation_file.exists():
        with open(ablation_file) as f:
            data = json.load(f)
    else:
        data = get_ablation_benchmark_results()

    table_report = format_ablation_tables(data)
    print(table_report)


if __name__ == "__main__":
    main()
