
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def select_threshold(y_validation, validation_scores):
    """Select the anomaly threshold that maximizes validation F1."""

    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        validation_scores,
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to select a threshold from validation data.")

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_idx = np.argmax(f1_scores)

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1_scores[best_idx]),
        "average_precision": float(
            average_precision_score(y_validation, validation_scores)
        ),
    }


def evaluate_predictions(y_true, anomaly_scores, threshold):
    """Evaluate anomaly scores using a supplied threshold."""

    predictions = (anomaly_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "average_precision": float(
            average_precision_score(y_true, anomaly_scores)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate_at_fpr_targets(
    y_validation,
    validation_scores,
    y_test,
    test_scores,
    targets=(0.01, 0.005, 0.001),
):
    """Evaluate test recall at FPR thresholds calibrated on legitimate validation data."""

    legitimate_validation_scores = validation_scores[
        np.asarray(y_validation) == 0
    ]

    results = []

    for target in targets:
        threshold = np.quantile(
            legitimate_validation_scores,
            1 - target,
        )

        metrics = evaluate_predictions(
            y_test,
            test_scores,
            threshold,
        )

        results.append({
            "target_fpr": float(target),
            "threshold": float(threshold),
            "test_fpr": metrics["false_positive_rate"],
            "fraud_recall": metrics["recall"],
            "fraud_detected": metrics["true_positives"],
            "fraud_missed": metrics["false_negatives"],
            "false_alarms": metrics["false_positives"],
        })

    return results


def evaluate_review_capacity(y_true, anomaly_scores, capacities=(50, 100, 200)):
    """Measure fraud detection within fixed investigation review capacities."""

    y_true = np.asarray(y_true)
    ranked_indices = np.argsort(anomaly_scores)[::-1]

    results = []

    for capacity in capacities:
        if capacity > len(y_true):
            raise ValueError("Review capacity exceeds the number of transactions.")

        selected_indices = ranked_indices[:capacity]
        fraud_found = int(y_true[selected_indices].sum())

        results.append({
            "review_capacity": int(capacity),
            "fraud_found": fraud_found,
            "legitimate_flagged": int(capacity - fraud_found),
            "precision": float(fraud_found / capacity),
        })

    return results
