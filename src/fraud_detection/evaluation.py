
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


def format_evaluation_report(model_name, test_metrics, fpr_results=None, review_results=None, threshold=None):
    """
    Generate a clear, polished, human-readable terminal report of evaluation metrics.
    Structured to be easy to read and explain in student project presentations.
    """
    lines = []
    separator = "=" * 65
    subsep = "-" * 65

    lines.append(separator)
    lines.append(f"  {model_name.upper()} - EVALUATION SUMMARY REPORT")
    lines.append(separator)

    if threshold is not None:
        lines.append(f" Decision Threshold (tuned on validation F1): {threshold:.4f}")
        lines.append(subsep)

    lines.append(" 1. CLASSIFICATION METRICS (Holdout Test Set):")
    lines.append(f"    - Average Precision (PR-AUC) : {test_metrics['average_precision']:.4f}")
    lines.append(f"    - Precision                  : {test_metrics['precision'] * 100:.2f}%")
    lines.append(f"    - Recall (Detection Rate)    : {test_metrics['recall'] * 100:.2f}%")
    lines.append(f"    - F1-Score                   : {test_metrics['f1']:.4f}")
    lines.append(f"    - False Positive Rate (FPR)  : {test_metrics['false_positive_rate'] * 100:.4f}%")

    lines.append("\n 2. TEST CONFUSION MATRIX:")
    tn = test_metrics['true_negatives']
    fp = test_metrics['false_positives']
    fn = test_metrics['false_negatives']
    tp = test_metrics['true_positives']
    total_test = tn + fp + fn + tp
    total_fraud = fn + tp
    total_legit = tn + fp

    lines.append(f"    {'':<26} Predicted Legit    Predicted Fraud")
    lines.append(f"    Actual Legit ({total_legit:>6,}):     {tn:>12,} (TN)     {fp:>12,} (FP)")
    lines.append(f"    Actual Fraud ({total_fraud:>6,}):     {fn:>12,} (FN)     {tp:>12,} (TP)")
    lines.append(f"    --> Detected {tp} of {total_fraud} frauds ({tp/total_fraud*100:.1f}%) with only {fp} false alarms ({fp/total_legit*100:.3f}%).")

    if fpr_results:
        lines.append(f"\n{subsep}")
        lines.append(" 3. OPERATIONAL FPR BUDGETING (Validation-Calibrated Thresholds):")
        lines.append("    Target FPR | Threshold | Test FPR | Recall | Fraud Caught | False Alarms")
        lines.append("    " + "-" * 60)
        for row in fpr_results:
            lines.append(
                f"    {row['target_fpr']*100:>8.2f}% | "
                f"{row['threshold']:>9.2f} | "
                f"{row['test_fpr']*100:>7.3f}% | "
                f"{row['fraud_recall']*100:>5.1f}% | "
                f"{row['fraud_detected']:>2} / {total_fraud:<2}       | "
                f"{row['false_alarms']:>6,}"
            )

    if review_results:
        lines.append(f"\n{subsep}")
        lines.append(" 4. FIXED REVIEW CAPACITY (Daily Triage Workload):")
        lines.append("    Capacity | Frauds Found | Legit Flagged | Precision (Hit Rate)")
        lines.append("    " + "-" * 56)
        for row in review_results:
            lines.append(
                f"    Top {row['review_capacity']:<4} | "
                f"{row['fraud_found']:>10}   | "
                f"{row['legitimate_flagged']:>12}  | "
                f"{row['precision']*100:>8.2f}%"
            )

    lines.append(separator)
    return "\n".join(lines)


def calculate_financial_loss(fp, fn, cost_per_false_alarm=5.0, avg_fraud_loss=120.0):
    """
    Compute operational cost for a financial fraud triage system.
    - cost_per_false_alarm: operational cost for an investigator/SMS verification per FP ($5 default)
    - avg_fraud_loss: unrecovered direct theft cost per missed fraud FN ($120 default)
    """
    investigation_cost = fp * cost_per_false_alarm
    unmitigated_theft_cost = fn * avg_fraud_loss
    total_cost = investigation_cost + unmitigated_theft_cost
    return {
        "investigation_cost": float(investigation_cost),
        "unmitigated_theft_loss": float(unmitigated_theft_cost),
        "total_cost": float(total_cost),
    }


