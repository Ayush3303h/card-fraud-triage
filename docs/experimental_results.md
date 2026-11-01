# Experimental Results — GMM Phase 1

## 1. Final model

| Parameter | Result |
|---|---:|
| Model | Gaussian Mixture Model |
| Components | 38 |
| Covariance type | diag |
| Converged | Yes |
| Iterations | 111 |
| Validation threshold | 94.7575 |

## 2. Validation results

The F1-maximising validation threshold was 94.7575.

| Metric | Validation |
|---|---:|
| Average Precision | 0.1859 |
| Precision | 31.65% |
| Recall | 45.45% |
| F1-score | 37.31% |

## 3. Test results

| Metric | Test result |
|---|---:|
| Average Precision | 0.2719 |
| Precision | 46.94% |
| Recall | 44.23% |
| F1-score | 45.54% |
| False-positive rate | 0.0612% |

### Confusion matrix

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 42,481 | 26 |
| Actual fraud | 29 | 23 |

## 4. Operational evaluation

### Recall at fixed FPR budgets

| Target FPR | Actual test FPR | Recall | Fraud detected | False alarms |
|---|---:|---:|---:|---:|
| 1.0% | 0.743% | 73.08% | 38 | 316 |
| 0.5% | 0.393% | 63.46% | 33 | 167 |
| 0.1% | 0.045% | 40.38% | 21 | 19 |

Thresholds are derived using legitimate validation scores. Actual test FPR can differ from the target.

### Top-K review capacity

| Review capacity | Fraud found | Legitimate flagged | Precision |
|---|---:|---:|---:|
| 50 | 24 | 26 | 48.00% |
| 100 | 28 | 72 | 28.00% |
| 200 | 33 | 167 | 16.50% |

## 5. Baseline comparison

Earlier K=11 baseline results:

| Metric | K=11 |
|---|---:|
| Average Precision | 0.1867 |
| Precision | 27.38% |
| Recall | 44.23% |
| F1-score | 33.82% |
| False-positive rate | 0.1435% |
| Fraud detected | 23 / 52 |
| False alarms | 61 |

The K=11 and K=38 experiments used the same test split. The comparison is useful for this project but does not constitute independent final holdout validation.

## 6. Interpretation

The K=38 model identifies 23 of 52 fraudulent transactions at the selected F1 threshold, with 26 false alarms among 42,507 legitimate transactions.

The Top-50 review experiment finds 24 fraudulent transactions among the 50 highest anomaly scores.

The operational results show the trade-off between fraud recall and false alarms.

## 7. Current phase

Phase 1: GMM density-based anomaly detection — implemented.

Phase 2: Autoencoder anomaly detection — planned.

A direct comparison between the two approaches remains future work.
