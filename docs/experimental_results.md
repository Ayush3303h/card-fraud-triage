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

## 7. Phase 2 — Autoencoder Deep Learning Results

### Architecture & Training
- Undercomplete Denoising Autoencoder: 30 -> 20 -> 12 -> 6 (latent bottleneck) -> 12 -> 20 -> 30.
- Regularization: LeakyReLU(0.1), BatchNorm1d, Dropout(0.05), Gaussian input noise (0.02).
- Anomaly score: Mean Squared Error (MSE) reconstruction loss.
- Validation threshold: 1.4821.

### Test Performance
| Metric | Autoencoder (Phase 2) |
|---|---:|
| Average Precision (PR-AUC) | **0.7248** |
| Precision | **78.72%** |
| Recall | **71.15%** (37 / 52 detected) |
| F1-score | **74.75%** |
| False-positive rate | **0.0235%** (10 false alarms) |

### Operational FPR Budgets (Autoencoder)
| Target FPR | Actual test FPR | Recall | Fraud detected | False alarms |
|---|---:|---:|---:|---:|
| 1.0% | 0.813% | 86.54% | 45 / 52 | 345 |
| 0.5% | 0.412% | 80.77% | 42 / 52 | 175 |
| 0.1% | 0.075% | 75.00% | 39 / 52 | 32 |

### Fixed Review Capacity (Autoencoder)
| Review capacity | Fraud found | Legitimate flagged | Precision |
|---|---:|---:|---:|
| Top 50 | 36 / 52 | 14 | 72.00% |
| Top 100 | 41 / 52 | 59 | 41.00% |
| Top 200 | 46 / 52 | 154 | 23.00% |

## 8. Comparative Analysis: Phase 1 (GMM) vs. Phase 2 (Autoencoder)

| Evaluation Dimension | GMM ($K=38$) | Autoencoder ($D=6$) | Improvement |
|---|---:|---:|---:|
| **Average Precision (AP)** | 0.2719 | **0.7248** | **+0.4529** (+166%) |
| **Precision** | 46.94% | **78.72%** | **+31.78%** |
| **Recall (Detection Rate)** | 44.23% (23/52) | **71.15%** (37/52) | **+26.92%** (+14 frauds) |
| **F1-Score** | 45.54% | **74.75%** | **+29.21%** |
| **False Alarms** | 26 | **10** | **-61.5%** reduction |
| **Top-50 Review Precision** | 48.00% (24/52) | **72.00%** (36/52) | **+24.00%** |
| **Recall at 0.1% FPR Budget** | 40.38% (21/52) | **75.00%** (39/52) | **+34.62%** |

### Why the Autoencoder Outperforms the GMM
1. **Gaussian Cluster Assumption Violation:** GMM assumes data is distributed according to elliptical Gaussian densities. In 30-dimensional transaction space, legitimate behavior forms complex non-linear curved manifolds that diagonal Gaussians cannot efficiently capture without exploding component counts.
2. **Non-Linear Representation:** The deep autoencoder's non-linear bottleneck compresses multi-variable dependencies, learning legitimate transaction invariants. When an unusual transaction (fraud) is passed through, the decoder fails to reconstruct it, producing high anomaly signals with far fewer false alarms.

