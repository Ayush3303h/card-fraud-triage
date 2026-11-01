# 💳 Card Fraud Triage via Gaussian Mixture Density Scoring

**Project 26 — Applied Machine Learning (AML)**

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Phase%201-GMM%20Completed-brightgreen)

## Project Overview

This project investigates credit card fraud detection using a Gaussian Mixture Model (GMM) as a density-based anomaly detector.

The model learns the probability distribution of legitimate transactions. Transactions with unusually low likelihood under the learned distribution receive higher anomaly scores and can be flagged for review.

The GMM is fitted exclusively on legitimate training transactions. Fraud labels are used during validation to select a decision threshold and during evaluation to measure detection performance.

**Current implementation:** Phase 1 — Gaussian Mixture Model.
**Planned extension:** Phase 2 — Autoencoder-based anomaly detection.

## Team Members

| # | Enrollment Number | Name | Email |
|---|---|---|---|
| 1 | 2401010120 | Ayush Sahu | — |
| 2 | 2401010364 | Rachit Singh | — |
| 3 | 2401010110 | Ashar Asif Shaikh | — |

## Dataset

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymised credit card transactions:

| Property | Value |
|---|---:|
| Original transactions | 284,807 |
| Features | 30 |
| Target column | Class |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Original fraud prevalence | Approximately 0.1727% |
| Missing values | None |

Features include `Time`, `V1`–`V28`, and `Amount`. The target `Class` is 0 for legitimate transactions and 1 for fraud.

Download `creditcard.csv` from Kaggle and place it inside `data/`. The dataset is excluded from Git because of its size and distribution terms.

## Methodology

### 1. Data cleaning

- Removed 1,081 duplicate rows.
- Retained 283,726 transactions.
- Sorted transactions chronologically by `Time`.

| Split | Transactions |
|---|---:|
| Training (70%) | 198,608 |
| Validation (15%) | 42,559 |
| Test (15%) | 42,559 |
| Total | 283,726 |

The chronological split is intended to approximate evaluation on later transactions.

### 2. Preprocessing

- Used all 30 feature columns.
- Fitted `StandardScaler` only on legitimate training transactions.
- Applied the fitted scaler to validation and test data.
- Excluded fraud transactions from GMM fitting.

The legitimate training subset contains 198,242 transactions.

### 3. Gaussian Mixture Model

A GMM represents the legitimate transaction density as a weighted mixture of Gaussian distributions:

\[
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k)
\]

The final configuration is:

| Parameter | Value |
|---|---|
| Components | 38 |
| Covariance type | `diag` |
| Maximum iterations | 200 |
| Initialisations | 3 |
| Covariance regularisation | `1e-5` |
| Random state | 42 |

The model converged in 111 iterations.

### 4. Model selection

Models with component counts from K=1 to K=40 were investigated using AIC, BIC, and legitimate-validation log-likelihood.

K=36 achieved the best single-run validation log-likelihood among the evaluated configurations. K=38 was retained as the final configuration based on the broader information-criterion and repeated-seed stability investigation. It should not be interpreted as the winner of every individual selection criterion.

### 5. Anomaly scoring

The anomaly score is the negative log-likelihood:

\[
s(x)=-\log p(x)
\]

A higher score indicates that a transaction is less likely under the learned legitimate-transaction distribution.

### 6. Threshold selection

The decision threshold was selected by maximising validation F1.

Final validation threshold:

**94.7575**

The threshold was then applied to the test scores for the reported fixed-threshold test metrics.

## Results — GMM K=38

### Test-set classification results

| Metric | Result |
|---|---:|
| Average Precision (AP) | **0.2719** |
| Precision | **46.94%** |
| Recall | **44.23%** |
| F1-score | **45.54%** |
| False-positive rate | **0.0612%** |

Confusion matrix:

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 42,481 | 26 |
| Actual fraud | 29 | 23 |

The model detected 23 of the 52 fraudulent transactions in the test split, while flagging 26 legitimate transactions.

### Operational evaluation

Thresholds derived from validation legitimate-transaction scores were used to examine different false-positive-rate budgets.

| Target FPR | Actual test FPR | Fraud recall | Fraud detected | False alarms |
|---:|---:|---:|---:|---:|
| 1.0% | 0.743% | 73.08% | 38 / 52 | 316 |
| 0.5% | 0.393% | 63.46% | 33 / 52 | 167 |
| 0.1% | 0.045% | 40.38% | 21 / 52 | 19 |

Actual test FPR can differ from the target because thresholds are estimated from validation scores and score ties may occur.

### Fixed review capacity

| Review capacity | Fraud found | Legitimate flagged | Precision |
|---:|---:|---:|---:|
| Top 50 | 24 | 26 | 48.00% |
| Top 100 | 28 | 72 | 28.00% |
| Top 200 | 33 | 167 | 16.50% |

These results illustrate how ranking transactions by anomaly score can support a limited fraud-investigation review capacity.

## Repository Structure

```text
card-fraud-triage/
├── configs/
│   └── gmm_config.json
├── data/
│   └── README.md
├── docs/
│   ├── methodology.md
│   └── experimental_results.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_gmm_fraud_detection.ipynb
├── reports/
│   ├── ieee_phase1/
│   │   ├── main.tex
│   │   └── main.pdf
│   └── requirements.txt
├── scripts/
│   ├── train_gmm.py
│   └── evaluate_gmm.py
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── model.py
│       └── evaluation.py
├── tests/
│   ├── test_model.py
│   └── test_preprocessing.py
├── artifacts/
├── README.md
├── requirements.txt
└── .gitignore
