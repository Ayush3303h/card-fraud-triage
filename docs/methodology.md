# Methodology — GMM Fraud Detection

## 1. Objective

The objective is to identify potentially fraudulent credit card transactions by learning the probability distribution of legitimate transactions using a Gaussian Mixture Model.

## 2. Dataset preparation

- Original dataset: 284,807 transactions.
- Exact duplicate rows removed: 1,081.
- Cleaned dataset: 283,726 transactions.
- Features: Time, V1–V28, Amount.
- Target: Class (0 = legitimate, 1 = fraud).

Transactions are sorted by Time and split chronologically:

| Split | Rows |
|---|---:|
| Training | 198,608 |
| Validation | 42,559 |
| Test | 42,559 |

## 3. Preprocessing

The 30 input features are standardised using `StandardScaler`.

The scaler is fitted exclusively on legitimate training transactions (198,242 rows). Fraudulent training transactions are excluded from model fitting.

The fitted scaler is subsequently applied to validation and test features.

## 4. Gaussian Mixture Model

The GMM estimates the density of legitimate transactions:

\[
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k)
\]

The final configuration is:

| Hyperparameter | Value |
|---|---|
| Components | 38 |
| Covariance type | diag |
| Maximum iterations | 200 |
| Initialisations | 3 |
| Regularisation | 0.00001 |
| Random state | 42 |

The final model converged in 111 iterations.

## 5. Component selection

Candidate models from K=1 to K=40 were investigated using AIC, BIC, and validation log-likelihood.

K=36 achieved the highest single-run validation log-likelihood (-30.725857). K=38 was selected after considering information criteria and repeated-seed stability.

Repeated-seed comparison:

| Configuration | Mean validation LL | Standard deviation |
|---|---:|---:|
| K=36 | -31.4959 | 0.4901 |
| K=38 | -31.4838 | 0.4676 |

K=38 performed better in 3 of 5 paired seeds. The difference is small, so this is a comparative selection rather than proof of a globally optimal component count.

## 6. Anomaly scoring

For each transaction, the anomaly score is calculated as:

\[
s(x)=-\log p(x)
\]

Higher scores indicate lower likelihood under the legitimate-transaction density.

## 7. Threshold selection

The threshold is selected by maximising F1 on the labelled validation set.

Selected threshold: **94.7575**.

The validation labels are used for threshold selection, but the GMM itself is fitted only on legitimate training observations.

## 8. Evaluation

Evaluation includes:

- Average Precision (AP).
- Precision, recall, and F1-score.
- Confusion matrix and false-positive rate.
- Recall at validation-derived false-positive-rate budgets.
- Precision at fixed Top-K review capacities.

The test split was also examined during an earlier K=11 baseline experiment. It is therefore not a completely untouched final holdout.
