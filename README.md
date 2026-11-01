<h1 align="center">💳 Card Fraud Triage via Gaussian Mixture Density Scoring</h1>

<p align="center">
  <strong>Project 26 — Applied Machine Learning (AML)</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-Academic-blue" alt="License">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status">
</p>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Abstract](#-abstract)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Team Members](#-team-members)
- [Dataset Description](#-dataset-description)
- [Technologies and Libraries Used](#-technologies-and-libraries-used)
- [Project Architecture / Workflow](#-project-architecture--workflow)
- [Methodology](#-methodology)
  - [Data Preprocessing](#1-data-preprocessing)
  - [Feature Engineering and Standardization](#2-feature-engineering-and-standardization)
  - [Gaussian Mixture Model](#3-gaussian-mixture-model-gmm)
  - [AIC and BIC Model Selection](#4-aic-and-bic-model-selection)
  - [Anomaly Scoring](#5-anomaly-scoring-using-negative-log-likelihood)
  - [Threshold Selection](#6-threshold-selection)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
  - [GMM Component Selection](#gmm-component-selection-k--1-to-40)
  - [Final Test Results](#final-test-results--gmm-k38)
  - [Recall at Fixed False-Positive Rate](#recall-at-fixed-false-positive-rate-budgets)
  - [Precision at Fixed Review Capacity](#precision-at-fixed-review-capacity)
- [Visualizations](#-visualizations)
- [Project Directory Structure](#-project-directory-structure)
- [Installation and Setup](#-installation-and-setup)
- [How to Run the Project](#-how-to-run-the-project)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [References](#-references)

---

## 🔍 Project Overview

This project detects suspicious credit card transactions using **unsupervised anomaly detection** with a **Gaussian Mixture Model (GMM)**. The core idea is to learn the probability density of *legitimate* transactions and flag observations that fall in low-density regions as potential fraud.

The GMM is trained exclusively on legitimate transactions, making it a **semi-supervised** anomaly detection approach — the model learns what "normal" looks like without ever seeing labelled fraud during training. Anomalies are then identified at inference time using **negative log-likelihood scoring**, where transactions that are unlikely under the learned distribution receive high anomaly scores.

---

## 📝 Abstract

Credit card fraud is a critical problem in the financial domain, characterised by an extreme class imbalance — fraudulent transactions constitute only a tiny fraction (~0.17%) of all activity. Traditional supervised classifiers often struggle with such imbalance, overfitting to the majority class or requiring costly labelled fraud examples.

This project implements a **density-based anomaly detection pipeline** using Gaussian Mixture Models. The GMM estimates the multivariate probability distribution of legitimate transactions. At inference time, each transaction is scored by its negative log-likelihood under this distribution — transactions with high scores (low likelihood) are flagged as potentially fraudulent.

The pipeline includes rigorous **model selection** (searching K = 1 to 40 components using AIC, BIC, and validation log-likelihood), **F1-maximising threshold selection** on a held-out validation set, and **operationally relevant evaluation** including recall at fixed false-positive rate budgets and precision at fixed review capacity.

On the held-out test set, the final GMM (K = 38) achieves a **Precision of 46.94%**, **Recall of 44.23%**, and a **PR AUC of 0.2719** — detecting nearly half of all fraud with a false-positive rate of only 0.06%.

---

## ❗ Problem Statement

Financial institutions process millions of transactions daily. Fraudulent transactions represent an extremely small fraction of total volume (typically < 0.2%), yet cause significant financial damage. The challenge is to build a system that can:

1. **Detect fraudulent transactions** without relying on labelled fraud examples for training.
2. **Minimise false positives** — incorrectly flagging legitimate transactions erodes customer trust and wastes investigator time.
3. **Maximise recall on fraud** — missed fraud directly translates to financial loss.
4. **Operate under extreme class imbalance** where traditional accuracy metrics are misleading (a model predicting "legitimate" for every transaction achieves 99.83% accuracy).

---

## 🎯 Objectives

1. Perform exploratory data analysis to understand the dataset's structure, class distribution, and feature characteristics.
2. Preprocess the data with chronological splitting to simulate realistic deployment conditions.
3. Train a Gaussian Mixture Model on legitimate transactions to learn the density of normal behaviour.
4. Select the optimal number of GMM components using AIC, BIC, and validation log-likelihood.
5. Detect anomalies using negative log-likelihood scoring.
6. Evaluate detection performance with metrics appropriate for imbalanced classification: Precision-Recall AUC, recall at fixed FPR, and precision at fixed review capacity.

---

## 👥 Team Members

| # | Enrollment Number | Name | Email |
|:-:|:-:|:-:|:-:|
| 1 | 2401010120 | Ayush Sahu | — |
| 2 | 2401010364 | Rachit Singh | — |
| 3 | 2401010110 | Ashar Asif Shaikh | — |

---

## 📊 Dataset Description

| Property | Value |
|:--|:--|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Transactions** | 284,807 |
| **Features** | 31 (Time, V1–V28, Amount, Class) |
| **Legitimate (Class 0)** | 284,315 (99.83%) |
| **Fraudulent (Class 1)** | 492 (0.17%) |
| **Missing Values** | None |
| **Memory** | ~67.4 MB |

**Feature Details:**

- **Time** — Seconds elapsed from the first transaction in the dataset.
- **V1 – V28** — Principal components obtained via PCA transformation (original features are confidential). These features are already dimensionality-reduced and anonymised.
- **Amount** — Transaction amount in the original currency.
- **Class** — Binary label: `0` = legitimate, `1` = fraud.

> **Note:** Because V1–V28 are PCA-transformed, they are already decorrelated. The `Time` and `Amount` features are the only original-scale features and require standardisation.

---

## 🛠 Technologies and Libraries Used

| Library | Purpose |
|:--|:--|
| **Python 3.12+** | Programming language |
| **pandas** | Data loading, manipulation, and analysis |
| **NumPy** | Numerical computing |
| **scikit-learn** | GMM, StandardScaler, evaluation metrics |
| **Matplotlib** | Plotting and visualisation |
| **seaborn** | Statistical visualisation (EDA) |
| **Jupyter Notebook** | Interactive development and documentation |

---

## 🏗 Project Architecture / Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                       Raw Dataset                               │
│              creditcard.csv (284,807 × 31)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Exploratory Data Analysis                        │
│  • Class distribution  • Correlations  • Duplicate analysis     │
│  • Amount statistics   • Temporal patterns                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Preprocessing                          │
│  • Remove 1,081 duplicate rows                                  │
│  • Chronological sort by Time                                   │
│  • 70 / 15 / 15 chronological split                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               Feature Engineering & Scaling                     │
│  • StandardScaler fit on legitimate training data ONLY          │
│  • Transform train / validation / test sets                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                GMM Component Selection                          │
│  • Train GMM for K = 1 to 40 on legitimate training data        │
│  • Evaluate AIC, BIC, Validation Log-Likelihood                 │
│  • Select K = 38 as final model                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             Anomaly Scoring & Threshold Selection               │
│  • Score: -log P(x) (negative log-likelihood)                   │
│  • Threshold: F1-maximising on validation PR curve              │
│  • Optimal threshold = 94.7575                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Evaluation                                 │
│  • Precision-Recall curve & PR AUC                              │
│  • Confusion matrix & classification report                     │
│  • Recall @ fixed FPR budgets (1%, 0.5%, 0.1%)                  │
│  • Precision @ fixed review capacity (50, 100, 200)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📐 Methodology

### 1. Data Preprocessing

**Duplicate Removal:**
The raw dataset contains 1,081 duplicate rows (beyond the first occurrence). After analysis confirming no feature-identical rows with conflicting labels, all exact duplicates were removed:

| Stage | Rows |
|:--|--:|
| Original | 284,807 |
| After deduplication | 283,726 |
| Removed | 1,081 |

**Chronological Train / Validation / Test Split:**
To simulate a realistic deployment scenario where the model is trained on past data and evaluated on future data, the dataset is sorted by `Time` and split chronologically (not randomly):

| Split | Rows | Percentage |
|:--|--:|--:|
| Training | 198,608 | 70% |
| Validation | 42,559 | 15% |
| Testing | 42,559 | 15% |

> **Why chronological splitting?** Random splitting leaks temporal information — the model could see "future" transactions during training. Chronological splitting reflects how a fraud detection model would be deployed in production.

### 2. Feature Engineering and Standardization

All 30 feature columns (Time, V1–V28, Amount) are standardised using `StandardScaler`. Critically, the scaler is **fit only on legitimate training transactions** (198,242 rows, with 366 fraud rows excluded from training). This prevents fraud patterns from influencing the learned distribution of normal behaviour.

After scaling:
- Feature means ≈ 0
- Feature standard deviations = 1.0

### 3. Gaussian Mixture Model (GMM)

A **Gaussian Mixture Model** models the probability density of legitimate transactions as a weighted sum of K multivariate Gaussian components:

$$P(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where:
- $\pi_k$ are mixing weights ($\sum_k \pi_k = 1$)
- $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$ are the mean and covariance of the *k*-th component
- $\mathcal{N}$ denotes the multivariate Gaussian density

**GMM Hyperparameters:**

| Parameter | Value |
|:--|:--|
| Covariance type | `diag` (diagonal) |
| Max iterations | 200 |
| Number of initialisations | 3 |
| Regularisation (covariance) | 1 × 10⁻⁵ |
| Random state | 42 |

> **Why diagonal covariance?** With 30 features, a full covariance matrix has 30 × 31 / 2 = 465 free parameters per component. Diagonal covariance reduces this to 30 parameters per component, making the model computationally tractable for large K while the PCA-transformed features (V1–V28) are already decorrelated.

### 4. AIC and BIC Model Selection

The optimal number of mixture components K was selected by training GMMs for K = 1 to 40 and evaluating three criteria:

- **Akaike Information Criterion (AIC):** $\text{AIC} = 2p - 2 \ln \hat{L}$
- **Bayesian Information Criterion (BIC):** $\text{BIC} = p \ln n - 2 \ln \hat{L}$
- **Validation Log-Likelihood:** Average per-sample log-likelihood on legitimate validation transactions

where $p$ is the number of model parameters, $n$ is the number of training samples, and $\hat{L}$ is the maximised likelihood.

All 40 models converged successfully. The best validation log-likelihood was achieved at **K = 36** (Val LL = −30.7259), while **K = 38** was selected as the final model considering the AIC/BIC trade-off:

| K | AIC | BIC | Validation LL |
|:-:|:-:|:-:|:-:|
| 36 | 9.84 × 10⁶ | 9.87 × 10⁶ | **−30.7259** |
| 38 | 9.86 × 10⁶ | 9.88 × 10⁶ | −31.0523 |

### 5. Anomaly Scoring Using Negative Log-Likelihood

Each transaction is scored by its **negative log-likelihood** under the trained GMM:

$$\text{Anomaly Score}(\mathbf{x}) = -\log P(\mathbf{x})$$

- **Low score** → transaction is in a high-density region → likely **legitimate**
- **High score** → transaction is in a low-density region → likely **anomalous / fraudulent**

**Score distribution on the test set (K = 38):**

| Statistic | Value |
|:--|:--|
| Minimum | 16.47 |
| Maximum | 2,043.50 |
| Mean | 35.55 |

### 6. Threshold Selection

The anomaly threshold is selected to **maximise the F1-score** on the validation set:

1. Compute anomaly scores for all validation transactions.
2. Compute the Precision-Recall curve across all possible thresholds.
3. For each threshold, calculate $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
4. Select the threshold that maximises $F_1$.

**Optimal threshold (K = 38):** **94.7575**

---

## 📏 Evaluation Metrics

Given the extreme class imbalance (0.17% fraud), **accuracy is not a reliable metric** — a naive model predicting "legitimate" for every transaction achieves 99.83% accuracy while detecting zero fraud. Instead, this project uses:

| Metric | Description | Why It Matters |
|:--|:--|:--|
| **Precision** | $\frac{TP}{TP + FP}$ | Fraction of flagged transactions that are actually fraud. High precision reduces investigator wasted effort. |
| **Recall** | $\frac{TP}{TP + FN}$ | Fraction of actual fraud that is detected. High recall reduces financial loss. |
| **F1-Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonic mean balancing precision and recall. |
| **PR AUC** | Area under the Precision-Recall curve | Summarises the precision-recall trade-off across all thresholds. More informative than ROC AUC under severe class imbalance. |
| **FPR** | $\frac{FP}{FP + TN}$ | False positive rate — fraction of legitimate transactions incorrectly flagged. |
| **Recall @ Fixed FPR** | Recall when FPR is capped | Operationally critical — how much fraud is caught within an acceptable false alarm budget. |
| **Precision @ Top-N** | Precision in the top-N scored transactions | Simulates a fixed-capacity review team inspecting the N most suspicious transactions. |

---

## 📈 Results

### GMM Component Selection (K = 1 to 40)

All 40 GMM configurations were trained and evaluated. Key checkpoints:

| K | Validation Log-Likelihood | Notes |
|:-:|:-:|:--|
| 1 | −42.4064 | Baseline (single Gaussian) |
| 11 | −34.2410 | Best in K = 1–15 range |
| 21 | −32.6422 | Best in K = 16–25 range |
| 36 | **−30.7259** | Best overall validation LL |
| 38 | −31.0523 | **Selected as final model** |

### Baseline vs Final Model Comparison

| Metric | GMM K=11 (Baseline) | GMM K=38 (Final) |
|:--|:-:|:-:|
| Threshold | 87.97 | 94.76 |
| Test Precision | 14.85% | **46.94%** |
| Test Recall | 28.85% | **44.23%** |
| Test F1-Score | 19.61% | **45.54%** |
| Test PR AUC | 0.0735 | **0.2719** |
| Test FPR | 0.20% | **0.06%** |
| Fraud Detected | 15 / 52 | **23 / 52** |
| False Alarms | 85 | **26** |

### Final Test Results — GMM K=38

**Confusion Matrix:**

|  | Predicted Legitimate | Predicted Fraud |
|:--|:-:|:-:|
| **Actual Legitimate** | 42,481 (TN) | 26 (FP) |
| **Actual Fraud** | 29 (FN) | 23 (TP) |

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|:--|:-:|:-:|:-:|:-:|
| Legitimate (0) | 1.00 | 1.00 | 1.00 | 42,507 |
| Fraud (1) | 0.47 | 0.44 | 0.46 | 52 |

**Summary:**

| Metric | Value |
|:--|:-:|
| **Test Average Precision (PR AUC)** | **0.2719** |
| **Test Precision** | **46.94%** |
| **Test Recall** | **44.23%** |
| **Test F1-Score** | **45.54%** |
| **Test FPR** | **0.06%** |
| Total Test Transactions | 42,559 |
| Fraud Detected (TP) | 23 |
| Fraud Missed (FN) | 29 |
| False Alarms (FP) | 26 |
| Correctly Classified Legitimate (TN) | 42,481 |

### Recall at Fixed False-Positive Rate Budgets

This analysis answers the operational question: *"If we can only tolerate X% false alarms, how much fraud will we catch?"*

| Target FPR | Threshold | Actual Test FPR | Fraud Recall | Fraud Detected | Fraud Missed | False Alarms |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1.0% | 72.26 | 0.74% | **73.08%** | 38 / 52 | 14 | 316 |
| 0.5% | 77.35 | 0.39% | **63.46%** | 33 / 52 | 19 | 167 |
| 0.1% | 98.53 | 0.05% | **40.38%** | 21 / 52 | 31 | 19 |

> At a 1% FPR budget, the model catches **73%** of all fraud while incorrectly flagging only 316 out of 42,507 legitimate transactions.

### Precision at Fixed Review Capacity

This analysis answers the operational question: *"If an investigator can only review N flagged transactions, what fraction will be actual fraud?"*

| Review Capacity (Top-N) | Fraud Found | Legitimate Flagged | Precision |
|:-:|:-:|:-:|:-:|
| 50 | 24 | 26 | **48.00%** |
| 100 | 28 | 72 | **28.00%** |
| 200 | 33 | 167 | **16.50%** |

> When reviewing only the top 50 most suspicious transactions, **nearly half (48%)** are actual fraud — a 278× enrichment over the base fraud rate of 0.17%.

---

## 📊 Visualizations

The project generates the following visualizations across both notebooks:

### Exploratory Data Analysis (`01_eda.ipynb`)

1. **Class Distribution Bar Plot** — Visualises the extreme imbalance between legitimate and fraudulent transactions.
2. **Transaction Amount Box Plot** — Compares Amount distributions across classes.
3. **Feature Correlation Heatmap** — Full 31 × 31 correlation matrix of all features.
4. **Transaction Distribution Over Time** — Density histogram showing temporal patterns by class.
5. **Top Correlated Features** — Top 10 positively and negatively correlated features with the fraud label.

### GMM Fraud Detection (`02_gmm_fraud_detection.ipynb`)

6. **GMM Component Selection (K = 1–40)** — AIC, BIC, and Validation Log-Likelihood plotted across all K values.
7. **Precision-Recall Curve (Final K=38)** — PR curve with AP = 0.2719 and random baseline.

---

## 📁 Project Directory Structure

```
card-fraud-triage/
│
├── README.md                              # Project documentation (this file)
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── data/
│   └── creditcard.csv                     # Dataset (not tracked in Git)
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   └── 02_gmm_fraud_detection.ipynb       # GMM training, evaluation, and results
│
└── reports/
    ├── README.md
    └── requirements.txt
```

> **Note:** The dataset file `data/creditcard.csv` is excluded from version control via `.gitignore`. See [Installation and Setup](#-installation-and-setup) for download instructions.

---

## ⚙ Installation and Setup

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ayush3303h/card-fraud-triage.git
   cd card-fraud-triage
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   # .venv\Scripts\activate          # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**

   Download [`creditcard.csv`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it in the `data/` directory:

   ```
   data/
   └── creditcard.csv
   ```

5. **Register the Jupyter kernel (optional):**

   ```bash
   python -m ipykernel install --user --name card-fraud-triage --display-name "Python (card-fraud-triage)"
   ```

---

## 🚀 How to Run the Project

1. **Start Jupyter:**

   ```bash
   jupyter notebook
   ```

2. **Run the notebooks in order:**

   | Order | Notebook | Description |
   |:-:|:--|:--|
   | 1 | `notebooks/01_eda.ipynb` | Exploratory Data Analysis — data understanding, cleaning, and splitting |
   | 2 | `notebooks/02_gmm_fraud_detection.ipynb` | GMM training, model selection, anomaly scoring, and evaluation |

3. **Select the correct kernel:** Choose `Python (card-fraud-triage)` or your virtual environment kernel when prompted.

> **Important:** Run the notebooks sequentially. The GMM notebook repeats the preprocessing steps internally, but running the EDA notebook first provides important context for understanding the data.

---

## ⚠ Limitations

1. **Single model approach:** Only a Gaussian Mixture Model is implemented. Comparing with other anomaly detection methods (Autoencoders, Isolation Forest, One-Class SVM) would provide a more comprehensive evaluation.

2. **Diagonal covariance assumption:** The GMM uses diagonal covariance matrices, which assumes feature independence within each component. While PCA features are globally decorrelated, they may have local dependencies within clusters.

3. **Static threshold:** A single global threshold is used for all transactions. In practice, adaptive thresholds based on transaction context (time of day, merchant category) may perform better.

4. **PCA-transformed features:** The V1–V28 features are pre-transformed, limiting feature interpretability. It is not possible to explain *why* a specific transaction was flagged in terms of original transaction attributes.

5. **Dataset age:** The Kaggle Credit Card Fraud dataset, while widely used as a benchmark, represents a fixed snapshot. Real-world fraud patterns evolve over time (concept drift).

6. **Moderate recall:** At the F1-optimal threshold, the model detects 44% of fraud. While false positives are very low (0.06% FPR), more than half of fraudulent transactions are missed.

---

## 🔮 Future Improvements

1. **Autoencoder Anomaly Detection:** Implement an autoencoder neural network that learns to reconstruct legitimate transactions. Anomalies would be detected via reconstruction error:

   $$\text{MSE}(\mathbf{x}) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$

   where $\hat{x}_i$ is the autoencoder's reconstruction. Transactions with high reconstruction error would be flagged as anomalous. This would enable a direct comparison between density-based (GMM) and reconstruction-based (Autoencoder) anomaly detection.

2. **Ensemble Methods:** Combine GMM and Autoencoder scores for a more robust anomaly detection system.

3. **Isolation Forest Baseline:** Add a tree-based anomaly detector for comparison.

4. **Full Covariance GMM:** Experiment with `covariance_type="full"` for smaller K values where the parameter count remains manageable.

5. **Threshold Optimisation:** Explore cost-sensitive threshold selection where the relative costs of false positives and false negatives are explicitly modelled.

6. **Real-Time Scoring Pipeline:** Package the trained model into a REST API for real-time transaction scoring.

---

## 📚 References

1. **Dataset:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 2015.

2. **Kaggle Dataset Page:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

3. **Gaussian Mixture Models:** Christopher M. Bishop. *Pattern Recognition and Machine Learning*, Chapter 9. Springer, 2006.

4. **Model Selection (AIC/BIC):** Schwarz, G. *Estimating the Dimension of a Model.* The Annals of Statistics, 6(2), 1978.

5. **scikit-learn GMM Documentation:** [sklearn.mixture.GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

6. **Precision-Recall for Imbalanced Data:** Jesse Davis and Mark Goadrich. *The Relationship Between Precision-Recall and ROC Curves.* ICML, 2006.

---

<p align="center">
  <em>Developed as part of the Applied Machine Learning coursework — Project 26</em>
</p>
