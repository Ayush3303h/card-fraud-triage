# 🎓 Individual Viva Voce Preparation Guide
**Course:** CSAI — Applied Machine Learning (AML) & Deep Learning (DL)  
**Project 26:** Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection  
**Institution:** Newton School of Technology (NST) / Rishihood University  
**Faculty Evaluators:** Prof. Shobhit Johri, Prof. Arun Chauhan, Prof. Ashwin Tewary  

---

## 1. Team Composition & Individual Contribution Breakdown (5 Members)

Per the CSAI Project Instructions, each student must have an explicitly documented role and be prepared to defend both their specific component and the end-to-end pipeline.

| # | Student Name | Assigned Role | Primary Technical Ownership | Key Viva Defense Topic |
|---|---|---|---|---|
| **1** | **Ayush Sahu** | *Data Engineering & Preprocessing Lead* | Dataset cleaning (1,081 duplicate drop), chronological 70/15/15 split, leakage-proof `StandardScaler` fitted on legitimate training only. | Why chronological splitting is mandatory in fintech; how leakage invalidates naive k-fold cross-validation. |
| **2** | **Rachit Singh** | *AML Pipeline & GMM Density Lead* | Phase 1 GMM modeling, component search ($K=1..40$) via AIC/BIC/Log-Likelihood, covariance stability analysis. | EM algorithm formulation, E-step and M-step derivations, curse of dimensionality in GMM. |
| **3** | **Ashar Asif Shaikh** | *Deep Learning Architecture Lead* | Phase 2 Undercomplete Denoising Autoencoder, PyTorch architecture, latent bottleneck ($30 \to 6$), training loop with early stopping. | Manifold learning theory, reconstruction loss formulation, why non-linear activations outperform convex Gaussians. |
| **4** | **Team Member 4** | *Hybrid Innovation & Ablations Lead* | Latent-space GMM density estimation, score-level fusion ($\alpha=0.6$), ablation studies on latent dims, noise, and activations. | Technical justification of the Hybrid model (why Latent GMM solves high-dimensional covariance collapse). |
| **5** | **Team Member 5** | *Evaluation, Triage & Explainability Lead* | PR-AUC evaluation, validation threshold tuning (max F1), operational FPR budgeting (0.1%), Top-$K$ review triage, XAI attribution. | Why Accuracy & ROC-AUC are misleading in extreme imbalance; translation of AI scores into daily human review capacity. |

---

## 2. Core Viva Questions & Model Answers

### Question 1: Why did you frame fraud detection as unsupervised anomaly detection instead of a supervised classifier (like XGBoost or Random Forest)?
**Answer:**  
In financial card fraud:
1. **Extreme Class Imbalance:** Fraud represents only ~0.17% of transactions (473 out of 283,726). Supervised classifiers tend to overfit to the historical fraud examples.
2. **Concept Drift & Adversarial Evasion:** Fraudsters continuously modify their evasion techniques. A supervised classifier learns *past* fraud patterns; an unsupervised normality model learns the stable manifold of *legitimate* transactions. Anything that deviates from normality receives high anomaly priority, enabling zero-day fraud capture.

---

### Question 2: Why was chronological splitting mandatory? Why not use stratified 5-fold cross-validation?
**Answer:**  
Transactions are an ordered temporal sequence. If we used random cross-validation, future transactions would leak into the training set, allowing the model to peek into future spending trends and seasonal volume shifts. By sorting by `Time` and taking the first 70% for training, next 15% for validation, and last 15% for test, we accurately simulate real-world deployment where tomorrow's transactions must be triaged using only models trained on yesterday's history.

---

### Question 3: Why was the StandardScaler fitted ONLY on legitimate training transactions?
**Answer:**  
If the scaler computes the mean and variance across the entire dataset (or includes fraudulent instances), the extreme outlier values typical of fraud transactions would distort the normalization scale ($\mu$ and $\sigma$). Furthermore, fitting on validation or test rows causes data leakage. Standardizing strictly on `Class == 0` training rows ensures our normality baseline reflects genuine cardholder behavior.

---

### Question 4: Explain the mathematical objective and EM algorithm of the Gaussian Mixture Model.
**Answer:**  
The GMM models probability density as a weighted sum of $K$ Gaussians:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \text{where } \sum_{k=1}^K \pi_k = 1$$
Because latent component assignments are unobserved, it is solved iteratively via Expectation-Maximization (EM):
- **E-Step (Expectation):** Compute the posterior responsibility $\gamma_{ik}$ that component $k$ generated transaction $x_i$:
  $$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
- **M-Step (Maximization):** Update parameters using the responsibilities:
  $$\mu_k^{\text{new}} = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}, \quad \Sigma_k^{\text{new}} = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_i \gamma_{ik}}, \quad \pi_k^{\text{new}} = \frac{1}{N}\sum_i \gamma_{ik}$$
The anomaly score is the negative log-likelihood: $s(x) = -\log p(x)$.

---

### Question 5: Why did the GMM component search settle on K=38, and why did it perform worse than the Autoencoder?
**Answer:**  
1. **Model Selection:** We swept $K=1..40$. AIC and BIC favored $K=38$, showing lowest penalization and stable validation log-likelihood over repeated seeds.
2. **The Theoretical Gap:** In 30-dimensional space, GMM suffers from the *curse of dimensionality*. Legitimate transactions do not lie in convex ellipsoids; they form curved, non-linear manifolds. A diagonal covariance GMM cannot model complex non-linear correlations without an explosion of components, leading to an Average Precision of only **0.2719**.

---

### Question 6: Explain the Deep Autoencoder architecture and why MSE acts as an anomaly score.
**Answer:**  
- **Architecture:** An undercomplete autoencoder with an encoder ($30 \to 20 \to 12 \to 6$) and symmetric decoder ($6 \to 12 \to 20 \to 30$).
- **Bottleneck Rationale:** The 6-dimensional bottleneck forces the network to discover the core low-dimensional manifold of normal transactions.
- **Anomaly Scoring:** Trained solely on legitimate data, the network minimizes MSE:
  $$\mathcal{L} = \frac{1}{D} \sum_{j=1}^D (x_j - \hat{x}_j)^2$$
  When a legitimate transaction passes through, the network reconstructs it with near-zero error. When a fraudulent transaction with anomalous feature combinations passes through, the bottleneck cannot represent it, causing a sharp spike in reconstruction error ($s_{\text{AE}}(x) = \text{MSE}$).

---

### Question 7: Explain your Hybrid Approach and why it qualifies for Additional Marks.
**Answer:**  
Per the project guidelines, simply combining models without technical justification does not earn marks. Our **Hybrid Latent-GMM & Autoencoder** has a principled rationale:
1. **High-D Covariance Collapse Solved:** Fitting a GMM in 30D fails because of covariance sparsity. The Autoencoder strips away noise and maps 30D non-linear space into a compact 6D latent space.
2. **Latent Density Estimation:** Fitting an 8-component GMM on the 6D latent manifold is statistically stable and well-conditioned.
3. **Dual-Signal Score Fusion:**
   $$s_{\text{hybrid}}(x) = \alpha \cdot \tilde{s}_{\text{AE}}(x) + (1-\alpha) \cdot \tilde{s}_{\text{LatentGMM}}(x), \quad (\alpha=0.6)$$
   It flags transactions that *either* violate structural manifold reconstruction (high MSE) *or* land in sparse/unlikely areas of the normal manifold (low latent probability).
4. **Demonstrated Value:** Pushes Average Precision to **0.7612** (+180% over Phase 1 GMM; +5% over standalone AE) and achieves **76.0% precision in the Top-50 review queue**.

---

### Question 8: Why is ROC-AUC misleading for this project, and why did you use Average Precision (PR-AUC)?
**Answer:**  
In our test set, there are 42,507 legitimate transactions and only 52 frauds (prevalence 0.12%).
ROC-AUC plots True Positive Rate against False Positive Rate:
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
Because $\text{TN} = 42,507$, even if the model produces 425 false alarms, the FPR is only $\frac{425}{42,507} \approx 1.0\%$. The ROC curve looks deceptively outstanding ($\text{AUC} > 0.98$) even while false alarms outnumber true frauds by 8 to 1!
**Precision-Recall curves** evaluate:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
PR curves explicitly penalize false alarms regardless of how large the true negative pool is.

---

### Question 9: What is the practical value of your "Top-K Review Capacity" and "FPR Budgeting" metrics?
**Answer:**  
In commercial banking, fraud operations teams have fixed headcounts (e.g., analysts can manually review only 50 or 100 flagged transactions per day):
- At **Top-50 review capacity**, our Hybrid model delivers **38 confirmed frauds out of 50 reviews (76.0% hit rate)**. The GMM baseline only found 24 frauds.
- At an **FPR budget of 0.1%** (less than 1 false alert per 1,000 customers), our model catches **76.9% of all frauds**, preventing customer harassment while mitigating financial risk.

---

### Question 10: How do you explain the root cause of an alert to a human investigator (Explainable AI)?
**Answer:**  
We decompose the autoencoder's total reconstruction error into per-feature squared errors:
$$e_j = (x_j - \hat{x}_j)^2$$
For any flagged transaction, our XAI module ranks the features contributing the highest percentage to the error (e.g., `V14` behavioral shift contributing 35%, `Amount` unusual magnitude contributing 26%). This gives human analysts immediate insight into why the account was flagged.
