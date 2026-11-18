# 📽️ Project Presentation Guide & Slide Deck Outline
**Project:** Card Fraud Triage via Gaussian Mixture Density Scoring & Autoencoder Anomaly Detection  
**Evaluation:** Phase 1 & Phase 2 Project Presentation (CSAI)  
**Team Members:** 5 Members (Assigned sections below)  

---

## Slide 1: Title & Executive Overview
- **Title:** Card Fraud Triage: From Classical Density Scoring to Deep Manifold Learning & Hybrid Innovation
- **Subtitle:** Project 26 — Applied Machine Learning (AML) & Deep Learning (DL)
- **Institution:** Newton School of Technology (NST) / Rishihood University
- **Team Members & Enrollment IDs:**
  1. Ayush Sahu (Data Engineering & Preprocessing)
  2. Rachit Singh (Phase 1: GMM Lead)
  3. Ashar Asif Shaikh (Phase 2: Deep Autoencoder Lead)
  4. Team Member 4 (Phase 2+: Hybrid Model & Ablations)
  5. Team Member 5 (Evaluation, Triage & Explainable AI)
- **Core Message:** Financial fraud cannot be solved by naive supervised classification due to 0.17% class imbalance and adversarial evasion. We frame triage under the unsupervised normality paradigm.

---

## Slide 2: Problem Statement & Class Imbalance Challenges
- **Key Visual:** Class distribution pie chart / bar chart showing 99.83% legitimate vs. 0.17% fraud.
- **Talking Points:**
  - 284,807 transactions; only 492 fraud cases.
  - Why standard accuracy is a fatal metric: a naive model predicting "always legitimate" achieves 99.83% accuracy while catching 0 frauds!
  - Concept drift: fraudsters adapt their evasion methods.

---

## Slide 3: Rigorous Data Engineering (Leakage Prevention)
- **Key Visual:** Chronological timeline diagram (70% Train $\to$ 15% Val $\to$ 15% Test).
- **Talking Points:**
  - **Deduplication:** Dropped 1,081 exact duplicate rows (leaving 283,726 clean rows).
  - **Temporal Ordering:** Sorted strictly by `Time`. Evaluating future transactions avoids lookahead bias.
  - **Zero Leakage Scaler:** `StandardScaler` fitted strictly on legitimate training records (`Class == 0`). Fraud records and future splits never leak into scaling.

---

## Slide 4: Phase 1 — Gaussian Mixture Model (AML Baseline)
- **Key Visual:** AIC/BIC curves from $K=1$ to $40$, showing optimal penalization at $K=38$.
- **Talking Points:**
  - GMM assumes legitimate transactions form a mixture of $K$ multivariate Gaussian components.
  - Model fitting via Expectation-Maximization (EM).
  - Anomaly Score: Negative log-likelihood $s(x) = -\log p(x)$.
  - **Limitation Discovered:** GMM achieves $\text{AP} = 0.2719$ because 30D transaction spaces have complex curved manifolds that convex diagonal Gaussians cannot easily capture.

---

## Slide 5: Phase 2 — Undercomplete Denoising Autoencoder (Deep Learning)
- **Key Visual:** Neural network architecture diagram ($30 \to 20 \to 12 \to 6 \to 12 \to 20 \to 30$).
- **Talking Points:**
  - Bottleneck latent dimension ($d=6$) forces the network to learn the non-linear invariant structure of normal transactions.
  - Trained exclusively on legitimate spending with AdamW, LeakyReLU, and Gaussian noise jitter.
  - Anomaly Score: Sample-wise Mean Squared Error (MSE) reconstruction loss.

---

## Slide 6: Phase 2+ — Hybrid Innovation (Additional Marks)
- **Key Visual:** Diagram of Hybrid Autoencoder + Latent GMM pipeline with dual score fusion.
- **Talking Points:**
  - *Technical Justification:* Autoencoder eliminates high-dimensional noise and projects to 6D. GMM models probability density directly on this compact 6D latent manifold.
  - Dual Anomaly Score:
    $$s_{\text{hybrid}}(x) = 0.6 \cdot \tilde{s}_{\text{AE}}(x) + 0.4 \cdot \tilde{s}_{\text{LatentGMM}}(x)$$
  - Catches both structural reconstruction failures AND probabilistic low-density outliers.

---

## Slide 7: Systematic Ablation Studies
- **Key Visual:** Latent dimension ($d=3, 6, 12, 16$) and noise factor ($\sigma=0.0..0.10$) performance curves.
- **Talking Points:**
  - Demonstrates why $d=6$ is optimal (avoids undercompression and identity shortcut).
  - Demonstrates why LeakyReLU prevents dying neurons on zero-mean standardized inputs.

---

## Slide 8: Experimental Results & Precision-Recall Benchmark
- **Key Visual:** Comparative PR Curve (`artifacts/comparative_pr_curve.png`).
- **Talking Points:**
  - **Average Precision (PR-AUC):** GMM ($0.2719$) $\to$ Autoencoder ($0.7248$) $\to$ **Hybrid ($0.7612$)** (+180% overall gain).
  - **F1-Score:** From $0.4554$ to **$0.7677$**.
  - **False Alarms:** Reduced from 26 to only **9** out of 42,507 legitimate transactions.

---

## Slide 9: Operational Banking Triage (FPR & Top-K Review)
- **Key Visual:** Top-50 Review queue hit rate comparison table.
- **Talking Points:**
  - **Daily Review Quota (Top 50):** GMM caught 24 frauds; **Hybrid caught 38 frauds (76.0% precision)**.
  - **FPR Budget $\le 0.10\%$:** Hybrid catches 76.9% of all test frauds with fewer than 1 false alarm per 1,000 legitimate users.

---

## Slide 10: Explainable AI & Financial Cost Impact
- **Key Visual:** Bar chart of feature-level error contributions + Financial savings summary card.
- **Talking Points:**
  - Feature-level anomaly attribution answers *why* a customer account was flagged (e.g., `V14` behavioral shift + `Amount` anomaly).
  - Economic modeling: Saves **$4,560** on the test partition compared to unmonitored fraud loss.

---

## Slide 11: Live Web Demo & Codebase Architecture
- **Action:** 1-minute live demo of the Streamlit dashboard (`app.py`).
- **Talking Points:**
  - Modular package in `src/fraud_detection/`.
  - 16 automated unit tests passing (`make test`).
  - Interactive triage sliders and root-cause analysis.

---

## Slide 12: Conclusion & Future Scope
- **Key Takeaways:**
  1. Unsupervised normality modeling overcomes extreme class imbalance and adversarial evasion.
  2. Deep Autoencoders dramatically outperform classical GMMs in high-dimensional financial spaces.
  3. The Hybrid Latent-GMM approach successfully fuses structural reconstruction with probabilistic density.
- **Future Scope:** Temporal sequence modeling (LSTM / Transformer Autoencoders) and graph-neural-network cardholder networks.
