IEEE Phase 2 Report Package: Autoencoder Anomaly Detection
==========================================================

Files:
- main.tex: IEEEtran conference paper source for Phase 2 (Autoencoder Anomaly Detection)
- figures/: High-resolution figures used in the paper:
    * comparative_pr.png: PR curves comparing Phase 2 Autoencoder vs Phase 1 GMM
    * class_distribution.png: Imbalanced transaction class distribution
    * precision_recall.png, validation_log_likelihood.png, aic_bic.png

Key Highlights of Phase 2 Paper:
- Model: Undercomplete Denoising Autoencoder (30 -> 20 -> 12 -> 6 -> 12 -> 20 -> 30)
- Parameter Count: 1,826 trainable parameters
- Validation Threshold: 1.4821 (F1-maximizing)
- Test AP: 0.7248 (+166% relative gain over Phase 1 GMM baseline AP 0.2719)
- Test Precision: 78.72%, Recall: 71.15% (37 of 52 frauds detected with only 10 false alarms)
- Operational Triage: Top-50 precision 72.0%, Recall at 0.1% FPR budget = 75.0%
- Systematic Ablation Studies: Latent dimension (d=3,6,12,16), Noise factor (sigma=0.00,0.02,0.05,0.10), Activations (ReLU, LeakyReLU, GELU)

How to Compile:
1. Locally with pdflatex / latexmk:
     pdflatex main.tex
     pdflatex main.tex
2. On Overleaf:
     Upload the entire `reports/ieee_phase2/` folder (or zip it) to Overleaf and compile with TeX Live (pdflatex).

Before Submission:
1. Verify author names, affiliation, student IDs, and emails in the \author block.
2. Check page constraints required by your course or conference.
