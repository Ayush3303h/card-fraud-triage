.PHONY: test evaluate compare hybrid ablation app train-gmm train-ae help

PYTHON ?= /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12

help:
	@echo "Card Fraud Triage - Project Commands (CSAI Phase 1 & Phase 2)"
	@echo "------------------------------------------------------------"
	@echo "make test        : Run all 16 unit tests across all modules"
	@echo "make evaluate    : Run evaluation for both GMM & Autoencoder"
	@echo "make compare     : Print side-by-side comparative benchmark table"
	@echo "make hybrid      : Run Phase 2+ Hybrid Innovation (AE + Latent GMM)"
	@echo "make ablation    : Print systematic ablation study tables"
	@echo "make app         : Launch interactive Streamlit triage web dashboard"
	@echo "make train-gmm   : Train GMM baseline (requires data/creditcard.csv)"
	@echo "make train-ae    : Train Autoencoder model (requires data/creditcard.csv)"

test:
	$(PYTHON) -m unittest discover -s tests

evaluate:
	@echo "\n=== PHASE 1: GMM EVALUATION ==="
	$(PYTHON) scripts/evaluate_gmm.py
	@echo "\n=== PHASE 2: AUTOENCODER EVALUATION ==="
	$(PYTHON) scripts/evaluate_autoencoder.py

compare:
	$(PYTHON) scripts/compare_models.py

hybrid:
	$(PYTHON) scripts/run_hybrid.py

ablation:
	$(PYTHON) scripts/run_ablations.py


app:
	$(PYTHON) -m streamlit run app.py

train-gmm:
	$(PYTHON) scripts/train_gmm.py

train-ae:
	$(PYTHON) scripts/train_autoencoder.py
