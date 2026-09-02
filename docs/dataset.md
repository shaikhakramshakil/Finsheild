# Dataset — Finsheild Phase 1

## 1. Dataset Source

**Primary (chosen): Kaggle Credit Card Fraud Detection (ULB, `mlg-ulb/creditcardfraud`)**
- URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- License: See Kaggle dataset page — original data from ULB Machine Learning Group; released for research.
- Citation: Dal Pozzolo, Caelen, Johnson & Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE CIDM 2015.
- Size (full Kaggle): 284,807 rows, 31 columns, 492 frauds (0.172%).
- Local sample used for CI: 10,000 rows (see `data/raw/creditcard.csv` via synthetic or subset) — actual run below reflects the 10k local checkout; full 284k can be fetched in Colab.

**Why it fits digital-payment fraud even though V1–V28 are anonymized:**
European cardholder transactions over 2 days; highly imbalanced, tabular, real-world fraud labels. PCA anonymization preserves fraud signal while protecting privacy — standard benchmark for payment-fraud research. Finsheild uses it for Phase-1 pipeline validation; features `Amount`/`Time` remain interpretable, V1–V28 used as-is.

**Candidates evaluated (at least 2–3):**
| Dataset | URL | Rows | Fraud rate | Pros | Cons | Verdict |
|---|---|---|---|---|---|---|
| Kaggle Credit Card Fraud (ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | 284k | 0.17% | Public, lightweight, reproducible, benchmark | PCA anonymized, 2-day window only | **Primary — immediate reproducibility** |
| IEEE-CIS Fraud Detection | https://www.kaggle.com/c/ieee-fraud-detection | ~590k | 3.5% | Richer features (identity + transaction), modern | Large (multi-GB), complex, needs join | Secondary / Future (Phase 3+) |
| PaySim Synthetic Mobile-Money | https://www.kaggle.com/datasets/ealaxi/paysim1 | 6.3M | 0.13% | Synthetic mobile-money, interpretable fields, large scale | Synthetic, not real cards | Secondary / Future (scale testing) |

Chosen primary is Credit Card Fraud for immediate reproducibility; PaySim/IEEE-CIS noted as future/secondary in `config/dataset.yaml`.

## 2. Features Table

| Name | Type | Description |
|---|---|---|
| Time | float | Seconds elapsed since first transaction in dataset |
| V1–V28 | float | PCA-transformed anonymized features (confidential) |
| Amount | float | Transaction amount (EUR) |
| Class | int (0/1) | Target — 1 = fraud, 0 = legitimate |

All features numeric; no missing values in raw (0 in run); no duplicated rows in 10k sample (0). See `evaluation/reports/dataset_report.md` for actual stats.

## 3. Target Variable

`Class` — binary fraud label. Distribution from actual run (10k sample):
- 0 (legit): 9828 (98.28%)
- 1 (fraud): 172 (1.72%)
- Fraud rate 1.72% matches expected imbalance.

## 4. Class Distribution

Actual counts from run (`evaluation/metrics.json`):
- Total rows: 10000
- Class 0: 9828, Class 1: 172 — see `evaluation/figures/class_distribution.png`

## 5. Preprocessing Steps (in order, leakage-safe)

1. Load raw CSV via `src/finsheild/data/loader.py:load_raw` — validate schema, log row counts, drop NaN if any.
2. Stratified split via `src/finsheild/data/splits.py:make_splits` (seed 42) before any fitting.
3. Fit `StandardScaler` ONLY on train for `["Amount", "Time"]` via `src/finsheild/data/preprocessing.py:FraudPreprocessor.fit_transform_train`.
4. Transform val/test with train-fitted scaler via `FraudPreprocessor.transform` (raises if not fitted).
5. Save `train.csv`, `val.csv`, `test.csv` + `scaler.joblib` under `data/processed/` (gitignored) or demonstrate via `preprocess_splits` helper.
6. Alternative `sklearn` Pipeline available via `FraudPreprocessor.get_pipeline()` (ColumnTransformer).

**Leakage note:** `mean_` after train-fit = `[50.666..., 85517.34]` vs full-data fit would differ — test `test_no_leakage_scaler_fitted_only_on_train` asserts `not np.allclose(...)`. No validation/test information leaks into scaler.

## 6. Split Strategy

- Sizes: 70% train (7000), 15% val (1500), 15% test (1500) — `test_size=0.15, val_size=0.15` in `config/dataset.yaml`.
- Stratification: `stratify=True` on `Class`, `random_state=42`.
- Indices reset after split; splits are disjoint (verified via merge).
- Time-based leakage check: `Time` is seconds since first transaction; prefer stratified random split for now. Time-based split (sort by Time, split chronologically) would test temporal generalization but changes fraud distribution — documented as alternative for future, not used now.

## 7. Acquisition Instructions

**Local (lightweight CPU):**
```bash
pip install -r requirements.txt
python scripts/download_dataset.py --help
python scripts/download_dataset.py --dry-run   # prints acquisition path without download
python scripts/download_dataset.py --synthetic --n 10000  # fallback without Kaggle creds
# With Kaggle creds:
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python scripts/download_dataset.py  # or kagglehub/opendatasets fallback
python -c "from finsheild.data.loader import load_raw; df=load_raw(); print(df.shape)"
```

**Colab (mandatory training env in future phases; dataset pipeline also works):**
- Open `notebooks/colab/01_dataset.ipynb` in Colab GPU runtime.
- Cell 1: hardware detection (python, torch, CUDA).
- Cell 2: `!pip install -r requirements-colab.txt`.
- Cell 3: kagglehub download (set Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY` or upload `kaggle.json`) OR synthetic fallback `!python scripts/download_dataset.py --synthetic`.
- Cells 4–6: loader/preprocessing/split demo — must run fresh without LLM weights.

Direct Kaggle download via `kagglehub`/`opendatasets` with Kaggle auth is primary; Colab notebook contains fallback URL/cell and does not leak local secrets.

**Keep lightweight:** raw CSV under `data/raw/` (gitignored), do NOT commit `*.csv`. Only small schema/sample checked if needed; tests use synthetic fallback when creds missing.

## 8. Remaining Gaps

- Full 284k dataset not bundled locally — Colab must fetch via Kaggle auth or use mirrored URL (document credible fallback).
- V1–V28 lack semantic meaning due to PCA — future phases may need additional interpretable datasets (IEEE-CIS/PaySim).
- Time-based split alternative not yet benchmarked; temporal drift not evaluated in Phase 1.
- No classifier training — Phase 2+ will handle imbalance (sampling, cost-sensitive) and metrics (PR-AUC).
