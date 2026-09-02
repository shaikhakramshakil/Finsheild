# Finsheild

ML-first digital-payment fraud intelligence platform. Current scope: **Phase 1 (dataset) + Phase 2 (LogReg baseline) + Phase 3 (XGBoost primary)** of the project's 16-phase ML plan. The user-facing application is built later.

## Project Overview

Finsheild trains tabular ML models on labeled payment-transaction data to flag fraud. Pipeline:

```
data/raw/creditcard.csv
      ↓  loader (schema check, no leakage)
train / val / test (stratified 70/15/15, seed 42)
      ↓  FraudPreprocessor (scaler fit ONLY on train)
scaled features
      ↓  train.py → logreg | xgboost | lightgbm
models/{baseline,xgboost,baseline_gbm}/ + evaluation/{reports,figures}/
      ↓  inference.FraudPredictor
fraud_prob / is_fraud
```

The data task is binary fraud detection on the Kaggle Credit Card Fraud (ULB) dataset — see `docs/dataset.md` for source rationale, candidates evaluated, and known gaps.

### Phases shipped

- **Phase 1 — Dataset pipeline**: loader, leakage-safe scaler, stratified splits, EDA notebook, 9 tests
- **Phase 2 — Baseline (LogReg)**: training entry point, model registry, evaluation harness (precision, recall, F1, ROC-AUC, PR-AUC, recall@FPR, confusion matrix), `FraudPredictor` inference
- **Phase 3 — XGBoost (primary classifier)**: tree-based primary with PR-AUC early stopping, full hyperparameter control via CLI flags

Total: 31 tests (9 Phase 1 + 22 Phase 2/3). XGBoost-specific tests are skipped on environments without `xgboost` installed and run in CI.

### Phases NOT yet started

Phase 4 (synthetic environment), Phases 5–11 (features, profiling, anomaly, rules, graph, risk fusion), Phases 12–15 (LLM copilot), Phase 16 (final export). See `Finsheild - ML-FIRST DEVELOPMENT PLAN.md`.

## Development Setup

Local development is CPU-only and lightweight.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"   # adds pytest
```

Python 3.11+ required (developed on 3.12).

### Local sanity checks

```bash
pytest tests/ -v          # 31 tests, ~15s (3 xgb tests skip without xgboost installed)
python scripts/download_dataset.py --synthetic --n 5000
python -m finsheild.train --model logreg    # Phase 2 baseline
python -m finsheild.train --model xgboost   # Phase 3 primary
```

## Colab Setup (primary training environment)

**Do not train on your laptop** — Colab is the designated training runtime per `AGENTS.md` and the Phase 1 README.

1. Open `notebooks/Finsheild_Training.ipynb` in Google Colab.
2. In **Cell 2**, set `REPO_URL` to your fork.
3. **Cell 5** uses Kaggle creds from Colab Secrets when set; falls back to synthetic data otherwise.
4. **Cell 6** runs `python -m finsheild.train --model xgboost` (Phase 3 primary) and writes artifacts to `models/xgboost/`, `evaluation/reports/`, `evaluation/figures/`, then mirrors to Drive.

For Phase 2 baseline specifically, change `MODEL = "logreg"` in Cell 6.

## Dataset Setup

**Primary**: Kaggle Credit Card Fraud (ULB) — `mlg-ulb/creditcardfraud`. 284,807 rows, 31 columns, 0.17% fraud.

- Documented in `docs/dataset.md` with two alternative candidates evaluated (IEEE-CIS, PaySim)
- Raw CSV is **gitignored**. Expected at `data/raw/creditcard.csv` (override via `FINSHEILD_RAW_CSV` env var or `--raw-csv` flag).
- Acquire via:
  - `python scripts/download_dataset.py` (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`)
  - `python scripts/download_dataset.py --synthetic --n 20000` (no creds; smoke test only)

## Training

### Phase 2 — LogReg baseline

```bash
python -m finsheild.train --model logreg
```

### Phase 3 — XGBoost primary

```bash
# Defaults (n_estimators=500, lr=0.05, max_depth=6, aucpr early stopping)
python -m finsheild.train --model xgboost

# Tune
python -m finsheild.train --model xgboost \
    --xgb-n-estimators 1000 --xgb-learning-rate 0.03 --xgb-max-depth 8

# Available flags
python -m finsheild.train --model xgboost --help
```

### Comparison — LightGBM

```bash
python -m finsheild.train --model lightgbm   # → models/baseline_gbm/
```

### Output layout (strict, per Phase 16 plan)

```
models/baseline/                  # Phase 2 LogReg
  model.joblib, scaler.joblib, threshold.json, metrics.json, config.json
models/xgboost/                   # Phase 3 XGBoost (primary)
  model.joblib, scaler.joblib, threshold.json, metrics.json, config.json
models/baseline_gbm/              # LightGBM comparison
  same shape

evaluation/
  reports/{baseline,xgboost}_report.md        # human-readable
  reports/{baseline,xgboost}_metrics.json     # test metrics
  figures/{baseline,xgboost}_{pr_curve,roc_curve,confusion_matrix}.png
```

## Evaluation

Per the plan's Phase 2 metric set: **precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix**. Also reported: recall @ target FPR (default 1%) and the val-tuned operating threshold.

XGBoost uses `eval_metric="aucpr"` internally for early stopping — matches our primary metric.

## Inference

```python
from finsheild.inference import FraudPredictor

# Phase 3 primary
pred = FraudPredictor.load("models/xgboost/model.joblib",
                           "models/xgboost/scaler.joblib",
                           threshold=0.05)

# Single transaction
record = {"Time": 12345, "V1": -1.4, "V2": 0.3, ..., "V28": 0.1, "Amount": 87.50}
result = pred.predict_record(record)
# -> {"fraud_prob": 0.83, "threshold": 0.05, "is_fraud": 1}

# Batch
import pandas as pd
df = pd.read_csv("new_transactions.csv")
scored = pred.predict_df(df)
scored.to_csv("scored.csv", index=False)
```

`FraudPredictor` is currently a single-model wrapper. The multi-signal `RiskFusion` engine (Phase 10) will replace/extend this in a later phase.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: finsheild` | cwd not at repo root | `cd Finsheild && python -m finsheild.train ...` |
| `Raw dataset missing` | no CSV at expected path | Run `scripts/download_dataset.py` or `--synthetic`; or set `FINSHEILD_RAW_CSV` |
| `ModuleNotFoundError: xgboost` | xgboost not installed locally | `pip install xgboost` (already in `requirements.txt`); CI installs it |
| Drive mirror fails with `OSError: [Errno 5]` | Drive not mounted | Run notebook Cell 4 (mounts Drive) |
| Low PR-AUC on synthetic data | Synthetic CSV has no real fraud signal | Expected; use real Kaggle data for real metrics |
| Threshold tuned to 0.0 | Class too rare in val (target FPR unattainable) | Increase `--target-fpr` |

## Repository Layout

```
Finsheild/
├── README.md, AGENTS.md, pyproject.toml
├── LICENSE (MIT)
├── "Finsheild - ML-FIRST DEVELOPMENT PLAN.md"   # 16-phase plan
├── requirements.txt, requirements-colab.txt, .gitignore
├── config/dataset.yaml
├── data/{raw,processed}/                       # gitignored
├── src/finsheild/
│   ├── __init__.py, config.py
│   ├── data/                                   # Phase 1
│   ├── model.py                                # Phase 2/3 registry: logreg, xgboost, lightgbm
│   ├── evaluation.py                           # Phase 2/3: precision/recall/F1/ROC-AUC/PR-AUC/confusion + plots
│   ├── train.py                                # MODEL_OUTPUT_DIR routes to models/<phase>/
│   └── inference.py                            # FraudPredictor
├── tests/
│   ├── test_data_pipeline.py                   # 9 tests (Phase 1)
│   └── test_model_pipeline.py                  # 22 tests (Phase 2/3; 3 xgb skipped if not installed)
├── scripts/download_dataset.py
├── notebooks/
│   ├── 02_eda.ipynb, Finsheild_Training.ipynb
│   └── colab/01_dataset.ipynb
├── docs/dataset.md
├── evaluation/{reports,figures}/
└── models/
    ├── baseline/                               # Phase 2 LogReg
    ├── baseline_gbm/                           # LightGBM comparison
    ├── xgboost/                                # Phase 3 XGBoost
    ├── anomaly/, risk_fusion/, llm/adapter/    # Phase 7/10/15 stubs
```

## Compute Rule (do not violate)

- **Local**: writing code, editing, git, lightweight testing, repository management.
- **Colab**: any GPU/CPU-intensive workload — model training, evaluation, experiments.
- Do NOT download large pretrained weights or CUDA toolchains to your laptop.