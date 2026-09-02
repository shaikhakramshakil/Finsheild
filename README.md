# Finsheild

ML-first digital-payment fraud intelligence platform. The current scope is **Phase 1 (dataset pipeline) and Phase 2 (LogReg baseline)** of the project's 16-phase ML plan. The user-facing application is built later.

## Project Overview

Finsheild trains tabular ML models on labeled payment-transaction data to flag fraud. Pipeline:

```
data/raw/creditcard.csv
      ↓  loader (schema check, no leakage)
train / val / test (stratified 70/15/15, seed 42)
      ↓  FraudPreprocessor (scaler fit ONLY on train)
scaled features
      ↓  train.py → logreg
models/baseline/ + evaluation/{reports,figures}/
      ↓  inference.FraudPredictor
fraud_prob / is_fraud
```

The data task is binary fraud detection on the Kaggle Credit Card Fraud (ULB) dataset — see `docs/dataset.md` for source rationale, candidates evaluated, and known gaps.

### Phases shipped

- **Phase 1 — Dataset pipeline**: loader, leakage-safe scaler, stratified splits, EDA notebook, 9 tests
- **Phase 2 — Baseline (LogReg)**: training entry point, model registry (`logreg`, `lightgbm`), evaluation harness (precision, recall, F1, ROC-AUC, PR-AUC, recall@FPR, confusion matrix), `FraudPredictor` inference, 19 tests

### Phases NOT yet started

Phase 3 (XGBoost primary), Phase 4 (synthetic environment), Phases 5–11 (features, profiling, anomaly, rules, graph, risk fusion), Phases 12–15 (LLM copilot), Phase 16 (final export). See `Finsheild - ML-FIRST DEVELOPMENT PLAN.md`.

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
pytest tests/ -v          # 28 tests, ~15s
python scripts/download_dataset.py --synthetic --n 5000
python -m finsheild.train --model logreg    # trains Phase 2 baseline
```

## Colab Setup (primary training environment)

**Do not train on your laptop** — Colab is the designated training runtime per `AGENTS.md` and the Phase 1 README.

1. Open `notebooks/Finsheild_Training.ipynb` in Google Colab.
2. In **Cell 2**, set `REPO_URL` to your fork.
3. **Cell 5** uses Kaggle creds from Colab Secrets when set; falls back to synthetic data otherwise.
4. **Cell 6** runs `python -m finsheild.train --model logreg` (Phase 2) and writes artifacts to `models/baseline/`, `evaluation/reports/`, `evaluation/figures/`, then mirrors to Drive.

## Dataset Setup

**Primary**: Kaggle Credit Card Fraud (ULB) — `mlg-ulb/creditcardfraud`. 284,807 rows, 31 columns, 0.17% fraud.

- Documented in `docs/dataset.md` with two alternative candidates evaluated (IEEE-CIS, PaySim)
- Raw CSV is **gitignored**. Expected at `data/raw/creditcard.csv` (override via `FINSHEILD_RAW_CSV` env var or `--raw-csv` flag).
- Acquire via:
  - `python scripts/download_dataset.py` (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`)
  - `python scripts/download_dataset.py --synthetic --n 20000` (no creds; smoke test only)

## Training (Phase 2 baseline)

```bash
# Train the LogReg baseline (Phase 2)
python -m finsheild.train --model logreg

# Train the LightGBM comparison model (output: models/baseline_gbm/)
python -m finsheild.train --model lightgbm

# XGBoost is wired but not yet implemented — Phase 3
python -m finsheild.train --model xgboost   # fails loud until Phase 3 lands
```

### Output layout

Per-model directory (strict, from the plan's Phase 16):

```
models/baseline/                  # Phase 2 LogReg artifacts
  model.joblib                    # trained classifier
  scaler.joblib                   # FraudPreprocessor (Amount, Time scaled, fit on train only)
  threshold.json                  # tuned operating threshold (recall-max @ FPR=1%)
  metrics.json                    # test metrics: precision/recall/F1/ROC-AUC/PR-AUC/confusion
  config.json                     # full config snapshot

evaluation/
  reports/baseline_report.md      # human-readable report
  reports/baseline_metrics.json   # test metrics (JSON, same as models/baseline/metrics.json)
  figures/baseline_pr_curve.png
  figures/baseline_roc_curve.png
  figures/baseline_confusion_matrix.png
```

## Evaluation

Per the plan's Phase 2 metric set: **precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix**. Also reported for operational use: recall @ target FPR (default 1%) and the val-tuned operating threshold.

The report at `evaluation/reports/baseline_report.md` is the human-readable summary; the JSON at `models/baseline/metrics.json` is machine-readable.

## Inference

The app-facing API lives in `finsheild.inference.FraudPredictor`:

```python
from finsheild.inference import FraudPredictor

pred = FraudPredictor.load("models/baseline/model.joblib",
                           "models/baseline/scaler.joblib",
                           threshold=0.05)

# Single transaction — this is what the future app will call
record = {"Time": 12345, "V1": -1.4, "V2": 0.3, ..., "V28": 0.1, "Amount": 87.50}
result = pred.predict_record(record)
# -> {"fraud_prob": 0.83, "threshold": 0.05, "is_fraud": 1}

# Batch (CSV → scored CSV)
import pandas as pd
df = pd.read_csv("new_transactions.csv")
scored = pred.predict_df(df)
scored.to_csv("scored.csv", index=False)
```

Note: `FraudPredictor` is currently a single-model wrapper around the LogReg baseline. The multi-signal `RiskFusion` engine (Phase 10) is not yet built; this interface will be wrapped/extended in Phase 10.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: finsheild` | cwd not at repo root | `cd Finsheild && python -m finsheild.train ...` |
| `Raw dataset missing` | no CSV at expected path | Run `scripts/download_dataset.py` or `--synthetic`; or set `FINSHEILD_RAW_CSV` |
| `Unknown model 'xgboost'` then `not yet wired up` | Expected — Phase 3 | Wait for Phase 3; use `--model logreg` for Phase 2 |
| Drive mirror fails with `OSError: [Errno 5]` | Drive not mounted | Run notebook Cell 4 (mounts Drive) |
| Threshold tuned to 0.0 | Class is too rare in val (target FPR unattainable) | Increase `--target-fpr` or use more data |
| Low PR-AUC on synthetic data | Synthetic CSV has no real fraud signal | Expected; synthetic is for pipeline testing only |

## Repository Layout

```
Finsheild/
├── README.md, AGENTS.md, pyproject.toml
├── LICENSE (MIT)
├── requirements.txt, requirements-colab.txt, .gitignore
├── config/dataset.yaml                 # dataset config (paths, schema, splits)
├── data/{raw,processed}/               # gitignored
├── src/finsheild/
│   ├── __init__.py, config.py          # centralized paths + training defaults
│   ├── data/                           # Phase 1: loader, preprocessing, splits
│   ├── model.py                        # Phase 2+: registry (logreg, lightgbm; xgboost = Phase 3)
│   ├── evaluation.py                   # Phase 2: precision/recall/F1/ROC-AUC/PR-AUC/confusion + plots
│   ├── train.py                        # Phase 2: training entry, MODEL_OUTPUT_DIR routes to models/<phase>/
│   └── inference.py                    # FraudPredictor (single-model wrapper)
├── tests/
│   ├── test_data_pipeline.py           # 9 tests (Phase 1)
│   └── test_model_pipeline.py          # 19 tests (Phase 2)
├── scripts/download_dataset.py         # Kaggle + synthetic fallback
├── notebooks/
│   ├── 02_eda.ipynb                    # local EDA
│   ├── Finsheild_Training.ipynb        # Colab training orchestrator
│   └── colab/01_dataset.ipynb          # Phase 1 Colab data notebook
├── docs/dataset.md                     # dataset rationale + candidates + gaps
├── evaluation/                         # reports/*.md + metrics.json + figures/*.png
├── models/                             # per-model artifacts (Phase 16 layout)
│   ├── baseline/                       # Phase 2 LogReg (gitignored contents)
│   ├── baseline_gbm/                   # LightGBM comparison (gitignored contents)
│   ├── xgboost/                        # Phase 3 (empty)
│   ├── anomaly/                        # Phase 7 (empty)
│   ├── risk_fusion/                    # Phase 10 (empty)
│   └── llm/adapter/                    # Phase 15 (empty)
├── checkpoints/                        # (Phase 2 trains in one shot; no checkpoints yet)
└── results/                            # (Phase 16 final-export placeholder)
```

## Compute Rule (do not violate)

- **Local**: writing code, editing, git, lightweight testing, repository management.
- **Colab**: any GPU/CPU-intensive workload — model training, evaluation, experiments.
- Do NOT download large pretrained weights or CUDA toolchains to your laptop.