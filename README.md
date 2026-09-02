# Finsheild

ML-first digital-payment fraud intelligence platform. **Phase 1** (dataset pipeline) and **Phase 2** (model training + inference) are both implemented. The user-facing application is built later.

## Project Overview

Finsheild trains tabular ML models on labeled payment-transaction data to flag fraud. Pipeline:

```
data/raw/creditcard.csv
      ↓  loader (schema check, no leakage)
train / val / test (stratified 70/15/15, seed 42)
      ↓  FraudPreprocessor (scaler fit ONLY on train)
scaled features
      ↓  train.py → logreg | lightgbm
checkpoints/<exp>/ + models/<exp>/ + results/<exp>/
      ↓  inference.FraudPredictor
fraud_prob / is_fraud
```

The data task is binary fraud detection on the Kaggle Credit Card Fraud (ULB) dataset — see `docs/dataset.md` for source rationale, candidates evaluated, and known gaps.

### What's in each phase

- **Phase 1 — Dataset pipeline**: loader, leakage-safe scaler, stratified splits, EDA notebook, 9 tests
- **Phase 2 — Modeling**: model registry (`logreg`, `lightgbm`), training loop with checkpoint + resume, evaluation harness (PR-AUC + recall@FPR + threshold tuning), per-experiment config snapshot, Colab training notebook, `FraudPredictor` for inference, 16 tests

## Development Setup

Local development is CPU-only and lightweight.

```bash
# Recommended: use a virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ required (developed on 3.12).

### Local sanity checks (no training)

```bash
pytest tests/ -v          # 25 tests, ~15s; uses data/raw/creditcard.csv if present
python scripts/download_dataset.py --dry-run        # prints acquisition plan
python scripts/download_dataset.py --synthetic --n 5000  # creates a tiny CSV for tests
```

## Colab Setup (primary training environment)

**Do not train on your laptop** — Colab is the designated training runtime per `AGENTS.md` and the README of Phase 1.

1. Open `notebooks/Finsheild_Training.ipynb` in Google Colab (GPU or CPU runtime — LightGBM is CPU-native; this dataset trains in seconds either way).
2. In **Cell 2**, set `REPO_URL` to your fork (e.g. `https://github.com/<you>/Finsheild.git`). Leave it blank if you uploaded the repo as a zip.
3. **Cell 5** uses Kaggle creds from Colab Secrets when set (`KAGGLE_USERNAME`, `KAGGLE_KEY`); falls back to synthetic data otherwise.
4. **Cell 6** runs `python -m finsheild.train` and writes artifacts to `models/`, `checkpoints/`, `results/` under the repo, then mirrors to `Drive/MyDrive/Finsheild/`.
5. **Cell 8** syncs the trained model + metrics to Drive.

If you can't pull from a remote yet, the notebook still runs end-to-end — just upload the repo via Colab's file panel.

## Dataset Setup

**Primary**: Kaggle Credit Card Fraud (ULB) — `mlg-ulb/creditcardfraud`. 284,807 rows, 31 columns, 0.17% fraud.

- Documented in `docs/dataset.md` with two alternative candidates evaluated (IEEE-CIS, PaySim)
- Raw CSV is **gitignored**. Expected at `data/raw/creditcard.csv` (override via `FINSHEILD_RAW_CSV` env var or `--raw-csv` flag).
- Acquire via:
  - `python scripts/download_dataset.py` (needs `KAGGLE_USERNAME`/`KAGGLE_KEY`)
  - `python scripts/download_dataset.py --synthetic --n 20000` (no creds; smoke test only)

The pipeline is **configurable** — paths come from `finsheild.config.ProjectPaths`. No hardcoded absolute paths.

## Training

```bash
# Train a fresh experiment
python -m finsheild.train --model lightgbm --experiment experiment_001

# Train the baseline
python -m finsheild.train --model logreg --experiment experiment_002

# Customize hyperparameters
python -m finsheild.train --model lightgbm --experiment experiment_003 \
    --lgbm-n-estimators 1000 --lgbm-learning-rate 0.03 --lgbm-num-leaves 63
```

Outputs (per experiment, never overwritten — pick a unique name):

```
models/experiment_001/
  model.joblib          # trained model
  scaler.joblib         # FraudPreprocessor (from data/processed/)
  threshold.json        # tuned operating threshold (recall-max @ FPR=1%)
results/experiment_001/
  config.json           # full config snapshot (paths, params, dataset info)
  metrics.json          # final test-set metrics
  pr_curve.png          # precision-recall curve
  roc_curve.png         # ROC curve
```

Each run also appends a summary line to `results/metrics.jsonl` for cross-experiment comparison.

## Resume Training

Colab runtimes can terminate mid-training. To resume:

```bash
python -m finsheild.train --model lightgbm --experiment experiment_001 \
    --resume --max-epochs 50
```

The `--resume` flag picks up the LightGBM booster from `checkpoints/<experiment>/last.estimator` and continues boosting. Set `FINSHEILD_CHECKPOINTS_DIR` to a Drive path (e.g. `/content/drive/MyDrive/Finsheild/checkpoints`) to persist across Colab sessions.

## Evaluation

Primary metric: **PR-AUC** (average precision) — best for extreme class imbalance.

Also reported:
- ROC-AUC (for sanity)
- **Recall @ target FPR** (default 1%) — closest to operational reality
- Precision / recall / F1 at the **val-tuned threshold**

Reproduce an evaluation on a saved model:

```python
from finsheild.inference import FraudPredictor
pred = FraudPredictor.load(
    "models/experiment_001/model.joblib",
    "models/experiment_001/scaler.joblib",
    threshold=0.5,  # or read from models/experiment_001/threshold.json
)
```

## Inference

The app-facing API lives in `finsheild.inference.FraudPredictor`:

```python
from finsheild.inference import FraudPredictor

pred = FraudPredictor.load("models/experiment_001/model.joblib",
                           "models/experiment_001/scaler.joblib",
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

The core shape is `predict_proba(X: pd.DataFrame) -> np.ndarray`. The thin wrappers share the same code path.

## Experiment Management

Experiments are addressed by name. Each experiment is independent and reproducible from:

- `results/<name>/config.json` — full config snapshot
- `models/<name>/` — trained artifact
- Dataset version pinned implicitly via the raw CSV (commit it to DVC / external storage for true reproducibility)

Cross-experiment metrics live in `results/metrics.jsonl` (one JSON object per line).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: finsheild` | cwd not at repo root | `cd Finsheild && python -m finsheild.train ...` (not `python src/finsheild/train.py`) |
| `Raw dataset missing` | no CSV at expected path | Run `scripts/download_dataset.py` or `--synthetic`; or set `FINSHEILD_RAW_CSV` |
| Drive mirror fails with `OSError: [Errno 5]` | Drive not mounted | Run notebook Cell 4 (mounts Drive) |
| Training interrupted | Colab runtime dropped | Re-run training with `--resume` |
| `lightgbm` install fails | Missing build deps on Python 3.14 | Use Python 3.11–3.12 (the venv in `requirements.txt` was tested on 3.12) |
| Low PR-AUC on synthetic data | Synthetic CSV has no real fraud signal | Expected; synthetic is for pipeline testing only — use real Kaggle data for real metrics |
| Threshold tuned to 0.0 | Class is too rare in val (target FPR unattainable) | Increase `--target-fpr` or use more data |

## Repository Layout

```
Finsheild/
├── README.md, AGENTS.md, pyproject.toml
├── requirements.txt, requirements-colab.txt, .gitignore
├── config/dataset.yaml                 # dataset config (paths, schema, splits)
├── data/
│   ├── raw/                            # gitignored
│   ├── processed/                      # gitignored
│   └── README.md
├── src/finsheild/
│   ├── __init__.py, config.py          # centralized paths + training defaults
│   ├── data/                           # loader, preprocessing, splits (Phase 1)
│   ├── model.py                        # registry: logreg, lightgbm
│   ├── evaluation.py                   # PR-AUC, ROC-AUC, recall@FPR, threshold tuning, plots
│   ├── train.py                        # training entry point + checkpoint/resume
│   └── inference.py                    # FraudPredictor for the future app
├── tests/
│   ├── test_data_pipeline.py           # 9 tests (Phase 1)
│   └── test_model_pipeline.py          # 16 tests (Phase 2)
├── scripts/download_dataset.py         # Kaggle + synthetic fallback
├── notebooks/
│   ├── 02_eda.ipynb                    # local EDA
│   ├── Finsheild_Training.ipynb        # Colab training orchestrator
│   └── colab/01_dataset.ipynb          # Colab dataset-only notebook (Phase 1)
├── docs/dataset.md                     # dataset rationale + candidates + gaps
├── evaluation/                         # Phase 1 evaluation artifacts (gitignored figures)
├── models/                             # final trained artifacts (gitignored, with README)
├── checkpoints/                        # resume state (gitignored, with README)
└── results/                            # per-experiment metrics + plots (gitignored, with README)
```

## Compute Rule (do not violate)

- **Local**: writing code, editing, git, lightweight testing, repository management.
- **Colab**: any GPU/CPU-intensive workload — dataset processing at scale, model training, evaluation, experiments, trained-model artifacts.
- Do NOT download large pretrained weights or CUDA toolchains to your laptop.

## Where to go next

Phase 2 is wired up. To advance further you'd want: a real training run in Colab on the full 284k dataset, feature engineering experiments, cost-sensitive learning, time-based split evaluation, and eventually the Finsheild application that consumes `FraudPredictor.predict_record`.