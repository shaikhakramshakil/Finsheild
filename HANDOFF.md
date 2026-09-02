# Finsheild — Handoff Document

**For whoever picks this up next (human or AI agent).**

## Current state

Repo: `https://github.com/shaikhakramshakil/Finsheild` (public, MIT)
Branch: `main`
Last commit: `06c012e` — "feat(phase3): XGBoost as primary supervised fraud classifier"
Working tree: clean

### What's done (verified by CI, 28 passing + 3 XGBoost-specific that need xgboost installed)

- **Phase 1 — Dataset pipeline** (commit `9b356f7`)
  - `src/finsheild/data/{loader,preprocessing,splits}.py` — Kaggle ULB Credit Card Fraud
  - `scripts/download_dataset.py` — kagglehub → opendatasets → synthetic fallback
  - 9 tests in `tests/test_data_pipeline.py`
  - `docs/dataset.md` — source rationale, 3 candidates evaluated
  - `notebooks/02_eda.ipynb` (local), `notebooks/colab/01_dataset.ipynb` (Colab)
  - `evaluation/metrics.json` + `evaluation/reports/dataset_report.md` (from a prior synthetic run)
- **Phase 2 — LogReg baseline** (commit `60f1484`)
  - `src/finsheild/{model,evaluation,train,inference}.py` (model.py + evaluation.py + train.py + inference.py)
  - `models/baseline/{model.joblib,scaler.joblib,threshold.json,metrics.json,config.json}` — LogReg baseline
  - `evaluation/reports/baseline_report.md` + 3 figures under `evaluation/figures/`
  - **Phase 2 numbers from synthetic 5k run, not real Kaggle data**
  - 19 tests in `tests/test_model_pipeline.py`
- **Phase 3 — XGBoost primary** (commit `06c012e`)
  - XGBoost wired into `model.py` registry + `train.py` with PR-AUC early stopping
  - 9 `--xgb-*` CLI flags (n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda)
  - `models/xgboost/` exists with only README.md — **no actual training run committed**
  - `requirements.txt` + `requirements-colab.txt` include `xgboost>=2.0` / `xgboost==2.1.2`

### CI status

Green on `main`. Workflow: `.github/workflows/tests.yml` — pytest on Python 3.11 + 3.12 with synthetic dataset bootstrap.

## What's NOT done (per the 16-phase plan)

Full plan reference: `Finsheild - ML-FIRST DEVELOPMENT PLAN.md` (in repo root).

- **Real Kaggle run** — Phase 2 + Phase 3 numbers are synthetic. A real Colab run has been attempted but artifacts were never retrieved.
- **Phase 4 — Synthetic digital payment environment** (users/accounts/devices/merchants/locations/transactions + 8 suspicious scenarios). This is the foundation for Phases 5–11.
- **Phase 5 — Feature engineering** (transaction/behavioral/velocity/location/device features)
- **Phase 6 — Behavioral profiling**
- **Phase 7 — Anomaly detection (Isolation Forest)**
- **Phase 8 — Rule engine**
- **Phase 9 — Graph intelligence (NetworkX)**
- **Phase 10 — Risk fusion** (multi-signal → GREEN/YELLOW/RED + APPROVE/STEP_UP/BLOCK/INVESTIGATE)
- **Phase 11 — SHAP explainability**
- **Phase 12 — LLM training dataset generation** from pipeline outputs
- **Phase 13 — Base LLM evaluation** (2–4B param open-weight, eval-only)
- **Phase 14 — QLoRA fine-tuning** (PyTorch + Transformers + PEFT + TRL + bitsandbytes)
- **Phase 15 — Fine-tuned LLM evaluation** vs base, on same held-out set
- **Phase 16 — Final model export** to `models/{baseline,xgboost,anomaly,risk_fusion,llm/adapter}/`

## What's BLOCKING next steps

### Blocker 1: No real training numbers

The plan requires actual model runs, not synthetic. The synthetic run gives `ROC-AUC=0.5927, PR-AUC=0.0267, F1=0.0` — that's "no signal" because synthetic data has no real fraud pattern. Real numbers come from Kaggle ULB (284,807 rows, 0.17% fraud).

**To unblock:**

1. Run `notebooks/Finsheild_Training.ipynb` in Colab (browser at https://colab.research.google.com, or via the VS Code "Google Colab" extension).
2. Set `REPO_URL = "https://github.com/shaikhakramshakil/Finsheild.git"` in Cell 2.
3. Set `KAGGLE_USERNAME` / `KAGGLE_KEY` in Colab Secrets (left panel 🔑), then uncomment the two lines in Cell 5 that pull from `userdata.get(...)`.
4. Run all cells. ~5–10 min on Colab CPU.
5. Cell 8 mirrors artifacts to Drive at `/content/drive/MyDrive/Finsheild/`.
6. Download these files (NOT `.joblib` — too big):
   - `models/xgboost/metrics.json`
   - `models/xgboost/config.json`
   - `models/xgboost/threshold.json`
   - `evaluation/reports/xgboost_report.md`
   - `evaluation/figures/xgboost_*.png`
7. Commit them into the repo (the easiest way: `git add` + commit + push from a Colab terminal cell, or copy them into your local repo and push).

**Expected realistic numbers** (on Kaggle ULB): PR-AUC ≈ 0.70–0.85, ROC-AUC ≈ 0.95–0.98 for XGBoost. If you see numbers in that range, the pipeline is working.

### Blocker 2: No Phase 2 vs Phase 3 comparison

The plan says "XGBoost as the primary supervised fraud classifier" — that requires showing it's better than the baseline. With synthetic data both will look bad and the comparison is meaningless.

Unblocked once Blocker 1 is done: read both `evaluation/reports/baseline_report.md` and `evaluation/reports/xgboost_report.md`, write `docs/phase3_comparison.md` with actual numbers.

## Environment notes

- **Python**: 3.11+ required (developed on 3.12). Pyproject.toml has `requires-python = ">=3.11"`.
- **Local**: this machine is the dev box. CPU-only. No GPU. Used for editing, git, tests, lightweight sanity checks.
- **Colab**: training environment. The plan's compute rule forbids downloading LLM weights or CUDA toolchains locally. Phase 14 (QLoRA) MUST run in Colab.
- **Network on this host**: pip downloads from PyPI have been timing out intermittently. CI runners are fine.

## Architecture map

```
src/finsheild/
  config.py            # ProjectPaths (env-overridable), TrainingDefaults
  data/                # Phase 1
    loader.py          # load_raw() — schema validation, missing-value handling
    preprocessing.py   # FraudPreprocessor — scaler fit ONLY on train
    splits.py          # make_splits() — stratified 70/15/15, seed 42
  model.py             # Phase 2/3 registry: logreg, xgboost, lightgbm
  evaluation.py        # EvalResult (precision/recall/F1/ROC-AUC/PR-AUC/confusion) + plots
  train.py             # python -m finsheild.train — routes --model to MODEL_OUTPUT_DIR
  inference.py         # FraudPredictor — predict_proba / predict / predict_record / predict_df

models/
  baseline/            # Phase 2 LogReg (committed: synthetic-run artifacts)
  baseline_gbm/        # LightGBM comparison (empty — never run)
  xgboost/             # Phase 3 (empty — needs Kaggle run)
  anomaly/             # Phase 7 stub
  risk_fusion/         # Phase 10 stub
  llm/adapter/         # Phase 15 stub
```

## Tests

```bash
PYTHONPATH=src pytest tests/ -v
```

- 9 Phase 1 data pipeline tests
- 22 Phase 2/3 model pipeline tests (3 XGBoost-specific skipped if xgboost not installed; CI installs it)
- Total: 28 passing + 3 conditional skips on a host without xgboost

## Plan discipline reminder

The plan (`Finsheild - ML-FIRST DEVELOPMENT PLAN.md`) explicitly says:

> Work on ONE phase at a time. After completing a phase:
> 1. Run tests.
> 2. Run the relevant experiment.
> 3. Save outputs.
> 4. Update documentation.
> 5. Report exactly what happened.
> 6. Report actual metrics if the experiment ran.
> 7. Clearly identify anything that remains incomplete.
> 8. STOP.
> Do not automatically proceed to the next phase.

Don't skip ahead. Don't fabricate metrics. Don't claim a model was trained when it wasn't.

## Quick start for next session

```bash
# Local sanity check
cd /home/akram/projects/Finsheild
PYTHONPATH=src pytest tests/ -v   # should pass 28 + 3 skipped locally
git log --oneline -10             # confirm you're on main, last commit 06c012e
```

Then either:

**Option A — finish the in-flight Colab run** (recommended if it's still possible)
1. Open Colab, find your most recent run / Drive sync
2. Pull artifacts from `/content/drive/MyDrive/Finsheild/` into the repo
3. Commit + push
4. Write `docs/phase3_comparison.md` from actual numbers

**Option B — start Phase 4** (synthetic env)
1. Read the plan's Phase 4 section carefully
2. Decide schema: users, accounts, devices, merchants, locations, transactions
3. Decide generator library (pure Python + numpy/pandas is simplest)
4. Generate synthetic fraud scenarios per the plan: account takeover, unusual amount, unusual time, velocity, new device, unusual location, device sharing, mule behavior, unusual merchant
5. Document clearly that this is synthetic, distinct from the public Kaggle ULB data

**Option C — Phase 5–11** (only after Option B works)

## Open questions for the human

- Is the Colab session still accessible, or do you need to re-run from scratch?
- Is real Kaggle data still the goal, or is synthetic-only acceptable going forward?
- What's the deadline / target for Phase 16? Affects scope decisions in Phase 4–11.

## Things to NOT do

- Don't add new frameworks (PyTorch, TensorFlow, etc.) without an explicit phase that needs them. PyTorch is needed at Phase 14 (QLoRA). Don't install it earlier.
- Don't create new top-level directories without consulting the plan. The layout in Phase 16 is strict.
- Don't write UI/API/FastAPI code — explicitly forbidden until Phase 16 is done.
- Don't merge Phase 4 work with Phase 3 cleanup in the same commit. One phase per commit.
- Don't push `.joblib` files for trained models. The plan doesn't commit trained model binaries — Drive is the storage.