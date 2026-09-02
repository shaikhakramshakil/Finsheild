# Finsheild

Research-oriented digital payment fraud intelligence platform — ML-first development. Phase 1 (Dataset pipeline) is local-only, no GitHub remote.

## Phase 1 — Dataset pipeline

Implements **Phase 1: Dataset** — local-only, lightweight CPU. Finds/configures public fraud dataset (Kaggle Credit Card Fraud Detection as primary), provides loader/preprocessing/splits with leakage prevention, EDA, and Colab-ready artifacts.

- **Primary dataset:** Kaggle Credit Card Fraud Detection (ULB, 284k rows, 0.17% fraud) — see `docs/dataset.md`.
- **Pipeline:** `src/finsheild/data/{loader,preprocessing,splits}.py` + `config/dataset.yaml`.
- **Verification:** `pytest` (≥6 checks), no leakage (scaler fit only on train), stratified 70/15/15, `random_state=42`.

## Reproduce

### Local (CPU-lightweight)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py --dry-run
python scripts/download_dataset.py --synthetic  # fallback without Kaggle creds
# or with Kaggle creds:
export KAGGLE_USERNAME=... KAGGLE_KEY=...
python scripts/download_dataset.py

python -c "from finsheild.data.loader import load_raw; from finsheild.data.splits import make_splits; df=load_raw(); print(df.shape)"

pytest -v
# Generates evaluation/reports/dataset_report.md + evaluation/metrics.json + figures
```

### Colab (mandatory training env for Phase 2+; dataset pipeline also works)

1. Open `notebooks/colab/01_dataset.ipynb` in Colab.
2. Cell 1 shows hardware (python/torch/CUDA/GPU).
3. Cell 2: `!pip install -r requirements-colab.txt`
4. Cell 3: download via `kagglehub` (set Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`) or fallback `!python scripts/download_dataset.py --synthetic`
5. Remaining cells demo loader/preprocessing/splits — no LLM weights, runs CPU or GPU runtime.

Local notebooks: `notebooks/02_eda.ipynb` for EDA (figures under `evaluation/figures/`).

## Exit Criteria (Phase 1)

- `src/finsheild/data/*.py` + configs + notebooks + docs + tests committed on `fm/fshld-003`
- `pytest` green (real run, not mocked)
- `docs/dataset.md` and `evaluation/reports/dataset_report.md` contain REAL numbers from actual run (or clear `BLOCKED: Kaggle credentials required in Colab, run notebooks/colab/01_dataset.ipynb` note)
- `.gitignore` prevents raw data commit (`data/raw/`, `*.csv`)

## Compute Rule

Google Colab is ONLY training environment. Local is lightweight CPU only. This phase (dataset loader/preprocessing/EDA/splits) is CPU-lightweight and may run locally, but every training/LLM/GPU artifact must be Colab-ready. Do NOT download LLM weights or CUDA locally.

## Project pointers

- Dataset docs: `docs/dataset.md`
- Config: `config/dataset.yaml`
- Data layout: `data/README.md`
- Tests: `tests/test_data_pipeline.py`
