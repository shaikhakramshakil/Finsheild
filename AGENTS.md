# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Build: `pip install -r requirements.txt` (local) / `pip install -r requirements-colab.txt` (Colab) — `pyproject.toml` defines package `finsheild` under `src/`.
- Test: `pytest -v` (requires `data/raw/creditcard.csv`; use `python scripts/download_dataset.py --synthetic` for CI fallback).
- Dataset: Primary Kaggle Credit Card Fraud (ULB) — see `docs/dataset.md` and `config/dataset.yaml`; raw under `data/raw/` (gitignored).
- Notebooks: `notebooks/02_eda.ipynb` (local), `notebooks/colab/01_dataset.ipynb` (Colab-ready — hardware detection, pip install, kagglehub/opendatasets download).
- Evaluation: `evaluation/reports/dataset_report.md` + `evaluation/metrics.json` (real run), figures under `evaluation/figures/` (gitignored).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
