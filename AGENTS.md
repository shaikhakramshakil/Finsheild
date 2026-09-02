# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Build: `pip install -r requirements.txt` (local) / `pip install -r requirements-colab.txt` (Colab) — `pyproject.toml` defines package `finsheild` under `src/`.
- Test: `pytest -v` (requires `data/raw/creditcard.csv`; use `python scripts/download_dataset.py --synthetic` for CI fallback).
- Dataset: Primary Kaggle Credit Card Fraud (ULB) — see `docs/dataset.md` and `config/dataset.yaml`; raw under `data/raw/` (gitignored).
- Notebooks: `notebooks/02_eda.ipynb` (local), `notebooks/colab/01_dataset.ipynb` (Colab-ready — hardware detection, pip install, kagglehub/opendatasets download).
- Evaluation: `evaluation/reports/dataset_report.md` + `evaluation/metrics.json` (real run), figures under `evaluation/figures/` (gitignored).
- Synthetic environment (Phase 4): `src/finsheild/synthetic_env/` — six tables (users, accounts, devices, merchants, locations, transactions) plus `account_devices` link. Eight inspectable scenarios; deterministic via `SyntheticEnvConfig(seed=…)`. Schema in `docs/synthetic_env_schema.md`. Generate with `PYTHONPATH=src python scripts/generate_synthetic_env.py --scale dev --out data/synthetic_env/dev`.
- Colab CLI: `colab` (v0.6.0) at `~/.local/bin/colab`. CPU sessions work out of the box; GPU sessions need `--gpu T4|L4|G4|H100|A100` (`colab new -s <name> --gpu T4`). Existing session name is `finsheild` (CPU); new ephemeral names auto-clean via `--session`. PyTorch in Colab comes as `2.11.0+cu128` and reports CUDA True on T4.
- Colab-side repo fetch: `git clone https://github.com/...` from inside the Colab base image fails (`could not read Username for 'https://github.com'`) even though `urllib`/`curl` reach `github.com`. Use `python scripts/fetch_repo.py --branch main --dest /content/Finsheild` (accepts `--ref <sha>`). Downloads the public tarball via `urllib`, extracts with `tarfile`, no extra deps. Does NOT bootstrap git history.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
