"""Centralized runtime config for training/evaluation/inference.

Paths are env-overridable so the same code runs locally and in Colab without edits.
Defaults follow the project's existing layout; Drive mount path is detected at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Repo root: src/finsheild/config.py -> parents[2] = repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ProjectPaths:
    """All on-disk paths the pipeline touches. Override via env vars.

    Env vars:
      FINSHEILD_ROOT          — repo root (default: auto-detected)
      FINSHEILD_RAW_CSV       — raw dataset CSV (default: <root>/data/raw/creditcard.csv)
      FINSHEILD_PROCESSED_DIR — processed splits dir (default: <root>/data/processed)
      FINSHEILD_MODELS_DIR    — final model artifacts (default: <root>/models)
      FINSHEILD_CHECKPOINTS_DIR — training checkpoints (default: <root>/checkpoints)
      FINSHEILD_RESULTS_DIR   — experiment results (default: <root>/results)
      FINSHEILD_DRIVE_ROOT    — Colab Drive mount root (default: /content/drive/MyDrive/Finsheild)
    """

    root: Path = field(default_factory=lambda: _env_path("FINSHEILD_ROOT", REPO_ROOT))
    raw_csv: Path = field(default_factory=lambda: _env_path("FINSHEILD_RAW_CSV", REPO_ROOT / "data" / "raw" / "creditcard.csv"))
    processed_dir: Path = field(default_factory=lambda: _env_path("FINSHEILD_PROCESSED_DIR", REPO_ROOT / "data" / "processed"))
    models_dir: Path = field(default_factory=lambda: _env_path("FINSHEILD_MODELS_DIR", REPO_ROOT / "models"))
    checkpoints_dir: Path = field(default_factory=lambda: _env_path("FINSHEILD_CHECKPOINTS_DIR", REPO_ROOT / "checkpoints"))
    results_dir: Path = field(default_factory=lambda: _env_path("FINSHEILD_RESULTS_DIR", REPO_ROOT / "results"))
    drive_root: Path = field(default_factory=lambda: _env_path("FINSHEILD_DRIVE_ROOT", Path("/content/drive/MyDrive/Finsheild")))
    figures_dir: Path = field(default_factory=lambda: REPO_ROOT / "evaluation" / "figures")
    reports_dir: Path = field(default_factory=lambda: REPO_ROOT / "evaluation" / "reports")

    def experiment_dir(self, name: str) -> Path:
        return self.results_dir / name

    def ensure(self) -> "ProjectPaths":
        for d in (self.processed_dir, self.models_dir, self.checkpoints_dir, self.results_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class TrainingDefaults:
    """Defaults consumed by `train.py`. CLI flags override these."""

    model_name: str = "lightgbm"  # 'logreg' | 'lightgbm'
    experiment: str = "experiment_001"
    random_state: int = 42
    # Logreg
    logreg_C: float = 1.0
    logreg_max_iter: int = 1000
    # LightGBM
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 20
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    # Early stopping (LightGBM only)
    lgbm_early_stopping_rounds: int = 50
    # Threshold tuning
    target_fpr: float = 0.01  # recall @ this FPR is reported; threshold tuned on val


def detect_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda', 'mps', or 'cpu'. LightGBM uses CPU natively; this is for the registry/logging."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch  # noqa: F401
        import torch as _t
        if _t.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def snapshot_config(paths: ProjectPaths, training: TrainingDefaults, **extra: Any) -> dict:
    """Build a JSON-serializable config dict for experiment tracking."""
    snap: dict[str, Any] = {
        "paths": {k: str(v) for k, v in asdict(paths).items()},
        "training": asdict(training),
    }
    snap.update(extra)
    return snap