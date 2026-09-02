"""Phase 2 tests — model registry, evaluation, inference, training smoke."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from finsheild.config import ProjectPaths, TrainingDefaults, snapshot_config
from finsheild.evaluation import (
    EvalResult,
    evaluate,
    plot_pr_curve,
    plot_roc_curve,
    pr_auc,
    recall_at_fpr,
    roc_auc,
    tune_threshold,
)
from finsheild.inference import EXPECTED_INPUT_COLUMNS, FraudPredictor
from finsheild.model import MODEL_REGISTRY, build_model, list_models, predict_proba

EXPECTED_INPUTS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# --- helpers ---------------------------------------------------------------- #


def _synth_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = EXPECTED_INPUTS + ["Class"]
    data = {c: rng.normal(0, 1, n) for c in cols}
    data["Class"] = (rng.uniform(0, 1, n) < 0.05).astype(int)
    # Inject a tiny signal so a model can learn something
    for i in range(1, 29):
        data[f"V{i}"] = data[f"V{i}"] + 1.5 * data["Class"] * (1 if i % 2 else -1)
    return pd.DataFrame(data)


# --- model registry --------------------------------------------------------- #


def test_model_registry_has_logreg_and_lightgbm():
    names = list_models()
    assert "logreg" in names
    assert "lightgbm" in names
    for n in names:
        spec = MODEL_REGISTRY[n]
        m = build_model(n)
        assert hasattr(m, "predict_proba") or hasattr(m, "decision_function")


def test_build_model_supports_overrides():
    m = build_model("lightgbm", n_estimators=37, learning_rate=0.1)
    assert m.n_estimators == 37
    assert m.learning_rate == 0.1


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("nope-neural-net")


# --- evaluation ------------------------------------------------------------- #


def test_evaluate_returns_expected_metrics():
    rng = np.random.default_rng(0)
    y = (rng.uniform(0, 1, 500) < 0.1).astype(int)
    score = y * 0.7 + rng.normal(0, 0.2, 500)
    res = evaluate(y, score, target_fpr=0.05)
    assert isinstance(res, EvalResult)
    assert 0.0 <= res.pr_auc <= 1.0
    assert 0.0 <= res.roc_auc <= 1.0
    assert 0.0 <= res.recall_at_target_fpr <= 1.0
    assert 0.0 <= res.threshold <= 1.0


def test_pr_auc_perfect_score():
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    assert pr_auc(y, score) == pytest.approx(1.0)


def test_roc_auc_perfect_score():
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    assert roc_auc(y, score) == pytest.approx(1.0)


def test_recall_at_fpr_meets_target():
    y = np.array([0] * 99 + [1])
    score = np.concatenate([np.linspace(0, 0.5, 99), np.array([1.0])])
    rec, thr = recall_at_fpr(y, score, target_fpr=0.05)
    assert rec == 1.0


def test_tune_threshold_respects_fpr_cap():
    y = np.array([0, 0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8])
    thr, p, r, f1 = tune_threshold(y, score, target_fpr=0.0)  # FPR=0 means only score==1 allowed
    # Should pick a high threshold and only flag the highest-scoring positives
    assert thr >= 0.7


def test_plot_pr_curve_writes_file(tmp_path: Path):
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    out = plot_pr_curve(y, score, tmp_path / "pr.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_roc_curve_writes_file(tmp_path: Path):
    y = np.array([0, 1, 0, 1, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    out = plot_roc_curve(y, score, tmp_path / "roc.png")
    assert out.exists() and out.stat().st_size > 0


# --- inference -------------------------------------------------------------- #


def test_fraud_predictor_roundtrip(tmp_path: Path):
    """Train a small logreg, save artifacts, load via FraudPredictor, score a record."""
    from sklearn.linear_model import LogisticRegression

    df = _synth_df(n=400, seed=7)
    X = df[EXPECTED_INPUTS].to_numpy()
    y = df["Class"].to_numpy()
    model = LogisticRegression(max_iter=200).fit(X, y)

    from finsheild.data.preprocessing import FraudPreprocessor
    pre = FraudPreprocessor(scale_features=["Amount", "Time"]).fit(df)
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    pre.save(scaler_path)
    joblib.dump(model, model_path)

    pred = FraudPredictor.load(model_path, scaler_path, threshold=0.5)

    # Single record
    rec = {c: 0.0 for c in EXPECTED_INPUTS}
    out = pred.predict_record(rec)
    assert set(out.keys()) == {"fraud_prob", "threshold", "is_fraud"}
    assert 0.0 <= out["fraud_prob"] <= 1.0

    # Batch
    df_out = pred.predict_df(df.head(5))
    assert "fraud_prob" in df_out.columns
    assert "is_fraud" in df_out.columns
    assert len(df_out) == 5

    # predict_proba shape
    p = pred.predict_proba(df.head(3))
    assert p.shape == (3,)


def test_fraud_predictor_missing_columns_raises(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression
    df = _synth_df(n=200, seed=3)
    X = df[EXPECTED_INPUTS].to_numpy()
    y = df["Class"].to_numpy()
    model = LogisticRegression(max_iter=200).fit(X, y)

    from finsheild.data.preprocessing import FraudPreprocessor
    pre = FraudPreprocessor(scale_features=["Amount", "Time"]).fit(df)
    scaler_path = tmp_path / "scaler.joblib"
    model_path = tmp_path / "model.joblib"
    pre.save(scaler_path)
    joblib.dump(model, model_path)

    pred = FraudPredictor.load(model_path, scaler_path)
    with pytest.raises(ValueError, match="Missing required input columns"):
        pred.predict_proba(df.drop(columns=["V1"]))


def test_fraud_predictor_missing_artifacts_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FraudPredictor.load(tmp_path / "no_model.joblib", tmp_path / "no_scaler.joblib")


# --- config ----------------------------------------------------------------- #


def test_project_paths_ensure_creates_dirs():
    paths = ProjectPaths(
        root=Path("/tmp/_finsheild_paths_test"),
        raw_csv=Path("/tmp/_finsheild_paths_test/data/raw/x.csv"),
        processed_dir=Path("/tmp/_finsheild_paths_test/data/processed"),
        models_dir=Path("/tmp/_finsheild_paths_test/models"),
        checkpoints_dir=Path("/tmp/_finsheild_paths_test/checkpoints"),
        results_dir=Path("/tmp/_finsheild_paths_test/results"),
    )
    try:
        paths.ensure()
        assert paths.results_dir.exists()
        assert paths.checkpoints_dir.exists()
        assert paths.models_dir.exists()
    finally:
        import shutil
        shutil.rmtree("/tmp/_finsheild_paths_test", ignore_errors=True)


def test_snapshot_config_is_json_serializable():
    paths = ProjectPaths()
    td = TrainingDefaults()
    snap = snapshot_config(paths, td, extra={"foo": "bar"})
    # round-trip through json
    s = json.dumps(snap, default=str)
    assert "paths" in s and "training" in s and "foo" in s


# --- training smoke (no full run) ------------------------------------------ #


def test_train_lightgbm_smoke(tmp_path: Path, monkeypatch):
    """Run a tiny end-to-end training + resume cycle against synthetic data.

    Uses the real train.py entry point with FINSHEILD_* env vars redirected to tmp_path.
    """
    import subprocess
    import sys

    # Generate synthetic CSV via the existing script
    raw_csv = tmp_path / "raw.csv"
    subprocess.check_call([
        sys.executable, "scripts/download_dataset.py",
        "--synthetic", "--n", "1000", "--output", str(raw_csv),
    ], cwd=str(Path.cwd()))

    env = {
        "PYTHONPATH": "src",
        "FINSHEILD_RAW_CSV": str(raw_csv),
        "FINSHEILD_PROCESSED_DIR": str(tmp_path / "processed"),
        "FINSHEILD_MODELS_DIR": str(tmp_path / "models"),
        "FINSHEILD_CHECKPOINTS_DIR": str(tmp_path / "checkpoints"),
        "FINSHEILD_RESULTS_DIR": str(tmp_path / "results"),
    }
    cmd = [
        sys.executable, "-m", "finsheild.train",
        "--model", "lightgbm",
        "--experiment", "test_smoke",
        "--lgbm-n-estimators", "30",
        "--lgbm-early-stopping-rounds", "10",
    ]
    subprocess.check_call(cmd, env={**env, **__import__("os").environ}, cwd=str(Path.cwd()))

    # artifacts exist
    assert (tmp_path / "models" / "test_smoke" / "model.joblib").exists()
    assert (tmp_path / "models" / "test_smoke" / "scaler.joblib").exists()
    assert (tmp_path / "models" / "test_smoke" / "threshold.json").exists()
    assert (tmp_path / "results" / "test_smoke" / "metrics.json").exists()
    assert (tmp_path / "results" / "test_smoke" / "config.json").exists()
    assert (tmp_path / "results" / "test_smoke" / "pr_curve.png").exists()
    assert (tmp_path / "results" / "test_smoke" / "roc_curve.png").exists()

    # metrics parses
    m = json.loads((tmp_path / "results" / "test_smoke" / "metrics.json").read_text())
    assert "pr_auc" in m and 0.0 <= m["pr_auc"] <= 1.0
    assert "roc_auc" in m and 0.0 <= m["roc_auc"] <= 1.0
    assert "threshold" in m

    # resume runs without raising
    cmd_resume = cmd + ["--resume", "--max-epochs", "5"]
    subprocess.check_call(cmd_resume, env={**env, **__import__("os").environ}, cwd=str(Path.cwd()))