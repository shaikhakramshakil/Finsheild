"""Phase 2 + Phase 3 tests — model registry, evaluation, inference, training smoke."""

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
    confusion_dict,
    evaluate,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    pr_auc,
    recall_at_fpr,
    roc_auc,
    tune_threshold,
)
from finsheild.inference import EXPECTED_INPUT_COLUMNS, FraudPredictor
from finsheild.model import MODEL_REGISTRY, build_model, list_models, predict_proba
from finsheild.train import MODEL_OUTPUT_DIR

EXPECTED_INPUTS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

try:
    import xgboost  # noqa: F401
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

xgb_required = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed in this env")


# --- helpers ---------------------------------------------------------------- #


def _synth_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = EXPECTED_INPUTS + ["Class"]
    data = {c: rng.normal(0, 1, n) for c in cols}
    data["Class"] = (rng.uniform(0, 1, n) < 0.05).astype(int)
    for i in range(1, 29):
        data[f"V{i}"] = data[f"V{i}"] + 1.5 * data["Class"] * (1 if i % 2 else -1)
    return pd.DataFrame(data)


# --- model registry (Phase 2 / always-installable) ------------------------- #


def test_model_registry_has_phase2_models():
    """logreg + lightgbm build without external ML deps beyond sklearn."""
    names = list_models()
    assert "logreg" in names
    assert "lightgbm" in names
    for n in ("logreg", "lightgbm"):
        m = build_model(n)
        assert hasattr(m, "predict_proba") or hasattr(m, "decision_function")


def test_build_model_supports_overrides():
    m = build_model("lightgbm", n_estimators=37, learning_rate=0.1)
    assert m.n_estimators == 37
    assert m.learning_rate == 0.1


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("nope-neural-net")


# --- Phase 3 XGBoost -------------------------------------------------------- #


@xgb_required
def test_model_registry_includes_xgboost_buildable():
    assert "xgboost" in list_models()
    m = build_model("xgboost")
    assert hasattr(m, "predict_proba")
    # Defaults match Phase 3 plan
    assert m.n_estimators == 500
    assert m.learning_rate == 0.05
    assert m.max_depth == 6
    assert m.eval_metric == "aucpr"  # PR-AUC for early stopping


@xgb_required
def test_build_xgboost_supports_overrides():
    m = build_model("xgboost", n_estimators=42, max_depth=4, learning_rate=0.2)
    assert m.n_estimators == 42
    assert m.max_depth == 4
    assert m.learning_rate == 0.2


# --- train.py MODEL_OUTPUT_DIR routing ------------------------------------- #


def test_model_output_dir_matches_phase_plan():
    assert MODEL_OUTPUT_DIR["logreg"] == "baseline"
    assert MODEL_OUTPUT_DIR["lightgbm"] == "baseline_gbm"
    assert MODEL_OUTPUT_DIR["xgboost"] == "xgboost"


# --- evaluation ------------------------------------------------------------- #


def test_evaluate_returns_phase2_metric_set():
    rng = np.random.default_rng(0)
    y = (rng.uniform(0, 1, 500) < 0.1).astype(int)
    score = y * 0.7 + rng.normal(0, 0.2, 500)
    res = evaluate(y, score, target_fpr=0.05)
    assert isinstance(res, EvalResult)
    for field in ("pr_auc", "roc_auc", "precision", "recall", "f1",
                  "recall_at_target_fpr", "threshold", "confusion_matrix"):
        assert hasattr(res, field), f"missing field {field}"
    assert 0.0 <= res.pr_auc <= 1.0
    assert 0.0 <= res.roc_auc <= 1.0
    cm = res.confusion_matrix
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)


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
    thr, p, r, f1 = tune_threshold(y, score, target_fpr=0.0)
    assert thr >= 0.7


def test_confusion_dict_correctness():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    cm = confusion_dict(y_true, y_pred)
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}


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


def test_plot_confusion_matrix_writes_file(tmp_path: Path):
    cm = {"tn": 700, "fp": 30, "fn": 50, "tp": 70}
    out = plot_confusion_matrix(cm, tmp_path / "cm.png")
    assert out.exists() and out.stat().st_size > 0


# --- inference -------------------------------------------------------------- #


def test_fraud_predictor_roundtrip(tmp_path: Path):
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

    rec = {c: 0.0 for c in EXPECTED_INPUTS}
    out = pred.predict_record(rec)
    assert set(out.keys()) == {"fraud_prob", "threshold", "is_fraud"}
    assert 0.0 <= out["fraud_prob"] <= 1.0

    df_out = pred.predict_df(df.head(5))
    assert "fraud_prob" in df_out.columns
    assert "is_fraud" in df_out.columns
    assert len(df_out) == 5

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
    s = json.dumps(snap, default=str)
    assert "paths" in s and "training" in s and "foo" in s


# --- training smoke --------------------------------------------------------- #


def test_train_logreg_smoke(tmp_path: Path):
    """End-to-end logreg baseline run; writes to models/baseline/ + evaluation/."""
    import os
    import subprocess
    import sys

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
        "FINSHEILD_RESULTS_DIR": str(tmp_path / "results"),
    }
    cmd = [sys.executable, "-m", "finsheild.train", "--model", "logreg"]
    subprocess.check_call(cmd, env={**env, **__import__("os").environ}, cwd=str(Path.cwd()))

    base = tmp_path / "models" / "baseline"
    assert (base / "model.joblib").exists()
    assert (base / "scaler.joblib").exists()
    assert (base / "threshold.json").exists()
    assert (base / "metrics.json").exists()
    assert (base / "config.json").exists()

    m = json.loads((base / "metrics.json").read_text())
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix"):
        assert key in m
    assert set(m["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


@xgb_required
def test_train_xgboost_smoke(tmp_path: Path):
    """End-to-end XGBoost training; writes to models/xgboost/ + evaluation/."""
    import os
    import subprocess
    import sys

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
        "FINSHEILD_RESULTS_DIR": str(tmp_path / "results"),
    }
    cmd = [
        sys.executable, "-m", "finsheild.train",
        "--model", "xgboost",
        "--xgb-n-estimators", "50",
    ]
    subprocess.check_call(cmd, env={**env, **__import__("os").environ}, cwd=str(Path.cwd()))

    xgb_dir = tmp_path / "models" / "xgboost"
    assert (xgb_dir / "model.joblib").exists()
    assert (xgb_dir / "scaler.joblib").exists()
    assert (xgb_dir / "threshold.json").exists()
    assert (xgb_dir / "metrics.json").exists()
    assert (xgb_dir / "config.json").exists()

    m = json.loads((xgb_dir / "metrics.json").read_text())
    for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix"):
        assert key in m
    assert set(m["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}