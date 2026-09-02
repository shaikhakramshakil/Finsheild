"""Training entry point — `python src/train.py`.

Supports two model families (see `finsheild.model`):
  - logreg    — sklearn LogisticRegression baseline (no resume; refits in one shot)
  - lightgbm  — LightGBM classifier with early stopping on val PR-AUC

Workflow per run:
  1. Load raw dataset (data/raw/creditcard.csv or --raw-csv override)
  2. Stratified 70/15/15 split (seed 42)
  3. Fit StandardScaler on train Amount/Time only (leakage-safe)
  4. Train model on train, evaluate on val
  5. Tune decision threshold on val (recall-max @ target_fpr)
  6. Save final model + scaler + threshold to models/<experiment>/
  7. Save metrics + plots + config snapshot to results/<experiment>/

Checkpointing (LightGBM only):
  checkpoints/<experiment>/last.estimator  — init_model-compatible booster
  checkpoints/<experiment>/last.json       — epoch, best metric, RNG state
  --resume checkpoints/<experiment>        — continues training from last.estimator

Outputs never overwrite; experiments are addressed by name (default experiment_001).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from finsheild.config import ProjectPaths, TrainingDefaults, detect_device, snapshot_config
from finsheild.data.loader import load_raw
from finsheild.data.preprocessing import FraudPreprocessor, preprocess_splits
from finsheild.data.splits import make_splits
from finsheild.evaluation import (
    EvalResult,
    evaluate,
    plot_pr_curve,
    plot_roc_curve,
    write_metrics,
)
from finsheild.model import build_model, predict_proba

logger = logging.getLogger("finsheild.train")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import lightgbm as lgb  # noqa: F401
        lgb.set_seed = seed  # no-op guard so import isn't "unused"
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #


def checkpoint_dir(paths: ProjectPaths, experiment: str) -> Path:
    return paths.checkpoints_dir / experiment


def save_checkpoint(
    paths: ProjectPaths,
    experiment: str,
    model: object,
    epoch: int,
    best_metric: float,
    seed: int,
) -> Path:
    ckpt_dir = checkpoint_dir(paths, experiment)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    est_path = ckpt_dir / "last.estimator"
    joblib.dump(model, est_path)
    state = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "seed": int(seed),
        "saved_at": time.time(),
        "model_type": type(model).__name__,
    }
    (ckpt_dir / "last.json").write_text(json.dumps(state, indent=2))
    logger.info("Checkpoint saved: %s (epoch=%d, best=%.4f)", est_path, epoch, best_metric)
    return est_path


def load_checkpoint(paths: ProjectPaths, experiment: str) -> tuple[object, dict] | None:
    ckpt_dir = checkpoint_dir(paths, experiment)
    est_path = ckpt_dir / "last.estimator"
    state_path = ckpt_dir / "last.json"
    if not (est_path.exists() and state_path.exists()):
        return None
    model = joblib.load(est_path)
    state = json.loads(state_path.read_text())
    logger.info("Resumed from %s (epoch=%d, best=%.4f)", est_path, state["epoch"], state["best_metric"])
    return model, state


# --------------------------------------------------------------------------- #
# Training core
# --------------------------------------------------------------------------- #


def train_logreg(X_tr, y_tr, X_va, y_va, params: dict, init_model=None):
    """One-shot fit (sklearn LR has no warm-start support across fits)."""
    model = build_model("logreg", **params)
    model.fit(X_tr, y_tr)
    proba_va = predict_proba(model, X_va)
    metric = evaluate(y_va, proba_va, target_fpr=params.get("target_fpr", 0.01)).pr_auc
    return model, proba_va, metric


def train_lightgbm(
    X_tr, y_tr, X_va, y_va, params: dict, init_model=None, max_epochs: int = 1
):
    """Incremental LightGBM training with PR-AUC early stopping on val.

    `max_epochs` is the number of additional boosting rounds in this run.
    Returns (model, proba_va, best_pr_auc).
    """
    import lightgbm as lgb

    n_estimators = params.get("n_estimators", 500)
    early_stopping_rounds = params.get("early_stopping_rounds", 50)

    base_kwargs = {
        "n_estimators": n_estimators,
        "learning_rate": params.get("learning_rate", 0.05),
        "num_leaves": params.get("num_leaves", 31),
        "min_child_samples": params.get("min_child_samples", 20),
        "subsample": params.get("subsample", 0.8),
        "subsample_freq": 1,
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "random_state": params.get("random_state", 42),
        "n_jobs": -1,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**base_kwargs)
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)]
    if init_model is not None:
        # Warm-start: prepend the prior booster's tree count to the new fit.
        try:
            prior_trees = init_model.booster_.current_iteration()
            model.set_params(n_estimators=prior_trees + max_epochs)
        except Exception:
            logger.warning("Could not read prior iteration count; fitting cold.")
            init_model = None

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average-precision",
        callbacks=callbacks,
        init_model=init_model.booster_ if init_model is not None else None,
    )
    proba_va = predict_proba(model, X_va)
    metric = evaluate(y_va, proba_va, target_fpr=params.get("target_fpr", 0.01)).pr_auc
    return model, proba_va, metric


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> int:
    paths = ProjectPaths().ensure()
    training = TrainingDefaults(
        model_name=args.model,
        experiment=args.experiment,
        random_state=args.seed,
    )
    set_seed(training.random_state)
    device = detect_device(prefer_gpu=False)
    logger.info("Device: %s", device)

    # 1) Load + split
    raw_path = Path(args.raw_csv) if args.raw_csv else paths.raw_csv
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset missing at {raw_path}. "
            "Run `python scripts/download_dataset.py --synthetic` for a quick start, "
            "or `python scripts/download_dataset.py` with Kaggle creds."
        )
    df = load_raw(raw_path)
    train, val, test = make_splits(df, random_state=training.random_state)
    train_t, val_t, test_t, pre = preprocess_splits(
        train, val, test,
        scale_features=["Amount", "Time"],
        save_scaler_path=paths.processed_dir / "scaler.joblib",
    )
    X_tr = train_t[FEATURE_COLS].to_numpy()
    y_tr = train_t["Class"].to_numpy()
    X_va = val_t[FEATURE_COLS].to_numpy()
    y_va = val_t["Class"].to_numpy()
    X_te = test_t[FEATURE_COLS].to_numpy()
    y_te = test_t["Class"].to_numpy()

    # 2) Optional resume
    init_model = None
    start_epoch = 0
    best_metric = -1.0
    if args.resume:
        ck = load_checkpoint(paths, training.experiment)
        if ck is None:
            raise FileNotFoundError(f"--resume set but no checkpoint under {checkpoint_dir(paths, training.experiment)}")
        init_model, state = ck
        start_epoch = state["epoch"]
        best_metric = state["best_metric"]
        if training.model_name != "lightgbm":
            logger.warning("--resume only effective for lightgbm; refitting %s cold.", training.model_name)

    # 3) Train
    t0 = time.time()
    if training.model_name == "logreg":
        params = {"C": args.C, "max_iter": args.max_iter, "target_fpr": args.target_fpr}
        model, proba_va, val_pr_auc = train_logreg(X_tr, y_tr, X_va, y_va, params, init_model=init_model)
    elif training.model_name == "lightgbm":
        params = {
            "n_estimators": args.lgbm_n_estimators,
            "learning_rate": args.lgbm_learning_rate,
            "num_leaves": args.lgbm_num_leaves,
            "min_child_samples": args.lgbm_min_child_samples,
            "subsample": args.lgbm_subsample,
            "colsample_bytree": args.lgbm_colsample_bytree,
            "early_stopping_rounds": args.lgbm_early_stopping_rounds,
            "random_state": training.random_state,
            "target_fpr": args.target_fpr,
        }
        model, proba_va, val_pr_auc = train_lightgbm(
            X_tr, y_tr, X_va, y_va, params, init_model=init_model, max_epochs=args.max_epochs
        )
    else:
        raise SystemExit(f"Unknown model {training.model_name!r}; expected 'logreg' or 'lightgbm'")

    train_seconds = time.time() - t0
    new_best = val_pr_auc > best_metric
    if new_best:
        best_metric = val_pr_auc
    if training.model_name == "lightgbm":
        save_checkpoint(paths, training.experiment, model, epoch=start_epoch + 1, best_metric=best_metric, seed=training.random_state)

    # 4) Tune threshold on val
    val_result = evaluate(y_va, proba_va, target_fpr=args.target_fpr)
    threshold = val_result.threshold

    # 5) Evaluate on test
    proba_te = predict_proba(model, X_te)
    test_result = evaluate(y_te, proba_te, target_fpr=args.target_fpr)

    # 6) Persist final model + scaler + threshold
    exp_models = paths.models_dir / training.experiment
    exp_models.mkdir(parents=True, exist_ok=True)
    model_path = exp_models / "model.joblib"
    joblib.dump(model, model_path)
    import shutil
    shutil.copy2(paths.processed_dir / "scaler.joblib", exp_models / "scaler.joblib")
    (exp_models / "threshold.json").write_text(json.dumps({"threshold": threshold, "target_fpr": args.target_fpr}, indent=2))

    # 7) Experiment results
    exp_results = paths.results_dir / training.experiment
    exp_results.mkdir(parents=True, exist_ok=True)
    write_metrics(test_result, exp_results / "metrics.json")
    # Append per-run summary to metrics.jsonl
    with (paths.results_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({
            "experiment": training.experiment,
            "model": training.model_name,
            "val_pr_auc": val_result.pr_auc,
            "test_pr_auc": test_result.pr_auc,
            "test_roc_auc": test_result.roc_auc,
            "recall_at_fpr": test_result.recall_at_target_fpr,
            "threshold": threshold,
            "train_seconds": train_seconds,
            "resumed": bool(args.resume),
            "timestamp": time.time(),
        }) + "\n")
    plot_pr_curve(y_te, proba_te, exp_results / "pr_curve.png", title=f"PR — {training.experiment}")
    plot_roc_curve(y_te, proba_te, exp_results / "roc_curve.png", title=f"ROC — {training.experiment}")

    # 8) Config snapshot
    snap = snapshot_config(
        paths, training,
        dataset={
            "raw_csv": str(raw_path),
            "rows": int(len(df)),
            "fraud_rate": float(df["Class"].mean()),
            "splits": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        },
        metrics={"val": val_result.to_dict(), "test": test_result.to_dict()},
    )
    (exp_results / "config.json").write_text(json.dumps(snap, indent=2))

    logger.info(
        "DONE exp=%s model=%s val_pr_auc=%.4f test_pr_auc=%.4f test_roc_auc=%.4f threshold=%.4f (%.1fs)",
        training.experiment, training.model_name,
        val_result.pr_auc, test_result.pr_auc, test_result.roc_auc, threshold, train_seconds
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finsheild training entry point")
    p.add_argument("--model", choices=["logreg", "lightgbm"], default=TrainingDefaults.model_name)
    p.add_argument("--experiment", default=TrainingDefaults.experiment)
    p.add_argument("--seed", type=int, default=TrainingDefaults.random_state)
    p.add_argument("--raw-csv", default=None, help="Override raw CSV path (default: data/raw/creditcard.csv)")
    p.add_argument("--target-fpr", type=float, default=TrainingDefaults.target_fpr)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints/<experiment>/")
    # Logreg
    p.add_argument("--C", type=float, default=TrainingDefaults.logreg_C)
    p.add_argument("--max-iter", type=int, default=TrainingDefaults.logreg_max_iter)
    # LightGBM
    p.add_argument("--lgbm-n-estimators", type=int, default=TrainingDefaults.lgbm_n_estimators)
    p.add_argument("--lgbm-learning-rate", type=float, default=TrainingDefaults.lgbm_learning_rate)
    p.add_argument("--lgbm-num-leaves", type=int, default=TrainingDefaults.lgbm_num_leaves)
    p.add_argument("--lgbm-min-child-samples", type=int, default=TrainingDefaults.lgbm_min_child_samples)
    p.add_argument("--lgbm-subsample", type=float, default=TrainingDefaults.lgbm_subsample)
    p.add_argument("--lgbm-colsample-bytree", type=float, default=TrainingDefaults.lgbm_colsample_bytree)
    p.add_argument("--lgbm-early-stopping-rounds", type=int, default=TrainingDefaults.lgbm_early_stopping_rounds)
    p.add_argument("--max-epochs", type=int, default=50, help="Additional boosting rounds when resuming")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())