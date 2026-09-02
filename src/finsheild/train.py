"""Training entry point — `python -m finsheild.train`.

Per the project's ML plan (Phase 2 = LogReg baseline, Phase 3 = XGBoost, comparison model = LightGBM):
  --model logreg    → models/baseline/
  --model lightgbm  → models/baseline_gbm/
  --model xgboost   → models/xgboost/   (Phase 3)

Workflow per run:
  1. Load raw dataset (data/raw/creditcard.csv or --raw-csv override)
  2. Stratified 70/15/15 split (seed 42)
  3. Fit StandardScaler on train Amount/Time only (leakage-safe)
  4. Train model on train, evaluate on val
  5. Tune decision threshold on val (recall-max @ target_fpr)
  6. Save final model + scaler + threshold to the per-model directory
  7. Save metrics + plots to evaluation/{reports,figures}/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
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
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    write_metrics,
)
from finsheild.model import build_model, predict_proba

logger = logging.getLogger("finsheild.train")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Map model name -> output directory under models/
MODEL_OUTPUT_DIR = {
    "logreg": "baseline",
    "lightgbm": "baseline_gbm",
    "xgboost": "xgboost",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run(args: argparse.Namespace) -> int:
    paths = ProjectPaths().ensure()
    training = TrainingDefaults(
        model_name=args.model,
        random_state=args.seed,
    )
    set_seed(training.random_state)
    device = detect_device(prefer_gpu=False)
    logger.info("Device: %s", device)

    if args.model not in MODEL_OUTPUT_DIR:
        raise SystemExit(f"Unknown model '{args.model}'; expected one of {list(MODEL_OUTPUT_DIR)}")
    out_dir_name = MODEL_OUTPUT_DIR[args.model]
    phase_name = f"phase2_{out_dir_name}"

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

    # Train
    t0 = time.time()
    if args.model == "logreg":
        params = {"C": args.C, "max_iter": args.max_iter}
        model = build_model("logreg", **params)
        model.fit(X_tr, y_tr)
        proba_va = predict_proba(model, X_va)
    elif args.model == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            min_child_samples=args.lgbm_min_child_samples,
            subsample=args.lgbm_subsample,
            subsample_freq=1,
            colsample_bytree=args.lgbm_colsample_bytree,
            random_state=training.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average-precision",
            callbacks=[lgb.early_stopping(stopping_rounds=args.lgbm_early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)],
        )
        proba_va = predict_proba(model, X_va)
    elif args.model == "xgboost":
        # Phase 3 will fill in the real XGBoost training loop; for now fail loud.
        raise SystemExit("XGBoost training is not yet wired up — that's Phase 3.")
    else:
        raise SystemExit(f"Unknown model {args.model!r}")
    train_seconds = time.time() - t0

    # Evaluate val (tune threshold) and test (apply tuned threshold)
    val_result = evaluate(y_va, proba_va, target_fpr=args.target_fpr)
    proba_te = predict_proba(model, X_te)
    test_result = evaluate(y_te, proba_te, target_fpr=args.target_fpr, threshold=val_result.threshold)

    # Persist model + scaler + threshold
    out_models = paths.models_dir / out_dir_name
    out_models.mkdir(parents=True, exist_ok=True)
    model_path = out_models / "model.joblib"
    joblib.dump(model, model_path)
    import shutil
    shutil.copy2(paths.processed_dir / "scaler.joblib", out_models / "scaler.joblib")
    (out_models / "threshold.json").write_text(json.dumps({"threshold": val_result.threshold, "target_fpr": args.target_fpr}, indent=2))

    # Persist metrics + plots + config snapshot
    out_eval_reports = paths.reports_dir
    out_eval_figures = paths.figures_dir
    out_eval_reports.mkdir(parents=True, exist_ok=True)
    out_eval_figures.mkdir(parents=True, exist_ok=True)
    write_metrics(test_result, out_models / "metrics.json")
    write_metrics(test_result, out_eval_reports / f"{out_dir_name}_metrics.json")
    plot_pr_curve(y_te, proba_te, out_eval_figures / f"{out_dir_name}_pr_curve.png", title=f"PR — {out_dir_name}")
    plot_roc_curve(y_te, proba_te, out_eval_figures / f"{out_dir_name}_roc_curve.png", title=f"ROC — {out_dir_name}")
    plot_confusion_matrix(test_result.confusion_matrix, out_eval_figures / f"{out_dir_name}_confusion_matrix.png", title=f"Confusion — {out_dir_name}")

    snap = snapshot_config(
        paths, training,
        dataset={
            "raw_csv": str(raw_path),
            "rows": int(len(df)),
            "fraud_rate": float(df["Class"].mean()),
            "splits": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        },
        metrics={"val": val_result.to_dict(), "test": test_result.to_dict()},
        phase=phase_name,
    )
    (out_models / "config.json").write_text(json.dumps(snap, indent=2, default=str))

    # Human-readable report (Phase 2 baseline report)
    _write_baseline_report(out_eval_reports / f"{out_dir_name}_report.md", args.model, out_dir_name, val_result, test_result, train_seconds, snap)

    logger.info(
        "DONE phase=%s model=%s test_pr_auc=%.4f test_roc_auc=%.4f precision=%.4f recall=%.4f f1=%.4f threshold=%.4f (%.1fs)",
        phase_name, args.model,
        test_result.pr_auc, test_result.roc_auc, test_result.precision, test_result.recall, test_result.f1,
        test_result.threshold, train_seconds
    )
    return 0


def _write_baseline_report(path: Path, model_name: str, out_dir: str, val_res: EvalResult, test_res: EvalResult, train_seconds: float, snap: dict) -> Path:
    cm = test_res.confusion_matrix
    lines = [
        f"# Baseline Report — {out_dir}",
        "",
        f"Model: `{model_name}`",
        f"Phase: 2 (baseline classifier)",
        "",
        "## Test metrics",
        f"- Precision: {test_res.precision:.4f}",
        f"- Recall:    {test_res.recall:.4f}",
        f"- F1:        {test_res.f1:.4f}",
        f"- ROC-AUC:   {test_res.roc_auc:.4f}",
        f"- PR-AUC:    {test_res.pr_auc:.4f}",
        f"- Recall @ FPR={test_res.target_fpr:.2%}: {test_res.recall_at_target_fpr:.4f}",
        "",
        "## Confusion matrix (test, threshold=" + f"{test_res.threshold:.4f})",
        f"|        | Predicted legit | Predicted fraud |",
        f"|--------|-----------------|-----------------|",
        f"| Actual legit | {cm['tn']} | {cm['fp']} |",
        f"| Actual fraud | {cm['fn']} | {cm['tp']} |",
        "",
        f"Support: {test_res.support_neg} legit, {test_res.support_pos} fraud",
        f"Threshold tuned on val: {val_res.threshold:.4f}",
        "",
        "## Training",
        f"- Train seconds: {train_seconds:.1f}",
        f"- Splits: {snap['dataset']['splits']}",
        "",
        "Figures: `evaluation/figures/{out_dir}_pr_curve.png`, `{out_dir}_roc_curve.png`, `{out_dir}_confusion_matrix.png`",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finsheild training entry point")
    p.add_argument("--model", choices=["logreg", "lightgbm", "xgboost"], default="logreg")
    p.add_argument("--seed", type=int, default=TrainingDefaults.random_state)
    p.add_argument("--raw-csv", default=None, help="Override raw CSV path (default: data/raw/creditcard.csv)")
    p.add_argument("--target-fpr", type=float, default=TrainingDefaults.target_fpr)
    # Logreg
    p.add_argument("--C", type=float, default=TrainingDefaults.logreg_C)
    p.add_argument("--max-iter", type=int, default=TrainingDefaults.logreg_max_iter)
    # LightGBM (comparison model)
    p.add_argument("--lgbm-n-estimators", type=int, default=TrainingDefaults.lgbm_n_estimators)
    p.add_argument("--lgbm-learning-rate", type=float, default=TrainingDefaults.lgbm_learning_rate)
    p.add_argument("--lgbm-num-leaves", type=int, default=TrainingDefaults.lgbm_num_leaves)
    p.add_argument("--lgbm-min-child-samples", type=int, default=TrainingDefaults.lgbm_min_child_samples)
    p.add_argument("--lgbm-subsample", type=float, default=TrainingDefaults.lgbm_subsample)
    p.add_argument("--lgbm-colsample-bytree", type=float, default=TrainingDefaults.lgbm_colsample_bytree)
    p.add_argument("--lgbm-early-stopping-rounds", type=int, default=TrainingDefaults.lgbm_early_stopping_rounds)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())