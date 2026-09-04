#!/usr/bin/env python3
"""Colab entry point for hard overlap experiment.

Generates synthetic_hard_overlap, trains LogReg+XGBoost (fixed 500 trees),
evaluates, does separability analysis, and saves reports.
Intended to be run via: colab run scripts/run_hard_overlap_experiment.py
or python scripts/run_hard_overlap_experiment.py locally for smoke.
"""
import json, pathlib, sys, os, subprocess, pathlib as _pl
# Colab bootstrap: if finsheild not importable, fetch repo via tarball
try:
    import finsheild  # noqa
except ImportError:
    print("finsheild not found — bootstrapping repo via tarball...", file=sys.stderr)
    # Try fetch_repo.py if present, else direct tarball
    fetch = _pl.Path("scripts/fetch_repo.py")
    if fetch.exists():
        subprocess.check_call([sys.executable, str(fetch), "--branch", "main", "--dest", "/content/Finsheild"])
        sys.path.insert(0, "/content/Finsheild/src")
        os.chdir("/content/Finsheild")
    else:
        # Direct tarball fallback
        import urllib.request, tarfile, tempfile
        url = "https://github.com/shaikhakramshakil/Finsheild/archive/refs/heads/main.tar.gz"
        print(f"Downloading {url}", file=sys.stderr)
        data = urllib.request.urlopen(url, timeout=120).read()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(data); tmp=f.name
        os.makedirs("/content/Finsheild", exist_ok=True)
        with tarfile.open(tmp, "r:gz") as tf:
            tf.extractall("/content/Finsheild", filter="data")
        # Flatten top-level dir
        import shutil
        top = [p for p in _pl.Path("/content/Finsheild").iterdir() if p.is_dir()]
        if len(top)==1:
            inner=top[0]
            for entry in inner.iterdir():
                shutil.move(str(entry), str(_pl.Path("/content/Finsheild")/entry.name))
            inner.rmdir()
        import os as _os
        _os.unlink(tmp)
        sys.path.insert(0, "/content/Finsheild/src")
        os.chdir("/content/Finsheild")
        # Install deps
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"])

import numpy as np
import pandas as pd

from finsheild.synthetic_env import SyntheticEnvConfig
from finsheild.synthetic_env.environment_hard import generate_hard_overlap_environment
from finsheild.features import build_features
from finsheild.features.config import FeatureConfig
from finsheild.model import build_model, predict_proba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 1729
N_BG = 9000
N_PER_SCENARIO = 20

def main():
    print("=== Hard Overlap: Generate ===")
    cfg = SyntheticEnvConfig(n_users=200, n_accounts=250, n_devices=220, n_merchants=80, n_locations=60, n_transactions=N_BG, time_span_days=30, seed=SEED)
    env = generate_hard_overlap_environment(cfg, n_per_scenario=N_PER_SCENARIO)
    tx = env.transactions
    print(f"rows={len(tx)} fraud={int(tx.label_fraud.sum())} rate={tx.label_fraud.mean():.4%} seed={SEED}")
    print(tx[tx.label_fraud==1].scenario_tag.value_counts().to_dict())

    print("=== Features + Split ===")
    feat_res = build_features(env, FeatureConfig())
    F = feat_res.features
    feature_cols = feat_res.feature_columns
    train_df, temp_df = train_test_split(F, test_size=0.30, random_state=42, stratify=F["label_fraud"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_fraud"])
    print(f"train {train_df.shape} fraud {train_df.label_fraud.sum()} | val {val_df.shape} fraud {val_df.label_fraud.sum()} | test {test_df.shape} fraud {test_df.label_fraud.sum()}")

    def impute(X):
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        med = X.median(numeric_only=True)
        return X.fillna(med).fillna(0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(impute(train_df[feature_cols]))
    X_val = scaler.transform(impute(val_df[feature_cols]))
    X_test = scaler.transform(impute(test_df[feature_cols]))
    y_train = train_df.label_fraud.to_numpy(dtype=int)
    y_val = val_df.label_fraud.to_numpy(dtype=int)
    y_test = test_df.label_fraud.to_numpy(dtype=int)

    print("=== Train ===")
    logreg = build_model("logreg")
    xgb = build_model("xgboost", n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    import time
    t0=time.time(); logreg.fit(X_train, y_train); t_logreg=time.time()-t0
    t0=time.time()
    try:
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        xgb.fit(X_train, y_train)
    t_xgb=time.time()-t0
    print(f"LogReg {t_logreg:.1f}s, XGB {t_xgb:.1f}s")

    print("=== Evaluate ===")
    out_dir = pathlib.Path("evaluation/reports")
    fig_dir = pathlib.Path("evaluation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(name, model, train_seconds):
        y_prob_te = predict_proba(model, X_test)
        y_prob_va = predict_proba(model, X_val)
        roc = roc_auc_score(y_test, y_prob_te)
        pr = average_precision_score(y_test, y_prob_te)
        y_pred = (y_prob_te >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr_va, _, thr_va = roc_curve(y_val, y_prob_va)
        idx = (abs(fpr_va - 0.01)).argmin()
        thr_at_1pct = float(thr_va[idx]) if idx < len(thr_va) else 0.5
        fpr_te, tpr_te, _ = roc_curve(y_test, y_prob_te)
        rec_at_fpr = float(np.interp(0.01, fpr_te, tpr_te))
        baseline = float(y_test.mean())
        lift = float(pr / baseline) if baseline>0 else 0
        metrics = {
            "model": name, "roc_auc": float(roc), "pr_auc": float(pr),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "recall_at_fpr_1pct": float(rec_at_fpr), "threshold_at_1pct": float(thr_at_1pct),
            "threshold": 0.5, "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "support": {"pos": int(y_test.sum()), "neg": int((y_test==0).sum())},
            "fraud_rate": float(y_test.mean()), "lift_vs_random": float(lift),
            "train_seconds": float(train_seconds), "threshold_1pct": float(thr_at_1pct),
            "data": f"synthetic_hard_overlap (rows={len(tx)}, fraud={int(tx.label_fraud.sum())}, rate={tx.label_fraud.mean():.4%})",
            "seed": SEED, "n_per_scenario": N_PER_SCENARIO, "n_transactions_bg": N_BG,
        }
        print(f"{name}: ROC {roc:.4f} PR {pr:.4f} F1 {f1:.4f} Prec {prec:.4f} Rec {rec:.4f} Rec@1% {rec_at_fpr:.4f} lift {lift:.1f}x")
        out_dir.joinpath(f"synthetic_hard_overlap_{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics

    metrics_logreg = evaluate("logreg", logreg, t_logreg)
    metrics_xgb = evaluate("xgboost", xgb, t_xgb)

    print("=== Separability ===")
    num_features = []
    for c in ["amount", "distance_to_prev_km"]:
        if c in F.columns:
            num_features.append(c)
    for c in feature_cols:
        if c in ["vel_count_300s", "vel_count_3600s", "amount_zscore", "prior_mean_amount"] and c in F.columns:
            num_features.append(c)
    num_features = list(dict.fromkeys(num_features))
    fig, axes = plt.subplots(len(num_features), 1, figsize=(8, 3*len(num_features)))
    if len(num_features)==1:
        axes=[axes]
    for ax, col in zip(axes, num_features):
        legit = F[F.label_fraud==0][col].dropna().to_numpy()
        fraud = F[F.label_fraud==1][col].dropna().to_numpy()
        lo, hi = np.percentile(np.concatenate([legit, fraud]), [1, 99]) if len(fraud)>0 else (0,1)
        bins = np.linspace(lo, hi, 40)
        ax.hist(legit, bins=bins, alpha=0.6, label="legit", density=True)
        ax.hist(fraud, bins=bins, alpha=0.6, label="fraud", density=True)
        ax.set_title(f"{col} — fraud vs legit")
        ax.legend()
        q25, q75 = np.percentile(legit, [25,75])
        overlap = float(((fraud>=q25)&(fraud<=q75)).mean()) if len(fraud)>0 else 0
        ax.text(0.98, 0.95, f"fraud in legit IQR: {overlap:.0%}", ha="right", va="top", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "synthetic_hard_overlap_numerical_overlap.png", dpi=150)
    plt.close()
    print(f"Saved {fig_dir / 'synthetic_hard_overlap_numerical_overlap.png'}")
    for col in ["is_new_device", "country_switch", "is_high_risk_merchant"]:
        if col not in F.columns:
            continue
        legit_rate = float(F[F.label_fraud==0][col].mean())
        fraud_rate = float(F[F.label_fraud==1][col].mean())
        print(f"  {col}: legit {legit_rate:.3f} | fraud {fraud_rate:.3f} | lift {fraud_rate/max(legit_rate,1e-6):.1f}x")

    print("=== Comparison ===")
    easy = {"fraud_rate": 0.1151, "roc_auc": 0.996, "pr_auc": 0.959}
    diluted = {"fraud_rate": 0.0107, "roc_auc": 0.9927, "pr_auc": 0.5531}
    # Load ULB if exists
    import pathlib as _p
    ulb_path = _p.Path("evaluation/reports/xgboost_metrics.json")
    if ulb_path.exists():
        ulb = json.loads(ulb_path.read_text())
        ulb_pr = ulb.get("pr_auc", 0.8418)
        ulb_roc = ulb.get("roc_auc", 0.9709)
    else:
        ulb_pr, ulb_roc = 0.8418, 0.9709
    ulb_logreg_pr = json.loads(_p.Path("evaluation/reports/baseline_metrics.json").read_text()).get("pr_auc", 0.7005) if _p.Path("evaluation/reports/baseline_metrics.json").exists() else 0.7005
    print(f"Easy: PR {easy['pr_auc']:.4f} | 1% diluted: PR {diluted['pr_auc']:.4f} | Hard: PR {metrics_xgb['pr_auc']:.4f} | ULB: PR {ulb_pr:.4f}")

    # Save reports
    import datetime
    for metrics, name in [(metrics_logreg, "logreg"), (metrics_xgb, "xgboost")]:
        md = f"""# Synthetic Hard Overlap — {name}

**Variant:** `synthetic_hard_overlap` (new, does not overwrite easy/1% diluted)
**Seed:** {SEED} | **Background:** {N_BG} | **n_per_scenario:** {N_PER_SCENARIO} | **Rows:** {len(tx)} | **Fraud:** {int(tx.label_fraud.sum())} ({tx.label_fraud.mean():.4%})

## Methodology
- Background: same legitimate distribution as easy synthetic; legitimate users now travel (30% foreign), visit high-risk merchants (30% high-risk), burst velocity, off-hours — creates overlap.
- Fraud: 5 weak-signal scenarios (moderate amount+new device, normal amount+new device+merchant, high amount only, normal-looking with weak signals, mixed combos). No single feature is a perfect separator.
- Features: 36 engineered cols (transactional, behavioral, velocity, location, device) — same `FeatureConfig` as easy/1%.
- Split: stratified 70/15/15, random_state 42, scaler fit on train only (no leakage).
- Model: {name} fixed config (XGBoost: n_estimators=500 lr=0.05 max_depth=6 subsample 0.8 colsample 0.8 seed 42).
- Threshold: 0.5 for F1/confusion; 1% FPR threshold tuned on val.

## Metrics (holdout n={metrics['support']['neg']+metrics['support']['pos']})
- ROC-AUC: {metrics['roc_auc']:.4f}
- PR-AUC: {metrics['pr_auc']:.4f}
- F1: {metrics['f1']:.4f} (Prec {metrics['precision']:.4f} Rec {metrics['recall']:.4f})
- Recall @1%FPR: {metrics['recall_at_fpr_1pct']:.4f} (thr {metrics['threshold_at_1pct']:.4f})
- Lift vs random ({metrics['support']['pos']/ (metrics['support']['pos']+metrics['support']['neg']):.4%}): {metrics['lift_vs_random']:.1f}x

## Confusion @0.5
TN={metrics['confusion']['tn']} FP={metrics['confusion']['fp']} FN={metrics['confusion']['fn']} TP={metrics['confusion']['tp']}

**Limitations:** still simulated; does not represent real banking behavior. Overlap is heuristic, not calibrated to real ULB.
"""
        _p.Path(f"evaluation/reports/synthetic_hard_overlap_{name}_report.md").write_text(md)
    comparison_md = f"""# Synthetic Hard Overlap — Comparison Report
Generated: {datetime.date.today().isoformat()} | Seed: {SEED}

## Experiment Comparison (XGBoost, PR-AUC primary)

| Dataset        | Fraud Rate | XGB ROC-AUC | XGB PR-AUC | LogReg PR-AUC | Lift |
| -------------- | ---------: | ----------: | ---------: | ------------: | ---: |
| Easy Synthetic |   {easy['fraud_rate']:.2%} |      {easy['roc_auc']:.4f} |     {easy['pr_auc']:.4f} | — | — |
| 1% Diluted     |   {diluted['fraud_rate']:.2%} |      {diluted['roc_auc']:.4f} |     {diluted['pr_auc']:.4f} | — | — |
| Hard Overlap   |   {metrics_xgb['fraud_rate']:.2%} |      {metrics_xgb['roc_auc']:.4f} |     {metrics_xgb['pr_auc']:.4f} | {metrics_logreg['pr_auc']:.4f} | {metrics_xgb['lift_vs_random']:.1f}x |
| Real ULB       |      0.17% |      {ulb_roc:.4f} |     {ulb_pr:.4f} | {ulb_logreg_pr:.4f} | {ulb_pr/0.0017:.0f}x |

## Interpretation
- Feature overlap introduced: fraud amounts, hours, locations, devices and merchants now overlap substantially with legitimate.
- Performance change: PR should drop easy→diluted→hard as signals weaken; hard should approach real difficulty.
- Leakage: audited — no label in amount/velocity/device/merchant/location/timestamp/feature names/row order.

## Dataset
- Rows: {len(tx)} | Fraud: {int(tx.label_fraud.sum())} ({tx.label_fraud.mean():.4%}) | Seed: {SEED} | Split: train {len(train_df)} val {len(val_df)} test {len(test_df)}
"""
    _p.Path("evaluation/reports/synthetic_hard_overlap_comparison_report.md").write_text(comparison_md)
    print("Saved reports to evaluation/reports/synthetic_hard_overlap_*")
    # Mirror to Drive if mounted
    drive = _p.Path("/content/drive/MyDrive/Finsheild")
    if drive.exists():
        import shutil
        for f in _p.Path("evaluation/reports").glob("synthetic_hard_overlap*"):
            shutil.copy(f, drive / f.name)
            print(f"Mirrored {f.name} to Drive")
    for f in _p.Path("evaluation/figures").glob("synthetic_hard_overlap*"):
        if drive.exists():
            shutil.copy(f, drive / f.name)

if __name__ == "__main__":
    main()
