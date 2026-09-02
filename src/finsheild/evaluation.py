"""Evaluation metrics + plotting for binary fraud detection.

Per the project's ML plan (Phase 2): precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
Also reported for operational use: recall at target FPR + precision/recall/F1 at the val-tuned threshold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    pr_auc: float
    roc_auc: float
    precision: float          # at the val-tuned threshold
    recall: float             # at the val-tuned threshold
    f1: float                 # at the val-tuned threshold
    recall_at_target_fpr: float
    target_fpr: float
    threshold: float
    confusion_matrix: dict    # {"tn": ..., "fp": ..., "fn": ..., "tp": ...}
    support_pos: int
    support_neg: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Find the smallest threshold giving FPR <= target_fpr, return (recall, threshold).

    If the score distribution cannot meet the FPR target, returns (recall@min_fpr, +inf).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not mask.any():
        return float(tpr[0]), float("inf")
    idx = np.argmax(tpr[mask])
    true_indices = np.where(mask)[0]
    chosen = true_indices[idx]
    return float(tpr[chosen]), float(thr[chosen])


def confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> tuple[float, float, float, float]:
    """Pick threshold that maximizes recall while keeping FPR <= target_fpr on val.

    Returns (threshold, precision, recall, f1) at the chosen threshold.
    """
    from sklearn.metrics import precision_recall_fscore_support
    candidates = np.unique(np.concatenate(([0.0, 1.0], y_score)))
    best = (0.0, 0.0, 0.0, 0.0, 0.0)  # (recall, threshold, precision, f1, fpr)
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fpr = fp / max(1, fp + tn)
        if fpr > target_fpr:
            continue
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if r > best[0]:
            best = (float(r), float(t), float(p), float(f1), float(fpr))
    _, threshold, precision, f1, _ = best
    recall = best[0]
    return threshold, precision, recall, f1


def evaluate(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01, threshold: float | None = None) -> EvalResult:
    """Compute the full Phase 2 metric set.

    If `threshold` is None, it's tuned on the same y_true/y_score (typically called on val).
    If a threshold is passed in (test-time), it's reused for the precision/recall/F1 trio.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pra = pr_auc(y_true, y_score)
    rau = roc_auc(y_true, y_score)
    recall_fpr, _thr_fpr = recall_at_fpr(y_true, y_score, target_fpr)

    if threshold is None:
        thr, prec, rec, f1 = tune_threshold(y_true, y_score, target_fpr)
    else:
        thr = float(threshold)
        y_pred = (y_score >= thr).astype(int)
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        prec, rec = float(p), float(r)
        f1 = float(f1)

    cm = confusion_dict(y_true, (y_score >= thr).astype(int))
    return EvalResult(
        pr_auc=pra,
        roc_auc=rau,
        precision=prec,
        recall=rec,
        f1=f1,
        recall_at_target_fpr=recall_fpr,
        target_fpr=target_fpr,
        threshold=thr,
        confusion_matrix=cm,
        support_pos=int(y_true.sum()),
        support_neg=int((1 - y_true).sum()),
    )


def write_metrics(result: EvalResult, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Wrote metrics to %s", out_path)
    return out_path


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str = "Precision-Recall") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str = "ROC") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: dict[str, int], out_path: Path, title: str = "Confusion Matrix") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path