"""Evaluation metrics + plotting for binary fraud detection.

Primary metric: PR-AUC (average_precision_score).
Also reported: ROC-AUC, recall@FPR=target, and precision/recall/F1 at the val-tuned threshold.
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
    recall_at_target_fpr: float
    target_fpr: float
    threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
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
    # fpr/tpr are sorted ascending by threshold; first threshold with fpr <= target
    mask = fpr <= target_fpr
    if not mask.any():
        return float(tpr[0]), float("inf")
    idx = np.argmax(tpr[mask])  # highest recall among thresholds meeting FPR target
    # Re-index into the original arrays
    true_indices = np.where(mask)[0]
    chosen = true_indices[idx]
    return float(tpr[chosen]), float(thr[chosen])


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> tuple[float, float, float, float]:
    """Pick threshold that maximizes recall while keeping FPR <= target_fpr on val.

    Returns (threshold, precision, recall, f1) at the chosen threshold.
    """
    from sklearn.metrics import precision_recall_fscore_support
    # Try thresholds at every observed score + a couple of sentinels.
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


def evaluate(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pra = pr_auc(y_true, y_score)
    rau = roc_auc(y_true, y_score)
    recall_fpr, _thr_fpr = recall_at_fpr(y_true, y_score, target_fpr)
    thr, prec, rec, f1 = tune_threshold(y_true, y_score, target_fpr)
    return EvalResult(
        pr_auc=pra,
        roc_auc=rau,
        recall_at_target_fpr=recall_fpr,
        target_fpr=target_fpr,
        threshold=thr,
        precision_at_threshold=prec,
        recall_at_threshold=rec,
        f1_at_threshold=f1,
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