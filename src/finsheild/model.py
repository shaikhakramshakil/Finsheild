"""Model registry — single place that maps a name to an unwrapped estimator.

Per the project's ML plan:
  Phase 2: logreg    (interpretable baseline)
  Phase 3: xgboost   (primary supervised classifier)
  Comparison: lightgbm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[], Any]
    # Whether the estimator supports incremental fitting (for checkpoint resume).
    supports_resume: bool
    # Whether the estimator natively exposes predict_proba.
    has_proba: bool


def _build_logreg():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=None,
    )


def _build_lightgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        # class_weight=None — imbalance is handled via threshold tuning on val.
        verbose=-1,
    )


def _build_xgboost():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",  # PR-AUC for early stopping (matches our primary metric)
        tree_method="hist",  # CPU-friendly; switch to "gpu_hist" when GPU available
        random_state=42,
        n_jobs=-1,
        # No scale_pos_weight — imbalance handled via threshold tuning on val.
        verbosity=0,
    )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        name="logreg",
        builder=_build_logreg,
        supports_resume=False,  # sklearn LR refits in one shot
        has_proba=True,
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        builder=_build_lightgbm,
        supports_resume=True,  # LGBM supports init_model= for warm start
        has_proba=True,
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        builder=_build_xgboost,
        supports_resume=False,  # XGBoost sklearn API refits via xgb_model=; not wired in train.py
        has_proba=True,
    ),
}


def build_model(name: str, **overrides: Any) -> Any:
    """Instantiate a model by name. Overrides are passed as kwargs to the builder."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    if not overrides:
        return MODEL_REGISTRY[name].builder()

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        base = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}
        base.update(overrides)
        return LogisticRegression(**base)
    if name == "lightgbm":
        import lightgbm as lgb
        base = {
            "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
            "colsample_bytree": 0.8, "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        base.update(overrides)
        return lgb.LGBMClassifier(**base)
    if name == "xgboost":
        import xgboost as xgb
        base = {
            "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
            "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
            "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "objective": "binary:logistic", "eval_metric": "aucpr",
            "tree_method": "hist", "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        base.update(overrides)
        return xgb.XGBClassifier(**base)
    raise KeyError(f"Unknown model '{name}'")


def list_models() -> list[str]:
    return list(MODEL_REGISTRY)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return P(fraud=1) as a 1-D array. Falls back to decision_function when needed."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X))
    raise RuntimeError(f"Model {type(model).__name__} exposes neither predict_proba nor decision_function")