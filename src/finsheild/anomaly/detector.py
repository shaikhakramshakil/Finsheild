"""Anomaly detection — Phase 7 (Isolation Forest).

Wraps :class:`sklearn.ensemble.IsolationForest` with FinSheild conventions:

* Trained **only** on legitimate / background transactions.
* :meth:`AnomalyDetector.score_samples` returns a calibrated score in
  ``[0, 1]`` where ``1`` = most anomalous (monotonic with the raw
  Isolation Forest score).
* Handles ``NaN`` / ``inf`` produced by behavioural features for new
  users (imputed with training medians).
* Persists via :mod:`joblib` (``save`` / ``load``).

Helper functions :func:`train_anomaly_detector` and
:func:`score_transactions` provide ergonomic entry-points on top of
``FeatureBuildResult`` (Phase 5) so callers do not need to manually
slice the feature matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Type alias for the Phase-5 result — imported lazily to avoid cycles.
_FeatureBuildResult = object  # runtime duck-typed


class AnomalyDetector:
    """Isolation-Forest wrapper exposing a ``[0, 1]`` anomaly score.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies.  Fixed to ``0.05`` per Phase 7
        spec; exposed as a parameter for testability.
    random_state:
        RNG seed.  Fixed to ``42`` per spec.
    feature_columns:
        Ordered list of feature names.  Captured at :meth:`fit` time if
        not supplied up-front and exposed publicly for downstream
        consumers (risk fusion, explainability).
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
        feature_columns: list[str] | None = None,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.feature_columns: list[str] | None = list(feature_columns) if feature_columns is not None else None
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._medians: np.ndarray | None = None
        self._raw_min: float | None = None
        self._raw_max: float | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _coerce_to_array(self, X) -> tuple[np.ndarray, list[str] | None]:
        """Return ``(array, columns)`` from flexible input.

        Accepts ``np.ndarray``, ``pd.DataFrame``, or
        ``FeatureBuildResult`` (duck-typed: has ``.features`` and
        ``.feature_columns``).  Columns are returned when they can be
        inferred.
        """
        # FeatureBuildResult duck-type
        if hasattr(X, "features") and hasattr(X, "feature_columns"):
            # X is a FeatureBuildResult
            cols = list(X.feature_columns)  # type: ignore[attr-defined]
            # If detector already has columns, respect them; else use result's
            use_cols = self.feature_columns if self.feature_columns is not None else cols
            arr = X.features[use_cols].to_numpy(dtype="float64")  # type: ignore[attr-defined]
            return arr, use_cols

        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            # If detector knows its columns, slice to those.
            if self.feature_columns is not None:
                # allow subset — caller may pass full DataFrame
                available = [c for c in self.feature_columns if c in X.columns]
                if available:
                    arr = X[available].to_numpy(dtype="float64")
                    return arr, available
            arr = X.to_numpy(dtype="float64")
            return arr, cols

        # ndarray / list-like
        arr = np.asarray(X, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr, None

    def _impute(self, arr: np.ndarray, fit: bool = False) -> np.ndarray:
        """Replace ``NaN`` / ``inf`` with training medians.

        When *fit* is True, compute and store medians.
        """
        # Convert inf → nan so nanmedian handles it.
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan

        if fit:
            # nanmedian may warn on all-nan slice; suppress by handling.
            with np.errstate(all="ignore"):
                medians = np.nanmedian(arr, axis=0)
            # Columns that are all-nan → median is nan → replace with 0.
            medians = np.where(np.isnan(medians), 0.0, medians)
            self._medians = medians
        else:
            if self._medians is None:
                raise RuntimeError("AnomalyDetector has not been fitted yet.")
            medians = self._medians
            # Handle width mismatch (e.g. caller passes fewer columns) — fallback to 0
            if medians.shape[0] != arr.shape[1]:
                # Re-compute medians for this shape as zeros if mismatch;
                # this keeps the call from crashing; training code always
                # matches shape.
                medians = np.zeros(arr.shape[1], dtype=float)

        # Fill NaNs
        mask = np.isnan(arr)
        if np.any(mask):
            # np.take handles per-column median broadcast
            # for each nan position, fill with median of that column
            rows, cols = np.where(mask)
            arr[rows, cols] = medians[cols]
        return arr

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, feature_columns: list[str] | None = None) -> "AnomalyDetector":
        """Fit the Isolation Forest on *legitimate* transactions.

        Parameters
        ----------
        X:
            Feature matrix.  May be ``np.ndarray``, ``pd.DataFrame``, or
            ``FeatureBuildResult``.  When a ``FeatureBuildResult`` or
            ``DataFrame`` is passed, ``feature_columns`` are inferred if
            not already set on the detector.
        feature_columns:
            Optional override for the column names.  When supplied it
            becomes the detector's ``feature_columns``.

        Returns
        -------
        self
        """
        if feature_columns is not None:
            self.feature_columns = list(feature_columns)

        arr, inferred_cols = self._coerce_to_array(X)

        if self.feature_columns is None and inferred_cols is not None:
            self.feature_columns = list(inferred_cols)

        arr_imputed = self._impute(arr, fit=True)

        self.model.fit(arr_imputed)
        self._is_fitted = True

        # Cache training raw-score range for stable [0,1] calibration.
        # raw = -score_samples → larger = more anomalous
        raw = -self.model.score_samples(arr_imputed)
        self._raw_min = float(np.min(raw))
        self._raw_max = float(np.max(raw))

        return self

    def predict(self, X) -> np.ndarray:
        """Return Isolation Forest labels: ``1`` = inlier, ``-1`` = outlier.

        Parameters
        ----------
        X:
            Feature matrix (same flexible types as :meth:`fit`).
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector has not been fitted yet — call fit() first.")
        arr, _ = self._coerce_to_array(X)
        arr_imputed = self._impute(arr, fit=False)
        return self.model.predict(arr_imputed)

    def score_samples(self, X) -> np.ndarray:
        """Return calibrated anomaly scores in ``[0, 1]``.

        ``1`` = most anomalous, ``0`` = most normal.  The score is
        monotonic with the underlying Isolation Forest anomaly score.

        Normalisation uses the training-set raw-score range when
        available (so scores are comparable across calls); otherwise
        falls back to per-batch min-max with clipping.

        Parameters
        ----------
        X:
            Feature matrix (same flexible types as :meth:`fit`).
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyDetector has not been fitted yet — call fit() first.")
        arr, _ = self._coerce_to_array(X)
        arr_imputed = self._impute(arr, fit=False)
        raw = -self.model.score_samples(arr_imputed)

        # Prefer training-range normalisation for stability.
        if self._raw_min is not None and self._raw_max is not None and self._raw_max > self._raw_min:
            norm = (raw - self._raw_min) / (self._raw_max - self._raw_min + 1e-9)
            norm = np.clip(norm, 0.0, 1.0)
            return norm.astype(float)

        # Fallback: batch min-max
        rmin, rmax = float(np.min(raw)), float(np.max(raw))
        if rmax > rmin:
            norm = (raw - rmin) / (rmax - rmin + 1e-9)
            return np.clip(norm, 0.0, 1.0).astype(float)
        # Constant batch → neutral score
        return np.full_like(raw, 0.5, dtype=float)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> Path:
        """Persist the detector (including sklearn model) via joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        """Load a detector saved with :meth:`save`."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            # Allow loading plain IsolationForest dumps by wrapping — but
            # the canonical path is dumping the whole AnomalyDetector.
            raise TypeError(f"Loaded object is {type(obj).__name__}, expected AnomalyDetector")
        return obj

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        fitted = "fitted" if self._is_fitted else "unfitted"
        cols = len(self.feature_columns) if self.feature_columns else "?"
        return (
            f"AnomalyDetector(contamination={self.contamination}, "
            f"random_state={self.random_state}, n_features={cols}, {fitted})"
        )


# ---------------------------------------------------------------------- #
# Helper functions operating on Phase-5 FeatureBuildResult
# ---------------------------------------------------------------------- #

def _extract_legit_matrix(
    result,
    feature_columns: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Return ``(X_legit, columns)`` from a ``FeatureBuildResult`` or ``DataFrame``.

    Legitimacy filter: ``label_fraud == 0`` when that column exists;
    otherwise all rows are treated as legitimate (the caller is assumed
    to have pre-filtered).
    """
    # FeatureBuildResult duck-type
    if hasattr(result, "features") and hasattr(result, "feature_columns"):
        cols = list(feature_columns) if feature_columns is not None else list(result.feature_columns)  # type: ignore[attr-defined]
        df: pd.DataFrame = result.features  # type: ignore[attr-defined]
        if "label_fraud" in df.columns:
            legit = df[df["label_fraud"] == 0]
        else:
            legit = df
        X = legit[cols].to_numpy(dtype="float64")
        return X, cols

    if isinstance(result, pd.DataFrame):
        df = result
        if feature_columns is not None:
            cols = list(feature_columns)
        else:
            # Use all numeric columns except known non-feature columns
            exclude = {"label_fraud", "scenario_tag", "txn_id", "account_id", "ts", "amount"}
            cols = [c for c in df.columns if c not in exclude]
        if "label_fraud" in df.columns:
            legit = df[df["label_fraud"] == 0]
        else:
            legit = df
        X = legit[cols].to_numpy(dtype="float64")
        return X, cols

    # ndarray — assume already legit-only
    arr = np.asarray(result, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    cols = list(feature_columns) if feature_columns is not None else [f"f{i}" for i in range(arr.shape[1])]
    return arr, cols


def _extract_matrix_for_scoring(detector: AnomalyDetector, data) -> np.ndarray:
    """Return matrix to score from flexible *data*."""
    # FeatureBuildResult
    if hasattr(data, "features") and hasattr(data, "feature_columns"):
        df: pd.DataFrame = data.features  # type: ignore[attr-defined]
        cols = detector.feature_columns if detector.feature_columns is not None else list(data.feature_columns)  # type: ignore[attr-defined]
        # Guard: if detector cols missing, fall back to data cols intersection
        use_cols = [c for c in cols if c in df.columns]
        if not use_cols:
            use_cols = cols
        return df[use_cols].to_numpy(dtype="float64")

    if isinstance(data, pd.DataFrame):
        df = data
        if detector.feature_columns is not None:
            use_cols = [c for c in detector.feature_columns if c in df.columns]
            if use_cols:
                return df[use_cols].to_numpy(dtype="float64")
            # fallback: try feature_columns directly (may raise KeyError if missing)
            return df[detector.feature_columns].to_numpy(dtype="float64")
        return df.to_numpy(dtype="float64")

    arr = np.asarray(data, dtype="float64")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def train_anomaly_detector(
    result,
    feature_columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyDetector:
    """Train an :class:`AnomalyDetector` on legitimate transactions.

    Parameters
    ----------
    result:
        A :class:`FeatureBuildResult` (``build_features`` output), a
        ``DataFrame`` containing at least ``label_fraud`` + feature
        columns, or a raw ``ndarray`` that is already legit-only.
    feature_columns:
        Optional override for the feature column list.  When *result*
        is a ``FeatureBuildResult`` and this is ``None``, the result's
        own ``feature_columns`` are used.
    contamination, random_state:
        Passed to :class:`AnomalyDetector` (defaults match spec).

    Returns
    -------
    AnomalyDetector
        Fitted detector.
    """
    X_legit, cols = _extract_legit_matrix(result, feature_columns)
    detector = AnomalyDetector(
        contamination=contamination,
        random_state=random_state,
        feature_columns=cols,
    )
    detector.fit(X_legit)
    return detector


def score_transactions(detector: AnomalyDetector, data) -> np.ndarray:
    """Score transactions with a fitted detector.

    Parameters
    ----------
    detector:
        Fitted :class:`AnomalyDetector`.
    data:
        ``FeatureBuildResult``, ``DataFrame``, or ``ndarray`` to score.
        When a ``FeatureBuildResult`` or ``DataFrame`` is supplied, the
        detector's ``feature_columns`` are used to select the correct
        columns automatically.

    Returns
    -------
    np.ndarray
        1-D array of anomaly scores in ``[0, 1]`` where ``1`` = most
        anomalous, aligned with the rows of *data*.
    """
    X = _extract_matrix_for_scoring(detector, data)
    return detector.score_samples(X)
