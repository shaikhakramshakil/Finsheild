"""Preprocessing — scaler fit only on train (leakage-safe)."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """Scaler for Amount/Time with strict train-only fitting.

    Usage:
        pre = FraudPreprocessor(scale_features=["Amount", "Time"])
        train_scaled = pre.fit_transform_train(train_df)
        val_scaled = pre.transform(val_df)
        test_scaled = pre.transform(test_df)
    """

    def __init__(self, scale_features: list[str] | None = None):
        if scale_features is None:
            scale_features = ["Amount", "Time"]
        self.scale_features = list(scale_features)
        self.scaler = StandardScaler()
        self._fitted = False
        self._feature_order = None

    def fit(self, df: pd.DataFrame) -> FraudPreprocessor:
        missing = set(self.scale_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing scale features in fit data: {missing}")
        self.scaler.fit(df[self.scale_features])
        self._fitted = True
        self._feature_order = list(df.columns)
        logger.info(
            "Fitted StandardScaler on %d rows; mean_= %s var_= %s",
            len(df),
            self.scaler.mean_.tolist(),
            self.scaler.var_.tolist(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted — call fit() on train first")
        missing = set(self.scale_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing scale features in transform data: {missing}")
        out = df.copy()
        out[self.scale_features] = self.scaler.transform(df[self.scale_features])
        return out

    def fit_transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scaler": self.scaler, "scale_features": self.scale_features}
        joblib.dump(payload, path)
        logger.info("Saved scaler to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> FraudPreprocessor:
        payload = joblib.load(path)
        obj = cls(scale_features=payload["scale_features"])
        obj.scaler = payload["scaler"]
        obj._fitted = True
        return obj

    def get_pipeline(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), self.scale_features)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return Pipeline([("preprocess", ct)])


def preprocess_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_features: list[str] | None = None,
    save_scaler_path: str | Path | None = None,
):
    pre = FraudPreprocessor(scale_features=scale_features)
    train_t = pre.fit_transform_train(train_df)
    val_t = pre.transform(val_df)
    test_t = pre.transform(test_df)
    if save_scaler_path:
        pre.save(save_scaler_path)
    return train_t, val_t, test_t, pre
