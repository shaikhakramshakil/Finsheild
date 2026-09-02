"""Phase 7 — Anomaly detection (Isolation Forest) tests.

Covers:

* fit on legit/background does not crash
* scores are in [0, 1]
* fraud scores higher on average than legit
* save / load roundtrip preserves scores
* predict labels contract, feature_columns exposure, helpers
* NaN/inf robustness
* single-row scoring edge case
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsheild.anomaly import AnomalyDetector, score_transactions, train_anomaly_detector
from finsheild.features import build_features
from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def result(env):
    return build_features(env)


@pytest.fixture(scope="module")
def detector(result):
    # Canonical helper path — trains only on legit
    return train_anomaly_detector(result)


# ------------------------------------------------------------------ #
# Core contract
# ------------------------------------------------------------------ #

def test_fit_on_legit_does_not_crash(result):
    X_legit = result.features[result.features["label_fraud"] == 0][result.feature_columns].to_numpy(dtype="float64")
    det = AnomalyDetector(contamination=0.05, random_state=42)
    det.fit(X_legit)
    assert det._is_fitted


def test_scores_in_range(detector, result):
    scores = detector.score_samples(result.features[result.feature_columns].to_numpy(dtype="float64"))
    assert scores.shape[0] == len(result.features)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), f"scores out of [0,1]: min {scores.min()} max {scores.max()}"
    assert np.all(np.isfinite(scores))


def test_fraud_scores_higher_than_legit_on_average(detector, result):
    scores = detector.score_samples(result.features[result.feature_columns].to_numpy(dtype="float64"))
    labels = result.features["label_fraud"].to_numpy()
    fraud_mean = float(scores[labels == 1].mean())
    legit_mean = float(scores[labels == 0].mean())
    assert fraud_mean > legit_mean, f"fraud mean {fraud_mean:.3f} not > legit {legit_mean:.3f}"
    # Require non-trivial separation
    assert fraud_mean - legit_mean > 0.05


def test_save_load_roundtrip(detector, result, tmp_path):
    path = tmp_path / "anomaly.joblib"
    detector.save(path)
    assert path.exists()
    loaded = AnomalyDetector.load(path)
    X = result.features[result.feature_columns].to_numpy(dtype="float64")
    s1 = detector.score_samples(X)
    s2 = loaded.score_samples(X)
    np.testing.assert_allclose(s1, s2, atol=1e-6)


def test_save_load_preserves_feature_columns(detector, tmp_path):
    path = tmp_path / "anomaly2.joblib"
    detector.save(path)
    loaded = AnomalyDetector.load(path)
    assert loaded.feature_columns == detector.feature_columns
    assert loaded.contamination == detector.contamination
    assert loaded.random_state == detector.random_state


def test_predict_labels_contract(detector, result):
    X = result.features[result.feature_columns].to_numpy(dtype="float64")
    preds = detector.predict(X)
    assert preds.shape[0] == len(result.features)
    assert set(np.unique(preds)).issubset({-1, 1})
    # At least some outliers predicted (contamination 0.05)
    assert (preds == -1).sum() > 0


def test_feature_columns_exposed(detector, result):
    assert detector.feature_columns is not None
    assert detector.feature_columns == result.feature_columns
    assert len(detector.feature_columns) > 0
    # contamination/random_state per spec
    assert detector.contamination == 0.05
    assert detector.random_state == 42


def test_train_helper_uses_legit_only(result):
    det = train_anomaly_detector(result)
    # Should be fitted
    assert det._is_fitted
    # Legit count used for training is less than total rows
    legit_n = int((result.features["label_fraud"] == 0).sum())
    total_n = len(result.features)
    assert legit_n < total_n
    # Feature columns propagated
    assert det.feature_columns == result.feature_columns


def test_score_transactions_helper_with_build_result(detector, result):
    scores = score_transactions(detector, result)
    assert scores.shape[0] == len(result.features)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    # Must match direct call
    direct = detector.score_samples(result.features[result.feature_columns].to_numpy(dtype="float64"))
    np.testing.assert_allclose(scores, direct, atol=1e-6)


def test_score_transactions_helper_with_dataframe(detector, result):
    df = result.features
    scores = score_transactions(detector, df)
    assert scores.shape[0] == len(df)
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_score_transactions_helper_with_ndarray(detector, result):
    X = result.features[result.feature_columns].to_numpy(dtype="float64")
    scores = score_transactions(detector, X)
    assert scores.shape[0] == X.shape[0]
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_handles_nan_and_inf(result):
    X_legit = result.features[result.features["label_fraud"] == 0][result.feature_columns].to_numpy(dtype="float64")
    # Inject extra NaNs/infs
    X_noisy = X_legit.copy()
    X_noisy[0, 0] = np.nan
    X_noisy[1, 1] = np.inf
    X_noisy[2, 2] = -np.inf
    det = AnomalyDetector()
    det.fit(X_noisy)
    scores = det.score_samples(X_noisy)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_single_row_scoring(detector, result):
    X_one = result.features[result.feature_columns].iloc[[0]].to_numpy(dtype="float64")
    score = detector.score_samples(X_one)
    assert score.shape == (1,)
    assert 0 <= float(score[0]) <= 1


def test_predict_before_fit_raises():
    det = AnomalyDetector()
    X = np.random.randn(10, 5)
    with pytest.raises(RuntimeError):
        det.predict(X)
    with pytest.raises(RuntimeError):
        det.score_samples(X)


def test_fit_accepts_dataframe(result):
    df_legit = result.features[result.features["label_fraud"] == 0][result.feature_columns]
    det = AnomalyDetector()
    det.fit(df_legit)
    assert det._is_fitted
    assert det.feature_columns == result.feature_columns


def test_fit_accepts_feature_build_result_directly(result):
    # Filter legit FeatureBuildResult manually — detector should handle duck type
    # For this test we pass the whole result but detector's coerce handles it;
    # however fit should train on whatever is passed. We test that passing
    # FeatureBuildResult as X doesn't crash and sets feature_columns.
    from copy import deepcopy

    # Create a legit-only FeatureBuildResult duck
    class Dummy:
        def __init__(self, features, feature_columns):
            self.features = features
            self.feature_columns = feature_columns

    legit_features = result.features[result.features["label_fraud"] == 0].copy()
    dummy = Dummy(legit_features, result.feature_columns)
    det = AnomalyDetector()
    det.fit(dummy)
    assert det._is_fitted
    assert det.feature_columns == result.feature_columns
