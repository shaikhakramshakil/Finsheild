"""Phase 11 — SHAP Explainability tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsheild.explain import explain_batch, explain_transaction, top_evidence
from finsheild.explain.explainer import evidence_from_features
from finsheild.features import build_features
from finsheild.model import build_model
from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def result(env):
    return build_features(env)


@pytest.fixture(scope="module")
def trained(result):
    X = result.features[result.feature_columns].to_numpy(dtype=float)
    y = result.features["label_fraud"].to_numpy(dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    model = build_model("xgboost", n_estimators=30, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    return model


# ------------------------------------------------------------------ #
# shape & type contracts
# ------------------------------------------------------------------ #

def test_explain_transaction_returns_dict_shape(result, trained):
    row = result.features.iloc[0]
    out = explain_transaction(trained, row, result.feature_columns)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(result.feature_columns)
    assert len(out) == len(result.feature_columns)
    for v in out.values():
        assert isinstance(v, float)
        assert np.isfinite(v)


def test_explain_transaction_accepts_dict_and_ndarray(result, trained):
    row_dict = result.features.iloc[1][result.feature_columns].to_dict()
    out1 = explain_transaction(trained, row_dict, result.feature_columns)
    assert set(out1.keys()) == set(result.feature_columns)

    arr = result.features.iloc[2][result.feature_columns].to_numpy(dtype=float)
    out2 = explain_transaction(trained, arr, result.feature_columns)
    assert len(out2) == len(result.feature_columns)

    # DataFrame single row
    df_row = result.features.iloc[[3]][result.feature_columns]
    out3 = explain_transaction(trained, df_row, result.feature_columns)
    assert len(out3) == len(result.feature_columns)


def test_explain_batch_returns_dataframe(result, trained):
    X_df = result.features[result.feature_columns].iloc[:10]
    shap_df = explain_batch(trained, X_df, result.feature_columns)
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == (10, len(result.feature_columns))
    assert list(shap_df.columns) == result.feature_columns
    # index preserved
    pd.testing.assert_index_equal(shap_df.index, X_df.index)
    assert np.all(np.isfinite(shap_df.to_numpy()))

    # ndarray input
    X_np = X_df.to_numpy(dtype=float)
    shap_df2 = explain_batch(trained, X_np, result.feature_columns)
    assert shap_df2.shape == (10, len(result.feature_columns))
    assert list(shap_df2.columns) == result.feature_columns


def test_top_evidence_sorted_by_abs(result, trained):
    row = result.features.iloc[0]
    sv = explain_transaction(trained, row, result.feature_columns)
    top3 = top_evidence(sv, result.feature_columns, k=3)
    assert len(top3) == 3
    # sorted descending by abs
    abs_vals = [abs(v) for _, v in top3]
    assert abs_vals == sorted(abs_vals, reverse=True)
    # all features are valid
    for feat, val in top3:
        assert feat in result.feature_columns
        assert isinstance(val, float)

    # test with array input
    arr = np.array(list(sv.values()), dtype=float)
    top_arr = top_evidence(arr, result.feature_columns, k=3)
    assert len(top_arr) == 3
    assert top_arr[0][0] == top3[0][0]  # same ordering for same values

    # k larger than features
    top_all = top_evidence(sv, result.feature_columns, k=100)
    assert len(top_all) == len(result.feature_columns)


# ------------------------------------------------------------------ #
# grounded evidence
# ------------------------------------------------------------------ #

def test_evidence_from_features_grounded(result):
    # Pick a row that has multiple risk indicators to verify grounding
    df = result.features
    # Find a row with velocity burst or new device
    cand = df[(df["vel_count_300s"] >= 3) | (df["is_new_device"] == 1) | (df["is_unusual_location"] == 1)]
    if len(cand) == 0:
        cand = df
    row = cand.iloc[0]
    evidence = evidence_from_features(row)
    assert isinstance(evidence, list)
    for e in evidence:
        assert isinstance(e, str) and len(e) > 0

    # Every numeric value mentioned in evidence must come from the actual row
    # Check velocity burst grounding
    vc5 = int(row["vel_count_300s"])
    if vc5 >= 3:
        assert any(str(vc5) in ev and "5min" in ev for ev in evidence), f"vc5={vc5} not grounded: {evidence}"
    if row["is_new_device"] == 1:
        assert any("new device" in ev for ev in evidence)
    if row["is_unusual_location"] == 1:
        assert any("unusual location" in ev for ev in evidence)
    if row["country_switch"] == 1:
        assert any("country switch" in ev for ev in evidence)
    # distance grounding
    dist = float(row["distance_to_prev_km"])
    if dist > 100:
        assert any("distance" in ev for ev in evidence)
        # check the numeric distance is not invented: evidence should contain round(dist)
        assert any(str(int(round(dist))) in ev or f"{dist:.0f}" in ev for ev in evidence)


def test_evidence_from_features_no_invention_for_benign(result):
    # Construct a completely benign row (all zeros / no flags)
    benign = {c: 0.0 for c in result.feature_columns}
    benign["prior_mean_amount"] = 100.0
    benign["amount_log"] = float(np.log1p(50.0))
    benign["amount_zscore"] = 0.1
    benign["amount_log_ratio"] = 0.05
    benign["hour"] = 12
    # Ensure no flags fire
    benign["is_new_device"] = 0
    benign["is_unusual_location"] = 0
    benign["country_switch"] = 0
    benign["is_high_value"] = 0
    benign["is_offhours"] = 0
    benign["is_high_risk_merchant"] = 0
    benign["is_new_user"] = 0
    benign["device_is_shared"] = 0
    benign["vel_count_300s"] = 0
    benign["vel_count_3600s"] = 0
    benign["vel_high_value_count_3600s"] = 0
    benign["distance_to_prev_km"] = 0.0
    benign["is_new_device"] = 0

    ev = evidence_from_features(benign)
    # Should be empty or minimal — must not contain invented high-risk tokens
    assert "new device" not in ev
    assert not any("velocity burst" in s for s in ev)
    assert not any("unusual location" in s for s in ev)
    assert not any("high-value" in s for s in ev)


def test_evidence_amount_ratio_grounded(result):
    # Craft a row with known amount ratio 5x
    row = {c: 0 for c in result.feature_columns}
    row["prior_mean_amount"] = 20.0
    row["amount_log"] = float(np.log1p(100.0))  # 100/20 = 5x
    row["amount_zscore"] = 4.5
    row["amount_log_ratio"] = float(np.log1p(100.0) - np.log1p(20.0))
    row["hour"] = 10
    ev = evidence_from_features(row)
    assert any("5.0x" in s or "5x" in s for s in ev), f"expected 5x ratio in {ev}"
    assert any("z-score" in s for s in ev)


# ------------------------------------------------------------------ #
# high-risk positive SHAP
# ------------------------------------------------------------------ #

def test_high_risk_features_positive_shap_for_fraud(result, trained):
    # Find fraud rows vs legit rows and compare SHAP
    fraud_df = result.features[result.features["label_fraud"] == 1]
    legit_df = result.features[result.features["label_fraud"] == 0]
    assert len(fraud_df) > 0 and len(legit_df) > 0

    # Use a fraud row that actually has risk flags
    fraud_flagged = fraud_df[
        (fraud_df["is_new_device"] == 1)
        | (fraud_df["is_unusual_location"] == 1)
        | (fraud_df["vel_count_300s"] >= 1)
    ]
    if len(fraud_flagged) == 0:
        fraud_flagged = fraud_df
    fraud_row = fraud_flagged.iloc[0]
    fraud_sv = explain_transaction(trained, fraud_row, result.feature_columns)

    # At least one known risk feature should have positive contribution for fraud
    risk_features = [
        "is_new_device",
        "is_unusual_location",
        "country_switch",
        "vel_count_300s",
        "vel_count_3600s",
        "is_high_value",
        "is_high_risk_merchant",
        "distance_to_prev_km",
    ]
    risk_available = [c for c in risk_features if c in result.feature_columns]
    positive_count = sum(1 for c in risk_available if fraud_sv.get(c, 0.0) > 0)
    # Require at least one positive risk SHAP, or sum of risk SHAPs positive
    risk_sum = sum(fraud_sv.get(c, 0.0) for c in risk_available)
    assert positive_count >= 1 or risk_sum > 0, (
        f"Expected positive SHAP for risk features on fraud row: { {c: fraud_sv[c] for c in risk_available} }"
    )

    # Top evidence for fraud should contain a risk feature
    top = top_evidence(fraud_sv, result.feature_columns, k=5)
    top_feats = {f for f, _ in top}
    assert len(top_feats & set(risk_available)) >= 1, f"top {top} missing risk feature"


# ------------------------------------------------------------------ #
# fallback
# ------------------------------------------------------------------ #

def test_fallback_works_when_shap_missing(result, trained, monkeypatch):
    # Force fallback path
    monkeypatch.setattr("finsheild.explain.explainer._HAS_SHAP", False)
    row = result.features.iloc[0]
    out = explain_transaction(trained, row, result.feature_columns)
    assert isinstance(out, dict)
    assert len(out) == len(result.feature_columns)
    assert all(np.isfinite(v) for v in out.values())

    # batch fallback
    X_df = result.features[result.feature_columns].iloc[:5]
    batch = explain_batch(trained, X_df, result.feature_columns)
    assert batch.shape == (5, len(result.feature_columns))

    # fallback model without feature_importances_ should still work
    class DummyModel:
        pass

    dummy = DummyModel()
    out2 = explain_transaction(dummy, row, result.feature_columns)
    assert len(out2) == len(result.feature_columns)
    batch2 = explain_batch(dummy, X_df.iloc[:2].to_numpy(), result.feature_columns)
    assert batch2.shape == (2, len(result.feature_columns))


def test_handles_nan_and_inf(result, trained):
    row = result.features.iloc[0].copy()
    # inject inf/nan in a couple of fields (zscore often inf)
    row["amount_zscore"] = np.inf
    row["prior_std_amount"] = np.nan
    row["vel_amount_300s"] = np.inf
    out = explain_transaction(trained, row, result.feature_columns)
    assert all(np.isfinite(v) for v in out.values())

    # evidence_from_features should handle NaN gracefully
    ev = evidence_from_features(row)
    assert isinstance(ev, list)

    # batch with NaN/inf
    X_df = result.features[result.feature_columns].iloc[:3].copy()
    X_df.iloc[0, X_df.columns.get_loc("amount_zscore")] = np.nan
    X_df.iloc[1, X_df.columns.get_loc("amount_zscore")] = np.inf
    shap_df = explain_batch(trained, X_df, result.feature_columns)
    assert np.all(np.isfinite(shap_df.to_numpy()))


def test_evidence_from_features_accepts_dataframe(result):
    df_row = result.features.iloc[[0]]
    ev = evidence_from_features(df_row)
    assert isinstance(ev, list)
    # also Series
    s = result.features.iloc[0]
    ev2 = evidence_from_features(s)
    assert isinstance(ev2, list)
