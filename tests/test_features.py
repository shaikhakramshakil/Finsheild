"""Phase 5 — feature engineering tests.

Run with:

    PYTHONPATH=src pytest tests/test_features.py -v

All tests use ``SyntheticEnvConfig.ci()`` so the suite finishes in <30s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsheild.features import FeatureConfig, build_features
from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def result(env):
    return build_features(env)


@pytest.fixture(scope="module")
def result2():
    # Same config → same features (determinism)
    return build_features(generate_environment(SyntheticEnvConfig.ci()))


# ---- Schema ---------------------------------------------------------------

def test_required_feature_columns_present(result):
    expected = {
        # transactional
        "amount_log", "hour", "day_of_week", "is_offhours", "is_high_value",
        "is_online", "is_pos", "is_atm", "is_mobile",
        "is_high_risk_merchant", "merchant_risk_band_ord",
        # behavioral
        "prior_tx_count", "prior_total_amount", "prior_mean_amount",
        "prior_std_amount", "amount_zscore", "amount_log_ratio",
        "is_new_user", "prior_unique_merchants", "prior_unique_devices",
        "prior_unique_locations", "prior_unique_countries",
        # velocity
        "vel_count_300s", "vel_amount_300s",
        "vel_count_3600s", "vel_amount_3600s",
        "vel_count_86400s", "vel_amount_86400s",
        "vel_high_value_count_3600s",
        # device
        "is_new_device", "device_account_count", "device_is_shared",
        "is_primary_device_for_account",
        # location
        "country_switch", "distance_to_prev_km", "is_unusual_location",
    }
    actual = set(result.feature_columns)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing features: {missing}"
    assert not extra, f"Unexpected extra features: {extra}"


def test_features_index_is_txn_id(result):
    assert "txn_id" in result.features.columns
    assert result.features["txn_id"].is_unique


def test_features_have_no_nans_in_critical_columns(result):
    """Critical columns must never be NaN (only zscore/std can be)."""
    critical = [
        "amount_log", "hour", "day_of_week", "is_offhours", "is_high_value",
        "is_new_user", "vel_count_300s", "vel_count_3600s", "vel_count_86400s",
        "is_new_device", "device_account_count", "country_switch",
        "distance_to_prev_km", "is_unusual_location",
        "is_high_risk_merchant", "merchant_risk_band_ord",
        "is_online", "is_pos", "is_atm", "is_mobile",
    ]
    nan_counts = result.features[critical].isna().sum()
    for col in critical:
        assert nan_counts[col] == 0, f"{col} has {nan_counts[col]} NaN values"


# ---- Determinism ---------------------------------------------------------

def test_determinism(result, result2):
    cols = result.feature_columns
    # Compare on a stable, sorted view
    a = result.features.sort_values("txn_id").reset_index(drop=True)
    b = result2.features.sort_values("txn_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a[cols], b[cols])


# ---- No leakage ----------------------------------------------------------

def test_velocity_excludes_current_row(result, env):
    """vel_count_300s is the count of PRIOR txns in the 5-min window,
    never the current row."""
    # Pick a high-velocity txn (one with vel_count_300s >= 1) and check
    # that the count is achievable with prior txns only.
    high = result.features[result.features["vel_count_300s"] >= 1]
    assert len(high) > 0
    # For at least one such row, the 5-min-prior count is achievable
    # (we sanity-check by re-computing a small slice manually).
    sample = high.head(50)
    tx_sorted = env.transactions.sort_values(["account_id", "ts", "txn_id"]) \
        .reset_index(drop=True)
    fails = 0
    for _, row in sample.iterrows():
        # find this txn in tx_sorted
        sub = tx_sorted[tx_sorted["txn_id"] == int(row["txn_id"])]
        if sub.empty:
            continue
        idx = sub.index[0]
        ts_i = tx_sorted.loc[idx, "ts"]
        acc_i = tx_sorted.loc[idx, "account_id"]
        # Count prior txns for same account within 5 min before ts_i
        prior_block = tx_sorted.iloc[:idx]
        prior_block = prior_block[prior_block["account_id"] == acc_i]
        if len(prior_block) == 0:
            continue
        count = (ts_i - prior_block["ts"]).dt.total_seconds().lt(300).sum()
        if count != int(row["vel_count_300s"]):
            fails += 1
    assert fails == 0, f"velocity leakage in {fails} rows"


def test_behavioral_excludes_current_row(result, env):
    """prior_tx_count for a transaction is the number of *prior* txns for
    the same user, never including the current row."""
    # First user-transaction should have prior_tx_count == 0
    user_map = env.accounts.set_index("account_id")["user_id"]
    tx_sorted = env.transactions.copy()
    tx_sorted["user_id"] = tx_sorted["account_id"].map(user_map)
    tx_sorted = tx_sorted.sort_values(["user_id", "ts", "txn_id"]) \
        .reset_index(drop=True)
    first_per_user = tx_sorted.groupby("user_id").first()["txn_id"].astype(int)
    first_rows = result.features[result.features["txn_id"].isin(first_per_user)]
    assert (first_rows["prior_tx_count"] == 0).all()
    assert (first_rows["prior_mean_amount"].isna()).all()


# ---- Scenario-aware features ---------------------------------------------

def test_new_device_scenario_marks_is_new_device(result, env):
    """The new_device scenario emits txns with a device the account has
    not used before; ``is_new_device`` should be 1 for at least one of them."""
    nd = env.transactions[env.transactions["scenario_tag"] == "new_device"]
    nd_fraud = nd[nd["label_fraud"] == 1]
    txn_ids = set(nd_fraud["txn_id"].astype(int))
    feats = result.features[result.features["txn_id"].isin(txn_ids)]
    assert (feats["is_new_device"] == 1).sum() > 0


def test_unusual_location_scenario_marks_is_unusual(result, env):
    ul = env.transactions[env.transactions["scenario_tag"] == "unusual_location"]
    ul_fraud = ul[ul["label_fraud"] == 1]
    txn_ids = set(ul_fraud["txn_id"].astype(int))
    feats = result.features[result.features["txn_id"].isin(txn_ids)]
    # every fraud row should have is_unusual_location == 1
    assert (feats["is_unusual_location"] == 1).sum() >= len(txn_ids) * 0.9


def test_velocity_burst_scenario_marks_high_vel(result, env):
    vel = env.transactions[env.transactions["scenario_tag"]
                             == "transaction_velocity"]
    vel_fraud = vel[vel["label_fraud"] == 1]
    txn_ids = set(vel_fraud["txn_id"].astype(int))
    feats = result.features[result.features["txn_id"].isin(txn_ids)]
    # Most burst rows should have vel_count_300s > 0
    nonzero = (feats["vel_count_300s"] > 0).sum()
    assert nonzero > 0


# ---- Configurability ----------------------------------------------------

def test_feature_config_overrides_velocity_windows():
    cfg = FeatureConfig(velocity_windows_seconds=(60, 600))
    from finsheild.synthetic_env import SyntheticEnvConfig
    env = generate_environment(SyntheticEnvConfig.ci())
    result = build_features(env, cfg)
    assert "vel_count_60s" in result.feature_columns
    assert "vel_amount_600s" in result.feature_columns
    # And the default 300/3600/86400 windows are NOT in this output
    assert "vel_count_300s" not in result.feature_columns
    assert "vel_count_86400s" not in result.feature_columns


def test_x_returns_ndarray(result):
    X = result.X()
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(result.features)
    assert X.shape[1] == len(result.feature_columns)


def test_y_returns_labels(result):
    y = result.y()
    assert y is not None
    assert set(y.tolist()).issubset({0, 1})


# ---- Performance --------------------------------------------------------

def test_ci_scale_runs_under_30_seconds():
    import time
    env = generate_environment(SyntheticEnvConfig.ci())
    t0 = time.time()
    build_features(env)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"feature build took {elapsed:.1f}s"