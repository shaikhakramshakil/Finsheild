"""Phase 6 — behavioral profiling tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsheild.behavioral import BehavioralProfile, build_profiles, score_transaction
from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def profiles(env):
    return build_profiles(env.transactions, env.accounts)


# ---- Profile building ---------------------------------------------------

def test_build_profiles_returns_dict(profiles, env):
    assert isinstance(profiles, dict)
    assert len(profiles) > 0
    # keys should be user_ids
    user_ids = set(env.users["user_id"].tolist())
    for uid in profiles:
        assert uid in user_ids


def test_profile_fields_present_and_types(profiles):
    # pick any profile
    profile = next(iter(profiles.values()))
    assert isinstance(profile, BehavioralProfile)
    assert isinstance(profile.user_id, int)
    assert isinstance(profile.mean_amount, float)
    assert isinstance(profile.std_amount, float)
    assert isinstance(profile.median_amount, float)
    assert isinstance(profile.txn_count, int)
    assert profile.txn_count > 0
    # hour_histogram: 24 bins
    hist = np.asarray(profile.hour_histogram)
    assert hist.shape == (24,)
    assert hist.sum() == profile.txn_count
    # common_merchants: <=3
    assert isinstance(profile.common_merchants, list)
    assert len(profile.common_merchants) <= 3
    # known_devices / locations are sets
    assert isinstance(profile.known_devices, set)
    assert isinstance(profile.known_locations, set)
    assert len(profile.known_devices) > 0
    assert len(profile.known_locations) > 0
    # avg daily frequency
    assert isinstance(profile.avg_daily_frequency, float)
    assert profile.avg_daily_frequency > 0


def test_profiles_cover_users_with_transactions(env, profiles):
    # Every user who has at least one transaction should have a profile
    acct_to_user = env.accounts.set_index("account_id")["user_id"].to_dict()
    tx_users = env.transactions["account_id"].map(acct_to_user).dropna().unique()
    for uid in tx_users:
        assert int(uid) in profiles


def test_profile_statistics_reasonable(profiles):
    for p in profiles.values():
        # mean should be positive (amounts are positive)
        assert p.mean_amount > 0
        assert p.median_amount > 0
        assert p.std_amount >= 0
        # std 0 iff single txn
        if p.txn_count == 1:
            assert p.std_amount == 0.0
        if p.txn_count >= 2:
            # std could be 0 if all amounts identical (rare but possible), so allow
            assert p.std_amount >= 0


def test_build_profiles_empty_inputs():
    empty_tx = pd.DataFrame(columns=["account_id", "amount", "ts", "merchant_id", "device_id", "location_id"])
    empty_acct = pd.DataFrame(columns=["account_id", "user_id"])
    assert build_profiles(empty_tx, empty_acct) == {}
    # empty transactions but valid accounts
    accts = pd.DataFrame({"account_id": [1], "user_id": [1]})
    assert build_profiles(empty_tx, accts) == {}


# ---- Scoring ------------------------------------------------------------

def test_score_transaction_returns_correct_types(profiles):
    profile = next(iter(profiles.values()))
    txn = {
        "amount": profile.mean_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": next(iter(profile.known_devices)),
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
        "hour": 12,
    }
    # ensure hour 12 is plausible; if profile has no hist at 12, is_unusual_hour may still be bool
    result = score_transaction(txn, profile)
    assert set(result.keys()) == {
        "amount_zscore", "is_unusual_hour", "is_new_merchant",
        "is_new_device", "is_new_location", "frequency_anomaly",
    }
    assert isinstance(result["amount_zscore"], float)
    assert isinstance(result["is_unusual_hour"], bool)
    assert isinstance(result["is_new_merchant"], bool)
    assert isinstance(result["is_new_device"], bool)
    assert isinstance(result["is_new_location"], bool)
    assert isinstance(result["frequency_anomaly"], bool)
    # normal amount => zscore near 0
    assert abs(result["amount_zscore"]) < 2.0
    assert result["is_new_merchant"] is False
    assert result["is_new_device"] is False
    assert result["is_new_location"] is False


def test_unusual_amount_detected(profiles):
    # Use a profile with at least 2 txns so std > 0
    profile = None
    for p in profiles.values():
        if p.txn_count >= 5 and p.std_amount > 1e-6:
            profile = p
            break
    assert profile is not None, "No suitable profile with std>0"
    huge_amount = profile.mean_amount + 10 * max(profile.std_amount, 1.0)
    txn = {
        "amount": huge_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": next(iter(profile.known_devices)),
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    }
    result = score_transaction(txn, profile)
    assert result["amount_zscore"] > 5.0


def test_new_device_detected(profiles):
    profile = next(iter(profiles.values()))
    # pick a device id guaranteed not in known_devices
    all_possible = set(range(1, 100000))
    unseen = (all_possible - profile.known_devices)
    new_device = next(iter(unseen))
    txn = {
        "amount": profile.mean_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": new_device,
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    }
    result = score_transaction(txn, profile)
    assert result["is_new_device"] is True
    # known device should be false
    txn2 = dict(txn)
    txn2["device_id"] = next(iter(profile.known_devices))
    result2 = score_transaction(txn2, profile)
    assert result2["is_new_device"] is False


def test_new_location_detected(profiles):
    profile = next(iter(profiles.values()))
    unseen = set(range(1, 100000)) - profile.known_locations
    new_loc = next(iter(unseen))
    txn = {
        "amount": profile.mean_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": next(iter(profile.known_devices)),
        "location_id": new_loc,
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    }
    assert score_transaction(txn, profile)["is_new_location"] is True


def test_new_merchant_detected(profiles):
    profile = next(iter(profiles.values()))
    # find merchant not in top-3
    unseen_merchant = 999999
    # ensure not in common_merchants
    while unseen_merchant in profile.common_merchants:
        unseen_merchant += 1
    txn = {
        "amount": profile.mean_amount,
        "merchant_id": unseen_merchant,
        "device_id": next(iter(profile.known_devices)),
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    }
    assert score_transaction(txn, profile)["is_new_merchant"] is True
    # known merchant => False
    if profile.common_merchants:
        txn["merchant_id"] = profile.common_merchants[0]
        assert score_transaction(txn, profile)["is_new_merchant"] is False


def test_unusual_hour_detected():
    # Construct a synthetic profile with history only at hour 10
    hist = np.zeros(24, dtype=np.int64)
    hist[10] = 20
    hist[11] = 15
    hist[9] = 10
    # top 12 will include 10,11,9 etc. Hour 3 should be unusual
    profile = BehavioralProfile(
        user_id=1,
        mean_amount=100.0,
        std_amount=10.0,
        median_amount=100.0,
        txn_count=45,
        hour_histogram=hist,
        common_merchants=[1, 2, 3],
        known_devices={1},
        known_locations={1},
        avg_daily_frequency=2.0,
    )
    txn_usual = {"amount": 100.0, "merchant_id": 1, "device_id": 1, "location_id": 1, "ts": pd.Timestamp("2024-01-15 10:30:00")}
    txn_unusual = {"amount": 100.0, "merchant_id": 1, "device_id": 1, "location_id": 1, "ts": pd.Timestamp("2024-01-15 03:30:00")}
    assert score_transaction(txn_usual, profile)["is_unusual_hour"] is False
    assert score_transaction(txn_unusual, profile)["is_unusual_hour"] is True


def test_frequency_anomaly(profiles):
    profile = next(iter(profiles.values()))
    # avg_daily_frequency e.g. 2 => threshold 4, daily_count 10 => anomaly
    txn = {
        "amount": profile.mean_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": next(iter(profile.known_devices)),
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    }
    # provide daily_count via row
    txn_high = dict(txn)
    txn_high["daily_count"] = int(profile.avg_daily_frequency * 3) + 5
    assert score_transaction(txn_high, profile)["frequency_anomaly"] is True
    # low daily count => no anomaly
    txn_low = dict(txn)
    txn_low["daily_count"] = 1
    assert score_transaction(txn_low, profile)["frequency_anomaly"] is False
    # also test explicit param
    assert score_transaction(txn, profile, daily_count=int(profile.avg_daily_frequency * 3) + 5)["frequency_anomaly"] is True


def test_score_with_series_input(profiles):
    profile = next(iter(profiles.values()))
    txn_series = pd.Series({
        "amount": profile.mean_amount,
        "merchant_id": profile.common_merchants[0] if profile.common_merchants else 1,
        "device_id": next(iter(profile.known_devices)),
        "location_id": next(iter(profile.known_locations)),
        "ts": pd.Timestamp("2024-01-15 12:00:00"),
    })
    result = score_transaction(txn_series, profile)
    assert isinstance(result["amount_zscore"], float)


def test_score_with_none_profile():
    txn = {"amount": 100.0, "merchant_id": 1, "device_id": 1, "location_id": 1, "ts": pd.Timestamp("2024-01-15 12:00:00")}
    result = score_transaction(txn, None)
    assert result["is_new_device"] is True
    assert result["is_new_location"] is True
    assert result["is_new_merchant"] is True
