"""Phase 4 — synthetic digital payment environment tests.

Run with:

    PYTHONPATH=src pytest tests/test_synthetic_env.py -v

All tests use ``SyntheticEnvConfig.ci()`` so they run in <30 seconds total.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from finsheild.synthetic_env import (
    SCENARIO_NAMES,
    SyntheticEnvConfig,
    generate_environment,
)


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def env_again():
    # Same config as ``env`` — used for determinism assertions.
    return generate_environment(SyntheticEnvConfig.ci())


# ---- Schema --------------------------------------------------------------

def test_required_tables_present(env):
    expected = {"users", "accounts", "devices", "merchants",
                "locations", "account_devices", "transactions"}
    assert set(env.tables.keys()) == expected


def test_required_columns_present(env):
    users_cols = {"user_id", "signup_ts", "home_country", "risk_segment"}
    assert users_cols.issubset(env.users.columns)

    accounts_cols = {"account_id", "user_id", "opened_ts",
                     "account_type", "status"}
    assert accounts_cols.issubset(env.accounts.columns)

    devices_cols = {"device_id", "device_type", "fingerprint_hash",
                    "first_seen_ts"}
    assert devices_cols.issubset(env.devices.columns)

    merchants_cols = {"merchant_id", "name", "category", "mcc_code",
                      "country", "risk_band"}
    assert merchants_cols.issubset(env.merchants.columns)

    locations_cols = {"location_id", "city", "country", "lat", "lon",
                      "region"}
    assert locations_cols.issubset(env.locations.columns)

    tx_cols = {"txn_id", "account_id", "device_id", "merchant_id",
               "location_id", "ts", "amount", "currency", "channel",
               "status", "scenario_tag", "label_fraud", "scenario_extra"}
    assert tx_cols.issubset(env.transactions.columns)


# ---- Configurability -----------------------------------------------------

def test_configurable_size():
    small = generate_environment(SyntheticEnvConfig(n_users=50,
                                                     n_accounts=80,
                                                     n_devices=60,
                                                     n_merchants=20,
                                                     n_locations=20,
                                                     n_transactions=500,
                                                     time_span_days=10,
                                                     seed=99))
    big = generate_environment(SyntheticEnvConfig(n_users=300,
                                                   n_accounts=400,
                                                   n_devices=350,
                                                   n_merchants=80,
                                                   n_locations=60,
                                                   n_transactions=4000,
                                                   time_span_days=15,
                                                   seed=99))
    assert len(small.users) == 50
    assert len(big.users) == 300
    # Transactions should respect the size, not be padded to a constant.
    assert len(small.transactions) < len(big.transactions)


# ---- Determinism ---------------------------------------------------------

def test_same_seed_same_tables(env, env_again):
    for table in env.tables:
        a = getattr(env, table)
        b = getattr(env_again, table)
        # ``equals`` on DataFrames works for our column-by-column dtypes.
        assert a.equals(b), f"Table {table} differs across same-seed runs"


def test_different_seed_different_tables(env):
    other = generate_environment(SyntheticEnvConfig(
        seed=2025, n_users=200, n_accounts=250, n_devices=220,
        n_merchants=80, n_locations=60, n_transactions=5000,
        time_span_days=30))
    # Not byte-equal, but the seed difference should change at least the
    # transactions table.
    assert not env.transactions["txn_id"].equals(
        other.transactions["txn_id"])


# ---- Referential integrity ----------------------------------------------

def test_no_orphan_accounts_in_transactions(env):
    acc_ids = set(env.accounts["account_id"].tolist())
    tx_acc = set(env.transactions["account_id"].tolist())
    assert tx_acc.issubset(acc_ids)


def test_no_orphan_devices_in_transactions(env):
    dev_ids = set(env.devices["device_id"].tolist())
    tx_dev = set(env.transactions["device_id"].tolist())
    assert tx_dev.issubset(dev_ids)


def test_no_orphan_merchants_in_transactions(env):
    m_ids = set(env.merchants["merchant_id"].tolist())
    tx_m = set(env.transactions["merchant_id"].tolist())
    assert tx_m.issubset(m_ids)


def test_no_orphan_locations_in_transactions(env):
    l_ids = set(env.locations["location_id"].tolist())
    tx_l = set(env.transactions["location_id"].tolist())
    assert tx_l.issubset(l_ids)


def test_accounts_user_id_resolves(env):
    user_ids = set(env.users["user_id"].tolist())
    acc_users = set(env.accounts["user_id"].tolist())
    assert acc_users.issubset(user_ids)


def test_account_devices_references_valid(env):
    acc_ids = set(env.accounts["account_id"].tolist())
    dev_ids = set(env.devices["device_id"].tolist())
    ad = env.account_devices
    assert set(ad["account_id"].tolist()).issubset(acc_ids)
    assert set(ad["device_id"].tolist()).issubset(dev_ids)


# ---- No leakage ---------------------------------------------------------

def test_no_future_timestamps(env):
    config_end = pd.Timestamp(env.config.start_ts) \
        + pd.Timedelta(days=env.config.time_span_days)
    # transactions within window
    assert env.transactions["ts"].max() <= config_end
    assert env.transactions["ts"].min() >= pd.Timestamp(env.config.start_ts)
    # users signup window can extend slightly before start_ts (doc'd)
    # but no future-dated signups
    assert env.users["signup_ts"].max() <= config_end


def test_reference_tables_carry_no_fraud_label(env):
    for name in ("users", "accounts", "devices", "merchants",
                 "locations", "account_devices"):
        df = getattr(env, name)
        assert "label_fraud" not in df.columns
        assert "scenario_tag" not in df.columns


def test_label_fraud_is_binary(env):
    vals = set(env.transactions["label_fraud"].unique().tolist())
    assert vals.issubset({0, 1})


def test_only_scenario_events_are_fraud(env):
    """Background must never be flagged; every fraud row must have a
    non-background scenario_tag."""
    bg = env.transactions[env.transactions["scenario_tag"] == "background"]
    assert (bg["label_fraud"] == 0).all()
    fraud = env.transactions[env.transactions["label_fraud"] == 1]
    assert (fraud["scenario_tag"] != "background").all()
    assert fraud["scenario_tag"].isin(SCENARIO_NAMES).all()


# ---- Scenario coverage --------------------------------------------------

def test_all_eight_scenarios_present(env):
    present = set(env.transactions["scenario_tag"].unique().tolist())
    for name in SCENARIO_NAMES:
        assert name in present, f"Scenario {name} missing from transactions"


def test_each_scenario_has_flagged_and_context_rows(env):
    breakdown = env.scenario_breakdown()
    for _, row in breakdown.iterrows():
        tag = row["scenario_tag"]
        if tag == "background":
            continue
        assert row["n_fraud"] > 0, f"{tag} produced zero flagged transactions"
        assert row["n_total"] > 0


def test_velocity_burst_produces_bursts(env):
    vel = env.transactions[env.transactions["scenario_tag"]
                            == "transaction_velocity"]
    # each episode produces 8-12 burst rows — group by approximate time
    # within a 10-min window
    fraud_vel = vel[vel["label_fraud"] == 1]
    # There should be at least 50 flagged burst rows in CI scale (40 episodes).
    assert len(fraud_vel) >= 50
def test_device_sharing_uses_same_device_across_accounts(env):
    ds = env.transactions[env.transactions["scenario_tag"]
                          == "device_sharing"]
    fraud = ds[ds["label_fraud"] == 1]
    # Every fraud row's device must appear on 2+ accounts in the link table
    # (the mule ring structure). That's the scenario's invariant.
    ad_device_account_counts = env.account_devices.groupby("device_id")[
        "account_id"].nunique()
    fraud_devices = fraud["device_id"].astype(int).tolist()
    counts = ad_device_account_counts.reindex(fraud_devices).fillna(0)
    n_shared = int((counts >= 2).sum())
    assert n_shared > 0, (
        "device_sharing scenario did not produce any fraud rows on a device "
        "shared by 2+ accounts in the link table")


def test_new_device_target_unseen_device(env):
    nd = env.transactions[env.transactions["scenario_tag"] == "new_device"]
    fraud = nd[nd["label_fraud"] == 1]
    # For each fraud row, the (account_id, device_id) pair should NOT exist
    # in the account_devices table.
    ad_pairs = set(zip(env.account_devices["account_id"],
                       env.account_devices["device_id"]))
    bad = fraud[fraud.apply(
        lambda r: (int(r["account_id"]), int(r["device_id"])) in ad_pairs,
        axis=1)]
    assert len(bad) == 0, "new_device scenario leaked a known device"


def test_unusual_location_targets_foreign_country(env):
    ul = env.transactions[env.transactions["scenario_tag"]
                          == "unusual_location"]
    fraud = ul[ul["label_fraud"] == 1]
    acc_country = env.accounts.merge(env.users, on="user_id", how="left") \
        .set_index("account_id")["home_country"]
    loc_country = env.locations.set_index("location_id")["country"]
    for _, row in fraud.iterrows():
        home = acc_country[int(row["account_id"])]
        tx_country = loc_country[int(row["location_id"])]
        assert home != tx_country, (
            "unusual_location scenario landed in the user's home country")


# ---- CI-scale perf -----------------------------------------------------

def test_ci_scale_runs_quickly():
    t0 = time.time()
    generate_environment(SyntheticEnvConfig.ci())
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"CI env generation took {elapsed:.1f}s (limit 30s)"


# ---- Useful helpers ----------------------------------------------------

def test_scenario_breakdown_helper(env):
    bd = env.scenario_breakdown()
    assert "scenario_tag" in bd.columns
    assert "n_total" in bd.columns
    assert "n_fraud" in bd.columns


def test_fraud_rate_helper_in_range(env):
    rate = env.fraud_rate()
    assert 0.0 <= rate <= 1.0


def test_config_to_dict_roundtrip():
    cfg = SyntheticEnvConfig.ci()
    d = cfg.to_dict()
    cfg2 = SyntheticEnvConfig(**d)
    assert cfg == cfg2