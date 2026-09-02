"""Phase 9 — Graph Intelligence tests."""

from __future__ import annotations

import types

import pandas as pd
import pytest
import networkx as nx

from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment
from finsheild.graph import build_graph, compute_graph_features, GraphFeatures


@pytest.fixture(scope="module")
def env():
    return generate_environment(SyntheticEnvConfig.ci())


@pytest.fixture(scope="module")
def graph(env):
    return build_graph(env)


@pytest.fixture(scope="module")
def features(env, graph):
    return compute_graph_features(env, graph)


# ---- Graph construction -------------------------------------------------

def test_build_graph_returns_networkx_graph(graph):
    assert isinstance(graph, nx.Graph)


def test_graph_has_expected_node_types(graph):
    types_set = {data.get("node_type") for _, data in graph.nodes(data=True)}
    assert "user" in types_set, f"missing user nodes, found {types_set}"
    assert "account" in types_set, f"missing account nodes, found {types_set}"
    assert "device" in types_set, f"missing device nodes, found {types_set}"
    assert "merchant" in types_set, f"missing merchant nodes, found {types_set}"
    assert types_set.issubset({"user", "account", "device", "merchant"})


def test_graph_has_expected_edge_types(graph):
    edge_types = {data.get("edge_type") for _, _, data in graph.edges(data=True)}
    assert "owns" in edge_types
    assert "uses" in edge_types
    assert "transacts_with" in edge_types
    assert "shares_device" in edge_types


def test_graph_node_counts_reasonable(env, graph):
    user_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "user"]
    acct_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "account"]
    dev_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "device"]
    merch_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "merchant"]
    assert len(user_nodes) == len(env.users)
    assert len(acct_nodes) >= len(env.accounts)
    assert len(dev_nodes) >= len(env.devices)
    assert len(merch_nodes) >= len(env.merchants)


def test_graph_has_edges(graph):
    assert graph.number_of_edges() > 0
    owns = [(u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "owns"]
    assert len(owns) > 0


def test_graph_owns_edges_match_accounts(env, graph):
    for _, row in env.accounts.head(10).iterrows():
        u = f"user:{int(row['user_id'])}"
        a = f"account:{int(row['account_id'])}"
        assert graph.has_edge(u, a), f"missing owns edge {u} - {a}"


def test_graph_shares_device_edges_exist(graph):
    shares = [(u, v) for u, v, d in graph.edges(data=True) if d.get("edge_type") == "shares_device"]
    assert len(shares) > 0, "expected at least one shares_device edge"


# ---- Feature frame ------------------------------------------------------

def test_features_indexed_by_txn_id(features, env):
    assert features.index.name == "txn_id"
    assert len(features) == len(env.transactions)
    assert set(features.index.tolist()) == set(env.transactions["txn_id"].tolist())


def test_features_have_expected_columns(features):
    expected = {
        "account_degree",
        "device_degree",
        "shared_device_count",
        "merchant_degree",
        "account_suspicious_neighbor_count",
    }
    assert set(features.columns.tolist()) == expected
    assert set(GraphFeatures.COLUMNS) == expected


def test_features_have_no_nans(features):
    assert not features.isnull().values.any(), f"NaNs found:\n{features.isnull().sum()}"


def test_features_dtypes_numeric(features):
    for col in features.columns:
        assert pd.api.types.is_numeric_dtype(features[col]), f"{col} not numeric"


# ---- Scenario-aware graph signals ----------------------------------------

def test_shared_device_count_positive_for_device_sharing(env, features):
    """device_sharing fraud txns should predominantly use a shared device.

    The synthetic generator's fallback path can produce a small minority
    of device_sharing fraud rows where the injected device happens to be
    exclusive in account_devices (no link-table entry).  We therefore check
    that the *majority* are shared rather than requiring 100%.
    """
    fraud = env.transactions[
        (env.transactions["scenario_tag"] == "device_sharing") & (env.transactions["label_fraud"] == 1)
    ]
    assert len(fraud) > 0, "no device_sharing fraud rows — seed changed?"
    vals = features.loc[fraud["txn_id"].tolist(), "shared_device_count"]
    assert (vals > 0).any(), "no device_sharing fraud txn has shared_device_count>0"
    assert (vals > 0).mean() >= 0.5, f"only {(vals>0).mean():.0%} of device_sharing fraud are shared, expected >=50%"

def test_shared_device_count_zero_for_exclusive_device(env, features):
    """Background txns on devices exclusive in account_devices should have 0."""
    dev_counts = env.account_devices.groupby("device_id")["account_id"].nunique()
    exclusive_devs = dev_counts[dev_counts == 1].index.tolist()
    assert len(exclusive_devs) > 0
    bg = env.transactions[
        (env.transactions["scenario_tag"] == "background")
        & (env.transactions["device_id"].isin(exclusive_devs))
    ]
    if len(bg) == 0:
        pytest.skip("no background txn on exclusive device — unlikely at ci scale")
    for txn_id in bg["txn_id"].head(5).tolist():
        val = features.loc[txn_id, "shared_device_count"]
        assert val == 0, f"txn {txn_id} expected exclusive device shared_device_count==0 got {val}"


def test_suspicious_neighbor_count_detects_fraud_neighbors(env, graph, features):
    """At least one txn should have a suspicious neighbor (graph shares_device + fraud)."""
    assert (features["account_suspicious_neighbor_count"] > 0).any(), (
        "expected at least one txn with suspicious neighbor >0"
    )
    fraud_accounts = set(
        env.transactions[env.transactions["label_fraud"] == 1]["account_id"].tolist()
    )
    found = False
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "account":
            continue
        acc_id = int(attrs["entity_id"])
        if acc_id in fraud_accounts:
            continue
        for nbr in graph.neighbors(node):
            nbr_attrs = graph.nodes[nbr]
            if nbr_attrs.get("node_type") == "account" and int(nbr_attrs["entity_id"]) in fraud_accounts:
                txns = env.transactions[env.transactions["account_id"] == acc_id]
                if len(txns) == 0:
                    continue
                txn_id = int(txns.iloc[0]["txn_id"])
                assert features.loc[txn_id, "account_suspicious_neighbor_count"] >= 1
                found = True
                break
        if found:
            break
    assert found, "could not find a non-fraud account sharing device with a fraud account"


def test_suspicious_neighbor_synthetic_mini_case():
    """Controlled mini-env: two accounts share a device, one is fraud."""
    users = pd.DataFrame({"user_id": [1, 2], "signup_ts": [pd.Timestamp("2025-01-01")] * 2, "home_country": ["US"] * 2, "risk_segment": ["standard"] * 2})
    accounts = pd.DataFrame({"account_id": [1, 2], "user_id": [1, 2], "opened_ts": [pd.Timestamp("2025-01-01")] * 2, "account_type": ["checking"] * 2, "status": ["active"] * 2})
    devices = pd.DataFrame({"device_id": [10], "device_type": ["mobile_ios"], "fingerprint_hash": ["abc"], "first_seen_ts": [pd.Timestamp("2025-01-01")]})
    merchants = pd.DataFrame({"merchant_id": [100], "name": ["M1"], "category": ["grocery"], "mcc_code": [1], "country": ["US"], "risk_band": ["low"]})
    account_devices = pd.DataFrame(
        {
            "account_id": [1, 2],
            "device_id": [10, 10],
            "first_used_ts": [pd.Timestamp("2025-01-01")] * 2,
            "last_used_ts": [pd.Timestamp("2025-01-02")] * 2,
            "is_primary": [True, True],
        }
    )
    transactions = pd.DataFrame(
        {
            "txn_id": [1, 2],
            "account_id": [1, 2],
            "device_id": [10, 10],
            "merchant_id": [100, 100],
            "location_id": [1, 1],
            "ts": [pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-11")],
            "amount": [10.0, 999.0],
            "currency": ["USD"] * 2,
            "channel": ["online"] * 2,
            "status": ["settled"] * 2,
            "scenario_tag": ["background", "device_sharing"],
            "label_fraud": [0, 1],
            "scenario_extra": ["", ""],
        }
    )
    locations = pd.DataFrame({"location_id": [1], "city": ["NYC"], "country": ["US"], "lat": [40.0], "lon": [-74.0], "region": ["NA"]})
    env = types.SimpleNamespace(
        users=users,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        account_devices=account_devices,
        transactions=transactions,
        locations=locations,
    )
    g = build_graph(env)
    assert g.has_edge("account:1", "account:2")
    assert g.edges["account:1", "account:2"]["edge_type"] == "shares_device"

    feats = compute_graph_features(env, g)
    assert feats.loc[1, "account_suspicious_neighbor_count"] == 1
    assert feats.loc[2, "account_suspicious_neighbor_count"] == 0
    assert feats.loc[1, "shared_device_count"] == 1
    assert feats.loc[2, "shared_device_count"] == 1
    assert feats.loc[1, "account_degree"] > 0
    assert feats.loc[1, "device_degree"] > 0


def test_graph_determinism(env):
    g1 = build_graph(env)
    g2 = build_graph(env)
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
    assert set(g1.nodes()) == set(g2.nodes())
    assert set(g1.edges()) == set(g2.edges())


def test_account_degree_positive_for_all_transactions(features):
    assert (features["account_degree"] > 0).all(), "some account_degree == 0"


def test_empty_transactions_returns_empty_features():
    """Empty transaction frame should yield empty feature frame with correct columns."""
    users = pd.DataFrame({"user_id": [1], "signup_ts": [pd.Timestamp("2025-01-01")], "home_country": ["US"], "risk_segment": ["standard"]})
    accounts = pd.DataFrame({"account_id": [1], "user_id": [1], "opened_ts": [pd.Timestamp("2025-01-01")], "account_type": ["checking"], "status": ["active"]})
    devices = pd.DataFrame({"device_id": [1], "device_type": ["mobile_ios"], "fingerprint_hash": ["abc"], "first_seen_ts": [pd.Timestamp("2025-01-01")]})
    merchants = pd.DataFrame({"merchant_id": [1], "name": ["M1"], "category": ["grocery"], "mcc_code": [1], "country": ["US"], "risk_band": ["low"]})
    account_devices = pd.DataFrame({"account_id": [1], "device_id": [1], "first_used_ts": [pd.Timestamp("2025-01-01")], "last_used_ts": [pd.Timestamp("2025-01-02")], "is_primary": [True]})
    transactions = pd.DataFrame(columns=["txn_id", "account_id", "device_id", "merchant_id", "location_id", "ts", "amount", "currency", "channel", "status", "scenario_tag", "label_fraud", "scenario_extra"])
    locations = pd.DataFrame({"location_id": [1], "city": ["NYC"], "country": ["US"], "lat": [40.0], "lon": [-74.0], "region": ["NA"]})
    env = types.SimpleNamespace(
        users=users,
        accounts=accounts,
        devices=devices,
        merchants=merchants,
        account_devices=account_devices,
        transactions=transactions,
        locations=locations,
    )
    g = build_graph(env)
    feats = compute_graph_features(env, g)
    assert len(feats) == 0
    assert list(feats.columns) == GraphFeatures.COLUMNS
    assert feats.index.name == "txn_id"
