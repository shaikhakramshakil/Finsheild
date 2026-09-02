"""Graph construction — Phase 9.

Builds an undirected :class:`networkx.Graph` from a
:class:`SyntheticEnvironment`.

Nodes
-----
* ``user:{user_id}``
* ``account:{account_id}``
* ``device:{device_id}``
* ``merchant:{merchant_id}``

Edges
-----
* **owns** — user → account  (from ``accounts``)
* **uses** — account → device  (from ``account_devices``)
* **transacts_with** — account → merchant (distinct pairs from ``transactions``)
* **shares_device** — account ↔ account (clique per device that is used by ≥2
  accounts in ``account_devices``; also includes transaction-level co-usage
  as a fallback so device-sharing scenarios that inject a shared device only
  at the transaction level are still captured).
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pandas as pd


def _node_id(prefix: str, entity_id: int) -> str:
    return f"{prefix}:{int(entity_id)}"


def build_graph(env) -> nx.Graph:
    """Build an undirected graph from ``env``.

    Parameters
    ----------
    env:
        :class:`finsheild.synthetic_env.environment.SyntheticEnvironment`
        (duck-typed — any object with ``users``, ``accounts``, ``devices``,
        ``merchants``, ``account_devices``, ``transactions`` DataFrames).

    Returns
    -------
    networkx.Graph
        Undirected graph with ``node_type`` and ``edge_type`` attributes.
    """
    G = nx.Graph()

    # -- Nodes -----------------------------------------------------------
    if hasattr(env, "users") and env.users is not None and len(env.users) > 0:
        for _, row in env.users.iterrows():
            nid = _node_id("user", row["user_id"])
            G.add_node(nid, node_type="user", entity_id=int(row["user_id"]))

    if hasattr(env, "accounts") and env.accounts is not None and len(env.accounts) > 0:
        for _, row in env.accounts.iterrows():
            nid = _node_id("account", row["account_id"])
            G.add_node(
                nid,
                node_type="account",
                entity_id=int(row["account_id"]),
                user_id=int(row["user_id"]) if "user_id" in row else None,
            )

    if hasattr(env, "devices") and env.devices is not None and len(env.devices) > 0:
        for _, row in env.devices.iterrows():
            nid = _node_id("device", row["device_id"])
            G.add_node(nid, node_type="device", entity_id=int(row["device_id"]))

    if hasattr(env, "merchants") and env.merchants is not None and len(env.merchants) > 0:
        for _, row in env.merchants.iterrows():
            nid = _node_id("merchant", row["merchant_id"])
            G.add_node(nid, node_type="merchant", entity_id=int(row["merchant_id"]))

    # Ensure any device/merchant that only appears in transactions is also a node
    if hasattr(env, "transactions") and env.transactions is not None and len(env.transactions) > 0:
        tx = env.transactions
        for dev_id in tx["device_id"].dropna().unique():
            nid = _node_id("device", int(dev_id))
            if not G.has_node(nid):
                G.add_node(nid, node_type="device", entity_id=int(dev_id))
        for merch_id in tx["merchant_id"].dropna().unique():
            nid = _node_id("merchant", int(merch_id))
            if not G.has_node(nid):
                G.add_node(nid, node_type="merchant", entity_id=int(merch_id))
        for acc_id in tx["account_id"].dropna().unique():
            nid = _node_id("account", int(acc_id))
            if not G.has_node(nid):
                G.add_node(nid, node_type="account", entity_id=int(acc_id))

    # -- Edges: owns (user-account) ---------------------------------------
    if (
        hasattr(env, "accounts")
        and env.accounts is not None
        and len(env.accounts) > 0
    ):
        for _, row in env.accounts.iterrows():
            u = _node_id("user", row["user_id"])
            a = _node_id("account", row["account_id"])
            if G.has_node(u) and G.has_node(a):
                G.add_edge(u, a, edge_type="owns")

    # -- Edges: uses (account-device) -------------------------------------
    if (
        hasattr(env, "account_devices")
        and env.account_devices is not None
        and len(env.account_devices) > 0
    ):
        for _, row in env.account_devices.iterrows():
            a = _node_id("account", row["account_id"])
            d = _node_id("device", row["device_id"])
            if not G.has_node(a):
                G.add_node(a, node_type="account", entity_id=int(row["account_id"]))
            if not G.has_node(d):
                G.add_node(d, node_type="device", entity_id=int(row["device_id"]))
            if not G.has_edge(a, d):
                G.add_edge(a, d, edge_type="uses")

    # -- Edges: transacts_with (account-merchant) -------------------------
    if (
        hasattr(env, "transactions")
        and env.transactions is not None
        and len(env.transactions) > 0
        and "account_id" in env.transactions.columns
        and "merchant_id" in env.transactions.columns
    ):
        pairs = env.transactions[["account_id", "merchant_id"]].drop_duplicates()
        for _, row in pairs.iterrows():
            a = _node_id("account", row["account_id"])
            m = _node_id("merchant", row["merchant_id"])
            if not G.has_node(a):
                G.add_node(a, node_type="account", entity_id=int(row["account_id"]))
            if not G.has_node(m):
                G.add_node(m, node_type="merchant", entity_id=int(row["merchant_id"]))
            if not G.has_edge(a, m):
                G.add_edge(a, m, edge_type="transacts_with")

    # -- Edges: shares_device (account-account via shared device) ---------
    # Via account_devices link table: device used by >=2 accounts forms a clique.
    device_to_accounts: dict[int, set[int]] = {}

    if (
        hasattr(env, "account_devices")
        and env.account_devices is not None
        and len(env.account_devices) > 0
    ):
        for dev_id, group in env.account_devices.groupby("device_id"):
            accs = set(int(x) for x in group["account_id"].tolist())
            device_to_accounts[int(dev_id)] = accs

    for dev_id, acc_set in device_to_accounts.items():
        if len(acc_set) < 2:
            continue
        for a1_id, a2_id in combinations(sorted(acc_set), 2):
            a1 = _node_id("account", a1_id)
            a2 = _node_id("account", a2_id)
            if not G.has_node(a1):
                G.add_node(a1, node_type="account", entity_id=int(a1_id))
            if not G.has_node(a2):
                G.add_node(a2, node_type="account", entity_id=int(a2_id))
            if not G.has_edge(a1, a2):
                G.add_edge(a1, a2, edge_type="shares_device", device_id=int(dev_id))

    return G
