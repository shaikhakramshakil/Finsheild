"""Graph feature extraction — Phase 9.

Translates a :class:`networkx.Graph` (built by :func:`build_graph`) into a
per-transaction feature frame.  All features are simple graph statistics —
no GNN.

Features per transaction (indexed by ``txn_id``)
------------------------------------------------
* ``account_degree`` — degree of the account node
* ``device_degree`` — degree of the device node
* ``shared_device_count`` — number of *other* accounts that share this
  transaction's device (``0`` means exclusive use)
* ``merchant_degree`` — degree of the merchant node
* ``account_suspicious_neighbor_count`` — number of 1-hop neighbor accounts
  that have at least one ``label_fraud=1`` transaction in the same env
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import networkx as nx

GRAPH_FEATURE_COLUMNS = [
    "account_degree",
    "device_degree",
    "shared_device_count",
    "merchant_degree",
    "account_suspicious_neighbor_count",
]


@dataclass(frozen=True)
class GraphFeatures:
    """Schema / row container for graph-derived signals.

    The :func:`compute_graph_features` function returns a ``pd.DataFrame``
    indexed by ``txn_id`` with these columns; this dataclass documents the
    schema and can be used for type hints or single-row instantiation.
    """

    account_degree: int = 0
    device_degree: int = 0
    shared_device_count: int = 0
    merchant_degree: int = 0
    account_suspicious_neighbor_count: int = 0

    # Class-level accessor matching typical Phase-5 patterns
    COLUMNS = GRAPH_FEATURE_COLUMNS


def _node_id(prefix: str, entity_id: int) -> str:
    return f"{prefix}:{int(entity_id)}"


def compute_graph_features(env, graph: nx.Graph) -> pd.DataFrame:
    """Compute per-transaction graph signals.

    Parameters
    ----------
    env:
        :class:`SyntheticEnvironment` (or duck-typed equivalent).
    graph:
        Graph returned by :func:`finsheild.graph.build_graph`.

    Returns
    -------
    pd.DataFrame
        Indexed by ``txn_id`` with columns in :data:`GRAPH_FEATURE_COLUMNS`.
        Empty frame (with correct columns) if ``env.transactions`` is empty.
    """
    tx = env.transactions

    # Empty env guard
    if tx is None or len(tx) == 0:
        empty = pd.DataFrame(columns=GRAPH_FEATURE_COLUMNS)
        empty.index.name = "txn_id"
        return empty

    # -- Precompute helpers -----------------------------------------------
    # device_id -> number of accounts sharing that device (from account_devices only)
    # This is the canonical definition: static sharing via the link table.
    # Transaction-level sharing (new_device) is intentionally NOT counted here
    # so that exclusive devices remain 0 even if a fraudster later uses them.
    device_shared_counts: dict[int, int] = {}
    if (
        hasattr(env, "account_devices")
        and env.account_devices is not None
        and len(env.account_devices) > 0
    ):
        counts = env.account_devices.groupby("device_id")["account_id"].nunique()
        for dev_id, cnt in counts.items():
            device_shared_counts[int(dev_id)] = int(cnt) - 1  # other accounts
            if device_shared_counts[int(dev_id)] < 0:
                device_shared_counts[int(dev_id)] = 0

    # Set of accounts that had at least one fraud label in this env
    fraud_accounts: set[int] = set()
    if "label_fraud" in tx.columns and "account_id" in tx.columns:
        fraud_rows = tx[tx["label_fraud"] == 1]
        if len(fraud_rows) > 0:
            fraud_accounts = set(int(x) for x in fraud_rows["account_id"].tolist())

    # Precompute suspicious neighbor count per account node
    # neighbor fraud = count of directly connected account nodes that are in fraud_accounts
    account_suspicious: dict[int, int] = {}
    # Collect all account nodes in graph
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "account":
            continue
        acc_id = int(attrs.get("entity_id", node.split(":")[1]))
        # neighbors that are accounts and fraudulent
        count = 0
        for nbr in graph.neighbors(node):
            nbr_attrs = graph.nodes[nbr]
            if nbr_attrs.get("node_type") == "account":
                nbr_id = int(nbr_attrs.get("entity_id", nbr.split(":")[1]))
                if nbr_id in fraud_accounts and nbr_id != acc_id:
                    count += 1
        account_suspicious[acc_id] = count

    # -- Per-transaction rows ---------------------------------------------
    rows = []
    for _, r in tx.iterrows():
        txn_id = int(r["txn_id"])
        acc_id = int(r["account_id"])
        dev_id = int(r["device_id"]) if "device_id" in r and pd.notna(r["device_id"]) else None
        merch_id = int(r["merchant_id"]) if "merchant_id" in r and pd.notna(r["merchant_id"]) else None

        acc_node = _node_id("account", acc_id)
        dev_node = _node_id("device", dev_id) if dev_id is not None else None
        merch_node = _node_id("merchant", merch_id) if merch_id is not None else None

        # degree lookups (0 if node missing — shouldn't happen but defensive)
        try:
            acc_deg = int(graph.degree(acc_node)) if graph.has_node(acc_node) else 0
        except Exception:
            acc_deg = 0
        try:
            dev_deg = int(graph.degree(dev_node)) if dev_node is not None and graph.has_node(dev_node) else 0
        except Exception:
            dev_deg = 0
        try:
            merch_deg = int(graph.degree(merch_node)) if merch_node is not None and graph.has_node(merch_node) else 0
        except Exception:
            merch_deg = 0

        shared_cnt = device_shared_counts.get(int(dev_id), 0) if dev_id is not None else 0
        susp_cnt = account_suspicious.get(acc_id, 0)

        rows.append(
            {
                "txn_id": txn_id,
                "account_degree": acc_deg,
                "device_degree": dev_deg,
                "shared_device_count": int(shared_cnt),
                "merchant_degree": merch_deg,
                "account_suspicious_neighbor_count": int(susp_cnt),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        empty = pd.DataFrame(columns=GRAPH_FEATURE_COLUMNS)
        empty.index.name = "txn_id"
        return empty
    df = df.set_index("txn_id")
    # ensure column order
    df = df[GRAPH_FEATURE_COLUMNS]
    df.index.name = "txn_id"
    return df
