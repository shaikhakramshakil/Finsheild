"""In-memory demo store + deterministic scenario generator."""
from __future__ import annotations

import hashlib
import itertools
import random
from datetime import datetime, timezone

from ..schemas import Transaction

_counter = itertools.count(10042)
_store: dict[str, dict] = {}


def _uid(prefix: str, n: int) -> str:
    return f"{prefix}-{n:05d}"


SCENARIOS = ("normal", "suspicious", "fraud_ring", "ambiguous")


def make_transaction(scenario: str = "normal", seed: int | None = None) -> tuple[Transaction, dict]:
    rng = random.Random(seed if seed is not None else int(next(_counter)))
    i = next(_counter)
    txn_id = f"TXN-{i}"
    if scenario == "suspicious":
        txn = Transaction(
            transaction_id=txn_id, user_id=_uid("U", rng.randint(1, 500)),
            amount=50000.0, timestamp=datetime.now(timezone.utc).isoformat(),
            merchant="High-Value Electronics", merchant_category="electronics",
            device_id="DEV-NEW-" + str(rng.randint(100, 999)), location="400km from home",
            velocity=8,
        )
        ctx = {"scenario": "suspicious", "amount_deviation": 4.2, "recent_transaction_count": 8}
    elif scenario == "fraud_ring":
        dev = "DEV-X-SHARED"
        txn = Transaction(
            transaction_id=txn_id, user_id=_uid("U", rng.randint(1, 50)),
            amount=float(rng.choice([9200, 14500, 22100])), timestamp=datetime.now(timezone.utc).isoformat(),
            merchant="Common Merchant M-7", merchant_category="retail",
            device_id=dev, location="Same city", velocity=5,
        )
        ctx = {"scenario": "fraud_ring", "amount_deviation": 2.1, "recent_transaction_count": 5,
               "shared_device_accounts": 4}
    elif scenario == "ambiguous":
        txn = Transaction(
            transaction_id=txn_id, user_id=_uid("U", rng.randint(1, 500)),
            amount=9800.0, timestamp=datetime.now(timezone.utc).isoformat(),
            merchant="Online Marketplace", merchant_category="ecommerce",
            device_id="DEV-KNOWN-" + str(rng.randint(10, 99)), location="120km away",
            velocity=4,
        )
        ctx = {"scenario": "ambiguous", "amount_deviation": 1.6, "recent_transaction_count": 4}
    else:
        txn = Transaction(
            transaction_id=txn_id, user_id=_uid("U", rng.randint(1, 2000)),
            amount=float(rng.choice([1200, 2500, 4200, 3100])), timestamp=datetime.now(timezone.utc).isoformat(),
            merchant=rng.choice(["Grocery Store", "Fuel Station", "Pharmacy", "Cafe"]),
            merchant_category="everyday", device_id="DEV-KNOWN-" + str(rng.randint(10, 99)),
            location="Home city", velocity=rng.randint(1, 2),
        )
        ctx = {"scenario": "normal", "amount_deviation": 0.3, "recent_transaction_count": txn.velocity}
    return txn, ctx


def save_scored(txn: Transaction, ctx: dict, score: dict) -> dict:
    rec = {"transaction": txn.model_dump(), "context": ctx, "score": score}
    _store[txn.transaction_id] = rec
    return rec


def get(txn_id: str) -> dict | None:
    return _store.get(txn_id)


def list_all(limit: int = 50) -> list[dict]:
    return list(reversed(list(_store.values())))[-limit:]


def reset() -> None:
    _store.clear()


def tokenize(user_id: str) -> dict:
    h = hashlib.sha256(("finsheild-salt::" + user_id).encode()).hexdigest()
    return {
        "user_token": f"{h[:4]}...{h[-4:]}",
        "phone": "••••••••42",
        "document": "TOKENIZED",
        "status": "Verified",
        "method": "Prototype identity tokenization (salted SHA-256, NOT a zero-knowledge proof)",
        "kind": "DEMO_SIMULATION",
    }


def graph_for(txn_id: str) -> dict:
    rec = _store.get(txn_id)
    scenario = (rec or {}).get("context", {}).get("scenario", "normal")
    if scenario == "fraud_ring":
        return {
            "nodes": [
                {"id": "acct:A", "type": "account"}, {"id": "acct:B", "type": "account"},
                {"id": "acct:C", "type": "account"}, {"id": "dev:X", "type": "device"},
                {"id": txn_id, "type": "transaction"}, {"id": "merch:M-7", "type": "merchant"},
            ],
            "edges": [
                {"from": "acct:A", "to": "dev:X"}, {"from": "acct:B", "to": "dev:X"},
                {"from": "acct:C", "to": "dev:X"}, {"from": "acct:A", "to": txn_id},
                {"from": "acct:A", "to": "merch:M-7"},
            ],
            "kind": "DEMO_SIMULATION",
            "note": "Demo visualization with synthetic investigation data — not real banking data.",
        }
    txn = (rec or {}).get("transaction", {})
    u = txn.get("user_id", "U-00000")
    return {
        "nodes": [
            {"id": f"user:{u}", "type": "user"}, {"id": txn.get("device_id", "dev"), "type": "device"},
            {"id": txn_id, "type": "transaction"}, {"id": txn.get("merchant", "m"), "type": "merchant"},
        ],
        "edges": [
            {"from": f"user:{u}", "to": txn_id}, {"from": txn.get("device_id", "dev"), "to": txn_id},
        ],
        "kind": "DEMO_SIMULATION",
        "note": "Demo visualization with synthetic investigation data.",
    }
