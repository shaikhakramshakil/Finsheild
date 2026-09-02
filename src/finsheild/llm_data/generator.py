"""LLM Training Data Generation — Phase 12.

Generates instruction-tuning examples from actual fraud-pipeline outputs
(features → risk_fusion → explain). No invented values — every number in the
``input`` JSON comes from the feature row or risk-fusion scores.

Public API
----------
* ``SCENARIO_TO_FRAUD_TYPE`` – canonical mapping  scenario_tag → fraud_type.
* ``build_llm_example`` – single-row converter.
* ``generate_llm_dataset`` – stratified bulk generator (80/10/10 split).
* ``save_dataset`` / ``load_dataset`` – JSONL helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from finsheild.synthetic_env.scenarios import SCENARIO_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "You are a fraud investigation copilot. Given the structured transaction "
    "evidence, assess risk, identify fraud type, summarize findings, and "
    "recommend an action. Respond with valid JSON containing risk_level, "
    "fraud_type, summary, evidence, and recommended_action. Be grounded in "
    "the provided evidence and do not invent facts."
)

SCENARIO_TO_FRAUD_TYPE: Dict[str, str] = {
    "account_takeover": "ACCOUNT_TAKEOVER",
    "unusual_amount_time": "UNUSUAL_AMOUNT_TIME",
    "transaction_velocity": "TRANSACTION_VELOCITY",
    "new_device": "NEW_DEVICE",
    "unusual_location": "UNUSUAL_LOCATION",
    "device_sharing": "DEVICE_SHARING",
    "mule_behavior": "MULE_BEHAVIOR",
    "unusual_merchant": "UNUSUAL_MERCHANT",
    "background": "LEGITIMATE",
    "legitimate": "LEGITIMATE",
    "legit": "LEGITIMATE",
}

# Reverse for validation
FRAUD_TYPE_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_FRAUD_TYPE.items()}

DECISION_TO_ACTION: Dict[str, str] = {
    "APPROVE": "Approve transaction – no further action required.",
    "STEP_UP": "Require step-up authentication (OTP/biometric).",
    "INVESTIGATE": "Flag for manual investigation.",
    "BLOCK": "Block transaction and freeze account pending review.",
}

_RISK_LEVELS = {"GREEN", "YELLOW", "RED", "LOW", "MEDIUM", "HIGH"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(row: Any, key: str, default: Any = None) -> Any:
    """Safe accessor for dict / Series / object."""
    if isinstance(row, dict):
        v = row.get(key, default)
    elif isinstance(row, pd.Series):
        v = row.get(key, default) if key in row.index else default
    elif isinstance(row, pd.DataFrame):
        if len(row) == 0:
            return default
        return _get(row.iloc[0], key, default)
    else:
        try:
            v = getattr(row, key, default)
            if v is default and isinstance(row, dict):
                v = row.get(key, default)
        except Exception:
            v = default
    # handle NaN/NA
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(float(v))
    except Exception:
        return default


def _extract_triggered_rules(risk_result: Any) -> List[str]:
    """Derive triggered rule ids from RiskResult.evidence."""
    evidence: List[str] = []
    if hasattr(risk_result, "evidence"):
        evidence = list(getattr(risk_result, "evidence", []) or [])
    elif isinstance(risk_result, dict):
        evidence = list(risk_result.get("evidence", []) or [])
        # also allow explicit triggered_rules key
        if "triggered_rules" in risk_result and risk_result["triggered_rules"]:
            return sorted({str(x).upper() for x in risk_result["triggered_rules"]})

    rules: List[str] = []
    for e in evidence:
        if not isinstance(e, str):
            continue
        if e.startswith("rule:"):
            rules.append(e.split(":", 1)[1].strip().upper())
        elif e.lower().startswith("high_velocity"):
            rules.append("HIGH_VELOCITY")
        elif e.lower().startswith("burst_velocity"):
            rules.append("BURST_VELOCITY")
        elif e.lower().startswith("shared_device"):
            rules.append("SHARED_DEVICE")
        elif e.lower().startswith("new_device"):
            rules.append("NEW_DEVICE")
        elif e.lower().startswith("unusual_location"):
            rules.append("UNUSUAL_LOCATION")
        elif e.lower().startswith("country_switch"):
            rules.append("COUNTRY_SWITCH")
        elif e.lower().startswith("unusual_amount"):
            rules.append("UNUSUAL_AMOUNT")
        elif e.lower().startswith("offhours"):
            rules.append("OFFHOURS_HIGH_VALUE")
        elif e.lower().startswith("graph_risk"):
            rules.append("GRAPH_RISK")
        elif e.lower().startswith("ml_flagged"):
            rules.append("ML_FLAGGED")
        elif e.lower().startswith("anomalous"):
            rules.append("ANOMALOUS_PATTERN")
        elif e.lower().startswith("elevated_risk"):
            rules.append("ELEVATED_RISK")
        elif e.lower().startswith("new_user"):
            rules.append("NEW_USER")
    # dedup + sort for determinism
    return sorted(set(rules))


def _fraud_type_for_row(row: Any) -> str:
    tag = str(_get(row, "scenario_tag", "background") or "background").strip().lower()
    # background but labelled legit? treat as legitimate if label_fraud==0 and tag background
    # still mapping gives LEGITIMATE
    return SCENARIO_TO_FRAUD_TYPE.get(tag, tag.upper())


def _recommended_action_for_result(risk_result: Any) -> str:
    decision = None
    if hasattr(risk_result, "decision"):
        decision = getattr(risk_result, "decision", None)
    elif isinstance(risk_result, dict):
        decision = risk_result.get("decision")
    if isinstance(decision, str):
        return DECISION_TO_ACTION.get(decision.upper(), f"Decision: {decision}")
    return "Review manually."


def _build_summary(
    fraud_type: str,
    risk_level: str,
    inp: Dict[str, Any],
    evidence: List[str],
) -> str:
    amt = inp.get("transaction_amount", 0)
    avg = inp.get("historical_average", 0)
    nd = inp.get("new_device", False)
    dist = inp.get("location_distance", 0)
    rc = inp.get("recent_count", 0)
    xgb = inp.get("xgboost_score", 0)
    ano = inp.get("anomaly_score", 0)
    if fraud_type == "LEGITIMATE":
        return (
            f"Legitimate transaction of ${amt:.2f} (historical avg ${avg:.2f}). "
            f"No strong fraud indicators; recent_count={rc}, new_device={nd}, "
            f"distance={dist:.0f}km. Scores xgb={xgb:.2f} anomaly={ano:.2f}. "
            f"Risk {risk_level}."
        )
    # fraud case
    ev_snippet = "; ".join(evidence[:3]) if evidence else "no additional evidence"
    return (
        f"{fraud_type.replace('_',' ').title()} suspected: "
        f"amount ${amt:.2f} vs avg ${avg:.2f}, "
        f"new_device={nd}, distance {dist:.0f}km, recent_count {rc}, "
        f"xgb {xgb:.2f} anomaly {ano:.2f}. Evidence: {ev_snippet}. "
        f"Risk {risk_level}."
    )


# ---------------------------------------------------------------------------
# Public: build_llm_example
# ---------------------------------------------------------------------------

def build_llm_example(
    txn_features_row: Any,
    risk_result: Any,
    explain_result: Any,
) -> Dict[str, str]:
    """Convert a single pipeline output into an LLM instruction example.

    Parameters
    ----------
    txn_features_row:
        Feature row (``pd.Series`` / ``dict`` / single-row ``pd.DataFrame``).
        Must contain at least engineered feature columns and optionally
        ``amount``, ``scenario_tag``, ``prior_mean_amount``, etc.
    risk_result:
        ``RiskResult`` (or dict with same keys) from ``RiskFusionEngine``.
    explain_result:
        Evidence list (``List[str]``) from ``evidence_from_features`` or
        ``RiskResult.evidence``. If a dict of SHAP values is passed, its
        string representation is used.

    Returns
    -------
    dict with ``{instruction, input, output}`` where ``input`` and
    ``output`` are JSON-encoded strings.
    """
    # Normalize row
    if isinstance(txn_features_row, pd.DataFrame):
        if len(txn_features_row) == 0:
            raise ValueError("Empty feature row DataFrame")
        row = txn_features_row.iloc[0]
    else:
        row = txn_features_row

    # --- input evidence (grounded) ---------------------------------------
    # transaction_amount: prefer 'amount' else derive from amount_log
    amt_raw = _get(row, "amount", None)
    if amt_raw is None:
        amt_raw = _get(row, "amount_x", None)
    if amt_raw is None:
        amt_raw = _get(row, "amount_y", None)
    if amt_raw is None:
        al = _safe_float(_get(row, "amount_log", None), None)
        if al is not None and al != 0:
            try:
                amt_raw = float(np.expm1(al))
            except Exception:
                amt_raw = 0.0
        else:
            amt_raw = 0.0
    transaction_amount = _safe_float(amt_raw, 0.0)

    historical_average = _safe_float(_get(row, "prior_mean_amount", 0.0), 0.0)
    # also fallback to prior_total_amount / prior_tx_count if mean is 0?
    if historical_average == 0.0:
        # try alternative: prior_mean already 0 for new users – keep 0
        pass

    new_device = bool(_safe_int(_get(row, "is_new_device", 0), 0) == 1)

    location_distance = _safe_float(_get(row, "distance_to_prev_km", 0.0), 0.0)

    # recent_count: prefer vel_count_300s else 3600s
    rc = _get(row, "vel_count_300s", None)
    if rc is None:
        rc = _get(row, "vel_count_3600s", 0)
    recent_count = _safe_int(rc, 0)

    # scores from risk_result
    if hasattr(risk_result, "xgb_score"):
        xgboost_score = _safe_float(getattr(risk_result, "xgb_score", 0.0), 0.0)
    elif isinstance(risk_result, dict):
        xgboost_score = _safe_float(risk_result.get("xgb_score", 0.0), 0.0)
    else:
        xgboost_score = 0.0

    if hasattr(risk_result, "anomaly_score"):
        anomaly_score = _safe_float(getattr(risk_result, "anomaly_score", 0.0), 0.0)
    elif isinstance(risk_result, dict):
        anomaly_score = _safe_float(risk_result.get("anomaly_score", 0.0), 0.0)
    else:
        anomaly_score = 0.0

    triggered_rules = _extract_triggered_rules(risk_result)

    # graph_signals
    dac = _safe_int(_get(row, "device_account_count", 1), 1)
    graph_score = 0.0
    if hasattr(risk_result, "graph_score"):
        graph_score = _safe_float(getattr(risk_result, "graph_score", 0.0), 0.0)
    elif isinstance(risk_result, dict):
        graph_score = _safe_float(risk_result.get("graph_score", 0.0), 0.0)
    prior_unique_countries = _safe_int(_get(row, "prior_unique_countries", 0), 0)
    graph_signals: Dict[str, Any] = {
        "shared_device_accounts": int(dac) if _get(row, "device_is_shared", 0) == 1 else int(dac),
        "graph_score": round(float(graph_score), 4),
        "prior_unique_countries": int(prior_unique_countries),
    }
    # also include velocity for graph context if useful
    if recent_count >= 5:
        graph_signals["velocity_5min"] = int(recent_count)

    input_dict: Dict[str, Any] = {
        "transaction_amount": round(float(transaction_amount), 2),
        "historical_average": round(float(historical_average), 2),
        "new_device": bool(new_device),
        "location_distance": round(float(location_distance), 1),
        "recent_count": int(recent_count),
        "xgboost_score": round(float(np.clip(xgboost_score, 0, 1)), 4),
        "anomaly_score": round(float(np.clip(anomaly_score, 0, 1)), 4),
        "triggered_rules": list(triggered_rules),
        "graph_signals": dict(graph_signals),
    }

    # --- output -----------------------------------------------------------
    # risk_level
    if hasattr(risk_result, "risk_level"):
        risk_level = str(getattr(risk_result, "risk_level", "GREEN")).upper()
    elif isinstance(risk_result, dict):
        risk_level = str(risk_result.get("risk_level", "GREEN")).upper()
    else:
        risk_level = "GREEN"

    fraud_type = _fraud_type_for_row(row)

    # evidence: prefer explain_result if list, else risk_result.evidence
    evidence: List[str] = []
    if isinstance(explain_result, list):
        evidence = [str(x) for x in explain_result if isinstance(x, str) and x]
    elif isinstance(explain_result, dict):
        # dict of SHAP values -> convert to top evidence strings? Use keys
        evidence = [f"{k}: {float(v):.3f}" for k, v in explain_result.items()]
    else:
        # fallback: try to coerce to list
        try:
            evidence = [str(x) for x in list(explain_result)]  # type: ignore[arg-type]
        except Exception:
            evidence = []

    # if explain_result is empty, fall back to risk_result.evidence
    if not evidence:
        if hasattr(risk_result, "evidence"):
            evidence = [str(x) for x in (getattr(risk_result, "evidence", []) or [])]
        elif isinstance(risk_result, dict):
            evidence = [str(x) for x in (risk_result.get("evidence", []) or [])]

    # dedup preserve order
    seen = set()
    deduped: List[str] = []
    for e in evidence:
        if e not in seen:
            deduped.append(e)
            seen.add(e)
    evidence = deduped

    recommended_action = _recommended_action_for_result(risk_result)

    output_dict: Dict[str, Any] = {
        "risk_level": risk_level,
        "fraud_type": fraud_type,
        "summary": _build_summary(fraud_type, risk_level, input_dict, evidence),
        "evidence": evidence,
        "recommended_action": recommended_action,
    }

    return {
        "instruction": INSTRUCTION,
        "input": json.dumps(input_dict, ensure_ascii=False),
        "output": json.dumps(output_dict, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Hybrid return type for generate_llm_dataset
# ---------------------------------------------------------------------------

class SplitDataset(list):  # type: ignore[type-arg]
    """Hybrid list/dict return for ``generate_llm_dataset``.

    Behaves as a flat ``list`` (``isinstance(..., list) is True``) while
    also supporting ``dataset["train"]`` dict-style access. This satisfies
    both the ``list[dict]`` spec and the intuitive ``dict`` split API.
    """

    def __init__(self, splits: Dict[str, List[Dict[str, Any]]]):
        self.splits: Dict[str, List[Dict[str, Any]]] = dict(splits)
        flat: List[Dict[str, Any]] = []
        for split_name, items in self.splits.items():
            for ex in items:
                # inject split field if missing (non-destructive copy)
                if "split" not in ex:
                    ex = dict(ex)
                    ex["split"] = split_name
                flat.append(ex)
        super().__init__(flat)

    # dict-style access
    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            return self.splits[key]
        return super().__getitem__(key)

    def keys(self):  # type: ignore[override]
        return self.splits.keys()

    def get(self, key: str, default: Any = None) -> Any:
        return self.splits.get(key, default)

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        if isinstance(key, str):
            return key in self.splits
        return super().__contains__(key)  # type: ignore[arg-type]

    def items(self):  # type: ignore[override]
        return self.splits.items()

    def values(self):  # type: ignore[override]
        return self.splits.values()


# ---------------------------------------------------------------------------
# generate_llm_dataset
# ---------------------------------------------------------------------------

def generate_llm_dataset(
    env: Any,
    feature_result: Any,
    risk_fusion_engine: Any,
    n_per_scenario: int = 50,
    random_state: int = 42,
) -> SplitDataset:
    """Generate an LLM training dataset from actual pipeline outputs.

    Runs ``risk_fusion_engine.predict_batch`` on ``feature_result`` to obtain
    real risk scores, then derives grounded evidence via
    ``evidence_from_features``. Sampling is stratified by ``scenario_tag`` so
    every fraud scenario plus legitimate traffic is represented.

    Parameters
    ----------
    env:
        ``SyntheticEnvironment`` (needed if the engine is not yet fitted;
        will call ``fit`` automatically).
    feature_result:
        ``FeatureBuildResult`` or ``pd.DataFrame`` containing feature columns
        and ``scenario_tag`` / ``label_fraud`` metadata.
    risk_fusion_engine:
        Fitted ``RiskFusionEngine``. If not fitted, ``fit(env,
        feature_result)`` is called.
    n_per_scenario:
        Number of examples to sample per scenario tag (including
        legitimate). Default 50 → total 450 examples.
    random_state:
        RNG seed for deterministic sampling / splitting.

    Returns
    -------
    SplitDataset
        Hybrid object that is both a flat ``list`` (``len == 9 * n_per_scenario``)
        and a ``dict``-like with ``["train"]``, ``["val"]``, ``["test"]``
        keys (80/10/10 stratified).

    Notes
    -----
    * If a scenario has fewer than ``n_per_scenario`` rows available,
      sampling is done with replacement.
    * Splits are stratified — each scenario contributes 80/10/10 to
      train/val/test.
    """
    # --- resolve features DataFrame ---------------------------------------
    if hasattr(feature_result, "features") and hasattr(feature_result, "feature_columns"):
        features: pd.DataFrame = feature_result.features  # type: ignore[attr-defined]
    elif isinstance(feature_result, pd.DataFrame):
        features = feature_result
    else:
        raise TypeError("feature_result must be FeatureBuildResult or DataFrame")

    if len(features) == 0:
        raise ValueError("feature_result is empty")

    if "scenario_tag" not in features.columns and "label_fraud" not in features.columns:
        # At minimum need one of them to stratify; fallback to treating all as legitimate
        features = features.copy()
        features["scenario_tag"] = "background"

    # Ensure engine is fitted
    is_fitted = bool(getattr(risk_fusion_engine, "_is_fitted", False))
    if not is_fitted:
        # try to fit
        risk_fusion_engine.fit(env, feature_result)

    # Run risk_fusion
    try:
        risk_results = risk_fusion_engine.predict_batch(features)
    except TypeError:
        # fallback: per-row
        risk_results = [risk_fusion_engine.predict(features.iloc[i]) for i in range(len(features))]

    if len(risk_results) != len(features):
        raise RuntimeError(f"risk_results length {len(risk_results)} != features {len(features)}")

    # Lazy import evidence helper
    try:
        from finsheild.explain.explainer import evidence_from_features as _evidence_fn
    except Exception:
        _evidence_fn = None  # type: ignore[assignment]

    rng = np.random.RandomState(int(random_state))

    # Build pool per tag
    # All 8 scenarios + legitimate (background / label_fraud==0)
    target_tags: List[str] = list(SCENARIO_NAMES)  # 8

    # Helper to get indices for a tag
    def _indices_for_tag(tag: str) -> np.ndarray:
        mask = features["scenario_tag"] == tag
        return np.where(mask)[0]

    # Collect examples per tag group (before split)
    per_tag_examples: Dict[str, List[Dict[str, Any]]] = {t: [] for t in target_tags}
    per_tag_examples["legitimate"] = []

    # For each fraud scenario
    for tag in target_tags:
        idxs = _indices_for_tag(tag)
        if len(idxs) == 0:
            # No rows for this tag with exact match — try case-insensitive? keep empty
            continue
        if len(idxs) < n_per_scenario:
            chosen = rng.choice(idxs, size=n_per_scenario, replace=True)
        else:
            chosen = rng.choice(idxs, size=n_per_scenario, replace=False)
        for pos in chosen:
            row = features.iloc[pos]
            rr = risk_results[pos]
            ev: List[str] = []
            if _evidence_fn is not None:
                try:
                    ev = _evidence_fn(row)
                except Exception:
                    ev = list(getattr(rr, "evidence", []) or [])
            else:
                ev = list(getattr(rr, "evidence", []) or [])
            ex = build_llm_example(row, rr, ev)
            # keep original tag for stratification verification (ex already has fraud_type, but also store tag)
            # inject internal tag without polluting input/output JSON – use extra key for split logic
            ex = dict(ex)
            ex["_scenario_tag"] = tag
            ex["_pos"] = int(pos)
            per_tag_examples[tag].append(ex)

    # Legitimate: prefer background rows, else any label_fraud==0
    if "scenario_tag" in features.columns:
        bg_idxs = np.where(features["scenario_tag"] == "background")[0]
    else:
        bg_idxs = np.array([], dtype=int)

    if len(bg_idxs) >= n_per_scenario:
        legit_pool = bg_idxs
    else:
        # fallback to any legit label
        if "label_fraud" in features.columns:
            legit_pool = np.where(features["label_fraud"] == 0)[0]
        else:
            legit_pool = bg_idxs if len(bg_idxs) > 0 else np.arange(len(features))

    if len(legit_pool) == 0:
        # degenerate: use random rows as legit
        legit_pool = np.arange(len(features))

    if len(legit_pool) < n_per_scenario:
        legit_chosen = rng.choice(legit_pool, size=n_per_scenario, replace=True)
    else:
        legit_chosen = rng.choice(legit_pool, size=n_per_scenario, replace=False)

    for pos in legit_chosen:
        row = features.iloc[pos]
        rr = risk_results[pos]
        if _evidence_fn is not None:
            try:
                ev = _evidence_fn(row)
            except Exception:
                ev = list(getattr(rr, "evidence", []) or [])
        else:
            ev = list(getattr(rr, "evidence", []) or [])
        ex = build_llm_example(row, rr, ev)
        ex = dict(ex)
        # Override fraud_type to LEGITIMATE for legit pool if the row's own tag says otherwise?
        # But build_llm_example already maps scenario_tag "background" -> LEGITIMATE.
        # For legit_pool sampled from label_fraud==0 that may include scenario context rows
        # with a fraud tag but still legit label, we want to ensure fraud_type is LEGITIMATE.
        # So patch output JSON to force LEGITIMATE for this group.
        try:
            out_j = json.loads(ex["output"])
            out_j["fraud_type"] = "LEGITIMATE"
            # also patch summary if needed: regenerate summary to reflect LEGITIMATE
            inp_j = json.loads(ex["input"])
            out_j["summary"] = _build_summary("LEGITIMATE", out_j.get("risk_level", "GREEN"), inp_j, out_j.get("evidence", []))
            # recommended_action stays based on risk_level/decision
            ex["output"] = json.dumps(out_j, ensure_ascii=False)
        except Exception:
            pass
        ex["_scenario_tag"] = "legitimate"
        ex["_pos"] = int(pos)
        per_tag_examples["legitimate"].append(ex)

    # Now stratified split 80/10/10 per tag
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for tag, examples in per_tag_examples.items():
        if not examples:
            continue
        # shuffle per tag deterministically (rng already advancing)
        perm = rng.permutation(len(examples))
        shuffled = [examples[i] for i in perm]
        n = len(shuffled)
        n_train = int(round(n * 0.8))
        n_val = int(round(n * 0.1))
        # ensure at least 1 val/test when possible for any reasonable n
        if n >= 5:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            # ensure test gets at least 1 when n >=5 and n is not tiny
            if n - n_train - n_val == 0 and n >= 5:
                # borrow from train to give test at least 1
                if n_train > 1:
                    n_train -= 1
        elif n >= 3:
            # for very small n, just ensure train at least 1
            if n_train == 0:
                n_train = 1
        n_test = n - n_train - n_val
        # fix rounding drift: ensure n_test >=0, if negative adjust
        if n_test < 0:
            # reduce train
            n_train += n_test
            n_test = 0
        if n_train + n_val + n_test != n:
            n_test = n - n_train - n_val

        train_slice = shuffled[:n_train]
        val_slice = shuffled[n_train:n_train + n_val]
        test_slice = shuffled[n_train + n_val:]

        # assign split label
        for ex in train_slice:
            ex["split"] = "train"
            splits["train"].append(ex)
        for ex in val_slice:
            ex["split"] = "val"
            splits["val"].append(ex)
        for ex in test_slice:
            ex["split"] = "test"
            splits["test"].append(ex)

    # Global shuffle within each split for realism (still stratified overall)
    for k in splits:
        perm = rng.permutation(len(splits[k]))
        splits[k] = [splits[k][i] for i in perm]

    return SplitDataset(splits)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def save_dataset(dataset: Any, path: str | Path) -> Path:
    """Save a dataset to JSONL.

    Parameters
    ----------
    dataset:
        Flat ``list`` of examples or ``dict`` with ``train/val/test`` keys.
        ``SplitDataset`` is handled automatically.
    path:
        Output file path. If ``dataset`` is a split dict and ``path`` is a
        directory, three files ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``
        are written inside it. Otherwise all examples are written to the single
        file at ``path``.

    Returns
    -------
    Path of the written file (or directory if split).
    """
    p = Path(path)

    # Normalize to flat list
    if isinstance(dataset, dict) and any(k in dataset for k in ("train", "val", "test")):
        # If path is a directory, write splits separately
        if p.suffix == "" and (p.is_dir() or not p.exists()):
            # treat as directory
            p.mkdir(parents=True, exist_ok=True)
            for split_name in ("train", "val", "test"):
                if split_name in dataset:
                    out = p / f"{split_name}.jsonl"
                    with out.open("w", encoding="utf-8") as f:
                        for ex in dataset[split_name]:  # type: ignore[index]
                            # strip internal keys _* before saving?
                            to_save = {k: v for k, v in ex.items() if not k.startswith("_")}
                            f.write(json.dumps(to_save, ensure_ascii=False) + "\n")
            return p
        else:
            # flatten
            flat: List[Dict[str, Any]] = []
            for v in dataset.values():
                if isinstance(v, list):
                    flat.extend(v)
            dataset = flat
    elif hasattr(dataset, "splits") and isinstance(getattr(dataset, "splits", None), dict):
        # SplitDataset – flatten for single-file save unless path is dir
        splits = getattr(dataset, "splits")
        if p.suffix == "" and (p.is_dir() or not p.exists()):
            p.mkdir(parents=True, exist_ok=True)
            for split_name in ("train", "val", "test"):
                if split_name in splits:
                    out = p / f"{split_name}.jsonl"
                    with out.open("w", encoding="utf-8") as f:
                        for ex in splits[split_name]:
                            to_save = {k: v for k, v in ex.items() if not k.startswith("_")}
                            f.write(json.dumps(to_save, ensure_ascii=False) + "\n")
            return p
        dataset = list(dataset)  # type: ignore[arg-type]

    # Single file – ensure parent exists
    p.parent.mkdir(parents=True, exist_ok=True)
    # dataset is now list
    lst: List[Dict[str, Any]] = list(dataset) if not isinstance(dataset, list) else dataset  # type: ignore[assignment]
    with p.open("w", encoding="utf-8") as f:
        for ex in lst:
            to_save = {k: v for k, v in ex.items() if not k.startswith("_")}
            f.write(json.dumps(to_save, ensure_ascii=False) + "\n")
    return p


def load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL dataset.

    Parameters
    ----------
    path:
        File containing one JSON object per line. If ``path`` is a directory,
        all ``*.jsonl`` files inside are concatenated.

    Returns
    -------
    list[dict]
    """
    p = Path(path)
    out: List[Dict[str, Any]] = []
    if p.is_dir():
        # load all jsonl files, sorted for determinism
        for f in sorted(p.glob("*.jsonl")):
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    out.append(json.loads(line))
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


__all__ = [
    "INSTRUCTION",
    "SCENARIO_TO_FRAUD_TYPE",
    "build_llm_example",
    "generate_llm_dataset",
    "save_dataset",
    "load_dataset",
    "SplitDataset",
]
