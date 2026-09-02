"""Phase 13 — Base LLM Eval tests (mock-based, no model download)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from finsheild.llm_eval import BaseEvalResult, evaluate_base_model, evaluate_with_mock
from finsheild.llm_eval.base_eval import _extract_json, _mock_predict


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def _sample_dataset():
    """Return a small deterministic dataset covering multiple scenarios."""
    return [
        {
            "input": {
                "transaction_amount": 50000,
                "historical_average": 4200,
                "new_device": True,
                "location_distance_km": 10,
                "recent_transaction_count": 1,
                "xgboost_score": 0.91,
                "anomaly_score": 0.83,
                "triggered_rules": ["NEW_DEVICE_HIGH_VALUE", "HIGH_VELOCITY"],
                "graph_signals": {"shared_device_accounts": 1},
            },
            "output": {
                # mock will predict HIGH / ACCOUNT_TAKEOVER (new_device+amount>1000)
                "risk_level": "HIGH",
                "fraud_type": "ACCOUNT_TAKEOVER",
                "summary": "High risk account takeover",
                "evidence": [],
                "recommended_action": "BLOCK",
            },
        },
        {
            "input": {
                "transaction_amount": 120,
                "historical_average": 100,
                "new_device": False,
                "location_distance_km": 5,
                "recent_transaction_count": 1,
                "xgboost_score": 0.05,
                "anomaly_score": 0.1,
                "triggered_rules": [],
                "graph_signals": {"shared_device_accounts": 0},
            },
            "output": {
                "risk_level": "LOW",
                "fraud_type": "LEGITIMATE",
                "summary": "Legitimate",
                "evidence": [],
                "recommended_action": "APPROVE",
            },
        },
        {
            "input": {
                "transaction_amount": 320,
                "historical_average": 200,
                "new_device": False,
                "location_distance_km": 0,
                "recent_transaction_count": 6,
                "xgboost_score": 0.3,
                "anomaly_score": 0.2,
                "triggered_rules": ["HIGH_VELOCITY"],
                "graph_signals": {"shared_device_accounts": 0},
            },
            "output": {
                # mock: velocity >=5 => HIGH VELOCITY_ABUSE (ratio 1.6 < 3 so not UNUSUAL_AMOUNT)
                "risk_level": "HIGH",
                "fraud_type": "VELOCITY_ABUSE",
                "summary": "velocity",
                "evidence": [],
                "recommended_action": "BLOCK",
            },
        },
        {
            "input": {
                "transaction_amount": 300,
                "historical_average": 250,
                "new_device": False,
                "location_distance_km": 450,
                "recent_transaction_count": 1,
                "xgboost_score": 0.2,
                "anomaly_score": 0.15,
                "triggered_rules": [],
                "graph_signals": {"shared_device_accounts": 0},
            },
            "output": {
                # mock: distance >300 => MEDIUM UNUSUAL_LOCATION
                "risk_level": "MEDIUM",
                "fraud_type": "UNUSUAL_LOCATION",
                "summary": "location",
                "evidence": [],
                "recommended_action": "STEP_UP",
            },
        },
    ]


# ------------------------------------------------------------------ #
# 1 — mock evaluation works
# ------------------------------------------------------------------ #

def test_mock_evaluation_works():
    data = _sample_dataset()
    result = evaluate_with_mock(data)
    assert isinstance(result, BaseEvalResult)
    assert result.n_total == 4
    assert result.n_valid == 4
    assert result.json_valid_rate == 1.0
    # all examples crafted to match mock exactly
    assert result.risk_level_accuracy == 1.0
    assert result.fraud_type_accuracy == 1.0
    assert result.exact_match_rate == 1.0
    assert result.model_name == "mock"
    assert len(result.details) == 4
    for d in result.details:
        assert d["json_valid"] is True


# ------------------------------------------------------------------ #
# 2 — metrics computed correctly (partial match)
# ------------------------------------------------------------------ #

def test_metrics_computed_correctly_partial():
    # Only first 2 match, next 2 are intentionally wrong expected
    data = [
        {
            "input": {"transaction_amount": 50000, "historical_average": 4200, "new_device": True, "xgboost_score": 0.9},
            "output": {"risk_level": "HIGH", "fraud_type": "ACCOUNT_TAKEOVER"},
        },
        {
            "input": {"transaction_amount": 120, "historical_average": 100, "new_device": False, "xgboost_score": 0.05},
            "output": {"risk_level": "LOW", "fraud_type": "LEGITIMATE"},
        },
        {
            "input": {"transaction_amount": 120, "historical_average": 100, "new_device": False, "xgboost_score": 0.05},
            # wrong expected: mock will predict LOW LEGITIMATE but we say HIGH
            "output": {"risk_level": "HIGH", "fraud_type": "ACCOUNT_TAKEOVER"},
        },
        {
            "input": {"transaction_amount": 300, "historical_average": 100, "new_device": False, "location_distance_km": 500},
            # mock will predict MEDIUM UNUSUAL_LOCATION, we claim LOW LEGITIMATE
            "output": {"risk_level": "LOW", "fraud_type": "LEGITIMATE"},
        },
    ]
    result = evaluate_with_mock(data)
    assert result.n_total == 4
    # 2 correct out of 4
    assert result.risk_level_accuracy == pytest.approx(0.5)
    assert result.fraud_type_accuracy == pytest.approx(0.5)
    assert result.exact_match_rate == pytest.approx(0.5)
    assert result.json_valid_rate == 1.0


def test_mock_heuristic_deterministic():
    # same input always gives same output
    inp = {"transaction_amount": 10000, "historical_average": 1000, "new_device": True}
    a = _mock_predict(inp)
    b = _mock_predict(inp)
    assert a == b
    assert a["risk_level"] == "HIGH"
    assert a["fraud_type"] == "ACCOUNT_TAKEOVER"

    # json string input also works
    c = _mock_predict(json.dumps(inp))
    assert c["risk_level"] == "HIGH"


# ------------------------------------------------------------------ #
# 3 — BaseEvalResult serialization
# ------------------------------------------------------------------ #

def test_base_eval_result_serialization():
    r = BaseEvalResult(
        model_name="test-model",
        n_total=10,
        n_valid=8,
        json_valid_rate=0.8,
        risk_level_accuracy=0.7,
        fraud_type_accuracy=0.6,
        exact_match_rate=0.5,
    )
    d = r.to_dict()
    assert d["model_name"] == "test-model"
    assert d["n_total"] == 10
    assert d["json_valid_rate"] == 0.8
    assert d["exact_match"] == 0.5
    assert d["exact_match_rate"] == 0.5
    # to_json round-trip
    j = r.to_json()
    loaded = json.loads(j)
    assert loaded["risk_level_accuracy"] == 0.7
    assert loaded["n_valid"] == 8
    # skipped defaults
    assert loaded["skipped"] is False

    # skipped result
    s = BaseEvalResult(model_name="x", n_total=5, skipped=True, skip_reason="offline")
    sd = s.to_dict()
    assert sd["skipped"] is True
    assert sd["skip_reason"] == "offline"


def test_extract_json_helper():
    valid, ok = _extract_json('{"risk_level": "HIGH", "fraud_type": "X"}')
    assert ok and valid["risk_level"] == "HIGH"

    # markdown fence
    txt = '```json\n{"risk_level": "LOW"}\n```'
    valid, ok = _extract_json(txt)
    assert ok

    # embedded
    txt2 = 'Here is JSON: {"risk_level": "MEDIUM", "fraud_type": "Y"} done'
    valid, ok = _extract_json(txt2)
    assert ok and valid["fraud_type"] == "Y"

    # invalid
    valid, ok = _extract_json('not json at all')
    assert not ok and valid is None


# ------------------------------------------------------------------ #
# 4 — handles empty dataset
# ------------------------------------------------------------------ #

def test_handles_empty_dataset():
    r = evaluate_with_mock([])
    assert r.n_total == 0
    assert r.n_valid == 0
    assert r.json_valid_rate == 0.0
    assert r.risk_level_accuracy == 0.0
    assert r.fraud_type_accuracy == 0.0
    assert r.exact_match_rate == 0.0
    assert r.details == []

    # also via None
    r2 = evaluate_with_mock(None)
    assert r2.n_total == 0

    # base model empty should also not crash (no torch needed)
    r3 = evaluate_base_model([])
    assert r3.n_total == 0
    assert r3.json_valid_rate == 0.0


# ------------------------------------------------------------------ #
# 5 — file path loading (json + jsonl)
# ------------------------------------------------------------------ #

def test_load_from_json_file():
    data = _sample_dataset()[:2]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        r = evaluate_with_mock(path)
        assert r.n_total == 2
        # also via dataset_path kwarg
        r2 = evaluate_with_mock(dataset_path=path)
        assert r2.n_total == 2

        # base model with empty-ish but file path (mock torch missing -> skip)
        # we just verify it doesn't crash on file load for base; it will try to load model
        # so use empty file to test empty path handling without model load
        empty_path = Path(path).with_suffix(".empty.json")
        empty_path.write_text("[]", encoding="utf-8")
        r3 = evaluate_with_mock(dataset_path=str(empty_path))
        assert r3.n_total == 0
    finally:
        Path(path).unlink(missing_ok=True)
        Path(path).with_suffix(".empty.json").unlink(missing_ok=True)


def test_load_from_jsonl_file():
    data = _sample_dataset()[:2]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")
        path = f.name
    try:
        r = evaluate_with_mock(path)
        assert r.n_total == 2
        assert r.json_valid_rate == 1.0
    finally:
        Path(path).unlink(missing_ok=True)


# ------------------------------------------------------------------ #
# 6 — evaluate_base_model graceful skip when no torch/transformers
# ------------------------------------------------------------------ #

def test_evaluate_base_model_skips_gracefully_without_model():
    # This host has no torch, so evaluate_base_model should return skipped=True
    # unless torch is installed; either way it must not raise
    data = _sample_dataset()[:1]
    result = evaluate_base_model(data, model_name="nonexistent/model-xyz-999")
    assert isinstance(result, BaseEvalResult)
    assert result.n_total == 1
    # If torch/transformers missing it should be skipped
    # If they are present but model not cached offline, it also skips
    # So we just assert it defines the skipped field correctly (bool)
    assert isinstance(result.skipped, bool)
    if result.skipped:
        assert result.skip_reason != ""
        assert result.json_valid_rate == 0.0
    else:
        # if somehow model loaded, metrics are defined
        assert 0.0 <= result.json_valid_rate <= 1.0


def test_evaluate_base_model_handles_alternate_dataset_keys():
    # Phase 12 may use different key names; ensure flexibility
    data = [
        {"prompt": json.dumps({"transaction_amount": 120, "historical_average": 100}), "expected": {"risk_level": "LOW", "fraud_type": "LEGITIMATE"}},
        {"evidence": {"transaction_amount": 50000, "historical_average": 1000, "new_device": True}, "label": {"risk_level": "HIGH", "fraud_type": "ACCOUNT_TAKEOVER"}},
    ]
    r = evaluate_with_mock(data)
    assert r.n_total == 2
    # second should be HIGH ACCOUNT_TAKEOVER
    assert r.risk_level_accuracy == 1.0
