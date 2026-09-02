import pytest
from finsheild.compare_llm import compare_models

def test_compare_with_mock():
    dataset = [{"input": "test", "output": {"risk_level": "HIGH", "fraud_type": "ACCOUNT_TAKEOVER"}}] * 5
    result = compare_models(dataset)
    assert result.n_test == 5
    assert "risk_level_accuracy" in result.base
    assert "risk_level_accuracy" in result.finetuned

def test_compare_delta():
    dataset = [{"input": "x", "output": {"risk_level": "LOW"}}] * 3
    r = compare_models(dataset)
    assert r.delta["risk_level_accuracy"] >= 0

def test_markdown():
    dataset = [{"input": "x", "output": {"risk_level": "LOW"}}] * 2
    r = compare_models(dataset)
    md = r.markdown()
    assert "Phase 15" in md
    assert "Base" in md
