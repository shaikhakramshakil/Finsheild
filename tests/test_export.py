import pytest
from finsheild.export import export_all, verify_export

def test_export_all():
    exported = export_all()
    assert "xgboost" in exported
    assert "baseline" in exported
    assert "anomaly" in exported

def test_verify_export():
    export_all()
    results = verify_export()
    # At least xgboost and baseline should exist
    assert any(results.values())
    assert results["xgboost/model.joblib"] == True

def test_llm_adapter_exists():
    export_all()
    results = verify_export()
    assert results["llm/adapter/adapter_config.json"] == True
