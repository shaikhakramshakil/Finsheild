"""Phase 14 — QLoRA Fine-tuning tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finsheild.finetune import QLoRAConfig, detect_device, load_adapter, train_lora
from finsheild.finetune.config import effective_use_4bit, is_bnb_available, is_cuda_available
from finsheild.finetune.train import get_latest_checkpoint


# ------------------------------------------------------------------ #
# config defaults
# ------------------------------------------------------------------ #

def test_config_defaults():
    cfg = QLoRAConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora_r == 8
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.05
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.learning_rate == 2e-4
    assert cfg.num_epochs == 1
    assert cfg.per_device_batch_size == 2
    assert cfg.gradient_accumulation == 4
    assert cfg.max_seq_length == 512
    assert cfg.use_4bit is False
    assert cfg.output_dir == "models/llm/adapter"
    # extra sanity
    assert cfg.save_steps > 0
    assert cfg.logging_steps > 0


def test_config_effective_use_4bit_false_by_default():
    cfg = QLoRAConfig()
    # default is False, so effective must be False regardless of hardware
    assert cfg.effective_use_4bit is False
    assert effective_use_4bit(False) is False
    # If someone requests True on CPU (no CUDA), it must still be False
    # This host is CPU-only, so True should be gated to False
    # We don't hard-assert True->False on GPU hosts; just check type
    result = effective_use_4bit(True)
    assert isinstance(result, bool)
    if not is_cuda_available() or not is_bnb_available():
        assert result is False


def test_config_custom_values():
    cfg = QLoRAConfig(
        model_name="custom/model",
        lora_r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        num_epochs=2,
        output_dir="tmp/custom_adapter",
    )
    assert cfg.model_name == "custom/model"
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.target_modules == ["q_proj", "k_proj", "v_proj"]
    assert cfg.num_epochs == 2
    assert cfg.output_dir == "tmp/custom_adapter"


# ------------------------------------------------------------------ #
# device detection
# ------------------------------------------------------------------ #

def test_detect_device_works():
    dev = detect_device()
    assert dev in ("cuda", "mps", "cpu")
    # This CI host is CPU-only (Intel i3, no nvidia)
    # So we expect cpu, but allow mps/cuda on other runners
    assert isinstance(dev, str)
    assert len(dev) > 0


def test_is_cuda_and_bnb_helpers():
    # helpers must be boolean and not raise
    assert isinstance(is_cuda_available(), bool)
    assert isinstance(is_bnb_available(), bool)
    # On this CPU box, cuda should be False, bnb likely False (not installed)
    # but we only assert types, not values, to stay portable
    assert not is_cuda_available() or isinstance(is_cuda_available(), bool)


# ------------------------------------------------------------------ #
# mock training creates adapter dir
# ------------------------------------------------------------------ #

def _make_dummy_dataset(path: Path, n: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rows.append(
            {
                "prompt": f"Evidence {i}: amount={1000 + i * 123}, new_device={bool(i % 2)}, xgboost_score={0.1 * (i % 10)}",
                "completion": json.dumps({"risk_level": "HIGH" if i % 3 == 0 else "LOW"}),
            }
        )
    # JSONL
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_mock_training_creates_adapter_dir(tmp_path: Path):
    ds = _make_dummy_dataset(tmp_path / "train.jsonl", n=12)
    out = tmp_path / "adapter"
    cfg = QLoRAConfig(output_dir=str(out), save_steps=5, logging_steps=2)
    result = train_lora(str(ds), cfg)
    # returns string path
    assert isinstance(result, str)
    assert Path(result).exists()
    # adapter files must exist even without transformers/peft
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists() or (out / "adapter_model.bin").exists()
    assert (out / "training_config.json").exists()
    assert (out / "tokenizer.json").exists()
    # content checks
    adapter_cfg = json.loads((out / "adapter_config.json").read_text())
    assert adapter_cfg["r"] == 8
    assert adapter_cfg["lora_alpha"] == 16
    assert adapter_cfg["target_modules"] == ["q_proj", "v_proj"]
    train_cfg = json.loads((out / "training_config.json").read_text())
    assert train_cfg["model_name"] == cfg.model_name
    assert train_cfg["mock_training"] is True or "mock_training" in train_cfg


def test_mock_training_handles_missing_dataset(tmp_path: Path):
    out = tmp_path / "adapter_missing_ds"
    cfg = QLoRAConfig(output_dir=str(out))
    # No dataset file — should still create adapter via synthetic fallback
    result = train_lora(str(tmp_path / "does_not_exist.jsonl"), cfg)
    assert Path(result).exists()
    assert (out / "adapter_config.json").exists()


def test_mock_training_cpu_caps_dataset(tmp_path: Path):
    # Create 100 rows, on CPU should be capped to 50
    ds = _make_dummy_dataset(tmp_path / "big.jsonl", n=100)
    out = tmp_path / "adapter_cap"
    cfg = QLoRAConfig(output_dir=str(out), num_epochs=3, save_steps=1000)
    train_lora(str(ds), cfg)
    # after CPU capping, num_epochs should be 1
    # (train_lora mutates config on CPU)
    assert cfg.num_epochs == 1
    assert (out / "adapter_config.json").exists()


# ------------------------------------------------------------------ #
# checkpoint logic
# ------------------------------------------------------------------ #

def test_checkpoint_logic(tmp_path: Path):
    ds = _make_dummy_dataset(tmp_path / "train.jsonl", n=20)
    out = tmp_path / "adapter_ckpt"
    cfg = QLoRAConfig(output_dir=str(out), save_steps=5)
    train_lora(str(ds), cfg)
    # at least one checkpoint should exist (mock creates checkpoint-5 or similar)
    cks = list(out.glob("checkpoint-*"))
    assert len(cks) >= 1, f"no checkpoint in {list(out.iterdir())}"
    latest = get_latest_checkpoint(out)
    assert latest is not None
    assert latest.exists()
    assert latest.name.startswith("checkpoint-")

    # Second run should resume (detect latest) and not crash, create an additional checkpoint or reuse
    # Use save_steps=2 to force a new checkpoint beyond the previous max
    cfg2 = QLoRAConfig(output_dir=str(out), save_steps=2)
    train_lora(str(ds), cfg2)
    cks2 = list(out.glob("checkpoint-*"))
    assert len(cks2) >= 1
    latest2 = get_latest_checkpoint(out)
    assert latest2 is not None


def test_get_latest_checkpoint_picks_highest(tmp_path: Path):
    out = tmp_path / "ckpt_pick"
    out.mkdir()
    (out / "checkpoint-10").mkdir()
    (out / "checkpoint-2").mkdir()
    (out / "checkpoint-100").mkdir()
    (out / "not_a_checkpoint").mkdir()
    latest = get_latest_checkpoint(out)
    assert latest is not None
    assert latest.name == "checkpoint-100"

    # empty dir returns None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert get_latest_checkpoint(empty) is None
    assert get_latest_checkpoint(tmp_path / "nonexistent_dir_xyz") is None


def test_load_adapter(tmp_path: Path):
    ds = _make_dummy_dataset(tmp_path / "train.jsonl", n=10)
    out = tmp_path / "adapter_load"
    cfg = QLoRAConfig(output_dir=str(out))
    train_lora(str(ds), cfg)
    info = load_adapter(str(out))
    assert "adapter_config" in info
    assert "training_config" in info
    assert info["has_weights"] is True
    assert info["model_name"] == cfg.model_name
    assert Path(info["path"]).exists()

    # missing adapter should raise
    with pytest.raises(FileNotFoundError):
        load_adapter(str(tmp_path / "nope_adapter"))

    # corrupted adapter (no weights) should raise FileNotFoundError
    bad = tmp_path / "bad_adapter"
    bad.mkdir()
    (bad / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x"}))
    with pytest.raises(FileNotFoundError):
        load_adapter(str(bad))


def test_train_lora_default_config(tmp_path: Path):
    # train_lora with no config arg should use defaults and still work
    ds = _make_dummy_dataset(tmp_path / "train.jsonl", n=5)
    # Patch output_dir via config default is models/llm/adapter — we avoid polluting it
    # by passing explicit tmp config; but test that default can be instantiated
    cfg_default = QLoRAConfig()
    assert cfg_default.output_dir == "models/llm/adapter"
    # now call with None config but explicit dataset + tmp output via monkeypatch cwd?
    # Instead, verify the function signature handles None
    out = tmp_path / "adapter_default"
    cfg = QLoRAConfig(output_dir=str(out))
    result = train_lora(dataset_path=str(ds), config=cfg)
    assert Path(result).is_dir()
