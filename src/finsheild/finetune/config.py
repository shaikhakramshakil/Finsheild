"""QLoRA configuration — Phase 14.

Defaults are CPU/local-friendly.  4-bit quantization is only enabled when
both CUDA and bitsandbytes are available.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _is_cuda_available() -> bool:
    """Return True iff CUDA is available via PyTorch."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _is_bnb_available() -> bool:
    """Return True iff bitsandbytes is importable."""
    return importlib.util.find_spec("bitsandbytes") is not None


def is_cuda_available() -> bool:
    """Public helper — CUDA detection."""
    return _is_cuda_available()


def is_bnb_available() -> bool:
    """Public helper — bitsandbytes detection."""
    return _is_bnb_available()


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' (same contract as finsheild.config)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        # mps on Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
    except Exception:
        pass
    return "cpu"


def effective_use_4bit(use_4bit: bool) -> bool:
    """Return True only if use_4bit was requested AND CUDA+bnb are available.

    This is the gate used by training — 4-bit quantization is never enabled
    on CPU or when bitsandbytes is missing, even if the config asks for it.
    """
    if not use_4bit:
        return False
    return _is_cuda_available() and _is_bnb_available()


@dataclass
class QLoRAConfig:
    """QLoRA fine-tuning hyper-parameters.

    Attributes mirror the assignment defaults so tests can assert them exactly.
    The ``use_4bit`` flag defaults to False; it is only honoured when both
    CUDA and bitsandbytes are present (see :func:`effective_use_4bit`).
    """

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation: int = 4
    max_seq_length: int = 512
    use_4bit: bool = False
    output_dir: str = "models/llm/adapter"

    # checkpoint / logging granularity
    save_steps: int = 25
    logging_steps: int = 10

    def __post_init__(self) -> None:
        # Normalize target_modules — allow comma-separated string or list
        if isinstance(self.target_modules, str):  # type: ignore[unreachable]
            self.target_modules = [s.strip() for s in self.target_modules.split(",") if s.strip()]  # type: ignore[attr-defined]
        # basic validation
        if self.lora_r <= 0:
            raise ValueError("lora_r must be > 0")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be > 0")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")

    @property
    def effective_use_4bit(self) -> bool:
        """Whether 4-bit will actually be used given current hardware."""
        return effective_use_4bit(self.use_4bit)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["effective_use_4bit"] = self.effective_use_4bit
        d["detected_device"] = detect_device()
        return d
