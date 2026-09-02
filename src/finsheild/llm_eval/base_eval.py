"""Phase 13 — Base LLM evaluation (no GPU required).

Evaluates a small instruct model on the fraud-investigation copilot
dataset (Phase 12 outputs).  All GPU/transformers work is optional:
if torch/transformers are missing or the model is not cached and the
host is offline, evaluation is skipped gracefully and a
``BaseEvalResult(skipped=True)`` is returned.

Public surface
--------------
* :class:`BaseEvalResult` — dataclass with metrics + serialization
* :func:`evaluate_base_model` — real model evaluation
* :func:`evaluate_with_mock` — deterministic mock for CI / offline tests

Dataset shape
-------------
The dataset is a ``list[dict]`` or a JSON/JSONL file.  Each entry should
contain an input and an expected output.  Both of the common Phase-12
shapes are accepted:

* ``{"input": {...}, "output": {...}}``
* ``{"prompt": "...", "expected": {...}}``
* ``{"evidence": {...}, "label": {...}}``
* ``{"instruction": "...", "response": {...}}``

The ``output``/``expected``/``response``/``label`` dict is expected to
contain ``risk_level`` and ``fraud_type`` keys (case-insensitive).
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BaseEvalResult:
    """Metrics from a base-model evaluation run."""

    model_name: str = ""
    n_total: int = 0
    n_valid: int = 0
    json_valid_rate: float = 0.0
    risk_level_accuracy: float = 0.0
    fraud_type_accuracy: float = 0.0
    exact_match_rate: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    details: List[Dict[str, Any]] = field(default_factory=list)

    # alias expected by spec: exact_match == exact_match_rate
    @property
    def exact_match(self) -> float:
        return self.exact_match_rate

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep alias for consumers that check either key
        d["exact_match"] = self.exact_match_rate
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Helpers — dataset loading & JSON extraction
# ---------------------------------------------------------------------------

def _load_dataset(
    dataset: Union[str, Path, List[Dict[str, Any]], None],
    dataset_path: Union[str, Path, None] = None,
) -> List[Dict[str, Any]]:
    """Resolve dataset to a list of dicts.

    Accepts:
    * list already
    * str/Path pointing to .json or .jsonl file
    * two-arg form: ``evaluate_base_model(dataset_path=...)``
    """
    # two-arg compat: if first arg is path-like string and second is None, treat as path
    # caller may pass dataset=Path ; we handle both names
    raw: Any = dataset if dataset is not None else dataset_path
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    p = Path(str(raw))
    if not p.exists():
        warnings.warn(f"Dataset path does not exist: {p}")
        return []
    text = p.read_text(encoding="utf-8")
    # try JSON array first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # wrap single object
            return [data]
    except json.JSONDecodeError:
        pass
    # try JSONL
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _get_expected(example: Dict[str, Any]) -> Dict[str, Any]:
    """Extract expected output dict from an example with flexible keys."""
    for key in ("output", "expected", "response", "label", "target", "ground_truth"):
        v = example.get(key)
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                continue
    # fallback: the example itself might be the expected
    if "risk_level" in example or "fraud_type" in example:
        return example
    return {}


def _get_input(example: Dict[str, Any]) -> Any:
    for key in ("input", "prompt", "instruction", "evidence", "query", "context"):
        if key in example:
            return example[key]
    return example.get("input", example)


def _normalize_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().upper()


def _extract_json(text: str) -> tuple[Optional[Dict[str, Any]], bool]:
    """Try to parse *text* as JSON, with fallbacks for markdown fences."""
    if not text or not text.strip():
        return None, False
    # direct
    try:
        return json.loads(text), True
    except Exception:
        pass
    # strip markdown fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1)), True
        except Exception:
            pass
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet), True
        except Exception:
            pass
        # try to fix trailing commas
        snippet2 = re.sub(r",\s*([}\]])", r"\1", snippet)
        try:
            return json.loads(snippet2), True
        except Exception:
            pass
    return None, False


def _build_prompt(evidence: Any) -> str:
    """Build an instruct prompt for the fraud copilot task."""
    if isinstance(evidence, dict):
        evidence_str = json.dumps(evidence, indent=2)
    elif isinstance(evidence, str):
        evidence_str = evidence
    else:
        try:
            evidence_str = json.dumps(evidence, indent=2, default=str)
        except Exception:
            evidence_str = str(evidence)
    return (
        "You are a fraud investigation copilot. Given the structured fraud evidence below, "
        "output ONLY valid JSON with keys: risk_level, fraud_type, summary, evidence, recommended_action.\n"
        "risk_level must be one of LOW, MEDIUM, HIGH (or GREEN, YELLOW, RED).\n"
        "fraud_type examples: LEGITIMATE, ACCOUNT_TAKEOVER, VELOCITY_ABUSE, UNUSUAL_LOCATION, UNUSUAL_AMOUNT, DEVICE_COMPROMISE, MULE_ACCOUNT, MERCHANT_ANOMALY, COORDINATED_ACTIVITY.\n"
        f"Evidence:\n{evidence_str}\n"
        "Respond with JSON only:"
    )


# ---------------------------------------------------------------------------
# Mock evaluation (no model needed)
# ---------------------------------------------------------------------------

def _mock_predict(evidence: Any) -> Dict[str, Any]:
    """Deterministic heuristic mock.

    Rules (simple, reproducible):
    * HIGH if amount >> historical average, or high velocity, or new_device+high value, or shared_device_accounts >=3
    * MEDIUM if any moderate signal
    * else LOW

    fraud_type chosen from the strongest signal.
    """
    if not isinstance(evidence, dict):
        # try to parse if evidence is string containing json
        try:
            evidence = json.loads(evidence) if isinstance(evidence, str) else {}
        except Exception:
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}

    amount = 0.0
    hist_avg = 0.0
    for k in ("transaction_amount", "amount", "amount_x", "amount_y"):
        if k in evidence and evidence[k] is not None:
            try:
                amount = float(evidence[k])
                break
            except Exception:
                continue
    for k in ("historical_average", "historical_avg", "prior_mean_amount", "mean_amount", "avg_amount"):
        if k in evidence and evidence[k] is not None:
            try:
                hist_avg = float(evidence[k])
                break
            except Exception:
                continue

    new_device = bool(evidence.get("new_device", evidence.get("is_new_device", False)))
    vel = 0
    for k in ("recent_transaction_count", "vel_count_300s", "vel_count_3600s", "velocity", "tx_count"):
        if k in evidence and evidence[k] is not None:
            try:
                vel = int(evidence[k])
                break
            except Exception:
                continue
    shared = 0
    # handle nested graph_signals
    gs = evidence.get("graph_signals", {})
    if isinstance(gs, dict):
        shared = int(gs.get("shared_device_accounts", gs.get("shared_device_count", 0)) or 0)
    for k in ("shared_device_accounts", "device_account_count", "shared_device_count"):
        if k in evidence and evidence[k] is not None:
            try:
                shared = max(shared, int(evidence[k]))
            except Exception:
                pass
    xgb_score = float(evidence.get("xgboost_score", evidence.get("xgb_score", 0)) or 0)
    anomaly_score = float(evidence.get("anomaly_score", 0) or 0)
    distance = float(evidence.get("location_distance_km", evidence.get("distance_km", 0)) or 0)
    # booleans
    unusual_location = bool(evidence.get("unusual_location", False))
    # triggered rules
    rules = evidence.get("triggered_rules", evidence.get("rules", []))
    if isinstance(rules, str):
        rules = [rules]
    if not isinstance(rules, list):
        rules = []

    # decision logic
    fraud_type = "LEGITIMATE"
    risk_level = "LOW"

    # check strong signals first
    if shared >= 3 or any("SHARED" in str(r).upper() or "DEVICE_SHARING" in str(r).upper() for r in rules):
        risk_level = "HIGH"
        fraud_type = "DEVICE_COMPROMISE"
    elif new_device and amount > 1000:
        risk_level = "HIGH"
        fraud_type = "ACCOUNT_TAKEOVER"
    elif hist_avg > 0 and amount > hist_avg * 3:
        risk_level = "HIGH"
        fraud_type = "UNUSUAL_AMOUNT"
    elif vel >= 5 or any("VELOCITY" in str(r).upper() or "HIGH_VELOCITY" in str(r).upper() for r in rules):
        risk_level = "HIGH"
        fraud_type = "VELOCITY_ABUSE"
    elif xgb_score > 0.7 or anomaly_score > 0.7:
        risk_level = "HIGH"
        fraud_type = "ACCOUNT_TAKEOVER"
    elif distance > 300 or unusual_location:
        risk_level = "MEDIUM"
        fraud_type = "UNUSUAL_LOCATION"
    elif vel >= 2 or xgb_score > 0.4:
        risk_level = "MEDIUM"
        fraud_type = "VELOCITY_ABUSE"
    elif amount > hist_avg * 1.5 and hist_avg > 0:
        risk_level = "MEDIUM"
        fraud_type = "UNUSUAL_AMOUNT"
    else:
        risk_level = "LOW"
        fraud_type = "LEGITIMATE"

    return {
        "risk_level": risk_level,
        "fraud_type": fraud_type,
        "summary": f"Mock assessment: {risk_level} risk, {fraud_type}.",
        "evidence": [f"mock: amount={amount}, hist_avg={hist_avg}, new_device={new_device}, vel={vel}"],
        "recommended_action": "APPROVE" if risk_level == "LOW" else ("STEP_UP" if risk_level == "MEDIUM" else "BLOCK"),
    }


def evaluate_with_mock(
    dataset: Union[str, Path, List[Dict[str, Any]], None] = None,
    dataset_path: Union[str, Path, None] = None,
) -> BaseEvalResult:
    """Evaluate using deterministic mock — no model download required."""
    data = _load_dataset(dataset, dataset_path)
    n_total = len(data)
    if n_total == 0:
        return BaseEvalResult(model_name="mock", n_total=0, n_valid=0,
                              json_valid_rate=0.0, risk_level_accuracy=0.0,
                              fraud_type_accuracy=0.0, exact_match_rate=0.0,
                              details=[])

    n_valid = 0
    correct_risk = 0
    correct_fraud = 0
    exact = 0
    details: List[Dict[str, Any]] = []

    for ex in data:
        inp = _get_input(ex)
        expected = _get_expected(ex)
        pred = _mock_predict(inp)

        # mock always produces valid json
        n_valid += 1
        is_valid = True

        exp_risk = _normalize_str(expected.get("risk_level"))
        exp_fraud = _normalize_str(expected.get("fraud_type"))
        pred_risk = _normalize_str(pred.get("risk_level"))
        pred_fraud = _normalize_str(pred.get("fraud_type"))

        risk_ok = (exp_risk == "" or pred_risk == exp_risk)
        # if expected has no risk_level, don't count as error (accuracy denominator still total but we count as correct)
        # For metrics definition: accuracy = correct / total where missing expected counts as correct.
        # Safer: if expected empty, we don't penalize.
        if exp_risk and pred_risk == exp_risk:
            correct_risk += 1
        elif not exp_risk:
            correct_risk += 1

        if exp_fraud and pred_fraud == exp_fraud:
            correct_fraud += 1
        elif not exp_fraud:
            correct_fraud += 1

        # exact match: full pred dict equals expected (for comparable keys)
        # We consider exact if risk_level and fraud_type both match and if expected has summary we ignore summary differences unless exact
        # For testability: exact means pred == expected when expected keys subset of pred and values equal (case-insensitive for risk/fraud)
        is_exact = False
        if expected:
            # compare risk_level + fraud_type exactly; if expected has only those two keys, that's exact
            if exp_risk and exp_fraud:
                is_exact = (pred_risk == exp_risk and pred_fraud == exp_fraud and len(expected) <= 5)
                # if expected fully equals pred (deep), also exact
                if not is_exact:
                    # deep compare normalized
                    try:
                        norm_pred = {k.upper(): _normalize_str(v) if isinstance(v, str) else v for k, v in pred.items()}
                        norm_exp = {k.upper(): _normalize_str(v) if isinstance(v, str) else v for k, v in expected.items()}
                        # check expected subset matches
                        is_exact = all(norm_pred.get(k) == v for k, v in norm_exp.items())
                    except Exception:
                        is_exact = False
            else:
                is_exact = False
            if is_exact:
                exact += 1
        else:
            # no expected -> cannot be exact
            pass

        details.append({
            "input": inp,
            "expected": expected,
            "predicted": pred,
            "json_valid": is_valid,
            "risk_match": risk_ok,
            "fraud_match": (exp_fraud == "" or pred_fraud == exp_fraud),
            "exact_match": is_exact,
        })

    return BaseEvalResult(
        model_name="mock",
        n_total=n_total,
        n_valid=n_valid,
        json_valid_rate=n_valid / n_total if n_total else 0.0,
        risk_level_accuracy=correct_risk / n_total if n_total else 0.0,
        fraud_type_accuracy=correct_fraud / n_total if n_total else 0.0,
        exact_match_rate=exact / n_total if n_total else 0.0,
        details=details,
    )


# ---------------------------------------------------------------------------
# Real model evaluation
# ---------------------------------------------------------------------------

def evaluate_base_model(
    dataset: Union[str, Path, List[Dict[str, Any]], None] = None,
    dataset_path: Union[str, Path, None] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    fallback_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
) -> BaseEvalResult:
    """Evaluate an instruct model on the fraud copilot dataset.

    Parameters
    ----------
    dataset / dataset_path:
        Either a list of examples or a path to a JSON/JSONL file.
        ``dataset`` is preferred; ``dataset_path`` is accepted for
        backwards compat.
    model_name:
        Primary HF model id. Defaults to ``Qwen/Qwen2.5-0.5B-Instruct``.
    fallback_model:
        Used when *model_name* is not available locally (CPU/CUDA-less
        host). Defaults to ``HuggingFaceTB/SmolLM2-135M-Instruct``.
    max_new_tokens:
        Generation budget per sample (default 256).
    max_samples:
        Optional cap for quick smoke runs (``None`` = all samples).
    """
    data = _load_dataset(dataset, dataset_path)
    n_total = len(data)

    if n_total == 0:
        return BaseEvalResult(
            model_name=model_name,
            n_total=0,
            n_valid=0,
            json_valid_rate=0.0,
            risk_level_accuracy=0.0,
            fraud_type_accuracy=0.0,
            exact_match_rate=0.0,
            details=[],
        )

    if max_samples is not None and max_samples < n_total:
        data = data[:max_samples]
        n_total = len(data)

    # Try to import transformers + torch lazily
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as e:
        return BaseEvalResult(
            model_name=model_name,
            n_total=n_total,
            n_valid=0,
            json_valid_rate=0.0,
            risk_level_accuracy=0.0,
            fraud_type_accuracy=0.0,
            exact_match_rate=0.0,
            skipped=True,
            skip_reason=f"transformers/torch not installed: {e}",
            details=[],
        )

    # Detect device / dtype
    use_cuda = False
    try:
        use_cuda = bool(torch.cuda.is_available())
    except Exception:
        use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    # dtype: auto — float16 on cuda, float32 on cpu
    torch_dtype = torch.float16 if use_cuda else torch.float32

    # Attempt to load model; graceful fallback + offline handling
    tokenizer = None
    model = None
    chosen_model = model_name
    last_error: Optional[str] = None

    for candidate in (model_name, fallback_model):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate, trust_remote_code=trust_remote_code, local_files_only=False
            )
            # ensure pad token
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            # device_map auto requires accelerate; avoid on pure CPU without accelerate
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }
            # only pass torch_dtype if supported
            try:
                load_kwargs["torch_dtype"] = torch_dtype
            except Exception:
                pass
            # use device_map auto only if cuda or accelerate available
            if use_cuda:
                load_kwargs["device_map"] = "auto"
            else:
                # cpu: explicit, avoids accelerate requirement
                pass

            # Try with local_files_only first to avoid hanging on offline host, then retry without
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    candidate, local_files_only=True, **load_kwargs
                )
            except Exception:
                # network attempt (may hang briefly; set low timeout via env if needed)
                model = AutoModelForCausalLM.from_pretrained(
                    candidate, local_files_only=False, **load_kwargs
                )

            chosen_model = candidate
            if not use_cuda:
                try:
                    model = model.to(device)  # type: ignore[attr-defined]
                except Exception:
                    pass
            model.eval()  # type: ignore[attr-defined]
            last_error = None
            break
        except Exception as e:
            last_error = f"{candidate}: {e}"
            tokenizer = None
            model = None
            continue

    if model is None or tokenizer is None:
        warnings.warn(f"Could not load model. Last error: {last_error}. Skipping.")
        return BaseEvalResult(
            model_name=chosen_model,
            n_total=n_total,
            n_valid=0,
            json_valid_rate=0.0,
            risk_level_accuracy=0.0,
            fraud_type_accuracy=0.0,
            exact_match_rate=0.0,
            skipped=True,
            skip_reason=f"Model load failed: {last_error}. No internet or not cached; skipped gracefully.",
            details=[],
        )

    # Generation loop
    n_valid = 0
    correct_risk = 0
    correct_fraud = 0
    exact = 0
    details: List[Dict[str, Any]] = []

    for ex in data:
        inp = _get_input(ex)
        expected = _get_expected(ex)
        prompt = _build_prompt(inp)

        pred_dict: Optional[Dict[str, Any]] = None
        is_valid = False
        raw_text = ""
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)  # type: ignore[attr-defined]
            # move to device
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore
            except Exception:
                pass
            with torch.no_grad():  # type: ignore
                outputs = model.generate(  # type: ignore[attr-defined]
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,  # type: ignore[attr-defined]
                )
            # decode only new tokens
            input_len = inputs["input_ids"].shape[1]  # type: ignore
            gen_tokens = outputs[0][input_len:]  # type: ignore
            raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)  # type: ignore[attr-defined]
            pred_dict, is_valid = _extract_json(raw_text)
            if pred_dict is None:
                # try decoding full output fallback
                full = tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore[attr-defined]
                pred_dict, is_valid = _extract_json(full)
        except Exception as e:
            raw_text = f"__generation_error__ {e}"
            pred_dict = None
            is_valid = False

        if is_valid and pred_dict is not None:
            n_valid += 1
        else:
            # keep pred_dict as None for metrics (counts as invalid)
            pred_dict = pred_dict or {}

        exp_risk = _normalize_str(expected.get("risk_level")) if expected else ""
        exp_fraud = _normalize_str(expected.get("fraud_type")) if expected else ""
        pred_risk = _normalize_str(pred_dict.get("risk_level")) if pred_dict else ""
        pred_fraud = _normalize_str(pred_dict.get("fraud_type")) if pred_dict else ""

        risk_ok = bool(exp_risk == "" or (is_valid and pred_risk == exp_risk))
        fraud_ok = bool(exp_fraud == "" or (is_valid and pred_fraud == exp_fraud))

        if exp_risk and is_valid and pred_risk == exp_risk:
            correct_risk += 1
        elif not exp_risk:
            # missing expected -> count as correct to avoid penalizing incomplete labels
            correct_risk += 1

        if exp_fraud and is_valid and pred_fraud == exp_fraud:
            correct_fraud += 1
        elif not exp_fraud:
            correct_fraud += 1

        is_exact = False
        if expected and is_valid and pred_dict is not None:
            try:
                # exact if all expected keys match (normalized for strings)
                match = True
                for k, v in expected.items():
                    if isinstance(v, str) and isinstance(pred_dict.get(k), str):
                        if _normalize_str(pred_dict.get(k)) != _normalize_str(v):
                            match = False
                            break
                    else:
                        if pred_dict.get(k) != v:
                            match = False
                            break
                if match:
                    is_exact = True
                    exact += 1
            except Exception:
                is_exact = False

        details.append({
            "input": inp,
            "expected": expected,
            "raw_output": raw_text,
            "predicted": pred_dict,
            "json_valid": is_valid,
            "risk_match": risk_ok,
            "fraud_match": fraud_ok,
            "exact_match": is_exact,
        })

    return BaseEvalResult(
        model_name=chosen_model,
        n_total=n_total,
        n_valid=n_valid,
        json_valid_rate=n_valid / n_total if n_total else 0.0,
        risk_level_accuracy=correct_risk / n_total if n_total else 0.0,
        fraud_type_accuracy=correct_fraud / n_total if n_total else 0.0,
        exact_match_rate=exact / n_total if n_total else 0.0,
        details=details,
    )
