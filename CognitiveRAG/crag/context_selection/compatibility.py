from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from CognitiveRAG.crag.contracts.schemas import ContextCandidate


_NEGATIVE_MARKERS = {"not", "never", "cannot", "cant", "can't", "failed", "failure", "no", "disabled", "off"}
_POSITIVE_MARKERS = {"works", "working", "succeeds", "success", "yes", "enabled", "active", "on"}
_SUPPORTED_COMPAT_MODES = {"heuristic", "nli"}
_SUPPORTED_NLI_BACKENDS = {"transformers"}
_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(tok) >= 3}


def _polarity(text: str) -> str:
    toks = _tokens(text)
    if {"not", "never", "cannot", "cant", "can't", "no"} & toks:
        return "negative"
    has_neg = bool(toks & _NEGATIVE_MARKERS)
    has_pos = bool(toks & _POSITIVE_MARKERS)
    if has_neg and not has_pos:
        return "negative"
    if has_pos and not has_neg:
        return "positive"
    return "mixed"


@dataclass(frozen=True)
class CompatibilityDecision:
    conflict: bool
    reason: str | None = None
    engine: str = "heuristic"


@dataclass(frozen=True)
class RuntimeCompatibilityState:
    configured_mode: str
    configured_backend: str
    configured_model: str
    resolved_engine: str
    backend_available: bool
    fallback_active: bool
    reason_code: str = ""
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class CompatibilityEngine(Protocol):
    name: str

    def evaluate(self, candidate: ContextCandidate, selected: list[ContextCandidate]) -> CompatibilityDecision: ...


class PairwiseNLIAdapter(Protocol):
    def contradiction_score(self, left: str, right: str) -> float: ...


class HeuristicCompatibilityEngine:
    """Deterministic heuristic compatibility gate.

    Priority order:
    1) explicit provenance claim key/value mismatch
    2) opposite polarity over overlapping lexical terms
    """
    name = "heuristic"

    def evaluate(self, candidate: ContextCandidate, selected: list[ContextCandidate]) -> CompatibilityDecision:
        cand_prov: dict[str, Any] = dict(candidate.provenance or {})
        cand_claim_key = str(cand_prov.get("claim_key") or "").strip().lower()
        cand_claim_value = str(cand_prov.get("claim_value") or "").strip().lower()

        cand_tokens = _tokens(candidate.text)
        cand_polarity = _polarity(candidate.text)

        for existing in selected:
            existing_prov: dict[str, Any] = dict(existing.provenance or {})
            ex_claim_key = str(existing_prov.get("claim_key") or "").strip().lower()
            ex_claim_value = str(existing_prov.get("claim_value") or "").strip().lower()

            if cand_claim_key and ex_claim_key and cand_claim_key == ex_claim_key:
                if cand_claim_value and ex_claim_value and cand_claim_value != ex_claim_value:
                    return CompatibilityDecision(conflict=True, reason="compatibility_conflict", engine=self.name)

            if cand_polarity == "mixed":
                continue
            ex_polarity = _polarity(existing.text)
            if ex_polarity == "mixed" or ex_polarity == cand_polarity:
                continue
            overlap = cand_tokens & _tokens(existing.text)
            if len(overlap) >= 2:
                return CompatibilityDecision(conflict=True, reason="compatibility_conflict", engine=self.name)

        return CompatibilityDecision(conflict=False, reason=None, engine=self.name)


class NLIBackedCompatibilityEngine:
    """Optional NLI-backed gate with deterministic heuristic fallback.

    This keeps B3 truthful:
    - NLI path is real when an adapter is supplied.
    - Selector stays deterministic via heuristic fallback if adapter is absent/failing.
    """

    name = "nli"

    def __init__(
        self,
        *,
        adapter: PairwiseNLIAdapter | None,
        fallback_engine: CompatibilityEngine | None = None,
        contradiction_threshold: float = 0.75,
    ):
        self._adapter = adapter
        self._fallback = fallback_engine or HeuristicCompatibilityEngine()
        self._threshold = float(max(0.0, min(1.0, contradiction_threshold)))

    def evaluate(self, candidate: ContextCandidate, selected: list[ContextCandidate]) -> CompatibilityDecision:
        if self._adapter is None:
            return self._fallback.evaluate(candidate, selected)

        # Preserve deterministic explicit claim-key conflict handling.
        fallback = self._fallback.evaluate(candidate, selected)
        if fallback.conflict:
            return fallback

        try:
            for existing in selected:
                score = float(self._adapter.contradiction_score(existing.text, candidate.text))
                if score >= self._threshold:
                    return CompatibilityDecision(conflict=True, reason="compatibility_conflict_nli", engine=self.name)
        except Exception:
            return self._fallback.evaluate(candidate, selected)
        return CompatibilityDecision(conflict=False, reason=None, engine=self.name)


class TransformersNLIAdapter:
    """Optional adapter that uses a local transformers NLI pipeline if available."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        from transformers import pipeline  # type: ignore[import-not-found]

        self._classifier = pipeline("text-classification", model=model_name)

    def contradiction_score(self, left: str, right: str) -> float:
        # Existing HF NLI models generally emit CONTRADICTION/NEUTRAL/ENTAILMENT labels.
        result = self._classifier({"text": left, "text_pair": right}, truncation=True)
        if isinstance(result, list) and result:
            top = result[0]
        else:
            top = result
        label = str((top or {}).get("label", "")).strip().lower()
        score = float((top or {}).get("score", 0.0))
        if "contradiction" in label:
            return score
        if "entailment" in label:
            return 0.0
        return min(0.5, score)


def _has_local_model_asset(model_name: str) -> bool:
    text = str(model_name or "").strip()
    if not text:
        return False

    candidate_path = Path(text).expanduser()
    if candidate_path.is_dir():
        return True

    cache_key = f"models--{text.replace('/', '--')}"
    hub_dirs: list[Path] = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        hub_dirs.append(Path(hf_home).expanduser() / "hub")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        hub_dirs.append(Path(hub_cache).expanduser())
    hub_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[str] = set()
    for hub_dir in hub_dirs:
        key = str(hub_dir)
        if key in seen:
            continue
        seen.add(key)
        if (hub_dir / cache_key).exists():
            return True
    return False


def check_transformers_nli_backend(model_name: str = _DEFAULT_NLI_MODEL) -> dict[str, Any]:
    if importlib.util.find_spec("transformers") is None:
        return {
            "backend": "transformers",
            "available": False,
            "reason_code": "missing_dependency",
            "reason": "transformers_import_not_found",
            "model_name": model_name,
        }
    if not _has_local_model_asset(model_name):
        return {
            "backend": "transformers",
            "available": False,
            "reason_code": "missing_model_asset",
            "reason": "local_model_asset_not_found",
            "model_name": model_name,
        }
    return {
        "backend": "transformers",
        "available": True,
        "reason_code": "ok",
        "reason": "local_model_asset_available",
        "model_name": model_name,
    }


def resolve_compatibility_engine(
    *,
    mode: str = "heuristic",
    adapter: PairwiseNLIAdapter | None = None,
    fallback_to_heuristic: bool = True,
    contradiction_threshold: float = 0.75,
) -> CompatibilityEngine:
    normalized = str(mode or "heuristic").strip().lower()
    if normalized != "nli":
        return HeuristicCompatibilityEngine()
    fallback = HeuristicCompatibilityEngine() if fallback_to_heuristic else HeuristicCompatibilityEngine()
    return NLIBackedCompatibilityEngine(
        adapter=adapter,
        fallback_engine=fallback,
        contradiction_threshold=contradiction_threshold,
    )


def load_runtime_compatibility_engine_from_env() -> tuple[CompatibilityEngine, RuntimeCompatibilityState]:
    raw_mode = str(os.getenv("CRAG_COMPAT_ENGINE", "heuristic")).strip().lower() or "heuristic"
    mode = raw_mode if raw_mode in _SUPPORTED_COMPAT_MODES else "heuristic"

    raw_backend = str(os.getenv("CRAG_COMPAT_NLI_BACKEND", "transformers")).strip().lower() or "transformers"
    backend = raw_backend if raw_backend in _SUPPORTED_NLI_BACKENDS else "transformers"

    model_name = str(os.getenv("CRAG_COMPAT_NLI_MODEL", _DEFAULT_NLI_MODEL)).strip()
    if not model_name:
        model_name = _DEFAULT_NLI_MODEL

    raw_threshold = str(os.getenv("CRAG_COMPAT_NLI_THRESHOLD", "0.75")).strip()
    threshold_reason_code = ""
    threshold_reason = ""
    try:
        threshold = float(raw_threshold)
    except Exception:
        threshold = 0.75
        threshold_reason_code = "invalid_threshold"
        threshold_reason = f"invalid_threshold:{raw_threshold}"

    if mode != "nli":
        engine = HeuristicCompatibilityEngine()
        reason_code = "mode_not_nli" if raw_mode in _SUPPORTED_COMPAT_MODES else "invalid_mode"
        reason = "mode_not_nli"
        if reason_code == "invalid_mode":
            reason = f"invalid_mode:{raw_mode}"
        diagnostics: dict[str, Any] = {"raw_mode": raw_mode}
        if threshold_reason_code:
            diagnostics["threshold_warning"] = {
                "reason_code": threshold_reason_code,
                "reason": threshold_reason,
                "effective_value": threshold,
            }
        state = RuntimeCompatibilityState(
            configured_mode=mode,
            configured_backend=backend,
            configured_model=model_name,
            resolved_engine=engine.name,
            backend_available=False,
            fallback_active=False,
            reason_code=reason_code,
            reason=reason,
            diagnostics=diagnostics,
        )
        return engine, state

    adapter: PairwiseNLIAdapter | None = None
    backend_available = False
    reason_code = ""
    reason = ""
    diagnostics: dict[str, Any] = {
        "raw_mode": raw_mode,
        "raw_backend": raw_backend,
    }
    if threshold_reason_code:
        diagnostics["threshold_warning"] = {
            "reason_code": threshold_reason_code,
            "reason": threshold_reason,
            "effective_value": threshold,
        }

    if backend == "transformers":
        diagnostics["dependency_check"] = check_transformers_nli_backend(model_name=model_name)
        if bool(diagnostics["dependency_check"]["available"]):
            try:
                adapter = TransformersNLIAdapter(model_name=model_name)
                backend_available = True
                reason_code = "adapter_loaded"
                reason = "adapter_loaded"
            except Exception as exc:
                adapter = None
                backend_available = False
                reason_code = "adapter_init_failed"
                reason = f"adapter_init_failed:{type(exc).__name__}"
        else:
            reason_code = str(diagnostics["dependency_check"]["reason_code"])
            reason = str(diagnostics["dependency_check"]["reason"])
    else:
        reason_code = "invalid_backend"
        reason = f"invalid_backend:{raw_backend}"

    engine = resolve_compatibility_engine(
        mode="nli",
        adapter=adapter,
        contradiction_threshold=threshold,
    )
    state = RuntimeCompatibilityState(
        configured_mode=mode,
        configured_backend=backend,
        configured_model=model_name,
        resolved_engine=engine.name,
        backend_available=backend_available,
        fallback_active=not backend_available,
        reason_code=reason_code,
        reason=reason,
        diagnostics=diagnostics,
    )
    return engine, state


def compatibility_conflict_reason(
    candidate: ContextCandidate,
    selected: list[ContextCandidate],
    *,
    engine: CompatibilityEngine | None = None,
) -> str | None:
    active_engine = engine or HeuristicCompatibilityEngine()
    decision = active_engine.evaluate(candidate, selected)
    return decision.reason if decision.conflict else None
