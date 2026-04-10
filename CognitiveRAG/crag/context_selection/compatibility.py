from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from CognitiveRAG.crag.contracts.schemas import ContextCandidate


_NEGATIVE_MARKERS = {"not", "never", "cannot", "cant", "can't", "failed", "failure", "no", "disabled", "off"}
_POSITIVE_MARKERS = {"works", "working", "succeeds", "success", "yes", "enabled", "active", "on"}


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
    reason: str = ""


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
    mode = str(os.getenv("CRAG_COMPAT_ENGINE", "heuristic")).strip().lower() or "heuristic"
    backend = str(os.getenv("CRAG_COMPAT_NLI_BACKEND", "transformers")).strip().lower() or "transformers"
    model_name = str(os.getenv("CRAG_COMPAT_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")).strip()
    raw_threshold = str(os.getenv("CRAG_COMPAT_NLI_THRESHOLD", "0.75")).strip()
    try:
        threshold = float(raw_threshold)
    except Exception:
        threshold = 0.75

    if mode != "nli":
        engine = HeuristicCompatibilityEngine()
        state = RuntimeCompatibilityState(
            configured_mode=mode,
            configured_backend=backend,
            configured_model=model_name,
            resolved_engine=engine.name,
            backend_available=False,
            fallback_active=False,
            reason="mode_not_nli",
        )
        return engine, state

    adapter: PairwiseNLIAdapter | None = None
    backend_available = False
    reason = "adapter_not_loaded"
    if backend == "transformers":
        try:
            adapter = TransformersNLIAdapter(model_name=model_name)
            backend_available = True
            reason = "adapter_loaded"
        except Exception as exc:
            adapter = None
            backend_available = False
            reason = f"adapter_unavailable:{type(exc).__name__}"
    else:
        reason = "unknown_backend"

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
        reason=reason,
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
