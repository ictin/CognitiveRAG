from __future__ import annotations

import os

import pytest

from CognitiveRAG.crag.contracts.enums import IntentFamily, MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.compatibility import (
    check_transformers_nli_backend,
    load_runtime_compatibility_engine_from_env,
)
from CognitiveRAG.crag.context_selection.policies import get_policy
from CognitiveRAG.crag.context_selection.selector import select_context


def _cand(cid: str, text: str) -> ContextCandidate:
    return ContextCandidate(
        id=cid,
        lane=RetrievalLane.CORPUS,
        memory_type=MemoryType.CORPUS_CHUNK,
        text=text,
        tokens=20,
        provenance={"source": "real-nli-gate"},
        lexical_score=0.75,
        semantic_score=0.75,
        recency_score=0.5,
        freshness_score=0.5,
        trust_score=0.7,
        novelty_score=0.4,
        contradiction_risk=0.1,
        cluster_id=f"cluster-{cid}",
        must_include=False,
        compressible=True,
    )


@pytest.mark.skipif(
    os.getenv("CRAG_RUN_REAL_NLI_TESTS", "").strip() != "1",
    reason="set CRAG_RUN_REAL_NLI_TESTS=1 to run real transformers-backed compatibility path",
)
def test_real_transformers_nli_runtime_path_executes_when_assets_available(monkeypatch):
    model_name = str(os.getenv("CRAG_COMPAT_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")).strip()
    check = check_transformers_nli_backend(model_name=model_name)
    if not check["available"]:
        pytest.skip(f"real NLI unavailable: {check['reason_code']} ({check['reason']})")

    monkeypatch.setenv("CRAG_COMPAT_ENGINE", "nli")
    monkeypatch.setenv("CRAG_COMPAT_NLI_BACKEND", "transformers")
    monkeypatch.setenv("CRAG_COMPAT_NLI_MODEL", model_name)

    engine, state = load_runtime_compatibility_engine_from_env()
    assert engine.name == "nli"
    assert state.backend_available is True
    assert state.fallback_active is False
    assert state.reason_code == "adapter_loaded"

    a = _cand("a", "The feature flag is enabled for all users.")
    b = _cand("b", "The feature flag is not enabled for all users.")
    policy = get_policy(IntentFamily.INVESTIGATIVE)
    policy.lane_maxima[RetrievalLane.CORPUS.value] = 10

    runs = []
    for _ in range(2):
        selected, dropped, _ = select_context(
            candidates=[a, b],
            policy=policy,
            total_budget=100,
            reserved_tokens=0,
            intent_family=IntentFamily.INVESTIGATIVE,
            compatibility_engine=engine,
        )
        runs.append(([c.id for c, _ in selected], [(c.id, reason) for c, reason in dropped]))

    assert runs[0] == runs[1]
