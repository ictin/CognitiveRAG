import json
import os
import shutil

from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context


def _make_raw(session_id: str, count: int = 12):
    raw = [{"index": i, "text": f"message {i}", "meta": {}} for i in range(count)]
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(raw, f)


def test_assemble_context_emits_discovery_plan():
    session_id = 'm9-discovery-plan'
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    _make_raw(session_id, 18)

    out = assemble_context(
        session_id,
        fresh_tail_count=4,
        budget=900,
        query='Investigate what we should check about this topic.',
    )

    assert 'discovery_plan' in out
    plan = out['discovery_plan']
    assert plan['intent_family'] in {'investigative', 'corpus_overview', 'planning'}
    assert isinstance(plan['expected_lanes'], list)
    assert isinstance(plan['role_conditioned_probes'], list)
