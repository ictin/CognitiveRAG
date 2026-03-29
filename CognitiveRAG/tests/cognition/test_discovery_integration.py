import json
import os
import shutil

from CognitiveRAG.session_memory.context_window import WORKDIR, assemble_context


def _make_raw(session_id: str, count: int = 15):
    rows = [{"index": i, "text": f"Investigate migration risk and rollback sequence {i}", "meta": {}} for i in range(count)]
    os.makedirs(WORKDIR, exist_ok=True)
    with open(os.path.join(WORKDIR, f"raw_{session_id}.json"), 'w', encoding='utf-8') as f:
        json.dump(rows, f)


def test_discovery_payload_emitted_and_bounded_from_assemble_context():
    session_id = 'm10-discovery-int'
    shutil.rmtree(WORKDIR, ignore_errors=True)
    os.makedirs(WORKDIR, exist_ok=True)
    _make_raw(session_id, 20)

    out = assemble_context(
        session_id,
        fresh_tail_count=4,
        budget=1000,
        query='Investigate migration risks and what else matters.',
    )

    assert 'discovery_plan' in out
    assert 'discovery' in out
    discovery = out['discovery']
    assert discovery.get('bounded') is True
    assert discovery.get('used_tokens', 0) <= discovery.get('budget_tokens', 0)
