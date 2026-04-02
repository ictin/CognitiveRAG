from CognitiveRAG.crag.graph_memory.relations import (
    record_problem_signature_resolved_by,
    record_reasoning_pattern_supported_by,
    record_web_promoted_derived_from,
)
from CognitiveRAG.crag.graph_memory.schemas import GraphRelationType, stable_node_id
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore
from CognitiveRAG.schemas.memory import ReasoningPattern


def _pattern() -> ReasoningPattern:
    return ReasoningPattern(
        pattern_id='rp:graph-test-1',
        item_id='rp:graph-test-1',
        problem_signature='how to reduce context-window drift',
        reasoning_steps=['check runtime path', 'verify artifact fields'],
        solution_summary='Use runtime code-match proof and closure artifacts.',
        confidence=0.88,
        provenance=['{"source":"unit-test","kind":"reasoning"}'],
        memory_subtype='workflow_pattern',
    )


def test_reasoning_pattern_supported_by_relation_is_written_and_readable(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')
    pattern = _pattern()

    edge = record_reasoning_pattern_supported_by(
        store,
        pattern=pattern,
        source={'source_url': 'https://docs.example/runtime-proof', 'source': 'docs'},
        provenance={'writer': 'unit-test'},
    )

    got = store.get_edge(edge.edge_id)
    assert got is not None
    assert got.relation_type == GraphRelationType.SUPPORTED_BY
    assert got.provenance['writer'] == 'unit-test'

    reasoning_node_id = stable_node_id('reasoning_pattern', pattern.pattern_id)
    node_edges = store.get_edges_for_node(reasoning_node_id, direction='outgoing')
    assert [e.edge_id for e in node_edges] == [edge.edge_id]


def test_web_promoted_derived_from_relation_is_written_and_readable(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')

    edge = record_web_promoted_derived_from(
        store,
        promoted_id='wp:fact-2026-btc',
        source_url='https://example.com/btc',
        metadata={'freshness_state': 'hot'},
        provenance={'writer': 'unit-test'},
    )

    got = store.get_edge(edge.edge_id)
    assert got is not None
    assert got.relation_type == GraphRelationType.DERIVED_FROM
    assert got.properties['source_url'] == 'https://example.com/btc'
    assert got.provenance['writer'] == 'unit-test'

    edges = store.get_edges_by_relation(GraphRelationType.DERIVED_FROM)
    assert [e.edge_id for e in edges] == [edge.edge_id]


def test_problem_signature_resolved_by_reasoning_pattern_relation_is_written_and_readable(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')
    pattern = _pattern()

    edge = record_problem_signature_resolved_by(
        store,
        problem_signature='session recall stale path mismatch',
        pattern=pattern,
        provenance={'writer': 'unit-test'},
    )

    got = store.get_edge(edge.edge_id)
    assert got is not None
    assert got.relation_type == GraphRelationType.RESOLVED_BY
    assert got.provenance['writer'] == 'unit-test'

    ps_node = stable_node_id('problem_signature', 'session recall stale path mismatch')
    neighbors = store.get_neighbors(ps_node, relation_type=GraphRelationType.RESOLVED_BY, direction='outgoing')
    assert len(neighbors) == 1
    assert neighbors[0].node_type == 'reasoning_pattern'


def test_relation_writes_are_additive_and_deterministic(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')
    pattern = _pattern()

    edge1 = record_problem_signature_resolved_by(
        store,
        problem_signature='same-problem',
        pattern=pattern,
        provenance={'source': 'a'},
    )
    edge2 = record_problem_signature_resolved_by(
        store,
        problem_signature='same-problem',
        pattern=pattern,
        provenance={'source': 'b'},
    )

    assert edge1.edge_id == edge2.edge_id
    edges = store.get_edges_by_relation(GraphRelationType.RESOLVED_BY)
    assert len(edges) == 1
    assert edges[0].edge_id == edge1.edge_id
