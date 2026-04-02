from CognitiveRAG.crag.graph_memory.schemas import GraphEdge, GraphNode, GraphRelationType
from CognitiveRAG.crag.graph_memory.store import GraphMemoryStore


def test_graph_store_roundtrip_for_node_and_edge(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')

    n1 = GraphNode(
        node_id='gn:test:n1',
        node_type='reasoning_pattern',
        label='Reasoning #1',
        properties={'pattern_id': 'rp:1'},
        provenance={'source': 'unit'},
        created_at='2026-04-02T10:00:00Z',
        updated_at='2026-04-02T10:00:00Z',
    )
    n2 = GraphNode(
        node_id='gn:test:n2',
        node_type='provenance_source',
        label='https://example.com/source',
        properties={'source_url': 'https://example.com/source'},
        provenance={'origin': 'web'},
        created_at='2026-04-02T10:00:00Z',
        updated_at='2026-04-02T10:00:00Z',
    )
    e = GraphEdge(
        edge_id='ge:test:e1',
        source_node_id=n1.node_id,
        relation_type=GraphRelationType.SUPPORTED_BY,
        target_node_id=n2.node_id,
        properties={'weight': 1.0},
        provenance={'reason': 'unit-test'},
        created_at='2026-04-02T10:00:00Z',
        updated_at='2026-04-02T10:00:00Z',
    )

    store.upsert_nodes([n1, n2])
    store.upsert_edge(e)

    got_n1 = store.get_node(n1.node_id)
    got_e = store.get_edge(e.edge_id)

    assert got_n1 is not None
    assert got_n1.properties['pattern_id'] == 'rp:1'
    assert got_n1.provenance['source'] == 'unit'

    assert got_e is not None
    assert got_e.relation_type == GraphRelationType.SUPPORTED_BY
    assert got_e.provenance['reason'] == 'unit-test'


def test_graph_store_relation_and_neighbor_lookups_are_deterministic(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')
    nodes = [
        GraphNode(node_id='gn:a', node_type='problem_signature', properties={'problem_signature': 'p1'}),
        GraphNode(node_id='gn:b', node_type='reasoning_pattern', properties={'pattern_id': 'rp:2'}),
        GraphNode(node_id='gn:c', node_type='source_url', properties={'source_url': 'https://x'}),
    ]
    store.upsert_nodes(nodes)

    edges = [
        GraphEdge(edge_id='ge:1', source_node_id='gn:a', relation_type=GraphRelationType.RESOLVED_BY, target_node_id='gn:b'),
        GraphEdge(edge_id='ge:2', source_node_id='gn:b', relation_type=GraphRelationType.SUPPORTED_BY, target_node_id='gn:c'),
    ]
    store.upsert_edges(edges)

    for _ in range(3):
        rel_edges = store.get_edges_by_relation(GraphRelationType.RESOLVED_BY)
        assert [e.edge_id for e in rel_edges] == ['ge:1']

        node_edges = store.get_edges_for_node('gn:b', direction='both')
        assert [e.edge_id for e in node_edges] == ['ge:1', 'ge:2']

        neighbors = store.get_neighbors('gn:b')
        assert [n.node_id for n in neighbors] == ['gn:a', 'gn:c']


def test_graph_store_preserves_provenance_on_upsert_update(tmp_path):
    store = GraphMemoryStore(tmp_path / 'graph.sqlite3')
    node = GraphNode(
        node_id='gn:prov',
        node_type='web_promoted',
        properties={'promoted_id': 'wp:1'},
        provenance={'source_bundle': ['https://a.example']},
    )
    store.upsert_node(node)

    edge = GraphEdge(
        edge_id='ge:prov',
        source_node_id='gn:prov',
        relation_type=GraphRelationType.DERIVED_FROM,
        target_node_id='gn:url',
        provenance={'evidence': ['e1', 'e2']},
    )
    store.upsert_edge(edge)

    got_node = store.get_node('gn:prov')
    got_edge = store.get_edge('ge:prov')

    assert got_node is not None
    assert got_node.provenance['source_bundle'] == ['https://a.example']
    assert got_edge is not None
    assert got_edge.provenance['evidence'] == ['e1', 'e2']
