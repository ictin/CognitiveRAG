from CognitiveRAG.crag.graph_memory.schemas import GraphEdge, GraphNode, GraphRelationType, stable_edge_id, stable_node_id


def test_stable_node_id_is_deterministic_and_type_scoped():
    a1 = stable_node_id('reasoning_pattern', 'rp:123')
    a2 = stable_node_id('reasoning_pattern', 'rp:123')
    b = stable_node_id('problem_signature', 'rp:123')

    assert a1 == a2
    assert a1 != b
    assert a1.startswith('gn:reasoning_pattern:')


def test_stable_edge_id_is_deterministic_for_same_triplet():
    e1 = stable_edge_id('n1', GraphRelationType.SUPPORTED_BY, 'n2')
    e2 = stable_edge_id('n1', GraphRelationType.SUPPORTED_BY, 'n2')
    e3 = stable_edge_id('n1', GraphRelationType.RESOLVED_BY, 'n2')

    assert e1 == e2
    assert e1 != e3


def test_graph_node_and_edge_accept_provenance_and_properties():
    node = GraphNode(
        node_id='gn:test:1',
        node_type='reasoning_pattern',
        properties={'pattern_id': 'rp:1', 'confidence': 0.8},
        provenance={'source': 'unit-test'},
    )
    edge = GraphEdge(
        edge_id='ge:test:1',
        source_node_id='gn:test:1',
        relation_type=GraphRelationType.SUPPORTED_BY,
        target_node_id='gn:test:2',
        properties={'weight': 1.0},
        provenance={'source_ref': 's1'},
    )

    assert node.properties['pattern_id'] == 'rp:1'
    assert node.provenance['source'] == 'unit-test'
    assert edge.properties['weight'] == 1.0
    assert edge.provenance['source_ref'] == 's1'
