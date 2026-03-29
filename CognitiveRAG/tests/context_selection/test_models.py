from CognitiveRAG.crag.contracts.enums import IntentFamily, RetrievalLane, MemoryType
from CognitiveRAG.crag.contracts.schemas import ContextCandidate, SelectionExplanation


def test_typed_candidate_validates_required_shape():
    c = ContextCandidate(
        id="c1",
        lane=RetrievalLane.EPISODIC,
        memory_type=MemoryType.EPISODIC_RAW,
        text="hello world",
        tokens=0,
        provenance={"source": "test"},
    )
    assert c.tokens > 0
    assert c.provenance["source"] == "test"


def test_enums_exist_for_selector_contracts():
    assert IntentFamily.EXACT_RECALL.value == "exact_recall"
    assert RetrievalLane.CORPUS.value == "corpus"
    assert MemoryType.CORPUS_CHUNK.value == "corpus_chunk"


def test_explanation_artifact_validates():
    ex = SelectionExplanation(intent_family=IntentFamily.MEMORY_SUMMARY, total_budget=1024, reserved_tokens=128)
    assert ex.intent_family == IntentFamily.MEMORY_SUMMARY
    assert ex.total_budget == 1024
