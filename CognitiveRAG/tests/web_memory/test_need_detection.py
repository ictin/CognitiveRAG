from CognitiveRAG.crag.contracts.enums import IntentFamily
from CognitiveRAG.crag.web_memory.query_planner import detect_web_need


def test_need_detection_triggers_for_verification_when_local_sparse():
    need = detect_web_need(
        query="verify this claim with sources",
        intent_family=IntentFamily.INVESTIGATIVE,
        local_evidence_count=0,
    )
    assert need.needed is True
    assert "verification_sensitive" in need.reason
