from CognitiveRAG.crag.cognition.ledger import GlobalLedger
from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContradictionRecord, DiscoveryBranchRecord, DiscoveryEvidenceRef


def test_ledger_records_explored_rejected_and_unresolved():
    ledger = GlobalLedger()
    explored = DiscoveryBranchRecord(branch_id='b1', query='q1', status='explored', score=0.9, evidence_ids=['e1'])
    rejected = DiscoveryBranchRecord(branch_id='b2', query='q2', status='rejected', score=0.1, evidence_ids=[], reason='low_score')
    evidence = [
        DiscoveryEvidenceRef(
            evidence_id='e1',
            branch_id='b1',
            lane=RetrievalLane.EPISODIC,
            memory_type=MemoryType.EPISODIC_RAW,
            text='We succeeded yesterday',
            tokens=5,
            score=0.8,
        )
    ]

    ledger.record_explored(explored, evidence)
    ledger.record_rejected(rejected)
    ledger.add_contradictions([
        ContradictionRecord(left_evidence_id='e1', right_evidence_id='e2', reason='conflict', strength=0.6)
    ])
    ledger.add_unresolved('Need newer verification')

    snap = ledger.snapshot()
    assert len(snap.explored_branches) == 1
    assert len(snap.rejected_branches) == 1
    assert snap.evidence_bundles['b1'] == ['e1']
    assert len(snap.contradictions) == 1
    assert 'Need newer verification' in snap.unresolved_questions
