from CognitiveRAG.crag.contracts.enums import MemoryType, RetrievalLane
from CognitiveRAG.crag.contracts.schemas import ContextCandidate
from CognitiveRAG.crag.context_selection.lane_pruning import prune_lane_local


def _c(cid: str, text: str, tokens: int = 10, compressible: bool = True, cluster: str | None = None):
    return ContextCandidate(
        id=cid,
        lane=RetrievalLane.EPISODIC,
        memory_type=MemoryType.EPISODIC_RAW,
        text=text,
        tokens=tokens,
        provenance={},
        compressible=compressible,
        cluster_id=cluster,
    )


def test_lane_pruning_dedupes_and_splits_and_merges():
    huge = "word " * 600
    candidates = [
        _c("a", "same text"),
        _c("b", "same   text"),
        _c("c", huge, tokens=500, compressible=True),
        _c("d", "tiny one", tokens=5, cluster="k1"),
        _c("e", "tiny two", tokens=5, cluster="k1"),
    ]
    out = prune_lane_local(candidates, max_candidate_tokens=100)
    out_ids = [c.id for c in out]
    assert "b" not in out_ids  # deduped
    assert any("c#part" in i for i in out_ids)  # split
    assert any("d+e" == i for i in out_ids)  # tiny merge


def test_lane_pruning_does_not_merge_fresh_tail_items():
    a = _c("f1", "tail one", tokens=5, cluster="k1")
    b = _c("f2", "tail two", tokens=5, cluster="k1")
    a.lane = RetrievalLane.FRESH_TAIL
    b.lane = RetrievalLane.FRESH_TAIL
    out = prune_lane_local([a, b], max_candidate_tokens=100)
    out_ids = [c.id for c in out]
    assert "f1" in out_ids
    assert "f2" in out_ids
    assert not any("f1+f2" == i for i in out_ids)
