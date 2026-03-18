from dataclasses import dataclass


@dataclass
class RetrievalPolicy:
    mode: str
    allow_documents: bool
    allow_episodic: bool
    allow_profile: bool
    allow_task: bool
    allow_reasoning: bool
    allow_web: bool
    allow_graph: bool


def policy_for_mode(mode: str | None) -> RetrievalPolicy:
    m = mode or "full_memory"
    if m == "documents_only":
        return RetrievalPolicy(m, True, False, False, False, False, False, False)
    if m == "regression_test":
        return RetrievalPolicy(m, True, False, False, False, False, False, False)
    if m == "task_memory":
        return RetrievalPolicy(m, True, False, True, True, True, False, False)
    # default: full_memory
    return RetrievalPolicy(m, True, True, True, True, True, True, True)
