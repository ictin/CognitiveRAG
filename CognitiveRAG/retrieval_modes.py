from enum import Enum
from typing import Dict

class RetrievalMode(Enum):
    DOCUMENTS_ONLY = "documents_only"
    REGRESSION_TEST = "regression_test"
    TASK_MEMORY = "task_memory"
    FULL_MEMORY = "full_memory"

# Policy map: which sources are allowed per mode
# keys: 'bm25' (documents), 'vector' (semantic over docs),
# 'task_profile_reasoning' (non-episodic memories), 'episodic' (episodic SQL stores), 'web' (external web)
MODE_SOURCE_POLICY: Dict[RetrievalMode, Dict[str, bool]] = {
    RetrievalMode.DOCUMENTS_ONLY: {
        'bm25': True,
        'vector': True,
        'task_profile_reasoning': False,
        'episodic': False,
        'web': False,
    },
    RetrievalMode.REGRESSION_TEST: {
        'bm25': True,
        'vector': True,
        'task_profile_reasoning': False,
        'episodic': False,
        'web': False,
    },
    RetrievalMode.TASK_MEMORY: {
        'bm25': True,
        'vector': True,
        'task_profile_reasoning': True,
        'episodic': False,
        'web': False,
    },
    RetrievalMode.FULL_MEMORY: {
        'bm25': True,
        'vector': True,
        'task_profile_reasoning': True,
        'episodic': True,
        'web': True,
    },
}

def allowed_sources_for_mode(mode: RetrievalMode):
    return MODE_SOURCE_POLICY.get(mode, MODE_SOURCE_POLICY[RetrievalMode.FULL_MEMORY])
