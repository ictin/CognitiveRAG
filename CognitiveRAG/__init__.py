"""CognitiveRAG package init."""

import sys

# Some local test runners resolve this package as "CognitiveRAG.CognitiveRAG"
# in nested worktree layouts. Keep a stable alias for compatibility.
sys.modules.setdefault("CognitiveRAG.CognitiveRAG", sys.modules[__name__])
