from enum import Enum


class IntentFamily(str, Enum):
    EXACT_RECALL = "exact_recall"
    MEMORY_SUMMARY = "memory_summary"
    ARCHITECTURE_EXPLANATION = "architecture_explanation"
    CORPUS_OVERVIEW = "corpus_overview"
    PLANNING = "planning"
    INVESTIGATIVE = "investigative"


class RetrievalLane(str, Enum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    SYSTEM = "system"
    USER_TURN = "user_turn"
    FRESH_TAIL = "fresh_tail"
    EPISODIC = "episodic"
    SESSION_SUMMARY = "session_summary"
    PROMOTED = "promoted"
    CORPUS = "corpus"
    LARGE_FILE = "large_file"
    REASONING = "reasoning"
    DISCOVERY = "discovery"
    WEB = "web"
    ARCHITECTURE = "architecture"
    FALLBACK_MIRROR = "fallback_mirror"


class MemoryType(str, Enum):
    SYSTEM_INSTRUCTION = "system_instruction"
    USER_QUERY = "user_query"
    EPISODIC_RAW = "episodic_raw"
    SUMMARY = "summary"
    PROMOTED_FACT = "promoted_fact"
    CORPUS_CHUNK = "corpus_chunk"
    LARGE_FILE_EXCERPT = "large_file_excerpt"
    ARCHITECTURE_NOTE = "architecture_note"
    MIRROR_NOTE = "mirror_note"


class DiscoveryMode(str, Enum):
    OFF = "off"
    PASSIVE = "passive"
    ACTIVE = "active"


class FreshnessClass(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    STALE = "stale"


class ConflictLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
