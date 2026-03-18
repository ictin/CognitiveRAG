"""Repair decision helper utilities.

Provides simple heuristics to classify failure text and decide whether the
deterministic repair workflow should be used.

Only uses Python standard library so it is easy to test and import.
"""
from __future__ import annotations

from typing import Dict, Optional


def classify_failure(text: str) -> Dict[str, Optional[object]]:
    """Classify an error/log text blob for repair decision making.

    Args:
        text: Error or log text to inspect.

    Returns:
        A dictionary containing boolean flags and optional suggested target:
        - syntax_error: True if SyntaxError/IndentationError detected
        - import_error: True if ModuleNotFoundError/ImportError detected
        - py_compile_failure: True if py_compile-like message appears
        - pytest_failure: True if pytest failure markers are detected
        - exact_bytes_risk: True if the failure mentions __future__, shebang, encoding, or similar
        - likely_repair_target: heuristic filename or None
        - reason: short human-readable reason
    """
    lower = (text or "").lower()
    res: Dict[str, Optional[object]] = {
        "syntax_error": False,
        "import_error": False,
        "py_compile_failure": False,
        "pytest_failure": False,
        "exact_bytes_risk": False,
        "likely_repair_target": None,
        "reason": None,
    }

    if "syntaxerror" in lower or "syntax error" in lower or "indentationerror" in lower or "indentation error" in lower:
        res["syntax_error"] = True
        res["reason"] = "syntax error detected"

    if "modulenotfounderror" in lower or "module not found" in lower or "importerror" in lower or "import error" in lower:
        res["import_error"] = True
        if res.get("reason"):
            res["reason"] += "; import error detected"
        else:
            res["reason"] = "import error detected"

    if "py_compile" in lower or "py_compile" in text or "invalid syntax" in lower:
        res["py_compile_failure"] = True
        if not res.get("reason"):
            res["reason"] = "py_compile failure"

    if "pytest" in lower and ("failed" in lower or "failure" in lower or "assert" in lower):
        res["pytest_failure"] = True
        if not res.get("reason"):
            res["reason"] = "pytest reported failures"

    # Exact-bytes risk indicators
    exact_indicators = ["__future__", "__future__ import", "#\!", "shebang", "encoding", " -*- coding:", "malformed import", "from __future__"]
    if any(ind in text for ind in exact_indicators):
        res["exact_bytes_risk"] = True
        if res.get("reason"):
            res["reason"] += "; exact-bytes risk"
        else:
            res["reason"] = "exact-bytes risk detected"

    # Heuristic: attempt to pick a filename mentioned in the message
    import os
    for token in text.split():
        if token.endswith(".py"):
            # strip punctuation
            candidate = token.strip("',:();\n\r")
            if os.path.exists(candidate):
                res["likely_repair_target"] = candidate
                break
            # maybe relative path
            if candidate.startswith("./") or candidate.startswith("/"):
                res["likely_repair_target"] = candidate
                break

    return res


def should_use_deterministic_repair(classification: Dict[str, object]) -> bool:
    """Decide whether deterministic content-file based repair is preferred.

    Returns True when the classification suggests exact-syntax or on-disk
    verification is required (syntax errors, __future__ issues, shebangs,
    encoding lines, or py_compile failures).
    """
    if not classification:
        return False

    if classification.get("syntax_error"):
        return True
    if classification.get("py_compile_failure"):
        return True
    if classification.get("exact_bytes_risk"):
        return True
    # Import errors may or may not require deterministic repair; prefer it when exact bytes risk present
    if classification.get("import_error") and classification.get("exact_bytes_risk"):
        return True

    return False


def should_retry_mechanical_mismatch(workflow_result: Dict[str, object]) -> bool:
    """Decide whether to retry once on mechanical mismatch based on workflow_result.

    Returns True only when workflow_result indicates a non-destructive mechanical
    mismatch (e.g., prefix_mismatch) where a single automatic retry is reasonable.
    """
    if not workflow_result:
        return False
    if workflow_result.get("workflow_result"):
        wf = workflow_result["workflow_result"]
    else:
        wf = workflow_result

    if not isinstance(wf, dict):
        return False

    if wf.get("workflow_ok") is True:
        return False

    fix = wf.get("fix_result") or {}
    actions = fix.get("actions") or []

    # mechanical mismatch pattern: actions contain one or more 'prefix_mismatch'
    if any(a == "prefix_mismatch" for a in actions):
        return True

    return False
