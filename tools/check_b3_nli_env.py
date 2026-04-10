#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CognitiveRAG.crag.context_selection.compatibility import check_transformers_nli_backend


def main() -> int:
    model_name = str(os.getenv("CRAG_COMPAT_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")).strip()
    check = check_transformers_nli_backend(model_name=model_name)
    payload = {
        "model_name": model_name,
        "transformers_backend": check,
        "real_nli_tests_requested": os.getenv("CRAG_RUN_REAL_NLI_TESTS", "").strip() == "1",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(check.get("available")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
