from __future__ import annotations

import sys
from pathlib import Path


# Keep test imports deterministic in nested-repo layouts where parent paths can
# shadow the intended package root.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
package_root_str = str(PACKAGE_ROOT)
if package_root_str in sys.path:
    sys.path.remove(package_root_str)
sys.path.insert(0, package_root_str)
