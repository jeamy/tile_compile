from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
WEB_BACKEND_DIR = TESTS_DIR.parent

if str(WEB_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_BACKEND_DIR))
