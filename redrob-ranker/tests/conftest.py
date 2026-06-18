"""
conftest.py — pytest configuration for redrob-ranker tests.

Adds the project root to sys.path so `src` can be imported from any test.
"""

import sys
from pathlib import Path

# Project root is one level above tests/
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
