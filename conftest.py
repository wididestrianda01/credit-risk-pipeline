"""conftest.py — adds repo root to sys.path so pytest can import credit_engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
