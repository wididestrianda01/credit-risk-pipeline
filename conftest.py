"""conftest.py — adds repo root to sys.path and exposes src/ as credit_engine."""

import sys
from pathlib import Path

# Add project root so `import src` resolves.
sys.path.insert(0, str(Path(__file__).parent))

# Alias src/ as credit_engine so existing imports (from credit_engine.X import ...)
# continue to work while the source tree lives in src/.
import src  # noqa: E402
sys.modules["credit_engine"] = src
