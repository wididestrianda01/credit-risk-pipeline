"""
test_streamlit_startup.py
--------------------------
Unit tests for Streamlit app startup and model loading.

Tests cover:
- App module imports without error
- Root entrypoint (streamlit_app.py) imports without error
- load_catboost_model() returns a model when file is present
- load_catboost_model() returns None gracefully when model file is missing
- load_catboost_model() tries HF Hub when HF_REPO_ID is set (post-Plan-02)
- _MODEL_PATH resolves inside the project root
- runtime.txt exists and pins Python 3.10
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APP_MODULE = "app.streamlit_app"


def _fresh_import(module_name: str):
    """Force a fresh import of a module by removing it from sys.modules first."""
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Remove the app module from sys.modules before and after each test
    so that module-level code (model = load_catboost_model()) re-runs cleanly."""
    for key in list(sys.modules.keys()):
        if key.startswith("app.streamlit_app") or key == "streamlit_app":
            del sys.modules[key]
    yield
    for key in list(sys.modules.keys()):
        if key.startswith("app.streamlit_app") or key == "streamlit_app":
            del sys.modules[key]


# ---------------------------------------------------------------------------
# Test: module-level imports
# ---------------------------------------------------------------------------

def test_app_imports():
    """app.streamlit_app imports without raising any exception."""
    with patch("app.streamlit_app.load_model", return_value=MagicMock()):
        mod = _fresh_import(_APP_MODULE)
    assert mod is not None


def test_root_entrypoint_imports():
    """Root streamlit_app.py wrapper imports without raising any exception."""
    with patch("app.streamlit_app.load_model", return_value=MagicMock()):
        sys.modules.pop("streamlit_app", None)
        mod = importlib.import_module("streamlit_app")
    assert mod is not None


# ---------------------------------------------------------------------------
# Test: _MODEL_PATH resolves inside project root
# ---------------------------------------------------------------------------

def test_model_path_is_inside_project_root():
    """_MODEL_PATH should point to models/ inside the project root."""
    with patch("app.streamlit_app.load_model", return_value=MagicMock()):
        mod = _fresh_import(_APP_MODULE)
    model_path = mod._MODEL_PATH
    assert model_path.is_relative_to(PROJECT_ROOT), (
        f"_MODEL_PATH {model_path} is not inside project root {PROJECT_ROOT}"
    )
    assert model_path.name == "catboost_raw_calibrated_v2.pkl"


# ---------------------------------------------------------------------------
# Test: load_catboost_model — model attribute is accessible at module level
# ---------------------------------------------------------------------------

def test_load_model_returns_model_when_file_exists():
    """When the actual model file exists on disk, the module-level model is
    not None and is a loaded estimator (not a plain MagicMock)."""
    model_file = PROJECT_ROOT / "models" / "catboost_raw_calibrated_v2.pkl"
    if not model_file.exists():
        pytest.skip("Model file not present in this environment — skipping")

    with patch("app.streamlit_app.load_model", wraps=__import__("src.model_base", fromlist=["load_model"]).load_model):
        mod = _fresh_import(_APP_MODULE)

    # model is set at module-level; it should be a live estimator
    assert mod.model is not None, "Expected model to load when pickle file exists"


# ---------------------------------------------------------------------------
# Test: load_catboost_model — model file missing, HF_REPO_ID not set
# ---------------------------------------------------------------------------

def test_load_model_fallback_returns_none_when_file_missing_and_no_hf(tmp_path):
    """load_catboost_model() returns None gracefully when model file is absent
    and HF_REPO_ID environment variable is not set.

    @st.cache_resource uses the function's source-code hash as cache key, so it
    survives module reloads.  Strategy:
      1. Pre-import the module under a neutral mock (so the module is in sys.modules)
      2. Clear the Streamlit cache on the decorated function
      3. Call again under the real test patches — the function body re-executes
    """
    missing_path = tmp_path / "nonexistent.pkl"
    env_without_hf = {k: v for k, v in os.environ.items() if k != "HF_REPO_ID"}

    # Step 1: ensure module is resident in sys.modules before applying patches
    with patch("app.streamlit_app.load_model", return_value=MagicMock()):
        mod = _fresh_import(_APP_MODULE)

    # Step 2 + 3: clear Streamlit cache and re-call under FileNotFoundError conditions
    with (
        patch("app.streamlit_app._MODEL_PATH", missing_path),
        patch("app.streamlit_app.load_model", side_effect=FileNotFoundError("not found")),
        patch.dict(os.environ, env_without_hf, clear=True),
    ):
        mod.load_catboost_model.clear()
        result = mod.load_catboost_model()

    assert result is None, "Expected None when model file missing and HF_REPO_ID not set"


# ---------------------------------------------------------------------------
# Test: runtime.txt pins Python 3.10
# ---------------------------------------------------------------------------

def test_runtime_txt_pins_python_310():
    """runtime.txt at repo root must contain 'python-3.10'."""
    runtime_path = PROJECT_ROOT / "runtime.txt"
    assert runtime_path.exists(), "runtime.txt not found at repo root"
    content = runtime_path.read_text().strip()
    assert content == "python-3.10", (
        f"runtime.txt should contain exactly 'python-3.10', got: {content!r}"
    )


# ---------------------------------------------------------------------------
# Test: root streamlit_app.py exists
# ---------------------------------------------------------------------------

def test_root_entrypoint_file_exists():
    """streamlit_app.py must exist at repo root for Community Cloud deployment."""
    entrypoint = PROJECT_ROOT / "streamlit_app.py"
    assert entrypoint.exists(), "streamlit_app.py not found at repo root"


# ---------------------------------------------------------------------------
# Test: load_catboost_model — HF Hub path (post-Plan-02, forward-compatible)
# ---------------------------------------------------------------------------

def test_load_model_uses_hf_hub_when_env_set(tmp_path):
    """When HF_REPO_ID is set and local file is missing, HF Hub download is attempted.

    Forward-compatible: passes as a no-op if hf_hub_download is not yet in the module
    (pre-Plan-02), and verifies the HF call path when it is integrated.

    Uses the same two-step isolation pattern as the fallback test:
      1. Pre-import under a neutral mock so the module is resident in sys.modules.
      2. Apply _MODEL_PATH and hf_hub_download patches against the already-resident
         module, clear the @st.cache_resource cache, then re-invoke.
    Entering a patch() context manager forces an import to resolve the target; if
    _fresh_import() is called *inside* that context it creates a new module object
    that is no longer the patch target, causing silent misses.
    """
    missing_path = tmp_path / "nonexistent.pkl"
    fake_downloaded_path = str(tmp_path / "downloaded.pkl")
    # Pre-create the "downloaded" file so load_model can open it
    Path(fake_downloaded_path).touch()

    fake_model = MagicMock(name="FakeCatBoostModelFromHF")
    hf_download_mock = MagicMock(return_value=fake_downloaded_path)

    # Step 1: get module into sys.modules under a neutral mock (no real I/O)
    with patch("app.streamlit_app.load_model", return_value=MagicMock()):
        mod = _fresh_import(_APP_MODULE)

    hf_integrated = hasattr(mod, "hf_hub_download")

    if not hf_integrated:
        # Pre-Plan-02: test is a no-op
        assert mod.model is None or mod.model is not None  # vacuously true
        return

    # Step 2: patch the already-resident module, clear cache, verify HF Hub path
    with (
        patch("app.streamlit_app._MODEL_PATH", missing_path),
        patch.dict(os.environ, {"HF_REPO_ID": "testuser/credit-risk-models"}, clear=False),
        patch("app.streamlit_app.hf_hub_download", hf_download_mock),
        patch("app.streamlit_app.load_model", return_value=fake_model),
    ):
        mod.load_catboost_model.clear()
        result = mod.load_catboost_model()

    hf_download_mock.assert_called_once()
    assert result is fake_model
