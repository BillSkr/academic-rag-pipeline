"""Pytest configuration and shared fixtures for the RAG pipeline test suite.

This conftest.py is automatically loaded by pytest before any test module.
It is used to:
  1. Mock heavyweight third-party packages that are not installed (e.g. litellm)
     so that test collection does not fail with ImportError.
  2. Provide session-level or module-level fixtures available to all tests.

NOTE: litellm fails to install on Windows due to MAX_PATH limitations in
some of its bundled data files. Since no test makes real LLM API calls, we
replace the module with a lightweight MagicMock before any imports happen.
"""

import sys
from unittest.mock import MagicMock

# ── Stub litellm before any src.* imports ────────────────────────────────────
# Every node that calls `from litellm import completion` will receive this mock.
sys.modules['litellm'] = MagicMock()
sys.modules['litellm.completion'] = MagicMock()