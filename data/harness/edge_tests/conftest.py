"""Shared fixture: load artifact from ARTIFACT_PATH env var.

Each test file does:
    from conftest import artifact
and then accesses functions via artifact.function_name.

Run via evaluate.py which sets ARTIFACT_PATH before invoking pytest.
"""

import importlib.util
import os
import sys
import types

import pytest


def load_artifact() -> types.ModuleType:
    path = os.environ.get("ARTIFACT_PATH", "")
    if not path:
        pytest.skip("ARTIFACT_PATH not set — run via evaluate.py")
    spec = importlib.util.spec_from_file_location("_artifact", path)
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot load artifact from {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_artifact"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def artifact() -> types.ModuleType:
    return load_artifact()
