from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from openai_multi_backend.config import get_settings
from openai_multi_backend.models.registry import reset_registry_for_tests


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_ENVIRONMENT", "test")
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_OUTPUT_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_API_KEYS", "")
    get_settings.cache_clear()
    reset_registry_for_tests()
    from openai_multi_backend.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
    reset_registry_for_tests()
