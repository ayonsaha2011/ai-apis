from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from openai_multi_backend.api.openai_routes import stream_completion_chunks
from openai_multi_backend.api.schemas import (
    CompletionRequest,
    ImageGenerationRequest,
    ModelDownloadRequest,
    SpeechRequest,
)
from openai_multi_backend.config import (
    LTX_GEMMA_REPO_ID,
    REQUESTED_MODEL_IDS,
    Settings,
    get_settings,
)
from openai_multi_backend.models.base import ModelLoadError
from openai_multi_backend.models.download import (
    reset_downloader_for_tests,
    resolve_download_plan,
)
from openai_multi_backend.models.image import LTXCliMediaAdapter
from openai_multi_backend.models.registry import (
    MODEL_METADATA,
    ModelEntry,
    ModelRegistry,
    get_registry,
    reset_registry_for_tests,
)
from openai_multi_backend.models.speech import extract_pipeline_audio, resolve_voice_reference


def test_models_endpoint_lists_requested_models(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [item["id"] for item in payload["data"]] == list(REQUESTED_MODEL_IDS)
    assert "chat.completions" in next(
        item for item in payload["data"] if item["id"].startswith("llmfan46/")
    )["endpoints"]


def test_model_lookup_returns_openai_error_for_unknown_model(client: TestClient) -> None:
    response = client.get("/v1/models/not-a-model")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_image_size_validation() -> None:
    request = ImageGenerationRequest(
        model="dx8152/Flux2-Klein-9B-Consistency",
        prompt="a production API diagram",
        size="768x512",
    )
    assert request.dimensions() == (768, 512)


def test_registry_marks_disabled_models() -> None:
    settings = Settings(environment="test", enabled_models=["openai/whisper-large-v3-turbo"])
    registry = ModelRegistry(settings)
    states = registry.health()
    assert states["openai/whisper-large-v3-turbo"]["state"] == "configured"
    assert states["coqui/XTTS-v2"]["state"] == "disabled"


def test_production_auth_rejects_missing_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_ENVIRONMENT", "production")
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_API_KEYS", "secret-token")
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_OUTPUT_DIR", str(tmp_path / "generated"))
    get_settings.cache_clear()
    reset_registry_for_tests()
    from openai_multi_backend.main import create_app

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    get_settings.cache_clear()
    reset_registry_for_tests()


def test_completion_rejects_empty_prompt_list() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="model", prompt=[])


def test_local_voice_reference_must_stay_inside_safe_dir(tmp_path) -> None:
    safe_dir = tmp_path / "voices"
    safe_dir.mkdir()
    allowed_file = safe_dir / "speaker.wav"
    allowed_file.write_bytes(b"RIFF")
    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"RIFF")
    settings = Settings(environment="test", voice_reference_dir=safe_dir)

    allowed = SpeechRequest(model="coqui/XTTS-v2", input="hello", speaker_wav=str(allowed_file))
    blocked = SpeechRequest(model="coqui/XTTS-v2", input="hello", speaker_wav=str(outside_file))

    assert resolve_voice_reference(allowed, settings) == str(allowed_file.resolve())
    with pytest.raises(ValueError, match="inside the configured safe directory"):
        resolve_voice_reference(blocked, settings)


def test_voice_reference_url_requires_allowlisted_host() -> None:
    settings = Settings(environment="test")
    request = SpeechRequest(
        model="coqui/XTTS-v2",
        input="hello",
        voice_reference_url="https://example.com/speaker.wav",
    )

    with pytest.raises(ValueError, match="allowlist"):
        resolve_voice_reference(request, settings)


def test_extract_pipeline_audio_does_not_boolean_check_arrays() -> None:
    class AmbiguousAudio:
        def __bool__(self) -> bool:
            raise AssertionError("audio output should not be truth-tested")

    audio = AmbiguousAudio()
    extracted, sample_rate = extract_pipeline_audio({"audio": audio, "sampling_rate": 16000})

    assert extracted is audio
    assert sample_rate == 16000


def test_streaming_completions_use_text_completion_chunks() -> None:
    class FakeAdapter:
        def stream_completion(self, _request: CompletionRequest):
            yield "hello"

    async def collect_chunks() -> list[str]:
        entry = ModelEntry(
            metadata=MODEL_METADATA["llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic"]
        )
        request = CompletionRequest(model=entry.metadata.id, prompt="hi", stream=True)
        return [
            chunk
            async for chunk in stream_completion_chunks(
                entry, FakeAdapter(), request, Settings(environment="test")
            )
        ]

    chunks = asyncio.run(collect_chunks())
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "hello"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_image_model_load_failure_returns_openai_503(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_ENVIRONMENT", "test")
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_API_KEYS", "")
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_MULTI_BACKEND_OUTPUT_DIR", str(tmp_path / "generated"))
    get_settings.cache_clear()
    reset_registry_for_tests()

    class FailingRegistry(ModelRegistry):
        async def load_adapter(self, model_id: str, endpoint):
            raise ModelLoadError("Diffusers pipeline could not load the requested model")

    from openai_multi_backend.main import create_app

    app = create_app()
    app.dependency_overrides[get_registry] = lambda: FailingRegistry(Settings(environment="test"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/images/generations",
            json={
                "model": "dx8152/Flux2-Klein-9B-Consistency",
                "prompt": "test image",
                "size": "512x512",
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_load_failed"
    get_settings.cache_clear()
    reset_registry_for_tests()


def test_model_download_skips_existing_file(client: TestClient, monkeypatch, tmp_path) -> None:
    cached_file = tmp_path / "cached-model.safetensors"
    cached_file.write_bytes(b"weights")

    class FakeHub:
        calls = 0

        @staticmethod
        def try_to_load_from_cache(**_kwargs):
            return str(cached_file)

        @classmethod
        def hf_hub_download(cls, **_kwargs):
            cls.calls += 1
            raise AssertionError("download should not be called for cached files")

    import openai_multi_backend.models.download as download_module

    monkeypatch.setattr(download_module, "import_optional", lambda _name: FakeHub)
    reset_downloader_for_tests()

    response = client.post(
        "/v1/models/download",
        json={
            "model": "Lightricks/LTX-2.3",
            "files": ["ltx-2.3-22b-distilled-1.1.safetensors"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["already_cached"] is True
    assert payload["artifacts"][0]["status"] == "cached"
    assert payload["artifacts"][0]["local_path"] == str(cached_file)
    assert FakeHub.calls == 0
    reset_downloader_for_tests()


def test_model_download_downloads_missing_file(client: TestClient, monkeypatch, tmp_path) -> None:
    downloaded_file = tmp_path / "downloaded-model.safetensors"

    class FakeHub:
        @staticmethod
        def try_to_load_from_cache(**_kwargs):
            return None

        @staticmethod
        def hf_hub_download(**_kwargs):
            downloaded_file.write_bytes(b"weights")
            return str(downloaded_file)

    import openai_multi_backend.models.download as download_module

    monkeypatch.setattr(download_module, "import_optional", lambda _name: FakeHub)
    reset_downloader_for_tests()

    response = client.post(
        "/v1/models/download",
        json={
            "model": "Lightricks/LTX-2.3",
            "files": ["ltx-2.3-22b-distilled-1.1.safetensors"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["already_cached"] is False
    assert payload["artifacts"][0]["status"] == "downloaded"
    assert payload["artifacts"][0]["local_path"] == str(downloaded_file)
    reset_downloader_for_tests()


def test_ltx_fp8_config_switches_download_defaults() -> None:
    settings = Settings(environment="test", ltx_repo_id="Lightricks/LTX-2.3-fp8")
    request = ModelDownloadRequest(model="Lightricks/LTX-2.3")

    plan = resolve_download_plan(request, settings)

    assert settings.ltx_checkpoint_filename == "ltx-2.3-22b-distilled-fp8.safetensors"
    assert settings.ltx_spatial_upsampler_filename is None
    assert settings.ltx_pipeline_module == "ltx_pipelines.ti2vid_one_stage"
    assert plan.repo_id == "Lightricks/LTX-2.3-fp8"
    assert plan.files == ("ltx-2.3-22b-distilled-fp8.safetensors",)


def test_ltx_cli_load_discovers_module_without_importing_it(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "ltx_pipelines"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "distilled.py").write_text(
        "raise SystemExit('distilled CLI should not run during adapter load')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    checkpoint_path = tmp_path / "checkpoint.safetensors"
    checkpoint_path.write_bytes(b"weights")
    upsampler_path = tmp_path / "upsampler.safetensors"
    upsampler_path.write_bytes(b"weights")
    gemma_root = tmp_path / "gemma"
    gemma_root.mkdir()

    settings = Settings(
        environment="test",
        default_device="cpu",
        ltx_checkpoint_path=checkpoint_path,
        ltx_spatial_upsampler_path=upsampler_path,
        ltx_gemma_root=gemma_root,
    )
    adapter = LTXCliMediaAdapter("Lightricks/LTX-2.3", settings)

    adapter.load()

    assert adapter.checkpoint_path == checkpoint_path.resolve()
    assert "--distilled-checkpoint-path" in adapter.help_text
    assert "--distilled-lora" in adapter.help_text


def test_ltx_cli_load_downloads_default_gemma_snapshot(monkeypatch, tmp_path) -> None:
    package_dir = tmp_path / "ltx_pipelines"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("")
    (package_dir / "distilled.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    checkpoint_path = tmp_path / "checkpoint.safetensors"
    checkpoint_path.write_bytes(b"weights")
    upsampler_path = tmp_path / "upsampler.safetensors"
    upsampler_path.write_bytes(b"weights")
    gemma_snapshot = tmp_path / "gemma-snapshot"
    gemma_snapshot.mkdir()

    class FakeHub:
        snapshot_kwargs: dict[str, object] | None = None

        @classmethod
        def snapshot_download(cls, **kwargs):
            cls.snapshot_kwargs = kwargs
            return str(gemma_snapshot)

    import openai_multi_backend.models.image as image_module

    monkeypatch.setattr(image_module, "import_optional", lambda _name: FakeHub)
    settings = Settings(
        environment="test",
        default_device="cpu",
        model_cache_dir=tmp_path / "cache",
        hf_token="test-token",
        ltx_checkpoint_path=checkpoint_path,
        ltx_spatial_upsampler_path=upsampler_path,
    )
    adapter = LTXCliMediaAdapter("Lightricks/LTX-2.3", settings)

    adapter.load()

    assert adapter.gemma_root == gemma_snapshot
    assert FakeHub.snapshot_kwargs == {
        "repo_id": "google/gemma-3-12b-it-qat-q4_0-unquantized",
        "cache_dir": str(tmp_path / "cache"),
        "token": "test-token",
    }


def test_chat_model_download_uses_snapshot_plan(client: TestClient, monkeypatch, tmp_path) -> None:
    snapshot_file = tmp_path / "snapshot" / "config.json"

    class Sibling:
        rfilename = "config.json"

    class Info:
        siblings = [Sibling()]

    class FakeHub:
        @staticmethod
        def model_info(**_kwargs):
            return Info()

        @staticmethod
        def try_to_load_from_cache(**_kwargs):
            return None

        @staticmethod
        def snapshot_download(**_kwargs):
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_bytes(b"{}")
            return str(snapshot_file.parent)

    import openai_multi_backend.models.download as download_module

    monkeypatch.setattr(download_module, "import_optional", lambda _name: FakeHub)
    reset_downloader_for_tests()

    response = client.post(
        "/v1/models/download",
        json={"model": "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"][0]["filename"] == "config.json"
    assert payload["artifacts"][0]["status"] == "downloaded"
    reset_downloader_for_tests()


def test_ltx_gemma_download_uses_auxiliary_snapshot_plan(
    client: TestClient, monkeypatch, tmp_path
) -> None:
    snapshot_file = tmp_path / "gemma" / "config.json"

    class Sibling:
        rfilename = "config.json"

    class Info:
        siblings = [Sibling()]

    class FakeHub:
        @staticmethod
        def model_info(**_kwargs):
            return Info()

        @staticmethod
        def try_to_load_from_cache(**_kwargs):
            return None

        @staticmethod
        def snapshot_download(**kwargs):
            assert kwargs["repo_id"] == LTX_GEMMA_REPO_ID
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_bytes(b"{}")
            return str(snapshot_file.parent)

    import openai_multi_backend.models.download as download_module

    monkeypatch.setattr(download_module, "import_optional", lambda _name: FakeHub)
    reset_downloader_for_tests()

    response = client.post(
        "/v1/models/download",
        json={"model": LTX_GEMMA_REPO_ID},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == LTX_GEMMA_REPO_ID
    assert payload["artifacts"][0]["repo_id"] == LTX_GEMMA_REPO_ID
    assert payload["artifacts"][0]["filename"] == "config.json"
    reset_downloader_for_tests()


def test_audio_model_download_uses_snapshot_plan(client: TestClient, monkeypatch, tmp_path) -> None:
    snapshot_file = tmp_path / "snapshot" / "preprocessor_config.json"

    class Sibling:
        rfilename = "preprocessor_config.json"

    class Info:
        siblings = [Sibling()]

    class FakeHub:
        @staticmethod
        def model_info(**_kwargs):
            return Info()

        @staticmethod
        def try_to_load_from_cache(**_kwargs):
            return None

        @staticmethod
        def snapshot_download(**_kwargs):
            snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file.write_bytes(b"{}")
            return str(snapshot_file.parent)

    import openai_multi_backend.models.download as download_module

    monkeypatch.setattr(download_module, "import_optional", lambda _name: FakeHub)
    reset_downloader_for_tests()

    response = client.post(
        "/v1/models/download",
        json={"model": "openai/whisper-large-v3-turbo"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifacts"][0]["filename"] == "preprocessor_config.json"
    assert payload["artifacts"][0]["status"] == "downloaded"
    reset_downloader_for_tests()
