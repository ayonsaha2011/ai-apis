from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import Depends

from openai_multi_backend.config import REQUESTED_MODEL_IDS, Settings, get_settings
from openai_multi_backend.metrics import MODEL_LOADS, MODEL_STATE, STATE_VALUES
from openai_multi_backend.models.base import BaseModelAdapter, Endpoint, ModelError, ModelLoadError

AdapterFactory = Callable[[str, Settings], BaseModelAdapter]


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    owned_by: str
    priority: int
    modalities: tuple[str, ...]
    endpoints: tuple[Endpoint, ...]
    adapter_kind: str
    created: int = 1_714_521_600


@dataclass
class ModelEntry:
    metadata: ModelMetadata
    state: str = "configured"
    adapter: BaseModelAdapter | None = None
    last_error: str | None = None
    load_started_at: float | None = None
    load_duration_seconds: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inference_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))


MODEL_METADATA: dict[str, ModelMetadata] = {
    "Lightricks/LTX-2.3": ModelMetadata(
        id="Lightricks/LTX-2.3",
        owned_by="Lightricks",
        priority=1,
        modalities=("image", "video"),
        endpoints=("images.generations",),
        adapter_kind="ltx_media",
    ),
    "dx8152/Flux2-Klein-9B-Consistency": ModelMetadata(
        id="dx8152/Flux2-Klein-9B-Consistency",
        owned_by="dx8152",
        priority=2,
        modalities=("image",),
        endpoints=("images.generations",),
        adapter_kind="diffusers_image",
    ),
    "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic": ModelMetadata(
        id="llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic",
        owned_by="llmfan46",
        priority=3,
        modalities=("text",),
        endpoints=("chat.completions", "completions"),
        adapter_kind="causal_lm",
    ),
    "nvidia/parakeet-tdt-0.6b-v3": ModelMetadata(
        id="nvidia/parakeet-tdt-0.6b-v3",
        owned_by="nvidia",
        priority=4,
        modalities=("audio", "speech-to-text"),
        endpoints=("audio.transcriptions",),
        adapter_kind="parakeet_asr",
    ),
    "openai/whisper-large-v3-turbo": ModelMetadata(
        id="openai/whisper-large-v3-turbo",
        owned_by="openai",
        priority=5,
        modalities=("audio", "speech-to-text", "translation"),
        endpoints=("audio.transcriptions", "audio.translations"),
        adapter_kind="whisper_asr",
    ),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ModelMetadata(
        id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        owned_by="Qwen",
        priority=6,
        modalities=("audio", "text-to-speech", "voice-clone"),
        endpoints=("audio.speech",),
        adapter_kind="qwen_tts",
    ),
    "coqui/XTTS-v2": ModelMetadata(
        id="coqui/XTTS-v2",
        owned_by="coqui",
        priority=7,
        modalities=("audio", "text-to-speech", "voice-clone"),
        endpoints=("audio.speech",),
        adapter_kind="coqui_xtts",
    ),
}


def _adapter_factory(kind: str) -> AdapterFactory:
    if kind == "causal_lm":
        from openai_multi_backend.models.text import CausalLMAdapter

        return CausalLMAdapter
    if kind in {"diffusers_image", "ltx_media"}:
        from openai_multi_backend.models.image import DiffusersMediaAdapter, LTXCliMediaAdapter

        if kind == "ltx_media":
            return LTXCliMediaAdapter

        return DiffusersMediaAdapter
    if kind == "whisper_asr":
        from openai_multi_backend.models.audio import WhisperASRAdapter

        return WhisperASRAdapter
    if kind == "parakeet_asr":
        from openai_multi_backend.models.audio import ParakeetASRAdapter

        return ParakeetASRAdapter
    if kind == "qwen_tts":
        from openai_multi_backend.models.speech import QwenTTSAdapter

        return QwenTTSAdapter
    if kind == "coqui_xtts":
        from openai_multi_backend.models.speech import CoquiXTTSAdapter

        return CoquiXTTSAdapter
    raise ModelLoadError(f"No adapter factory registered for kind '{kind}'")


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_semaphore = asyncio.Semaphore(settings.max_concurrent_model_loads)
        self._entries: dict[str, ModelEntry] = {}
        for model_id in REQUESTED_MODEL_IDS:
            metadata = MODEL_METADATA[model_id]
            state = "configured" if model_id in settings.enabled_models else "disabled"
            entry = ModelEntry(
                metadata=metadata,
                state=state,
                inference_semaphore=asyncio.Semaphore(
                    settings.max_concurrent_inferences_per_model
                ),
            )
            MODEL_STATE.labels(model=model_id).set(STATE_VALUES[state])
            self._entries[model_id] = entry

    def list_metadata(self) -> list[ModelEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.metadata.priority)

    def get_entry(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise KeyError(model_id)
        return self._entries[model_id]

    def metadata_for_endpoint(self, endpoint: Endpoint) -> list[ModelMetadata]:
        return [
            entry.metadata
            for entry in self.list_metadata()
            if endpoint in entry.metadata.endpoints and entry.state != "disabled"
        ]

    async def load_adapter(self, model_id: str, endpoint: Endpoint) -> BaseModelAdapter:
        entry = self.get_entry(model_id)
        if entry.state == "disabled":
            raise ModelLoadError(f"Model '{model_id}' is disabled by configuration")
        if endpoint not in entry.metadata.endpoints:
            raise ModelLoadError(f"Model '{model_id}' does not support endpoint '{endpoint}'")
        async with entry.lock:
            if entry.adapter is not None and entry.state == "ready":
                return entry.adapter
            async with self._load_semaphore:
                entry.state = "loading"
                entry.load_started_at = time.monotonic()
                entry.last_error = None
                MODEL_STATE.labels(model=model_id).set(STATE_VALUES["loading"])
                try:
                    adapter = _adapter_factory(entry.metadata.adapter_kind)(model_id, self.settings)
                    await asyncio.to_thread(adapter.load)
                    entry.adapter = adapter
                    entry.state = "ready"
                    entry.load_duration_seconds = time.monotonic() - entry.load_started_at
                    MODEL_LOADS.labels(model=model_id, status="success").inc()
                    MODEL_STATE.labels(model=model_id).set(STATE_VALUES["ready"])
                    return adapter
                except ModelError as exc:
                    entry.state = "failed"
                    entry.last_error = str(exc)
                    entry.load_duration_seconds = time.monotonic() - entry.load_started_at
                    MODEL_LOADS.labels(model=model_id, status="failed").inc()
                    MODEL_STATE.labels(model=model_id).set(STATE_VALUES["failed"])
                    raise
                except Exception as exc:
                    entry.state = "failed"
                    entry.last_error = str(exc)
                    entry.load_duration_seconds = time.monotonic() - entry.load_started_at
                    MODEL_LOADS.labels(model=model_id, status="failed").inc()
                    MODEL_STATE.labels(model=model_id).set(STATE_VALUES["failed"])
                    raise ModelLoadError(f"Failed to load model '{model_id}': {exc}") from exc

    def health(self) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for model_id, entry in self._entries.items():
            payload[model_id] = {
                "state": entry.state,
                "modalities": list(entry.metadata.modalities),
                "endpoints": list(entry.metadata.endpoints),
                "last_error": entry.last_error,
                "load_duration_seconds": entry.load_duration_seconds,
                "device": entry.adapter.device if entry.adapter else None,
                "dtype": entry.adapter.dtype if entry.adapter else None,
            }
        return payload


_registry: ModelRegistry | None = None


def get_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    global _registry
    if _registry is None or _registry.settings is not settings:
        _registry = ModelRegistry(settings)
    return _registry


def reset_registry_for_tests() -> None:
    global _registry
    _registry = None
