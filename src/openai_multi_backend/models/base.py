from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai_multi_backend.config import Settings

Endpoint = Literal[
    "chat.completions",
    "completions",
    "images.generations",
    "audio.transcriptions",
    "audio.translations",
    "audio.speech",
]


class ModelError(RuntimeError):
    error_type = "server_error"
    code = "model_error"
    status_code = 500


class ModelLoadError(ModelError):
    code = "model_load_failed"
    status_code = 503


class ModelNotReadyError(ModelError):
    code = "model_not_ready"

    status_code = 503


class UnsupportedModelCapability(ModelError):
    error_type = "invalid_request_error"
    code = "unsupported_model_capability"
    status_code = 400


class OptionalDependencyError(ModelLoadError):
    code = "missing_optional_dependency"


@dataclass(frozen=True)
class TextGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


@dataclass(frozen=True)
class MediaItem:
    path: Path | None = None
    b64_json: str | None = None
    media_type: str = "image/png"
    revised_prompt: str | None = None
    extension: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class SpeechResult:
    path: Path
    media_type: str


def import_optional(module_name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = package_hint or module_name
        raise OptionalDependencyError(
            f"Optional dependency '{module_name}' is required. Install '{hint}' to use this model."
        ) from exc


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }


def media_type_for_format(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }.get(fmt, "application/octet-stream")


class BaseModelAdapter:
    def __init__(self, model_id: str, settings: Settings) -> None:
        self.model_id = model_id
        self.settings = settings
        self.device: str | None = None
        self.dtype: str | None = None

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        return None

    def resolve_device(self) -> str:
        if self.settings.default_device != "auto":
            return self.settings.default_device
        torch = import_optional("torch")
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def resolve_torch_dtype(self) -> Any:
        torch = import_optional("torch")
        if self.settings.torch_dtype == "float16":
            return torch.float16
        if self.settings.torch_dtype == "bfloat16":
            return torch.bfloat16
        if self.settings.torch_dtype == "float32":
            return torch.float32
        device = self.resolve_device()
        if device == "cuda":
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32

    def common_hf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "cache_dir": str(self.settings.model_cache_dir),
            "trust_remote_code": self.settings.trust_remote_code_for(self.model_id),
        }
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        return kwargs
