from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from openai_multi_backend.api.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ModelCard,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelList,
    ModelPermission,
    SpeechRequest,
    TranscriptionResponse,
    Usage,
)
from openai_multi_backend.config import Settings, get_settings
from openai_multi_backend.errors import OpenAIHTTPException
from openai_multi_backend.metrics import ACTIVE_INFERENCES, REQUEST_COUNT, REQUEST_LATENCY
from openai_multi_backend.models.audio import ParakeetASRAdapter, WhisperASRAdapter
from openai_multi_backend.models.base import (
    MediaItem,
    ModelError,
    SpeechResult,
    TextGeneration,
    TranscriptionResult,
)
from openai_multi_backend.models.download import get_downloader
from openai_multi_backend.models.image import MediaGenerationAdapter
from openai_multi_backend.models.registry import ModelEntry, ModelRegistry, get_registry
from openai_multi_backend.models.speech import CoquiXTTSAdapter, QwenTTSAdapter
from openai_multi_backend.models.text import CausalLMAdapter
from openai_multi_backend.security import require_api_key

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
T = TypeVar("T")


@router.get("/models", response_model=ModelList)
async def list_models(registry: ModelRegistry = Depends(get_registry)) -> ModelList:
    return ModelList(data=[model_card(entry) for entry in registry.list_metadata()])


@router.get("/models/{model_id:path}", response_model=ModelCard)
async def get_model(model_id: str, registry: ModelRegistry = Depends(get_registry)) -> ModelCard:
    try:
        return model_card(registry.get_entry(model_id))
    except KeyError as exc:
        raise OpenAIHTTPException(
            404, f"Model '{model_id}' was not found", code="model_not_found"
        ) from exc


@router.post("/models/download", response_model=ModelDownloadResponse)
async def download_model(
    request: ModelDownloadRequest,
    settings: Settings = Depends(get_settings),
) -> ModelDownloadResponse:
    downloader = get_downloader(settings)
    try:
        return await downloader.download(request)
    except KeyError as exc:
        raise OpenAIHTTPException(
            404, f"Model '{request.model}' was not found", code="model_not_found"
        ) from exc
    except ModelError as exc:
        raise OpenAIHTTPException(
            exc.status_code, str(exc), error_type=exc.error_type, code=exc.code
        ) from exc
    except Exception as exc:
        raise OpenAIHTTPException(
            503,
            f"Failed to download model '{request.model}': {exc}",
            error_type="server_error",
            code="model_download_failed",
        ) from exc


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> ChatCompletionResponse | StreamingResponse:
    entry, adapter = await load_typed_adapter(
        registry, request.model, "chat.completions", CausalLMAdapter
    )
    if request.stream:
        return StreamingResponse(
            stream_chat_chunks(entry, adapter, request, settings),
            media_type="text/event-stream",
        )
    result: TextGeneration = await infer_with_metrics(
        entry,
        "chat.completions",
        settings,
        adapter.generate_chat,
        request.messages,
        request.temperature,
        request.top_p,
        request.max_tokens,
        request.stop,
        request.frequency_penalty,
    )
    created = int(time.time())
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=created,
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@router.post("/completions", response_model=None)
async def completions(
    request: CompletionRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> CompletionResponse | StreamingResponse:
    entry, adapter = await load_typed_adapter(
        registry, request.model, "completions", CausalLMAdapter
    )
    if request.stream:
        return StreamingResponse(
            stream_completion_chunks(entry, adapter, request, settings),
            media_type="text/event-stream",
        )
    result: TextGeneration = await infer_with_metrics(
        entry,
        "completions",
        settings,
        adapter.generate_completion,
        request,
    )
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[CompletionChoice(text=result.text, index=0, finish_reason=result.finish_reason)],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@router.post("/images/generations", response_model=ImageGenerationResponse)
async def images_generations(
    request: ImageGenerationRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> ImageGenerationResponse:
    entry, adapter = await load_typed_adapter(
        registry, request.model, "images.generations", MediaGenerationAdapter
    )
    items: list[MediaItem] = await infer_with_metrics(
        entry,
        "images.generations",
        settings,
        adapter.generate,
        request,
    )
    return ImageGenerationResponse(
        created=int(time.time()),
        data=[image_data(item, settings, request.response_format) for item in items],
    )


@router.post("/audio/transcriptions", response_model=None)
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> JSONResponse | PlainTextResponse:
    audio_path = await persist_upload(file, settings.max_upload_bytes)
    try:
        if model == "nvidia/parakeet-tdt-0.6b-v3":
            entry, parakeet_adapter = await load_typed_adapter(
                registry, model, "audio.transcriptions", ParakeetASRAdapter
            )
            result: TranscriptionResult = await infer_with_metrics(
                entry, "audio.transcriptions", settings, parakeet_adapter.transcribe, audio_path
            )
        else:
            entry, whisper_adapter = await load_typed_adapter(
                registry, model, "audio.transcriptions", WhisperASRAdapter
            )
            result = await infer_with_metrics(
                entry,
                "audio.transcriptions",
                settings,
                whisper_adapter.transcribe,
                audio_path,
                language,
                prompt,
                "transcribe",
            )
        return transcription_response(result, response_format)
    finally:
        audio_path.unlink(missing_ok=True)


@router.post("/audio/translations", response_model=None)
async def audio_translations(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(default=None),
    response_format: Literal["json", "text", "verbose_json"] = Form(default="json"),
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> JSONResponse | PlainTextResponse:
    audio_path = await persist_upload(file, settings.max_upload_bytes)
    try:
        entry, adapter = await load_typed_adapter(
            registry, model, "audio.translations", WhisperASRAdapter
        )
        result: TranscriptionResult = await infer_with_metrics(
            entry,
            "audio.translations",
            settings,
            adapter.transcribe,
            audio_path,
            None,
            prompt,
            "translate",
        )
        return transcription_response(result, response_format)
    finally:
        audio_path.unlink(missing_ok=True)


@router.post("/audio/speech", response_model=None)
async def audio_speech(
    request: SpeechRequest,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    if request.model == "coqui/XTTS-v2":
        entry, coqui_adapter = await load_typed_adapter(
            registry, request.model, "audio.speech", CoquiXTTSAdapter
        )
        synthesize = coqui_adapter.synthesize
    else:
        entry, qwen_adapter = await load_typed_adapter(
            registry, request.model, "audio.speech", QwenTTSAdapter
        )
        synthesize = qwen_adapter.synthesize
    result: SpeechResult = await infer_with_metrics(
        entry,
        "audio.speech",
        settings,
        synthesize,
        request,
    )
    return FileResponse(path=result.path, media_type=result.media_type, filename=result.path.name)


def model_card(entry: ModelEntry) -> ModelCard:
    created = entry.metadata.created
    return ModelCard(
        id=entry.metadata.id,
        created=created,
        owned_by=entry.metadata.owned_by,
        permission=[
            ModelPermission(
                id=f"modelperm-{uuid.uuid5(uuid.NAMESPACE_URL, entry.metadata.id).hex}",
                created=created,
            )
        ],
        root=entry.metadata.id,
        modalities=list(entry.metadata.modalities),
        endpoints=list(entry.metadata.endpoints),
        status=entry.state,
    )


async def load_typed_adapter(
    registry: ModelRegistry,
    model_id: str,
    endpoint: str,
    expected_type: type[T],
) -> tuple[ModelEntry, T]:
    try:
        entry = registry.get_entry(model_id)
    except KeyError as exc:
        raise OpenAIHTTPException(
            404, f"Model '{model_id}' was not found", code="model_not_found"
        ) from exc
    adapter = await typed_loaded(registry, model_id, endpoint, expected_type)
    return entry, adapter


async def typed_loaded(
    registry: ModelRegistry,
    model_id: str,
    endpoint: str,
    expected_type: type[T],
) -> T:
    try:
        adapter = await registry.load_adapter(model_id, endpoint)  # type: ignore[arg-type]
    except KeyError as exc:
        raise OpenAIHTTPException(
            404, f"Model '{model_id}' was not found", code="model_not_found"
        ) from exc
    except ModelError as exc:
        raise OpenAIHTTPException(
            exc.status_code, str(exc), error_type=exc.error_type, code=exc.code
        ) from exc
    if not isinstance(adapter, expected_type):
        raise OpenAIHTTPException(
            400,
            f"Model '{model_id}' is not compatible with endpoint '{endpoint}'",
            code="model_endpoint_mismatch",
        )
    return cast(T, adapter)


async def infer_with_metrics(
    entry: ModelEntry,
    endpoint: str,
    settings: Settings,
    func: Any,
    *args: Any,
) -> T:
    start = time.monotonic()
    ACTIVE_INFERENCES.labels(model=entry.metadata.id).inc()
    async with entry.inference_semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args), timeout=settings.request_timeout_seconds
            )
            REQUEST_COUNT.labels(endpoint=endpoint, model=entry.metadata.id, status="success").inc()
            return result
        except ModelError as exc:
            REQUEST_COUNT.labels(endpoint=endpoint, model=entry.metadata.id, status="error").inc()
            raise OpenAIHTTPException(
                exc.status_code, str(exc), error_type=exc.error_type, code=exc.code
            ) from exc
        except TimeoutError as exc:
            REQUEST_COUNT.labels(endpoint=endpoint, model=entry.metadata.id, status="timeout").inc()
            raise OpenAIHTTPException(
                504, "Model inference timed out", error_type="server_error", code="timeout"
            ) from exc
        except Exception as exc:
            REQUEST_COUNT.labels(endpoint=endpoint, model=entry.metadata.id, status="error").inc()
            raise OpenAIHTTPException(
                500, str(exc), error_type="server_error", code="inference_failed"
            ) from exc
        finally:
            ACTIVE_INFERENCES.labels(model=entry.metadata.id).dec()
            REQUEST_LATENCY.labels(endpoint=endpoint, model=entry.metadata.id).observe(
                time.monotonic() - start
            )


async def stream_chat_chunks(
    entry: ModelEntry,
    adapter: CausalLMAdapter,
    request: ChatCompletionRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    async with entry.inference_semaphore:
        ACTIVE_INFERENCES.labels(model=entry.metadata.id).inc()
        try:
            iterator = adapter.stream_chat(
                request.messages,
                request.temperature,
                request.top_p,
                request.max_tokens,
                request.stop,
                request.frequency_penalty,
            )
            deadline = time.monotonic() + settings.request_timeout_seconds
            while True:
                if time.monotonic() > deadline:
                    raise OpenAIHTTPException(504, "Model inference timed out", code="timeout")
                sentinel = object()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                payload = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            final_payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_payload, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
            REQUEST_COUNT.labels(
                endpoint="chat.completions", model=entry.metadata.id, status="success"
            ).inc()
        finally:
            ACTIVE_INFERENCES.labels(model=entry.metadata.id).dec()


async def stream_completion_chunks(
    entry: ModelEntry,
    adapter: CausalLMAdapter,
    request: CompletionRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    chunk_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    async with entry.inference_semaphore:
        ACTIVE_INFERENCES.labels(model=entry.metadata.id).inc()
        try:
            iterator = adapter.stream_completion(request)
            deadline = time.monotonic() + settings.request_timeout_seconds
            while True:
                if time.monotonic() > deadline:
                    raise OpenAIHTTPException(504, "Model inference timed out", code="timeout")
                sentinel = object()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                payload = {
                    "id": chunk_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {"text": chunk, "index": 0, "logprobs": None, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
            final_payload = {
                "id": chunk_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_payload, separators=(',', ':'))}\n\n"
            yield "data: [DONE]\n\n"
            REQUEST_COUNT.labels(
                endpoint="completions", model=entry.metadata.id, status="success"
            ).inc()
        finally:
            ACTIVE_INFERENCES.labels(model=entry.metadata.id).dec()


def image_data(item: MediaItem, settings: Settings, response_format: str) -> ImageData:
    if item.path is None and item.b64_json is None:
        raise OpenAIHTTPException(500, "Generated media item has no content", code="empty_media")
    video_url = None
    if item.media_type.startswith("video") and item.path is not None:
        video_url = public_url(settings, item.path)
    if response_format == "b64_json":
        if item.b64_json is not None:
            encoded = item.b64_json
        elif item.path is not None:
            encoded = base64.b64encode(item.path.read_bytes()).decode("ascii")
        else:
            encoded = None
        return ImageData(
            b64_json=encoded,
            revised_prompt=item.revised_prompt,
            video_url=video_url,
            media_type=item.media_type,
        )
    url = public_url(settings, item.path) if item.path else None
    return ImageData(
        url=url,
        revised_prompt=item.revised_prompt,
        video_url=video_url,
        media_type=item.media_type,
    )


def public_url(settings: Settings, path: Path) -> str:
    filename = path.name
    if settings.external_base_url:
        return f"{settings.external_base_url}/generated/{filename}"
    return f"/generated/{filename}"


async def persist_upload(upload: UploadFile, max_bytes: int) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".wav"
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise OpenAIHTTPException(
            413, "Uploaded file exceeds configured size limit", code="file_too_large"
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return Path(handle.name)


def transcription_response(
    result: TranscriptionResult,
    response_format: Literal["json", "text", "verbose_json"],
) -> JSONResponse | PlainTextResponse:
    if response_format == "text":
        return PlainTextResponse(result.text)
    payload = TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        segments=result.segments if response_format == "verbose_json" else None,
    ).model_dump(exclude_none=True)
    return JSONResponse(payload)
