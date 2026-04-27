from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ModelPermission(OpenAIBaseModel):
    id: str
    object: Literal["model_permission"] = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    permission: list[ModelPermission] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None
    modalities: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    status: str = "configured"


class ModelList(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ModelDownloadRequest(OpenAIBaseModel):
    model: str
    files: list[str] | None = None
    revision: str | None = None
    force: bool = False
    allow_snapshot: bool = True
    local_files_only: bool = False

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("files must contain at least one non-empty path")
        for item in cleaned:
            if item.startswith("/") or ".." in item.split("/"):
                raise ValueError("files must be relative Hugging Face repository paths")
        return cleaned


class ModelDownloadArtifact(OpenAIBaseModel):
    repo_id: str
    filename: str | None = None
    local_path: str
    status: Literal["cached", "downloaded"]
    bytes: int | None = None


class ModelDownloadResponse(OpenAIBaseModel):
    object: Literal["model.download"] = "model.download"
    model: str
    revision: str | None = None
    already_cached: bool
    artifacts: list[ModelDownloadArtifact]


class ChatMessage(OpenAIBaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def text_content(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)


class ChatCompletionRequest(OpenAIBaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = None


class CompletionRequest(OpenAIBaseModel):
    model: str
    prompt: str | list[str]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, gt=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    user: str | None = None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, list) and not value:
            raise ValueError("prompt list must contain at least one item")
        return value

    def first_prompt(self) -> str:
        return self.prompt[0] if isinstance(self.prompt, list) else self.prompt


class Usage(OpenAIBaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class CompletionChoice(OpenAIBaseModel):
    text: str
    index: int
    logprobs: Any | None = None
    finish_reason: str | None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionResponse(OpenAIBaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ImageGenerationRequest(OpenAIBaseModel):
    model: str
    prompt: str = Field(min_length=1)
    n: int = Field(default=1, ge=1)
    size: str = "1024x1024"
    quality: str | None = None
    response_format: Literal["url", "b64_json"] = "url"
    user: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    guidance_scale: float | None = Field(default=None, ge=0.0)
    num_inference_steps: int | None = Field(default=None, ge=1)
    frames: int | None = Field(default=None, ge=1)
    duration: float | None = Field(default=None, gt=0.0)
    frame_rate: float | None = Field(default=None, gt=0.0)
    enhance_prompt: bool = True

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: str) -> str:
        parts = value.lower().split("x")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("size must use WIDTHxHEIGHT format")
        width, height = (int(part) for part in parts)
        if width < 64 or height < 64 or width > 4096 or height > 4096:
            raise ValueError("size dimensions must be between 64 and 4096 pixels")
        return value.lower()

    def dimensions(self) -> tuple[int, int]:
        width, height = self.size.split("x")
        return int(width), int(height)


class ImageData(OpenAIBaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None
    video_url: str | None = None
    media_type: str | None = None


class ImageGenerationResponse(OpenAIBaseModel):
    created: int
    data: list[ImageData]


class SpeechRequest(OpenAIBaseModel):
    model: str
    input: str = Field(min_length=1)
    voice: str = "default"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, gt=0.25, le=4.0)
    language: str | None = None
    voice_reference: str | None = None
    voice_reference_url: str | None = None
    speaker_wav: str | None = None


class TranscriptionResponse(OpenAIBaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict[str, Any]] | None = None


class ReadinessResponse(OpenAIBaseModel):
    status: Literal["ready", "degraded"]
    models: dict[str, dict[str, Any]]
