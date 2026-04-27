from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REQUESTED_MODEL_IDS: tuple[str, ...] = (
    "Lightricks/LTX-2.3",
    "dx8152/Flux2-Klein-9B-Consistency",
    "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic",
    "nvidia/parakeet-tdt-0.6b-v3",
    "openai/whisper-large-v3-turbo",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "coqui/XTTS-v2",
)

LTX_STANDARD_REPO_ID = "Lightricks/LTX-2.3"
LTX_FP8_REPO_ID = "Lightricks/LTX-2.3-fp8"
LTX_STANDARD_CHECKPOINT = "ltx-2.3-22b-distilled-1.1.safetensors"
LTX_STANDARD_UPSAMPLER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_FP8_CHECKPOINT = "ltx-2.3-22b-distilled-fp8.safetensors"
LTX_DISTILLED_MODULE = "ltx_pipelines.distilled"
LTX_ONE_STAGE_MODULE = "ltx_pipelines.ti2vid_one_stage"
LTX_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"


def _split_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_MULTI_BACKEND_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "production"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"

    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)
    hf_token: str | None = None
    model_cache_dir: Path = Path(".model-cache")
    output_dir: Path = Path(".generated")
    external_base_url: str | None = None
    voice_reference_dir: Path | None = None
    voice_reference_url_allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )

    default_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    enabled_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(REQUESTED_MODEL_IDS)
    )
    trust_remote_code_models: Annotated[list[str], NoDecode] = Field(default_factory=list)

    max_concurrent_model_loads: int = 1
    max_concurrent_inferences_per_model: int = 1
    max_upload_bytes: int = 100 * 1024 * 1024
    request_timeout_seconds: int = 900

    text_max_new_tokens_default: int = 1024
    text_max_new_tokens_limit: int = 8192
    image_max_batch_size: int = 4
    image_default_steps: int = 28
    image_max_steps: int = 100
    video_default_frames: int = 49
    video_max_frames: int = 257
    video_default_frame_rate: float = 24.0
    ltx_repo_id: Literal["Lightricks/LTX-2.3", "Lightricks/LTX-2.3-fp8"] = (
        "Lightricks/LTX-2.3"
    )
    ltx_pipeline_module: str = LTX_DISTILLED_MODULE
    ltx_checkpoint_path: Path | None = None
    ltx_checkpoint_filename: str = LTX_STANDARD_CHECKPOINT
    ltx_spatial_upsampler_path: Path | None = None
    ltx_spatial_upsampler_filename: str | None = LTX_STANDARD_UPSAMPLER
    ltx_gemma_root: Path | None = None
    ltx_gemma_repo_id: str = LTX_GEMMA_REPO_ID
    ltx_distilled_lora_path: Path | None = None
    audio_sample_rate: int = 24000

    @field_validator(
        "api_keys",
        "enabled_models",
        "trust_remote_code_models",
        "voice_reference_url_allowed_hosts",
        mode="before",
    )
    @classmethod
    def parse_csv_lists(cls, value: str | list[str] | tuple[str, ...] | None) -> list[str]:
        return _split_csv(value)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("external_base_url")
    @classmethod
    def normalize_base_url(cls, value: str | None) -> str | None:
        if not value:
            return None
        return value.rstrip("/")

    @field_validator("ltx_spatial_upsampler_filename")
    @classmethod
    def normalize_optional_ltx_filename(cls, value: str | None) -> str | None:
        return value or None

    @model_validator(mode="after")
    def validate_security(self) -> Settings:
        if self.environment == "production" and not self.api_keys:
            raise ValueError("OPENAI_MULTI_BACKEND_API_KEYS is required in production")
        unknown_models = sorted(set(self.enabled_models) - set(REQUESTED_MODEL_IDS))
        if unknown_models:
            raise ValueError(f"Unknown enabled model IDs: {', '.join(unknown_models)}")
        if self.ltx_repo_id == LTX_FP8_REPO_ID:
            if self.ltx_checkpoint_filename == LTX_STANDARD_CHECKPOINT:
                self.ltx_checkpoint_filename = LTX_FP8_CHECKPOINT
            if self.ltx_spatial_upsampler_filename == LTX_STANDARD_UPSAMPLER:
                self.ltx_spatial_upsampler_filename = None
            if self.ltx_pipeline_module == LTX_DISTILLED_MODULE:
                self.ltx_pipeline_module = LTX_ONE_STAGE_MODULE
        return self

    def trust_remote_code_for(self, model_id: str) -> bool:
        return model_id in self.trust_remote_code_models

    def prepare_directories(self) -> None:
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.voice_reference_dir is not None:
            self.voice_reference_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
