from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai_multi_backend.api.schemas import (
    ModelDownloadArtifact,
    ModelDownloadRequest,
    ModelDownloadResponse,
)
from openai_multi_backend.config import LTX_GEMMA_REPO_ID, Settings
from openai_multi_backend.models.base import (
    ModelLoadError,
    OptionalDependencyError,
    import_optional,
)
from openai_multi_backend.models.registry import MODEL_METADATA


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    files: tuple[str, ...]
    snapshot: bool


SNAPSHOT_DOWNLOAD_MODEL_IDS = {
    LTX_GEMMA_REPO_ID,
    "llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic",
    "nvidia/parakeet-tdt-0.6b-v3",
    "openai/whisper-large-v3-turbo",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "coqui/XTTS-v2",
}

AUXILIARY_DOWNLOAD_MODEL_IDS = {
    LTX_GEMMA_REPO_ID,
}


class ModelDownloader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def download(self, request: ModelDownloadRequest) -> ModelDownloadResponse:
        plan = resolve_download_plan(request, self.settings)
        lock = await self._lock_for(plan.repo_id)
        async with lock:
            artifacts = await asyncio.to_thread(self._download_sync, request, plan)
        return ModelDownloadResponse(
            model=request.model,
            revision=request.revision,
            already_cached=all(artifact.status == "cached" for artifact in artifacts),
            artifacts=artifacts,
        )

    async def _lock_for(self, repo_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            if repo_id not in self._locks:
                self._locks[repo_id] = asyncio.Lock()
            return self._locks[repo_id]

    def _download_sync(
        self, request: ModelDownloadRequest, plan: DownloadPlan
    ) -> list[ModelDownloadArtifact]:
        huggingface_hub = import_optional("huggingface_hub")
        if plan.snapshot:
            return self._download_snapshot(huggingface_hub, request, plan)
        return [
            self._download_file(huggingface_hub, request, plan.repo_id, filename)
            for filename in plan.files
        ]

    def _download_snapshot(
        self, huggingface_hub: Any, request: ModelDownloadRequest, plan: DownloadPlan
    ) -> list[ModelDownloadArtifact]:
        filenames = list(plan.files) or list_repo_files(
            huggingface_hub, plan.repo_id, self.settings, request.revision
        )
        if not filenames:
            raise ModelLoadError(f"Model repository '{plan.repo_id}' has no downloadable files")
        cached = [
            cached_file_path(
                huggingface_hub, plan.repo_id, filename, self.settings, request.revision
            )
            for filename in filenames
        ]
        if not request.force and all(path is not None and path.exists() for path in cached):
            return [
                artifact_response(plan.repo_id, filename, path, "cached")
                for filename, path in zip(filenames, cached, strict=True)
                if path is not None
            ]
        if request.local_files_only:
            missing = [
                filename
                for filename, path in zip(filenames, cached, strict=True)
                if path is None
            ]
            raise ModelLoadError(
                "Requested local-only model download but files are missing from cache: "
                + ", ".join(missing)
            )
        snapshot_path = huggingface_hub.snapshot_download(
            repo_id=plan.repo_id,
            revision=request.revision,
            cache_dir=str(self.settings.model_cache_dir),
            token=self.settings.hf_token,
            local_files_only=False,
            allow_patterns=list(filenames),
        )
        root = Path(snapshot_path)
        artifacts: list[ModelDownloadArtifact] = []
        for filename, before_path in zip(filenames, cached, strict=True):
            path = root / filename
            status: Literal["cached", "downloaded"] = (
                "cached" if before_path is not None and before_path.exists() else "downloaded"
            )
            artifacts.append(artifact_response(plan.repo_id, filename, path, status))
        return artifacts

    def _download_file(
        self, huggingface_hub: Any, request: ModelDownloadRequest, repo_id: str, filename: str
    ) -> ModelDownloadArtifact:
        cached = cached_file_path(
            huggingface_hub, repo_id, filename, self.settings, request.revision
        )
        if cached is not None and cached.exists() and not request.force:
            return artifact_response(repo_id, filename, cached, "cached")
        if request.local_files_only:
            raise ModelLoadError(
                f"Requested local-only model download but '{filename}' is missing from cache"
            )
        path = Path(
            huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=request.revision,
                cache_dir=str(self.settings.model_cache_dir),
                token=self.settings.hf_token,
                local_files_only=False,
                force_download=request.force,
            )
        )
        return artifact_response(repo_id, filename, path, "downloaded")


def resolve_download_plan(request: ModelDownloadRequest, settings: Settings) -> DownloadPlan:
    if request.model not in MODEL_METADATA and request.model not in AUXILIARY_DOWNLOAD_MODEL_IDS:
        raise KeyError(request.model)
    if request.files:
        return DownloadPlan(
            repo_id=download_repo_id_for_model(request.model, settings),
            files=tuple(request.files),
            snapshot=False,
        )
    if request.model == "Lightricks/LTX-2.3":
        files = [settings.ltx_checkpoint_filename]
        if settings.ltx_spatial_upsampler_filename:
            files.append(settings.ltx_spatial_upsampler_filename)
        return DownloadPlan(repo_id=settings.ltx_repo_id, files=tuple(files), snapshot=False)
    if request.model == "dx8152/Flux2-Klein-9B-Consistency":
        return DownloadPlan(
            repo_id=request.model,
            files=("Flux2-Klein-9B-consistency-V2.safetensors",),
            snapshot=False,
        )
    if request.model in SNAPSHOT_DOWNLOAD_MODEL_IDS:
        if not request.allow_snapshot:
            raise ModelLoadError(
                f"Model '{request.model}' requires a repository snapshot download"
            )
        return DownloadPlan(repo_id=request.model, files=(), snapshot=True)
    if not request.allow_snapshot:
        raise ModelLoadError(
            f"No default single-file artifacts are configured for '{request.model}'"
        )
    return DownloadPlan(repo_id=request.model, files=(), snapshot=True)


def download_repo_id_for_model(model_id: str, settings: Settings) -> str:
    if model_id == "Lightricks/LTX-2.3":
        return settings.ltx_repo_id
    return model_id


def cached_file_path(
    huggingface_hub: Any,
    repo_id: str,
    filename: str,
    settings: Settings,
    revision: str | None,
) -> Path | None:
    try_to_load_from_cache = getattr(huggingface_hub, "try_to_load_from_cache", None)
    if try_to_load_from_cache is None:
        raise OptionalDependencyError("huggingface_hub.try_to_load_from_cache is required")
    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(settings.model_cache_dir),
    )
    if cached is None or cached is False:
        return None
    path = Path(cached)
    return path if path.exists() else None


def list_repo_files(
    huggingface_hub: Any, repo_id: str, settings: Settings, revision: str | None
) -> list[str]:
    info = huggingface_hub.model_info(repo_id=repo_id, revision=revision, token=settings.hf_token)
    siblings = getattr(info, "siblings", [])
    filenames = [sibling.rfilename for sibling in siblings if not sibling.rfilename.endswith("/")]
    return [filename for filename in filenames if filename != ".gitattributes"]


def artifact_response(
    repo_id: str, filename: str, path: Path, status: Literal["cached", "downloaded"]
) -> ModelDownloadArtifact:
    return ModelDownloadArtifact(
        repo_id=repo_id,
        filename=filename,
        local_path=str(path),
        status=status,
        bytes=path.stat().st_size if path.exists() and path.is_file() else None,
    )


_downloader: ModelDownloader | None = None


def get_downloader(settings: Settings) -> ModelDownloader:
    global _downloader
    if _downloader is None or _downloader.settings is not settings:
        _downloader = ModelDownloader(settings)
    return _downloader


def reset_downloader_for_tests() -> None:
    global _downloader
    _downloader = None
