from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai_multi_backend.config import Settings
from openai_multi_backend.models.base import (
    BaseModelAdapter,
    MediaItem,
    ModelLoadError,
    OptionalDependencyError,
    filter_supported_kwargs,
    import_optional,
)

if TYPE_CHECKING:
    from openai_multi_backend.api.schemas import ImageGenerationRequest


class MediaGenerationAdapter(BaseModelAdapter):
    def generate(self, request: ImageGenerationRequest) -> list[MediaItem]:
        raise NotImplementedError

    def _output_path(self, suffix: str) -> Path:
        name = f"{uuid.uuid4().hex}.{suffix}"
        path = (self.settings.output_dir / name).resolve()
        output_root = self.settings.output_dir.resolve()
        if output_root not in path.parents:
            raise RuntimeError("Generated output path escaped configured output directory")
        return path


class DiffusersMediaAdapter(MediaGenerationAdapter):
    pipeline: Any

    def load(self) -> None:
        diffusers = import_optional("diffusers")
        torch = import_optional("torch")
        self.device = self.resolve_device()
        torch_dtype = self.resolve_torch_dtype()
        self.dtype = str(torch_dtype).replace("torch.", "")
        kwargs = self.common_hf_kwargs()
        kwargs["torch_dtype"] = torch_dtype
        self.pipeline = diffusers.DiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        if self.device != "cuda":
            self.pipeline.to(self.device)
        elif not getattr(self.pipeline, "hf_device_map", None):
            self.pipeline.to("cuda")
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)
        torch.set_grad_enabled(False)

    def generate(self, request: ImageGenerationRequest) -> list[MediaItem]:
        torch = import_optional("torch")
        width, height = request.dimensions()
        steps = min(
            request.num_inference_steps or self.settings.image_default_steps,
            self.settings.image_max_steps,
        )
        batch_size = min(request.n, self.settings.image_max_batch_size)
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self.device if self.device != "mps" else "cpu")
            generator.manual_seed(request.seed)
        frame_rate = resolve_video_frame_rate(request, self.settings)
        frames = resolve_video_frame_count(request, self.settings)
        raw_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": height,
            "width": width,
            "num_images_per_prompt": batch_size,
            "num_inference_steps": steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
            "num_frames": frames,
            "frame_count": frames,
            "frame_rate": frame_rate,
            "fps": frame_rate,
            "enhance_prompt": request.enhance_prompt,
        }
        kwargs = filter_supported_kwargs(self.pipeline.__call__, raw_kwargs)
        result = self.pipeline(**kwargs)
        items: list[MediaItem] = []
        images = getattr(result, "images", None)
        if images:
            for image in images[:batch_size]:
                items.append(self._save_image(image, request.prompt))
        frame_batches = getattr(result, "frames", None) or getattr(result, "videos", None)
        if frame_batches:
            normalized_batches = self._normalize_frame_batches(frame_batches)
            for frame_batch in normalized_batches[:batch_size]:
                items.append(self._save_video(frame_batch, request.prompt, frame_rate))
        if not items:
            raise RuntimeError(
                f"Diffusers pipeline for '{self.model_id}' returned no images or video frames"
            )
        return items

    def _save_image(self, image: Any, prompt: str) -> MediaItem:
        path = self._output_path("png")
        if hasattr(image, "save"):
            image.save(path)
        else:
            Image = import_optional("PIL.Image", "pillow")
            numpy = import_optional("numpy")
            Image.fromarray(numpy.asarray(image)).save(path)
        return MediaItem(path=path, media_type="image/png", revised_prompt=prompt)

    def _save_video(self, frames: list[Any], prompt: str, frame_rate: float) -> MediaItem:
        imageio = import_optional("imageio")
        numpy = import_optional("numpy")
        path = self._output_path("mp4")
        arrays = [numpy.asarray(frame) for frame in frames]
        imageio.mimsave(path, arrays, fps=frame_rate)
        return MediaItem(
            path=path,
            media_type="video/mp4",
            revised_prompt=prompt,
            extension={"video_url": str(path)},
        )

    @staticmethod
    def _normalize_frame_batches(frame_batches: Any) -> list[list[Any]]:
        if not isinstance(frame_batches, list):
            return [[frame_batches]]
        if not frame_batches:
            return []
        first = frame_batches[0]
        if isinstance(first, list):
            return frame_batches
        return [frame_batches]


class LTXCliMediaAdapter(MediaGenerationAdapter):
    checkpoint_path: Path
    spatial_upsampler_path: Path | None
    gemma_root: Path
    help_text: str

    def load(self) -> None:
        self._ensure_ltx_pipeline_module_available()
        self.device = self.resolve_device()
        self.dtype = "ltx-pipelines-default"
        self.checkpoint_path = self._resolve_ltx_file(
            configured_path=self.settings.ltx_checkpoint_path,
            filename=self.settings.ltx_checkpoint_filename,
        )
        self.spatial_upsampler_path = self._resolve_optional_ltx_file(
            configured_path=self.settings.ltx_spatial_upsampler_path,
            filename=self.settings.ltx_spatial_upsampler_filename,
        )
        self.gemma_root = self._resolve_ltx_directory(
            configured_path=self.settings.ltx_gemma_root,
            repo_id=self.settings.ltx_gemma_repo_id,
            description="Gemma text encoder",
        )
        self._validate_ltx_cli_config()
        self.help_text = self._supported_flags()

    def _ensure_ltx_pipeline_module_available(self) -> None:
        try:
            module_spec = importlib.util.find_spec(self.settings.ltx_pipeline_module)
        except (ImportError, AttributeError, ValueError) as exc:
            raise self._ltx_dependency_error() from exc
        if module_spec is None:
            raise self._ltx_dependency_error()

    @staticmethod
    def _ltx_dependency_error() -> OptionalDependencyError:
        return OptionalDependencyError(
            "Optional dependency 'ltx_pipelines' is required for LTX-2. "
            "Install it with: pip install -e '.[ltx]'. "
            "LTX-2 requires Python 3.12+ and the official package from "
            "https://github.com/Lightricks/LTX-2."
        )

    def generate(self, request: ImageGenerationRequest) -> list[MediaItem]:
        output_path = self._output_path("mp4")
        width, height = request.dimensions()
        frame_rate = resolve_video_frame_rate(request, self.settings)
        frames = resolve_video_frame_count(request, self.settings)
        steps = min(
            request.num_inference_steps or self.settings.image_default_steps,
            self.settings.image_max_steps,
        )
        args = [sys.executable, "-m", self.settings.ltx_pipeline_module]
        self._append_arg(args, self._checkpoint_flag(), str(self.checkpoint_path), required=True)
        if self.spatial_upsampler_path is not None:
            self._append_arg(args, "--spatial-upsampler-path", str(self.spatial_upsampler_path))
        self._append_arg(args, "--gemma-root", str(self.gemma_root), required=True)
        if self.settings.ltx_distilled_lora_path is not None:
            self._append_arg(args, "--distilled-lora", str(self.settings.ltx_distilled_lora_path))
        self._append_arg(args, "--prompt", request.prompt, required=True)
        self._append_arg(args, "--output-path", str(output_path), required=True)
        self._append_arg(args, "--height", str(height))
        self._append_arg(args, "--width", str(width))
        self._append_arg(args, "--num-frames", str(frames))
        self._append_arg(args, "--frame-rate", str(frame_rate))
        self._append_arg(args, "--num-inference-steps", str(steps))
        self._append_flag(args, "--enhance-prompt", request.enhance_prompt)
        if self.settings.ltx_repo_id.endswith("-fp8"):
            self._append_arg(args, "--quantization", "fp8-cast")
        if request.seed is not None:
            self._append_arg(args, "--seed", str(request.seed))

        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.settings.request_timeout_seconds,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"LTX-2 pipeline failed: {stderr}")
        if not output_path.exists():
            raise RuntimeError("LTX-2 pipeline completed without creating an output video")
        return [
            MediaItem(
                path=output_path,
                media_type="video/mp4",
                revised_prompt=request.prompt,
                extension={"video_url": str(output_path)},
            )
        ]

    def _append_arg(
        self, args: list[str], flag: str, value: str, required: bool = False
    ) -> None:
        if required or flag in self.help_text:
            args.extend([flag, value])

    def _append_flag(self, args: list[str], flag: str, enabled: bool) -> None:
        if enabled and flag in self.help_text:
            args.append(flag)

    def _checkpoint_flag(self) -> str:
        if self.settings.ltx_pipeline_module.endswith(".distilled"):
            return "--distilled-checkpoint-path"
        return "--checkpoint-path"

    def _validate_ltx_cli_config(self) -> None:
        if (
            self.settings.ltx_pipeline_module.endswith(".distilled")
            and self.spatial_upsampler_path is None
        ):
            raise ModelLoadError(
                "OPENAI_MULTI_BACKEND_LTX_SPATIAL_UPSAMPLER_PATH or "
                "OPENAI_MULTI_BACKEND_LTX_SPATIAL_UPSAMPLER_FILENAME is required "
                "for ltx_pipelines.distilled"
            )

    def _supported_flags(self) -> str:
        flags = {
            "--checkpoint-path",
            "--distilled-checkpoint-path",
            "--spatial-upsampler-path",
            "--gemma-root",
            "--distilled-lora",
            "--lora",
            "--prompt",
            "--output-path",
            "--height",
            "--width",
            "--num-frames",
            "--frame-rate",
            "--enhance-prompt",
            "--seed",
            "--quantization",
        }
        if not self.settings.ltx_pipeline_module.endswith(".distilled"):
            flags.add("--num-inference-steps")
        return "\n".join(sorted(flags))

    def _resolve_ltx_file(self, configured_path: Path | None, filename: str) -> Path:
        if configured_path is not None:
            path = configured_path.expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise ModelLoadError(f"Configured LTX file does not exist: {path}")
            return path
        huggingface_hub = import_optional("huggingface_hub")
        return Path(
            huggingface_hub.hf_hub_download(
                repo_id=self.settings.ltx_repo_id,
                filename=filename,
                cache_dir=str(self.settings.model_cache_dir),
                token=self.settings.hf_token,
            )
        )

    def _resolve_optional_ltx_file(
        self, configured_path: Path | None, filename: str | None
    ) -> Path | None:
        if configured_path is not None:
            return self._resolve_ltx_file(configured_path, filename or configured_path.name)
        if not filename:
            return None
        return self._resolve_ltx_file(None, filename)

    def _resolve_ltx_directory(
        self, configured_path: Path | None, repo_id: str, description: str
    ) -> Path:
        if configured_path is not None:
            path = configured_path.expanduser().resolve()
            if not path.exists() or not path.is_dir():
                raise ModelLoadError(
                    f"Configured LTX {description} directory does not exist: {path}"
                )
            return path
        huggingface_hub = import_optional("huggingface_hub")
        try:
            return Path(
                huggingface_hub.snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.settings.model_cache_dir),
                    token=self.settings.hf_token,
                )
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to download LTX {description} from '{repo_id}'. "
                "If this is a gated Hugging Face model, accept its license for the configured "
                "token or set OPENAI_MULTI_BACKEND_LTX_GEMMA_ROOT to a local snapshot directory."
            ) from exc


def resolve_video_frame_rate(request: ImageGenerationRequest, settings: Settings) -> float:
    return request.frame_rate or settings.video_default_frame_rate


def resolve_video_frame_count(request: ImageGenerationRequest, settings: Settings) -> int:
    if request.frames is not None:
        frames = request.frames
    elif request.duration is not None:
        frames = math.ceil(request.duration * resolve_video_frame_rate(request, settings))
    else:
        frames = settings.video_default_frames
    return max(1, min(frames, settings.video_max_frames))
