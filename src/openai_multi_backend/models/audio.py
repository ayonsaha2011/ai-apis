from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from openai_multi_backend.models.base import BaseModelAdapter, TranscriptionResult, import_optional


class WhisperASRAdapter(BaseModelAdapter):
    pipeline: Any

    def load(self) -> None:
        transformers = import_optional("transformers")
        self.device = self.resolve_device()
        torch_dtype = self.resolve_torch_dtype()
        self.dtype = str(torch_dtype).replace("torch.", "")
        device_arg = 0 if self.device == "cuda" else self.device
        hf_kwargs = self.common_hf_kwargs()
        trust_remote_code = hf_kwargs.pop("trust_remote_code", False)
        self.pipeline = transformers.pipeline(
            task="automatic-speech-recognition",
            model=self.model_id,
            torch_dtype=torch_dtype,
            device=device_arg,
            model_kwargs=hf_kwargs,
            trust_remote_code=trust_remote_code,
            return_timestamps=True,
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
        task: Literal["transcribe", "translate"],
    ) -> TranscriptionResult:
        generate_kwargs: dict[str, Any] = {"task": task}
        if language:
            generate_kwargs["language"] = language
        if prompt:
            generate_kwargs["prompt"] = prompt
        output = self.pipeline(str(audio_path), generate_kwargs=generate_kwargs)
        text = output["text"] if isinstance(output, dict) else str(output)
        segments = None
        if isinstance(output, dict) and isinstance(output.get("chunks"), list):
            segments = output["chunks"]
        return TranscriptionResult(
            text=text,
            language=language,
            duration=_audio_duration(audio_path),
            segments=segments,
        )


class ParakeetASRAdapter(BaseModelAdapter):
    model: Any

    def load(self) -> None:
        nemo_asr = import_optional("nemo.collections.asr.models", "nemo-toolkit[asr]")
        self.device = self.resolve_device()
        self.dtype = "model-default"
        self.model = nemo_asr.ASRModel.from_pretrained(model_name=self.model_id)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        output = self.model.transcribe([str(audio_path)])
        first = output[0] if isinstance(output, list) else output
        text = getattr(first, "text", None) or str(first)
        return TranscriptionResult(text=text, duration=_audio_duration(audio_path))


def _audio_duration(audio_path: Path) -> float | None:
    try:
        soundfile = import_optional("soundfile")
        info = soundfile.info(str(audio_path))
        return float(info.frames / info.samplerate)
    except Exception:
        return None
