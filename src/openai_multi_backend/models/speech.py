from __future__ import annotations

import ipaddress
import socket
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai_multi_backend.config import Settings
from openai_multi_backend.models.base import (
    BaseModelAdapter,
    SpeechResult,
    filter_supported_kwargs,
    import_optional,
    media_type_for_format,
)

if TYPE_CHECKING:
    from openai_multi_backend.api.schemas import SpeechRequest


class QwenTTSAdapter(BaseModelAdapter):
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
            task="text-to-speech",
            model=self.model_id,
            torch_dtype=torch_dtype,
            device=device_arg,
            model_kwargs=hf_kwargs,
            trust_remote_code=trust_remote_code,
        )

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        reference = resolve_voice_reference(request, self.settings)
        raw_kwargs: dict[str, Any] = {
            "text": request.input,
            "voice": request.voice,
            "speaker": request.voice,
            "language": request.language,
            "speed": request.speed,
            "speaker_wav": reference,
            "voice_reference": reference,
            "reference_audio": reference,
        }
        kwargs = filter_supported_kwargs(self.pipeline.__call__, raw_kwargs)
        if "text" not in kwargs:
            call_kwargs = {key: value for key, value in kwargs.items() if key != "text"}
            output = self.pipeline(request.input, **call_kwargs)
        else:
            output = self.pipeline(**kwargs)
        temporary_wav = output_path(self.settings.output_dir, "wav")
        save_pipeline_audio(output, temporary_wav)
        if request.response_format == "wav":
            return SpeechResult(path=temporary_wav, media_type=media_type_for_format("wav"))
        final_path = output_path(self.settings.output_dir, request.response_format)
        convert_audio(temporary_wav, final_path, request.response_format)
        return SpeechResult(
            path=final_path, media_type=media_type_for_format(request.response_format)
        )


class CoquiXTTSAdapter(BaseModelAdapter):
    tts: Any

    def load(self) -> None:
        tts_api = import_optional("TTS.api", "TTS")
        self.device = self.resolve_device()
        self.dtype = "model-default"
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.tts = tts_api.TTS(model_name=model_name)
        if hasattr(self.tts, "to"):
            self.tts.to(self.device)

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        reference = resolve_voice_reference(request, self.settings)
        temporary_wav = output_path(self.settings.output_dir, "wav")
        raw_kwargs: dict[str, Any] = {
            "text": request.input,
            "file_path": str(temporary_wav),
            "speaker_wav": reference,
            "language": request.language or "en",
            "speed": request.speed,
        }
        kwargs = filter_supported_kwargs(self.tts.tts_to_file, raw_kwargs)
        self.tts.tts_to_file(**kwargs)
        if request.response_format == "wav":
            return SpeechResult(path=temporary_wav, media_type=media_type_for_format("wav"))
        final_path = output_path(self.settings.output_dir, request.response_format)
        convert_audio(temporary_wav, final_path, request.response_format)
        return SpeechResult(
            path=final_path, media_type=media_type_for_format(request.response_format)
        )


def resolve_voice_reference(request: SpeechRequest, settings: Settings) -> str | None:
    reference = request.speaker_wav or request.voice_reference
    if reference:
        return resolve_local_voice_reference(reference, settings.voice_reference_dir)
    if request.voice_reference_url:
        return download_reference_audio(request.voice_reference_url, settings)
    return None


def resolve_local_voice_reference(reference: str, safe_dir: Path | None) -> str:
    if safe_dir is None:
        raise ValueError(
            "Local voice references require OPENAI_MULTI_BACKEND_VOICE_REFERENCE_DIR"
        )
    safe_root = safe_dir.expanduser().resolve()
    path = Path(reference).expanduser().resolve()
    if path != safe_root and safe_root not in path.parents:
        raise ValueError("Voice reference file must be inside the configured safe directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Voice reference file does not exist: {path}")
    return str(path)


def download_reference_audio(url: str, settings: Settings) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("voice_reference_url must be an HTTPS URL")
    if parsed.hostname not in settings.voice_reference_url_allowed_hosts:
        raise ValueError("voice_reference_url host is not in the configured allowlist")
    validate_public_hostname(parsed.hostname, parsed.port or 443)
    request = urllib.request.Request(url, headers={"User-Agent": "openai-multi-backend/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > settings.max_upload_bytes:
            raise ValueError("voice_reference_url exceeds configured upload size limit")
        data = response.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ValueError("voice_reference_url exceeds configured upload size limit")
    suffix = Path(url.split("?", 1)[0]).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return handle.name


def validate_public_hostname(hostname: str, port: int) -> None:
    addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise ValueError("voice_reference_url host did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("voice_reference_url host resolves to a disallowed address")


def output_path(output_dir: Path, suffix: str) -> Path:
    path = (output_dir / f"{uuid.uuid4().hex}.{suffix}").resolve()
    output_root = output_dir.resolve()
    if output_root not in path.parents:
        raise RuntimeError("Generated output path escaped configured output directory")
    return path


def save_pipeline_audio(output: Any, path: Path) -> None:
    soundfile = import_optional("soundfile")
    numpy = import_optional("numpy")
    audio, sample_rate = extract_pipeline_audio(output)
    if audio is None:
        raise RuntimeError("Text-to-speech pipeline returned no audio samples")
    array = numpy.asarray(audio)
    if array.ndim == 2 and array.shape[0] < array.shape[1]:
        array = array.T
    soundfile.write(str(path), array, sample_rate)


def extract_pipeline_audio(output: Any) -> tuple[Any, int]:
    if not isinstance(output, dict):
        return output, 24000
    audio = None
    for key in ("audio", "waveform", "wav"):
        if key in output and output[key] is not None:
            audio = output[key]
            break
    sample_rate = int(output.get("sampling_rate") or output.get("sample_rate") or 24000)
    return audio, sample_rate


def convert_audio(source: Path, destination: Path, fmt: str) -> None:
    if fmt in {"wav", "flac"}:
        soundfile = import_optional("soundfile")
        audio, sample_rate = soundfile.read(str(source))
        soundfile.write(str(destination), audio, sample_rate, format=fmt.upper())
        return
    torchaudio = import_optional("torchaudio")
    waveform, sample_rate = torchaudio.load(str(source))
    torchaudio.save(str(destination), waveform, sample_rate, format=fmt)
