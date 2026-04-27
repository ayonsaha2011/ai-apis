# OpenAI Multi-Backend API

Python FastAPI service exposing OpenAI-compatible `/v1` endpoints backed by local Hugging Face/PyTorch model adapters.

## Models

- `Lightricks/LTX-2.3`: media generation through the official LTX-2 pipeline package.
- `dx8152/Flux2-Klein-9B-Consistency`: image generation through Diffusers.
- `llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic`: chat and text completions through Transformers causal LM.
- `nvidia/parakeet-tdt-0.6b-v3`: transcription through NVIDIA NeMo.
- `openai/whisper-large-v3-turbo`: transcription and translation through Transformers ASR pipeline.
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: speech synthesis through Transformers text-to-speech pipeline.
- `coqui/XTTS-v2`: speech synthesis and voice cloning through Coqui TTS.

The service does not return placeholder responses. If a dependency, license, model card implementation, token, or device is missing, the API returns an OpenAI-style error explaining the failure.

Coqui `XTTS-v2` uses the upstream `TTS` package, which currently publishes wheels only for Python versions below 3.12. On Python 3.12+, `pip install '.[tts]'` skips `TTS` instead of failing; use Python 3.11 when Coqui XTTS-v2 runtime support is required.

`Lightricks/LTX-2.3` is not currently published as a Diffusers `model_index.json` repository. The adapter loads the published `.safetensors` files and executes the official `ltx_pipelines` CLI (`OPENAI_MULTI_BACKEND_LTX_PIPELINE_MODULE`, default `ltx_pipelines.distilled`). Install the LTX-2 repository packages from `https://github.com/Lightricks/LTX-2`; the adapter downloads the configured checkpoint, spatial upsampler, and Gemma text encoder from Hugging Face when local paths are not supplied. The default Gemma repo is `google/gemma-3-12b-it-qat-q4_0-unquantized`, which is gated, so the configured Hugging Face token must have accepted access or `OPENAI_MULTI_BACKEND_LTX_GEMMA_ROOT` must point to a local snapshot directory.

To use the quantized fp8 LTX variant while keeping the OpenAI-facing model name as `Lightricks/LTX-2.3`, set `OPENAI_MULTI_BACKEND_LTX_REPO_ID=Lightricks/LTX-2.3-fp8`. If checkpoint filenames are left at their defaults, the service automatically switches to `ltx-2.3-22b-distilled-fp8.safetensors` and disables the standard spatial upsampler artifact.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[asr,tts,dev]'
cp .env.example .env
```

Install LTX-2 support in Python 3.12+ environments:

```bash
pip install -e '.[ltx]'
```

Set `OPENAI_MULTI_BACKEND_API_KEYS` in `.env`. Set `OPENAI_MULTI_BACKEND_HF_TOKEN` for gated Hugging Face models. Add any model requiring custom Hugging Face code to `OPENAI_MULTI_BACKEND_TRUST_REMOTE_CODE_MODELS` only after reviewing its repository code.

LTX-2 local pipeline setup:

```bash
pip install -e '.[ltx]'
```

Run this API from an environment where the configured `ltx_pipelines` module is installed. If it is missing, LTX requests return `missing_optional_dependency` with setup guidance instead of attempting a fake response. The LTX-2 package currently requires Python 3.12+.

Voice cloning reference safety:

- Set `OPENAI_MULTI_BACKEND_VOICE_REFERENCE_DIR` to allow `speaker_wav` or `voice_reference` local files. Paths outside that directory are rejected.
- Set `OPENAI_MULTI_BACKEND_VOICE_REFERENCE_URL_ALLOWED_HOSTS` to a comma-separated HTTPS host allowlist for `voice_reference_url`. Private, loopback, link-local, multicast, reserved, and unspecified resolved IPs are rejected.

## Run

```bash
uvicorn openai_multi_backend.main:create_app --factory --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/models/download`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/images/generations`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

## Examples

```bash
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Lightricks/LTX-2.3"}'
```

```bash
curl http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic"}'
```

```bash
curl http://localhost:8000/v1/models/download \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/whisper-large-v3-turbo"}'
```

`POST /v1/models/download` checks the local Hugging Face cache before downloading. If the required artifact already exists, the response marks it as `cached` and does not call Hugging Face download APIs. LTX and Flux default to known `.safetensors` artifacts; chat, Whisper, Parakeet, Qwen TTS, and XTTS default to repository snapshot downloads. Pass `files` to download explicit repository files, `force: true` to re-download, or `local_files_only: true` to fail instead of downloading missing files.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic","messages":[{"role":"user","content":"Write a short deployment checklist."}],"max_tokens":256}'
```

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"dx8152/Flux2-Klein-9B-Consistency","prompt":"A clean architecture diagram for a model serving API","size":"1024x1024","response_format":"url"}'
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F model=openai/whisper-large-v3-turbo \
  -F file=@speech.wav
```

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -o speech.wav \
  -d '{"model":"coqui/XTTS-v2","input":"Production deployment complete.","voice":"default","response_format":"wav","language":"en","speaker_wav":"/secure/reference.wav"}'
```

## Verification

```bash
python -m compileall src tests
ruff check .
mypy src
pytest
```

Heavy model execution requires accepted model licenses, enough disk, and suitable GPU/accelerator memory. Run integration tests only on prepared hardware with real model access.
