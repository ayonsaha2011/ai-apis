FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/.venv

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir '.[ltx]'

RUN pip install --no-cache-dir '.[asr]'

EXPOSE 80

CMD ["uvicorn", "openai_multi_backend.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "80"]
