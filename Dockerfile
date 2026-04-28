FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3-pip \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python3.12 -m pip install --no-cache-dir --upgrade pip \
    && python3.12 -m pip install --no-cache-dir '.[asr,ltx]'

EXPOSE 80

CMD ["uvicorn", "openai_multi_backend.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "80"]
