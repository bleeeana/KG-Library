FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    wget \
    git \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip

RUN pip install poetry

WORKDIR /kg_library
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi --no-root && rm -rf ~/.cache/pip ~/.cache/poetry

RUN python -m spacy download en_core_web_lg && \
    python -m coreferee install en
COPY . .

RUN mkdir -p /kg_library/cache/datasets /kg_library/cache/whisper \
    /data/input /data/output /data/models
VOLUME ["/data/input", "/data/output", "/data/models", "/kg_library/cache/datasets", "/kg_library/cache/whisper"]
RUN chmod +x /kg_library/entrypoint.sh

ENTRYPOINT ["/kg_library/entrypoint.sh"]