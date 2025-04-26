FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc ffmpeg && pip install poetry && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /kg_library
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi --no-root && rm -rf ~/.cache/pip ~/.cache/poetry
RUN mkdir -p /kg_library/cache/datasets && \
    python -m spacy download en_core_web_lg && \
    python -m spacy download en_core_web_trf && \
    python -m coreferee install en && \
    python -c "from datasets import load_dataset; \
    load_dataset('kingkangkr/book_summary_dataset', cache_dir='/kg_library/cache/datasets')" && \
    python -c "import whisper; whisper.load_model('base',download_root='/kg_library/cache/whisper')"

COPY . .
CMD ["python", "-m", "kg_library.main"]