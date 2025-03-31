FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc && pip install poetry && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
WORKDIR /kg_library
COPY pyproject.toml poetry.lock ./
RUN python -m spacy download en_core_web_trf && python -m coreferee install en
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi --no-root && rm -rf ~/.cache/pip ~/.cache/poetry
COPY . .
CMD ["python", "kg_library/main.py"]