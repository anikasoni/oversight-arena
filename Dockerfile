FROM python:3.11-slim

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./
COPY openenv.yaml ./
COPY oversight_arena ./oversight_arena
COPY docs ./docs

RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONPATH=/app/env
ENV OVERSIGHT_DIFFICULTY=0.5
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "oversight_arena.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
