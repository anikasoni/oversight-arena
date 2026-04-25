FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY oversight_arena /app/oversight_arena
COPY openenv.yaml /app/

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
        fastapi==0.115.* uvicorn==0.30.* pydantic==2.* \
        numpy pyyaml requests

# HF Spaces requires port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "oversight_arena.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]