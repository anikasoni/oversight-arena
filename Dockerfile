# FROM python:3.11-slim

# WORKDIR /app/env

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git build-essential curl \
#     && rm -rf /var/lib/apt/lists/*

# COPY pyproject.toml ./
# COPY README.md ./
# COPY openenv.yaml ./
# COPY oversight_arena ./oversight_arena
# COPY docs ./docs

# RUN pip install --no-cache-dir -e .

# EXPOSE 7860

# ENV PYTHONPATH=/app/env
# ENV OVERSIGHT_DIFFICULTY=0.5
# ENV ENABLE_WEB_INTERFACE=true

# HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
#     CMD curl -f http://localhost:7860/health || exit 1

# CMD ["uvicorn", "oversight_arena.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
# ----------------------------
# Stage 1: Build Next.js UI
# ----------------------------
# ----------------------------
# Stage 1: Build Next.js UI
# ----------------------------
FROM node:20-bookworm-slim AS ui-builder

WORKDIR /app/ui

ENV NEXT_TELEMETRY_DISABLED=1

COPY oversight-arena-ui/package*.json ./
RUN npm ci

COPY oversight-arena-ui ./
RUN npm run build && test -d out && ls -la out


# ----------------------------
# Stage 2: Python OpenEnv backend + static UI
# ----------------------------
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
COPY scripts ./scripts
COPY results ./results

COPY --from=ui-builder /app/ui/out ./ui_out

RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONPATH=/app/env
ENV OVERSIGHT_DIFFICULTY=0.5
ENV ENABLE_WEB_INTERFACE=true
ENV FRONTEND_DIR=/app/env/ui_out

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "oversight_arena.server.app:app", "--host", "0.0.0.0", "--port", "7860"]