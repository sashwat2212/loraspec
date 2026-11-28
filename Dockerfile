# Multi-stage CPU-light loraspec Studio container (React 19 + FastAPI DEMO_REPLAY)
FROM node:22-slim AS ui-builder
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEMO_REPLAY=true \
    LORASPEC_UI_DIST=/app/ui/dist \
    LORASPEC_REPLAY_DIR=/app/assets/demo_replays

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir ".[studio,spectral]"

COPY assets/ ./assets/
COPY --from=ui-builder /app/ui/dist ./ui/dist/

EXPOSE 8000

HEALTHCHECK --interval=2s --timeout=3s --start-period=2s --retries=15 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "loraspec.serving.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
