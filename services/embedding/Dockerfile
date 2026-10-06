FROM python:3.12-slim AS build
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
COPY services/embedding/pyproject.toml services/embedding/uv.lock ./
COPY services/embedding/src ./src
RUN pip install --no-cache-dir uv==0.9.26 \
    && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ETAG_TIMEOUT=10 \
    HF_HUB_DOWNLOAD_TIMEOUT=60
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/models && chown app:app /data/models
USER app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=6 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ready', timeout=3).close()"]
CMD ["uvicorn", "rag_embedding.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "5", "--no-access-log"]
