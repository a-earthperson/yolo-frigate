ARG PYTHON_IMAGE="python:3.12-slim-bookworm"

FROM ${PYTHON_IMAGE} AS yolo-frigate-onnx-gpu-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-editable --extra onnx-gpu

FROM ${PYTHON_IMAGE} AS yolo-frigate-onnx-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_FRIGATE_RUNTIME=onnx
ENV YOLO_FRIGATE_MODEL_CACHE_DIR=/cache/yolo-frigate
ENV YOLO_CONFIG_DIR=/cache/Ultralytics
ENV PATH="/app/.venv/bin:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /cache/yolo-frigate /cache/Ultralytics /models

COPY --from=yolo-frigate-onnx-gpu-builder /app/.venv /app/.venv
COPY labelmap.txt /models/
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=60s --retries=10 CMD [ "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10)" ]

ENTRYPOINT ["yolo-frigate"]
