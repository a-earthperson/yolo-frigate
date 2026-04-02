FROM hobbsau/aria2 AS model-downloader
WORKDIR /downloads
COPY <<EOF /downloads/models.list
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt 
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt 
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt 
EOF
RUN aria2c -j32 -k 1M -i models.list -d models

FROM python:3.11.9-slim-bookworm
ARG CPU_ARCHITECTURE="amd64"
ARG DEBIAN_VERSION="bookworm"
ARG LIBEDGETPU_VERSION1="16.0"
ARG LIBEDGETPU_VERSION2="2.17.1-1"
ARG LIBEDGETPU_RELEASE_URL="https://github.com/feranick/libedgetpu/releases/download/${LIBEDGETPU_VERSION1}TF${LIBEDGETPU_VERSION2}/libedgetpu1-max_${LIBEDGETPU_VERSION1}tf${LIBEDGETPU_VERSION2}.${DEBIAN_VERSION}_${CPU_ARCHITECTURE}.deb"

RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 curl gnupg libgl1 libglib2.0-0 \
    && curl ${LIBEDGETPU_RELEASE_URL} --output /tmp/libedgetpu.deb -L --fail \
    && echo "yes" | dpkg -i /tmp/libedgetpu.deb \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends edgetpu-compiler \
    && rm /tmp/libedgetpu.deb && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/root/.cache/uv
ENV YOLO_FRIGATE_RUNTIME=tflite
ENV YOLO_FRIGATE_MODEL_CACHE_DIR=/cache/yolo-frigate
ENV YOLO_CONFIG_DIR=/cache/Ultralytics

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=model-downloader /downloads/models /models

RUN mkdir -p /cache/yolo-frigate /cache/Ultralytics

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --system ".[tflite]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "curl", "--fail", "--silent", "http://localhost:8000/health" ]

ENTRYPOINT ["yolo-frigate"]
