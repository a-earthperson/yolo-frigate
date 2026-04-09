# syntax=docker/dockerfile:1.10.0
FROM ubuntu:noble AS openvino-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
ENV CLANG_VERSION="20"
ENV CMAKE_INSTALL_PREFIX="/opt/level-zero-runtime"
ENV LEVEL_ZERO_PACKAGES="libclang-$CLANG_VERSION-dev \
    clang-tools-$CLANG_VERSION \
    libomp-$CLANG_VERSION-dev \
    llvm-$CLANG_VERSION-dev \
    lld-$CLANG_VERSION"

WORKDIR /build

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    file \
    git \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    lsb-release \
    ocl-icd-libopencl1 \
    opencl-headers \
    python-is-python3 \
    python3 \
    software-properties-common \
    wget \
    automake \
    cmake \
    make && \
    update-ca-certificates

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh "$CLANG_VERSION" && \
    apt-get update && \
    apt-get install -y --no-install-recommends $LEVEL_ZERO_PACKAGES && \
    rm -rf /var/lib/apt/lists/*

FROM hobbsau/aria2 AS intel-lz-debs
WORKDIR /build/intel-lz
COPY <<EOF /build/intel-lz/intel-lz-debs.list
https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-core_1.0.17537.20_amd64.deb
https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17537.20/intel-igc-opencl_1.0.17537.20_amd64.deb
https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-level-zero-gpu-legacy1_1.3.30872.22_amd64.deb
https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-level-zero-gpu_1.3.30872.22_amd64.deb
https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-opencl-icd-legacy1_24.35.30872.22_amd64.deb
https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/intel-opencl-icd_24.35.30872.22_amd64.deb
https://github.com/intel/compute-runtime/releases/download/24.35.30872.22/libigdgmm12_22.5.0_amd64.deb
EOF
RUN aria2c -j32 -k 1M -i intel-lz-debs.list -d debs

FROM openvino-builder AS yolo-frigate-openvino-builder

WORKDIR /build/intel-lz

COPY --from=intel-lz-debs /build/intel-lz/debs /tmp/intel-lz-debs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG CMAKE_BUILD_TYPE="Release"
ARG LEVEL_ZERO_VERSION="v1.28.2"

ENV CMAKE_C_COMPILER="/usr/bin/clang-$CLANG_VERSION"
ENV CC="$CMAKE_C_COMPILER"
ENV CMAKE_CXX_COMPILER="/usr/bin/clang++-$CLANG_VERSION"
ENV CXX="$CMAKE_CXX_COMPILER"

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    dpkg -i /tmp/intel-lz-debs/*.deb && \
    git clone --branch "$LEVEL_ZERO_VERSION" --depth 1 https://github.com/oneapi-src/level-zero.git && \
    mkdir -p level-zero/build && \
    cd level-zero/build && \
    cmake \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_CXX_FLAGS="-Wno-error=extern-c-compat" \
    -DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
    .. && \
    cmake --build . -j --target package && \
    cmake --build . -j --target install

RUN mkdir -p /opt/intel-runtime-root && \
    cp -a --parents \
    /etc/OpenCL/vendors/intel.icd \
    /etc/OpenCL/vendors/intel_legacy1.icd \
    /usr/lib/x86_64-linux-gnu/intel-opencl \
    /usr/lib/x86_64-linux-gnu/libigdgmm.so.12 \
    /usr/lib/x86_64-linux-gnu/libigdgmm.so.12.5.0 \
    /usr/lib/x86_64-linux-gnu/libocloc.so \
    /usr/lib/x86_64-linux-gnu/libocloc_legacy1.so \
    /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 \
    /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.3.30872.22 \
    /usr/lib/x86_64-linux-gnu/libze_intel_gpu_legacy1.so.1 \
    /usr/lib/x86_64-linux-gnu/libze_intel_gpu_legacy1.so.1.3.30872.22 \
    /usr/local/lib/libiga64.so \
    /usr/local/lib/libiga64.so.1 \
    /usr/local/lib/libiga64.so.1.0.17537.20 \
    /usr/local/lib/libigc.so \
    /usr/local/lib/libigc.so.1 \
    /usr/local/lib/libigc.so.1.0.17537.20 \
    /usr/local/lib/libigdfcl.so \
    /usr/local/lib/libigdfcl.so.1 \
    /usr/local/lib/libigdfcl.so.1.0.17537.20 \
    /usr/local/lib/libopencl-clang.so \
    /usr/local/lib/libopencl-clang.so.14 \
    /opt/level-zero-runtime \
    /opt/intel-runtime-root

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-editable --extra openvino && \
    uv pip install \
    --python /app/.venv/bin/python \
    --reinstall-package torch \
    --reinstall-package torchvision \
    --torch-backend cpu \
    "torch==2.11.0" \
    "torchvision==0.26.0" && \
    uv pip uninstall \
    --python /app/.venv/bin/python \
    nvidia-cublas \
    nvidia-cuda-cupti \
    nvidia-cuda-nvrtc \
    nvidia-cuda-runtime \
    nvidia-cudnn-cu13 \
    nvidia-cufft \
    nvidia-cufile \
    nvidia-curand \
    nvidia-cusolver \
    nvidia-cusparse \
    nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvjitlink \
    nvidia-nvshmem-cu13 \
    nvidia-nvtx \
    triton

FROM python:3.12-slim-bookworm AS yolo-frigate-openvino

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_FRIGATE_RUNTIME=openvino
ENV YOLO_FRIGATE_MODEL_CACHE_DIR=/cache/yolo-frigate
ENV YOLO_CONFIG_DIR=/cache/Ultralytics
ENV PATH="/app/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/level-zero-runtime/lib:/opt/level-zero-runtime/lib64"

WORKDIR /app

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    ocl-icd-libopencl1 && \
    ln -sf /usr/local/bin/python3 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /cache/yolo-frigate /cache/Ultralytics /models

COPY --from=yolo-frigate-openvino-builder /opt/intel-runtime-root/ /
COPY --from=yolo-frigate-openvino-builder /app/.venv /app/.venv
COPY labelmap.txt /models/
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=30s --retries=5 CMD [ "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10)" ]

ENTRYPOINT ["yolo-frigate"]
