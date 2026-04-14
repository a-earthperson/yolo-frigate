# yolo-frigate

Source: **[github.com/a-earthperson/yolo-frigate](https://github.com/a-earthperson/yolo-frigate)**.

HTTP object detection for **NVR-style stacks**—most often [Frigate](https://github.com/blakeblackshear/frigate) or anything else that speaks a **DeepStack-compatible** multipart `POST /detect` API. The service wraps Ultralytics YOLO models, optionally **lazy-exporting** Ultralytics `.pt` checkpoints into each image’s native runtime (TensorRT, OpenVINO, TFLite, EdgeTPU, ONNX) on first use.

Typical deployments:

- One **GPU-class** container per inference node (NVIDIA TensorRT or ONNX Runtime, Intel iGPU via OpenVINO, Coral EdgeTPU via TFLite, and so on).
- Shared **read-only model trees** and **writable export caches** (bind mounts, named volumes, or NFS when workers are spread across machines).
- **Orchestrator placement** so NVIDIA-class images (TensorRT, ONNX GPU) land on GPU hosts and OpenVINO on Intel-GPU hosts—Compose on a single box, Swarm/Kubernetes when models and cache live on shared storage.

## Runtime images

Each Docker image ships with a fixed `YOLO_FRIGATE_RUNTIME` and the dependencies for that stack only:

| Variant | Dockerfile | `YOLO_FRIGATE_RUNTIME` | Native artifacts | Typical devices |
|---------|------------|-------------------|------------------|-----------------|
| **NVIDIA TensorRT** | [`nvidia-trt.Dockerfile`](nvidia-trt.Dockerfile) | `tensorrt` | `.engine` | NVIDIA: `gpu`, `gpu:0`, … |
| **NVIDIA ONNX Runtime (GPU)** | [`onnx-gpu.Dockerfile`](onnx-gpu.Dockerfile) | `onnx` | `.onnx` | `cpu` (CPU EP); NVIDIA: `gpu`, `gpu:0`, … |
| **Intel GPU (OpenVINO)** | [`intel-gpu.Dockerfile`](intel-gpu.Dockerfile) | `openvino` | `*_openvino_model/` | CPU; Intel GPU: `gpu`, `gpu:0`, …; NPU where supported |
| **Coral EdgeTPU (TFLite)** | [`coral-tpu.Dockerfile`](coral-tpu.Dockerfile) | `tflite` | `.tflite` | `cpu`; Coral: `usb`, `pci`, … |

Release builds publish four images from [`.github/workflows/publish.yml`](.github/workflows/publish.yml). The **image name** is the GitHub `owner/repository` name plus a **variant suffix** that matches the Dockerfile stem (GHCR normalizes names to lowercase):

- `ghcr.io/a-earthperson/yolo-frigate-nvidia-trt` — from `nvidia-trt.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-onnx-gpu` — from `onnx-gpu.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-intel-gpu` — from `intel-gpu.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-coral-tpu` — from `coral-tpu.Dockerfile`

Forks and private mirrors use their own `owner/repo` in place of `a-earthperson/yolo-frigate`. Version **tags** (for example `v0.1.8`, `latest`) come from the GitHub release via `docker/metadata-action`. You may **retag or mirror** under other names if your registry layout requires it.

## API contract

Shared across all runtime profiles:

- `POST /detect` — multipart form-data with an `image` file field.
- Success responses follow the `Predictions` schema in `yolo_frigate.prediction`.
- `GET /health` — liveness (empty body).
- `POST /force_save/{state}` — toggles forced save behavior for debugging.

## Runtime selection and devices

Inside an image, runtime is determined by `YOLO_FRIGATE_RUNTIME` (set in the variant Dockerfile). Legacy images may still use `YOLOREST_RUNTIME`; the application reads both. You normally pass a `.pt` checkpoint to `--model_file` and let that image export lazily.

Overrides when running outside Docker:

```bash
uv run yolo-frigate --runtime=tensorrt --device=gpu:0 --model_file=/models/model.pt
uv run yolo-frigate --runtime=onnx --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=onnx --device=gpu:0 --model_file=/models/model.pt
uv run yolo-frigate --runtime=openvino --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=tflite --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=edgetpu --device=pci --model_file=/models/model.pt
```

The installable project is **`yolo-frigate`**; the Python import package is **`yolo_frigate`** (for example `python -m yolo_frigate`).

Pre-exported artifacts can be passed directly to `--model_file`: TensorRT `.engine`, ONNX `.onnx`, OpenVINO `*_openvino_model/`, TFLite / EdgeTPU `.tflite`.

Device strings are runtime-specific (TensorRT: `gpu`, `gpu:0`, …; ONNX: `cpu`, `gpu`, `gpu:0`, …; OpenVINO: `cpu`, `gpu`, `npu`, …; TFLite: `cpu`; EdgeTPU: `usb`, `pci`, …). `--runtime` is the only runtime selector; `--label_file` overrides embedded class names when needed.

## Lazy export and cache

When `--model_file` is a `.pt` checkpoint, export happens on first inference. Cached outputs live under `YOLO_FRIGATE_MODEL_CACHE_DIR` (default `/cache/yolo-frigate` in images; legacy `YOLOREST_MODEL_CACHE_DIR` and `/cache/yolorest` are still honored). The cache key includes profile, `--export_imgsz`, `--export_half`, `--export_int8`, `--export_dynamic`, `--export_nms`, `--export_batch`, `--export_workspace`, and calibration inputs.

Operational notes:

- Mount a **writable** `/cache` (or custom `YOLO_FRIGATE_MODEL_CACHE_DIR`) whenever the container filesystem is read-only.
- `--export_int8` uses `--export_data` when provided; otherwise TensorRT, OpenVINO, and TFLite bootstrap a cached deterministic Open Images V7 validation subset derived from `--label_file` (up to 512 images). Labels not present in Open Images are ignored with a warning.
- TensorRT INT8 caches are **GPU-generation sensitive**; export on hardware representative of production.
- EdgeTPU export expects an x86 Linux exporter environment.

Common export flags:

```bash
--export_imgsz=640
--export_half
--export_dynamic
--export_batch=1
--export_workspace=4
--export_int8 --export_data=/models/data.yaml
```

## Deployment patterns

### What every container needs

- **`/models`** — readable tree with checkpoints or pre-exported artifacts and optional `labelmap.txt` (or pass `--label_file`).
- **`/cache`** — writable export cache (omit only if you never lazy-export or you redirect `YOLO_FRIGATE_MODEL_CACHE_DIR` elsewhere).
- **Devices** — NVIDIA Container Toolkit + GPU reservation for TensorRT and ONNX GPU; `/dev/dri` (and often `video`/`render` groups) for Intel GPU OpenVINO; Coral device nodes for EdgeTPU.
- **Shared memory** — large `tmpfs` on `/dev/shm` can help some workloads when memory pressure appears during export or batching.

### Docker Compose (single host)

Minimal TensorRT service (image name ends with `-nvidia-trt` for releases from this repo):

```yaml
services:
  yolo-frigate-tensorrt:
    image: ghcr.io/a-earthperson/yolo-frigate-nvidia-trt:v1.2.3
    restart: on-failure
    read_only: true
    security_opt:
      - "no-new-privileges=true"
    volumes:
      - ./models:/models:ro
      - ./cache:/cache
    command:
      - "--device=gpu:0"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=640"
      - "--export_half"
      - "--export_dynamic"
    gpus: all
```

The ONNX GPU image (`yolo-frigate-onnx-gpu`) uses the same NVIDIA + `gpus` pattern; runtime is fixed to `onnx` in the Dockerfile.

For **Swarm**, use GPU reservations on `deploy.resources` instead of `gpus: all` (Compose ignores `deploy` on non-Swarm setups).

Intel GPU (OpenVINO image; image name ends with `-intel-gpu`):

```yaml
services:
  yolo-frigate-intel-gpu:
    image: ghcr.io/a-earthperson/yolo-frigate-intel-gpu:v1.2.3
    restart: on-failure
    read_only: true
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - "${GROUP_RENDER}"
    volumes:
      - ./models:/models:ro
      - ./cache:/cache
    command:
      - "--device=gpu"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=320"
      - "--export_half"
      - "--export_dynamic"
```

Coral EdgeTPU (image name ends with `-coral-tpu`):

```yaml
networks:
  yolo-frigate:
    driver: bridge
    internal: true

services:
  yolo-frigate:
    image: ghcr.io/a-earthperson/yolo-frigate-coral-tpu:v1.2.3
    restart: on-failure
    read_only: true
    group_add:
      - "${GROUP_CORAL}"
    security_opt:
      - "no-new-privileges=true"
    networks:
      - yolo-frigate
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./models:/models:ro
      - ./cache:/cache
    devices:
      - /dev/apex_0:/dev/apex_0
    command:
      - "--device=pci"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=320"
```

### Docker Swarm (multi-node, shared storage)

Swarm is useful when **Frigate and workers sit on different nodes** or when **models and export cache** should live on NFS (or another shared filesystem) so any worker can serve the same tree.

Pattern:

- **Overlay network** attached to Frigate and detector services so `http://<service>:8000/detect` resolves cluster-wide.
- **Named volumes** backed by NFS for `/models` and `/cache` so exports done on one node are visible to others (same cache key → same artifact).
- **Placement constraints** so NVIDIA GPU images (TensorRT, ONNX GPU) run only on GPU-labeled nodes and OpenVINO on Intel-GPU-labeled nodes.
- **Optional** `cap_add: [CAP_PERFMON]` when you want perf counters on Intel stacks; **NVIDIA** stacks often set `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` for full GPU feature exposure inside the container.

Illustrative `stack.yml` (replace NFS address, paths, images, and labels with yours):

```yaml
networks:
  cams:
    external: true

volumes:
  models:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.1,rw,nfsvers=4,async
      device: ":/path/on/nfs/to/yolo/models"
  cache:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.1,rw,nfsvers=4,async
      device: ":/path/on/nfs/to/yolo/cache"

x-base: &x-base
  cap_add:
    - CAP_PERFMON
  networks:
    - cams
  volumes:
    - /dev/dri:/dev/dri
    - /etc/localtime:/etc/localtime:ro
    - /etc/timezone:/etc/timezone:ro
    - models:/models
    - cache:/cache
    - type: tmpfs
      target: /dev/shm
      tmpfs:
        size: 1073741824

services:
  objectdetector:
    <<: *x-base
    image: ghcr.io/a-earthperson/yolo-frigate-nvidia-trt:0.1.8
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
    command:
      - "--device=gpu"
      - "--export_imgsz=640"
      - "--export_dynamic"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo26l.pt"
      - "--export_half"
    deploy:
      placement:
        constraints:
          - node.labels.nvidia_gpu_compute == true
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1200M

  yolov9-intel:
    <<: *x-base
    image: ghcr.io/a-earthperson/yolo-frigate-intel-gpu:0.1.7
    command:
      - "--device=gpu"
      - "--export_imgsz=320"
      - "--export_dynamic"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo26s.pt"
      - "--export_half"
    deploy:
      mode: replicated
      replicas: 0
      placement:
        constraints:
          - node.labels.intel_gpu_compute == true
      resources:
        limits:
          memory: 4096M
        reservations:
          memory: 1280M
```

Keep **one writable cache** per logical deployment; concurrent writers on the same NFS path can corrupt cache entries—usually you run a **single replica** per cache volume or partition cache per replica.

### Kubernetes and other orchestrators

The same container arguments and volume mounts apply: mount models and cache, pass GPU device plugins or `/dev/dri` as your platform requires, and expose port `8000` to clients that call `/detect`.

## Frigate

Frigate’s HTTP detector expects DeepStack-style JSON; point it at this service’s **`/detect`** endpoint on the **Docker network** Frigate shares with the detector (service name resolves under Compose; under Swarm use the stack service name on the overlay).

```yaml
detectors:
  http_detector:
    type: deepstack
    api_url: http://objectdetector:8000/detect
    api_timeout: 1.0
```

Tune `api_timeout` to your SLA: sub-second values work when the model is warm; allow more time for **cold lazy export** or slow hosts. Align Frigate’s `model.labelmap_path` (or Frigate’s label file) with the classes you serve—often the same `labelmap.txt` you pass to `--label_file`.

## Runtime-native scope

The service intentionally does one thing: resolve a runtime profile, lazily export `.pt` sources when needed, load with Ultralytics, and adapt outputs to the existing `Predictions` REST contract. It is **not** a generic server for arbitrary ONNX graphs or custom postprocessing.

## Manual export (optional)

Ultralytics model sources: e.g. [ultralytics/assets](https://github.com/ultralytics/assets).

Lazy export is the default. Prebuilding is useful for air-gapped or reproducible rollouts.

### TensorRT

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=engine half=True dynamic=True
```

### OpenVINO

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=openvino
```

### ONNX

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=onnx
```

### TFLite

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=tflite
```

### EdgeTPU

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=edgetpu
```

You can pass resulting `.engine`, `.onnx`, `*_openvino_model/`, or `.tflite` paths directly to `--model_file`.

## Development

Install [uv](https://docs.astral.sh/uv/). Sync the project (core dependencies plus the `dev` group for tests and formatters), then run the test suite:

```bash
uv sync --group dev
uv run python -m unittest discover -s tests -v
```

Optional inference extras (install only what you need locally):

```bash
uv sync --group dev --extra tflite
uv sync --group dev --extra tensorrt
uv sync --group dev --extra openvino
uv sync --group dev --extra onnx-gpu
```

Notes:

- Runtime extras are primarily intended for Linux environments that mirror the Docker images.
- Lazy export depends on `ultralytics`; TFLite export additionally needs TensorFlow, native OpenVINO export needs `openvino`, and the ONNX GPU image uses the `onnx-gpu` extra (`onnxruntime-gpu`).

Format and lint:

```bash
uv run ruff check src tests
uv run black src tests
```

The repo pins the default interpreter with [`.python-version`](.python-version); `uv` respects that for `uv sync` and `uv run`.
