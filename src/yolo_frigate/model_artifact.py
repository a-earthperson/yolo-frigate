from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yolo_frigate.calibration_dataset import ensure_open_images_v7_validation_dataset
from yolo_frigate.config import AppConfig
from yolo_frigate.runtime_profile import (
    ModelSource,
    RuntimeProfile,
    describe_model_source,
)
from yolo_frigate.ultralytics_support import (
    ensure_tensorrt_namespace,
    get_ultralytics_version,
    import_ultralytics_yoloe,
    resolve_ultralytics_checkpoint,
)


@dataclass(frozen=True)
class ResolvedModelArtifact:
    path: str
    cached: bool


@dataclass(frozen=True)
class ExportRequest:
    cache_key: str
    cache_root: Path
    work_dir: Path
    manifest_path: Path
    source: ModelSource
    runtime_profile: RuntimeProfile
    class_names: tuple[str, ...]
    hardware: dict[str, str | None]
    export_args: dict[str, Any]
    source_sha256: str


class ModelArtifactManager:
    def resolve(
        self,
        config: AppConfig,
        runtime_profile: RuntimeProfile,
        class_names: list[str] | None = None,
    ) -> ResolvedModelArtifact:
        source = describe_model_source(config.model_file)
        if source.kind != "checkpoint":
            return ResolvedModelArtifact(
                path=str(source.path),
                cached=False,
            )

        request = self._build_export_request(
            config, runtime_profile, source, class_names
        )
        artifact_path = self._ensure_exported(request)
        return ResolvedModelArtifact(
            path=str(artifact_path),
            cached=True,
        )

    def _build_export_request(
        self,
        config: AppConfig,
        runtime_profile: RuntimeProfile,
        source: ModelSource,
        class_names: list[str] | None,
    ) -> ExportRequest:
        self._validate_export_config(config, runtime_profile)

        source_path = self._resolve_checkpoint_source(source, class_names)
        export_data = self._resolve_export_data(config, runtime_profile, class_names)

        resolved_source = ModelSource(
            path=source_path,
            kind=source.kind,
        )
        resolved_class_names = tuple(class_names or ())
        source_sha256 = _sha256_file(source_path)
        export_args = self._build_export_args(config, runtime_profile, export_data)
        hardware = self._hardware_fingerprint(runtime_profile.name, config.device)
        payload = {
            "source_sha256": source_sha256,
            "runtime": runtime_profile.name,
            "format": runtime_profile.export_format,
            "class_names": resolved_class_names,
            "export_args": export_args,
            "hardware": hardware,
            "ultralytics_version": get_ultralytics_version(),
        }
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_root = Path(config.model_cache_dir).expanduser().resolve() / cache_key
        return ExportRequest(
            cache_key=cache_key,
            cache_root=cache_root,
            work_dir=cache_root / "work",
            manifest_path=cache_root / "manifest.json",
            source=resolved_source,
            runtime_profile=runtime_profile,
            class_names=resolved_class_names,
            hardware=hardware,
            export_args=export_args,
            source_sha256=source_sha256,
        )

    def _resolve_checkpoint_source(
        self,
        source: ModelSource,
        class_names: list[str] | None,
    ) -> Path:
        if source.kind != "checkpoint":
            return source.path.expanduser().resolve()

        source_path = source.path.expanduser()
        if not class_names:
            prompt_free_candidate = _resolve_prompt_free_checkpoint_candidate(
                source_path
            )
            if prompt_free_candidate is not None:
                source_path = prompt_free_candidate
        if source_path.is_file():
            return source_path.resolve()

        return resolve_ultralytics_checkpoint(str(source_path))

    def _ensure_exported(self, request: ExportRequest) -> Path:
        request.cache_root.mkdir(parents=True, exist_ok=True)
        with _locked_file(request.cache_root / ".lock"):
            manifest = self._read_manifest(request.manifest_path)
            if manifest is not None:
                artifact_path = request.cache_root / manifest["artifact_path"]
                if artifact_path.exists():
                    return artifact_path

            if request.work_dir.exists():
                shutil.rmtree(request.work_dir)
            request.work_dir.mkdir(parents=True, exist_ok=True)

            staged_source = request.work_dir / request.source.path.name
            shutil.copy2(request.source.path, staged_source)

            artifact_path = self._export_artifact(request, staged_source)
            relative_artifact_path = artifact_path.relative_to(request.cache_root)
            request.manifest_path.write_text(
                json.dumps(
                    {
                        "cache_key": request.cache_key,
                        "runtime": request.runtime_profile.name,
                        "format": request.runtime_profile.export_format,
                        "artifact_path": relative_artifact_path.as_posix(),
                        "source_model": str(request.source.path),
                        "source_sha256": request.source_sha256,
                        "class_names": list(request.class_names),
                        "hardware": request.hardware,
                        "export_args": request.export_args,
                        "ultralytics_version": get_ultralytics_version(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return artifact_path

    def _read_manifest(self, manifest_path: Path) -> dict[str, Any] | None:
        if not manifest_path.is_file():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _export_artifact(self, request: ExportRequest, staged_source: Path) -> Path:
        if request.runtime_profile.name == "tensorrt":
            ensure_tensorrt_namespace()
        yolo_cls = import_ultralytics_yoloe()
        model = yolo_cls(str(staged_source))
        if request.class_names:
            model.set_classes(list(request.class_names))
        elif _uses_prompt_free_head(model):
            _strip_prompt_embeddings(model)
        with _patch_yoloe_end2end_export_fuse(model):
            model.export(**request.export_args)

        artifact_path = self._find_export_artifact(request)
        if artifact_path is None:
            raise RuntimeError(
                f"Failed to locate exported artifact for runtime '{request.runtime_profile.name}'."
            )
        return artifact_path

    def _find_export_artifact(self, request: ExportRequest) -> Path | None:
        if request.runtime_profile.name == "tensorrt":
            return _single_match(request.work_dir.glob("*.engine"))
        if request.runtime_profile.name == "openvino":
            return _single_match(request.work_dir.glob("*_openvino_model"))
        if request.runtime_profile.name == "edgetpu":
            return _single_match(request.work_dir.rglob("*edgetpu*.tflite"))
        if request.runtime_profile.name == "tflite":
            return _single_match(
                path
                for path in request.work_dir.rglob("*.tflite")
                if "edgetpu" not in path.name.lower()
            )
        if request.runtime_profile.name == "onnx":
            return _single_match(request.work_dir.glob("*.onnx"))
        return None

    def _validate_export_config(
        self, config: AppConfig, runtime_profile: RuntimeProfile
    ) -> None:
        if runtime_profile.name == "tensorrt" and not config.device.startswith("gpu"):
            raise ValueError(
                "TensorRT export requires a GPU device such as gpu or gpu:<index>."
            )
        if runtime_profile.name == "edgetpu" and platform.machine().lower() not in {
            "x86_64",
            "amd64",
        }:
            raise ValueError(
                "EdgeTPU export requires an x86 Linux-compatible environment."
            )

    def _build_export_args(
        self,
        config: AppConfig,
        runtime_profile: RuntimeProfile,
        export_data: str | None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "format": runtime_profile.export_format,
            "imgsz": config.export_imgsz,
            "batch": config.export_batch,
            "nms": config.export_nms,
        }
        if runtime_profile.name != "edgetpu":
            args["half"] = config.export_half
        if runtime_profile.name in {"tensorrt", "openvino", "tflite", "onnx"}:
            args["int8"] = config.export_int8
        if runtime_profile.name in {"tensorrt", "openvino", "onnx"}:
            args["dynamic"] = config.export_dynamic
        if runtime_profile.name == "tensorrt":
            args["simplify"] = True
            args["device"] = _normalize_tensorrt_export_device(config.device)
            if config.export_workspace is not None:
                args["workspace"] = config.export_workspace
        if runtime_profile.name == "onnx":
            args["simplify"] = True
            if config.device == "cpu":
                args["device"] = "cpu"
            else:
                args["device"] = _normalize_tensorrt_export_device(config.device)
        if runtime_profile.name in {"openvino", "tflite", "edgetpu"}:
            args["device"] = "cpu"
        if export_data is not None:
            args["data"] = export_data
        if config.export_int8:
            args["fraction"] = config.export_fraction
        return args

    def _resolve_export_data(
        self,
        config: AppConfig,
        runtime_profile: RuntimeProfile,
        class_names: list[str] | None,
    ) -> str | None:
        if config.export_data is not None:
            return config.export_data
        if not config.export_int8:
            return None
        if runtime_profile.name not in {"tensorrt", "openvino", "tflite"}:
            raise ValueError(
                "--export_data is required when --export_int8 is enabled for "
                f"runtime '{runtime_profile.name}'. Automatic Open Images V7 "
                "calibration bootstrap is only available for TensorRT, OpenVINO, and TFLite."
            )
        return str(
            ensure_open_images_v7_validation_dataset(
                Path(config.model_cache_dir),
                class_names,
                config.export_calibration_max_samples,
            )
        )

    def _hardware_fingerprint(self, runtime: str, device: str) -> dict[str, str | None]:
        gpu_identity = _resolve_gpu_identity(device)
        return {
            "runtime": runtime,
            "requested_device": device,
            "machine": platform.machine(),
            "system": platform.system(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES"),
            "gpu_name": gpu_identity["name"],
            "gpu_compute_capability": gpu_identity["compute_capability"],
        }


@contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _single_match(paths) -> Path | None:
    matches = sorted(Path(path) for path in paths)
    if not matches:
        return None
    return matches[0]


def _resolve_prompt_free_checkpoint_candidate(source_path: Path) -> Path | None:
    if source_path.suffix.lower() != ".pt":
        return None

    stem = source_path.stem
    lower_stem = stem.lower()
    if "yoloe" not in lower_stem or lower_stem.endswith("-pf"):
        return None

    candidate = source_path.with_name(f"{stem}-pf{source_path.suffix}")
    if source_path.is_file() and not candidate.is_file():
        return None
    return candidate


def _uses_prompt_free_head(model: Any) -> bool:
    head = _resolve_yoloe_head(model)
    return head is not None and hasattr(head, "lrpc")


def _strip_prompt_embeddings(model: Any) -> None:
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "pe"):
        delattr(inner_model, "pe")


def _resolve_yoloe_head(model: Any) -> Any | None:
    inner_model = getattr(model, "model", None)
    layers = getattr(inner_model, "model", None)
    if layers is None:
        return None
    try:
        return layers[-1]
    except (IndexError, KeyError, TypeError):
        return None


@contextmanager
def _patch_yoloe_end2end_export_fuse(model: Any):
    head = _resolve_yoloe_head(model)
    inner_model = getattr(model, "model", None)
    if (
        head is None
        or not getattr(head, "end2end", False)
        or not hasattr(inner_model, "pe")
        or not hasattr(head, "fuse")
    ):
        yield
        return

    head_cls = type(head)
    original_fuse = head_cls.fuse

    def patched_fuse(self, txt_feats=None):
        if txt_feats is None:
            return original_fuse(self, txt_feats)

        if getattr(self, "is_fused", False):
            return

        if (
            getattr(self, "cv3", None) is not None
            and getattr(self, "cv4", None) is not None
        ):
            return original_fuse(self, txt_feats)

        one2one_cv3 = getattr(self, "one2one_cv3", None)
        one2one_cv4 = getattr(self, "one2one_cv4", None)
        if one2one_cv3 is None or one2one_cv4 is None:
            return original_fuse(self, txt_feats)

        assert not self.training
        txt_feats = txt_feats.float().squeeze(0)
        self._fuse_tp(txt_feats, one2one_cv3, one2one_cv4)
        if hasattr(self, "reprta"):
            del self.reprta
            try:
                from torch import nn

                self.reprta = nn.Identity()
            except ImportError:
                self.reprta = object()
        self.is_fused = True

    head_cls.fuse = patched_fuse
    try:
        yield
    finally:
        head_cls.fuse = original_fuse


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_tensorrt_export_device(device: str) -> str:
    if device == "gpu":
        return "0"
    if device.startswith("gpu:"):
        index = device.split(":", maxsplit=1)[1]
        if not index.isdigit():
            raise ValueError(
                f"Invalid GPU device '{device}'. Expected gpu or gpu:<index>."
            )
        return index
    raise ValueError(
        f"TensorRT export requires gpu or gpu:<index>, received '{device}'."
    )


def _resolve_gpu_identity(device: str) -> dict[str, str | None]:
    gpu_index = _normalize_gpu_lookup_index(device)
    if gpu_index is None:
        return {"name": None, "compute_capability": None}

    try:
        import torch
    except ImportError:
        return {"name": None, "compute_capability": None}

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return {"name": None, "compute_capability": None}

    try:
        if not cuda.is_available():
            return {"name": None, "compute_capability": None}
    except Exception:
        return {"name": None, "compute_capability": None}

    try:
        if gpu_index >= cuda.device_count():
            return {"name": None, "compute_capability": None}
    except Exception:
        return {"name": None, "compute_capability": None}

    name: str | None = None
    compute_capability: str | None = None
    try:
        name = str(cuda.get_device_name(gpu_index))
    except Exception:
        pass
    try:
        capability = cuda.get_device_capability(gpu_index)
        if len(capability) == 2:
            compute_capability = f"{capability[0]}.{capability[1]}"
    except Exception:
        pass
    return {
        "name": name,
        "compute_capability": compute_capability,
    }


def _normalize_gpu_lookup_index(device: str) -> int | None:
    if device == "gpu":
        return 0
    if not device.startswith("gpu:"):
        return None
    index = device.split(":", maxsplit=1)[1]
    if not index.isdigit():
        return None
    return int(index)
