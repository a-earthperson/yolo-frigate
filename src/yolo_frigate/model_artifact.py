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

from yolo_frigate.config import AppConfig
from yolo_frigate.runtime_profile import (
    ModelSource,
    RuntimeProfile,
    describe_model_source,
)
from yolo_frigate.ultralytics_support import (
    get_ultralytics_version,
    import_ultralytics_yolo,
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
    export_args: dict[str, Any]
    source_sha256: str


class ModelArtifactManager:
    def resolve(
        self, config: AppConfig, runtime_profile: RuntimeProfile
    ) -> ResolvedModelArtifact:
        source = describe_model_source(config.model_file)
        if source.kind != "checkpoint":
            return ResolvedModelArtifact(
                path=str(source.path),
                cached=False,
            )

        request = self._build_export_request(config, runtime_profile, source)
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
    ) -> ExportRequest:
        self._validate_export_config(config, runtime_profile)

        source_path = source.path.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Model source '{source_path}' does not exist.")

        resolved_source = ModelSource(
            path=source_path,
            kind=source.kind,
        )
        source_sha256 = _sha256_file(source_path)
        export_args = self._build_export_args(config, runtime_profile)
        payload = {
            "source_sha256": source_sha256,
            "runtime": runtime_profile.name,
            "format": runtime_profile.export_format,
            "export_args": export_args,
            "hardware": self._hardware_fingerprint(runtime_profile.name, config.device),
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
            export_args=export_args,
            source_sha256=source_sha256,
        )

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
        yolo_cls = import_ultralytics_yolo()
        model = yolo_cls(str(staged_source))
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
        if config.export_int8 and config.export_data is None:
            raise ValueError(
                "--export_data is required when --export_int8 is enabled so calibration is deterministic."
            )

    def _build_export_args(
        self, config: AppConfig, runtime_profile: RuntimeProfile
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "format": runtime_profile.export_format,
            "imgsz": config.export_imgsz,
            "batch": config.export_batch,
            "nms": config.export_nms,
        }
        if runtime_profile.name != "edgetpu":
            args["half"] = config.export_half
        if runtime_profile.name in {"tensorrt", "openvino", "tflite"}:
            args["int8"] = config.export_int8
        if runtime_profile.name in {"tensorrt", "openvino"}:
            args["dynamic"] = config.export_dynamic
        if runtime_profile.name == "tensorrt":
            args["simplify"] = True
            args["device"] = _normalize_tensorrt_export_device(config.device)
            if config.export_workspace is not None:
                args["workspace"] = config.export_workspace
        if runtime_profile.name in {"openvino", "tflite", "edgetpu"}:
            args["device"] = "cpu"
        if config.export_data is not None:
            args["data"] = config.export_data
        if config.export_int8:
            args["fraction"] = config.export_fraction
        return args

    def _hardware_fingerprint(self, runtime: str, device: str) -> dict[str, str | None]:
        return {
            "runtime": runtime,
            "requested_device": device,
            "machine": platform.machine(),
            "system": platform.system(),
            "nvidia_visible_devices": os.getenv("NVIDIA_VISIBLE_DEVICES"),
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
