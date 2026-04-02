from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yolo_frigate.config import AppConfig


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    export_format: str


@dataclass(frozen=True)
class ModelSource:
    path: Path
    kind: str


RUNTIME_PROFILES = {
    "tensorrt": RuntimeProfile(name="tensorrt", export_format="engine"),
    "openvino": RuntimeProfile(name="openvino", export_format="openvino"),
    "onnx": RuntimeProfile(name="onnx", export_format="onnx"),
    "tflite": RuntimeProfile(name="tflite", export_format="tflite"),
    "edgetpu": RuntimeProfile(name="edgetpu", export_format="edgetpu"),
}


def describe_model_source(model_file: str) -> ModelSource:
    path = Path(model_file)
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    as_posix = path.as_posix().lower()

    if suffix == ".pt":
        return ModelSource(path=path, kind="checkpoint")
    if suffix == ".engine":
        return ModelSource(path=path, kind="tensorrt")
    if suffix == ".onnx":
        return ModelSource(path=path, kind="onnx")
    if suffix == ".tflite":
        if "edgetpu" in lower_name:
            return ModelSource(path=path, kind="edgetpu")
        return ModelSource(path=path, kind="tflite")
    if as_posix.endswith("_openvino_model") or (
        path.suffix == "" and path.name.lower().endswith("_openvino_model")
    ):
        return ModelSource(path=path, kind="openvino")

    raise ValueError(
        f"Unsupported model source '{model_file}'. Expected a .pt checkpoint or a supported runtime artifact."
    )


def resolve_runtime_profile(config: AppConfig) -> RuntimeProfile:
    runtime_name = resolve_runtime_name(config)
    return RUNTIME_PROFILES[runtime_name]


def resolve_runtime_name(config: AppConfig) -> str:
    source = describe_model_source(config.model_file)
    return _resolve_runtime_name(config, source)


def _resolve_runtime_name(config: AppConfig, source: ModelSource) -> str:
    if config.runtime != "auto":
        runtime_name = config.runtime
    elif source.kind == "checkpoint":
        raise ValueError(
            "Unable to infer runtime for a .pt source model. Specify --runtime or set "
            "YOLO_FRIGATE_RUNTIME (or legacy YOLOREST_RUNTIME)."
        )
    elif source.kind == "tflite":
        runtime_name = _resolve_tflite_family_runtime(config.device)
    else:
        runtime_name = source.kind

    if (
        source.kind in RUNTIME_PROFILES
        and source.kind != "tflite"
        and source.kind != runtime_name
    ):
        raise ValueError(
            f"Model source '{config.model_file}' is tied to runtime '{source.kind}', "
            f"but runtime '{runtime_name}' was requested."
        )
    if source.kind == "tflite" and runtime_name not in {"tflite", "edgetpu"}:
        raise ValueError(
            f"Model source '{config.model_file}' is tied to the TFLite runtime family, "
            f"but runtime '{runtime_name}' was requested."
        )

    return runtime_name


def _resolve_tflite_family_runtime(device: str) -> str:
    return "tflite" if device == "cpu" else "edgetpu"
