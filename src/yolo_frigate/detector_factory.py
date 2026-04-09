from __future__ import annotations

from yolo_frigate.config import AppConfig
from yolo_frigate.detector_backend import DetectorBackend
from yolo_frigate.label import parse_classes
from yolo_frigate.model_artifact import ModelArtifactManager
from yolo_frigate.runtime_profile import resolve_runtime_profile
from yolo_frigate.ultralytics_detector import UltralyticsDetector


def resolve_runtime(config: AppConfig) -> str:
    return resolve_runtime_profile(config).name


def load_classes(label_file: str | None) -> list[str] | None:
    if label_file is None:
        return None
    return parse_classes(label_file)


def create_detector(
    config: AppConfig,
    artifact_manager: ModelArtifactManager | None = None,
) -> DetectorBackend:
    runtime_profile = resolve_runtime_profile(config)
    class_names = load_classes(config.label_file)
    resolved_artifact = (artifact_manager or ModelArtifactManager()).resolve(
        config, runtime_profile, class_names
    )

    return UltralyticsDetector(
        resolved_artifact.path,
        runtime_profile.name,
        class_names,
        config.confidence_threshold,
        config.iou_threshold,
        config.device,
    )
