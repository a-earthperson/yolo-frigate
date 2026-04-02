import unittest
from unittest.mock import patch

from yolo_frigate.config import AppConfig
from yolo_frigate.detector_factory import create_detector, resolve_runtime
from yolo_frigate.model_artifact import ResolvedModelArtifact


def make_config(**overrides) -> AppConfig:
    values = {
        "log_level": "warning",
        "runtime": "auto",
        "label_file": None,
        "model_file": "model.engine",
        "device": "cpu",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "export_imgsz": 640,
        "export_half": False,
        "export_int8": False,
        "export_dynamic": False,
        "export_nms": False,
        "export_batch": 1,
        "export_data": None,
        "export_fraction": 1.0,
        "export_workspace": None,
        "model_cache_dir": "/tmp/yolo-frigate-cache",
        "enable_save": False,
        "save_threshold": "0.75",
        "save_path": "./output",
        "host": "0.0.0.0",
        "port": 8000,
    }
    values.update(overrides)
    return AppConfig(**values)


class FakeArtifactManager:
    def __init__(self, resolved: ResolvedModelArtifact):
        self.resolved = resolved
        self.calls = []

    def resolve(self, config, runtime_profile):
        self.calls.append((config, runtime_profile.name))
        return self.resolved


class TestDetectorFactory(unittest.TestCase):
    def test_resolve_runtime_from_native_artifacts(self):
        self.assertEqual(
            resolve_runtime(make_config(model_file="model.engine")), "tensorrt"
        )
        self.assertEqual(
            resolve_runtime(make_config(model_file="model_openvino_model")), "openvino"
        )

    def test_onnx_artifacts_resolve_runtime(self):
        self.assertEqual(
            resolve_runtime(make_config(model_file="model.onnx")),
            "onnx",
        )

    def test_tflite_runtime_switches_to_edgetpu_for_non_cpu_devices(self):
        self.assertEqual(
            resolve_runtime(make_config(model_file="model.tflite")), "tflite"
        )
        self.assertEqual(
            resolve_runtime(make_config(model_file="model.tflite", device="pci")),
            "edgetpu",
        )

    def test_pt_sources_require_explicit_runtime(self):
        with self.assertRaises(ValueError):
            resolve_runtime(make_config(model_file="model.pt"))

    def test_runtime_mismatch_with_native_artifact_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_runtime(make_config(runtime="openvino", model_file="model.engine"))

    def test_create_detector_uses_resolved_artifact_and_labels(self):
        config = make_config(
            runtime="tensorrt",
            model_file="model.pt",
            label_file="labels.yaml",
            device="gpu:1",
        )
        manager = FakeArtifactManager(
            ResolvedModelArtifact(
                path="/cache/model.engine",
                cached=True,
            )
        )

        with (
            patch(
                "yolo_frigate.detector_factory.parse_labels", return_value={0: "person"}
            ),
            patch("yolo_frigate.detector_factory.UltralyticsDetector") as detector_cls,
        ):
            detector_cls.return_value = object()
            detector = create_detector(config, artifact_manager=manager)

        detector_cls.assert_called_once_with(
            "/cache/model.engine",
            "tensorrt",
            {0: "person"},
            0.25,
            0.45,
            "gpu:1",
        )
        self.assertEqual(manager.calls[0][1], "tensorrt")
        self.assertIs(detector, detector_cls.return_value)
