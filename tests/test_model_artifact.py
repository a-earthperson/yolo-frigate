import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

from yolo_frigate.config import AppConfig
from yolo_frigate.model_artifact import ModelArtifactManager
from yolo_frigate.runtime_profile import RuntimeProfile, resolve_runtime_profile


def make_config(**overrides) -> AppConfig:
    values = {
        "log_level": "warning",
        "runtime": "tensorrt",
        "label_file": None,
        "model_file": "model.pt",
        "device": "gpu:0",
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


class FakeYOLOE:
    export_calls = []
    set_classes_calls = []
    init_calls = []
    val_calls = []
    download_dir = None

    def __init__(self, model_file):
        self.model_file = model_file
        FakeYOLOE.init_calls.append(model_file)
        source_path = Path(model_file)
        if source_path.is_file():
            self.ckpt_path = str(source_path)
        elif FakeYOLOE.download_dir is not None:
            downloaded = Path(FakeYOLOE.download_dir) / source_path.name
            downloaded.parent.mkdir(parents=True, exist_ok=True)
            downloaded.write_bytes(b"downloaded-weights")
            self.ckpt_path = str(downloaded)
        else:
            self.ckpt_path = str(source_path)

    def set_classes(self, classes):
        FakeYOLOE.set_classes_calls.append((self.model_file, list(classes)))

    def export(self, **kwargs):
        FakeYOLOE.export_calls.append((self.model_file, kwargs))
        source_path = Path(self.model_file)
        parent = source_path.parent
        stem = source_path.stem
        fmt = kwargs["format"]
        if fmt == "engine":
            artifact = parent / f"{stem}.engine"
            artifact.write_text("engine", encoding="utf-8")
        elif fmt == "openvino":
            artifact = parent / f"{stem}_openvino_model"
            artifact.mkdir()
            (artifact / "model.xml").write_text("<xml />", encoding="utf-8")
            (artifact / "model.bin").write_bytes(b"bin")
        elif fmt == "onnx":
            artifact = parent / f"{stem}.onnx"
            artifact.write_bytes(b"onnx")
        elif fmt == "edgetpu":
            artifact = parent / f"{stem}_full_integer_quant_edgetpu.tflite"
            artifact.write_text("edgetpu", encoding="utf-8")
        else:
            artifact = parent / f"{stem}_float32.tflite"
            artifact.write_text("tflite", encoding="utf-8")
        return str(artifact)

    def val(self):
        FakeYOLOE.val_calls.append(self.model_file)
        return object()


class TestModelArtifactManager(unittest.TestCase):
    def tearDown(self):
        FakeYOLOE.export_calls.clear()
        FakeYOLOE.set_classes_calls.clear()
        FakeYOLOE.init_calls.clear()
        FakeYOLOE.val_calls.clear()
        FakeYOLOE.download_dir = None

    def test_checkpoint_export_is_cached_lazily(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            config = make_config(
                runtime="tensorrt",
                model_file=str(model_path),
                model_cache_dir=str(cache_dir),
            )
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                first = manager.resolve(
                    config, resolve_runtime_profile(config), ["person", "package"]
                )
                second = manager.resolve(
                    config, resolve_runtime_profile(config), ["person", "package"]
                )
                self.assertTrue(first.cached)
                self.assertTrue(Path(first.path).is_file())
                self.assertEqual(first.path, second.path)
                self.assertEqual(len(FakeYOLOE.export_calls), 1)
                self.assertEqual(
                    FakeYOLOE.set_classes_calls,
                    [(str(Path(first.path).parents[0] / "model.pt"), ["person", "package"])],
                )
                self.assertEqual(FakeYOLOE.val_calls, [])
                self.assertTrue(
                    (Path(first.path).parents[1] / "manifest.json").is_file()
                )

    def test_cache_key_changes_when_export_flags_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                fp32 = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                        export_half=False,
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["person"],
                )
                fp16 = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                        export_half=True,
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["person"],
                )

        self.assertNotEqual(fp32.path, fp16.path)
        self.assertEqual(len(FakeYOLOE.export_calls), 2)

    def test_cache_key_changes_when_class_names_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                first = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["person"],
                )
                second = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["package"],
                )

        self.assertNotEqual(first.path, second.path)
        self.assertEqual(len(FakeYOLOE.export_calls), 2)

    def test_named_checkpoint_is_downloaded_via_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            download_dir = Path(tmpdir) / "downloads"
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )
            FakeYOLOE.download_dir = download_dir

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                resolved = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file="yoloe-26l-seg.pt",
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["package"],
                )
                self.assertTrue(Path(resolved.path).is_file())
                self.assertTrue((download_dir / "yoloe-26l-seg.pt").is_file())
                self.assertEqual(FakeYOLOE.val_calls, [])

        self.assertEqual(FakeYOLOE.init_calls[0], "yoloe-26l-seg.pt")
        self.assertEqual(Path(FakeYOLOE.export_calls[0][0]).name, "yoloe-26l-seg.pt")

    def test_cache_key_changes_when_gpu_model_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with (
                unittest.mock.patch.dict(
                    sys.modules, {"ultralytics": ultralytics_module}
                ),
                unittest.mock.patch(
                    "yolo_frigate.model_artifact._resolve_gpu_identity",
                    side_effect=[
                        {
                            "name": "NVIDIA GeForce RTX 3090",
                            "compute_capability": "8.6",
                        },
                        {
                            "name": "NVIDIA GeForce RTX 4090",
                            "compute_capability": "8.9",
                        },
                    ],
                ),
            ):
                first = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["person"],
                )
                second = manager.resolve(
                    make_config(
                        runtime="tensorrt",
                        model_file=str(model_path),
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tensorrt", "engine"),
                    ["person"],
                )

        self.assertNotEqual(first.path, second.path)
        self.assertEqual(len(FakeYOLOE.export_calls), 2)

    def test_int8_requires_calibration_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            config = make_config(
                runtime="tensorrt",
                model_file=str(model_path),
                export_int8=True,
            )

            with self.assertRaises(ValueError):
                ModelArtifactManager().resolve(
                    config, RuntimeProfile("tensorrt", "engine"), ["person"]
                )

    def test_onnx_export_accepts_cpu_device(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            config = make_config(
                runtime="onnx",
                model_file=str(model_path),
                device="cpu",
                model_cache_dir=str(cache_dir),
            )
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                resolved = manager.resolve(
                    config, RuntimeProfile("onnx", "onnx"), ["person"]
                )

        self.assertTrue(str(resolved.path).endswith(".onnx"))
        self.assertEqual(FakeYOLOE.export_calls[-1][1]["device"], "cpu")

    def test_openvino_directory_artifact_is_passed_through(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model_openvino_model"
            model_dir.mkdir()
            config = make_config(
                runtime="openvino",
                model_file=str(model_dir),
                device="cpu",
            )

            resolved = ModelArtifactManager().resolve(
                config, RuntimeProfile("openvino", "openvino"), ["person"]
            )

        self.assertFalse(resolved.cached)
        self.assertEqual(resolved.path, str(model_dir))

    def test_tflite_and_edgetpu_exports_produce_distinct_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model_path.write_bytes(b"weights")
            cache_dir = Path(tmpdir) / "cache"
            manager = ModelArtifactManager()
            ultralytics_module = types.SimpleNamespace(
                YOLOE=FakeYOLOE, __version__="8.3.0"
            )

            with unittest.mock.patch.dict(
                sys.modules, {"ultralytics": ultralytics_module}
            ):
                tflite = manager.resolve(
                    make_config(
                        runtime="tflite",
                        model_file=str(model_path),
                        device="cpu",
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("tflite", "tflite"),
                    ["person"],
                )
                edgetpu = manager.resolve(
                    make_config(
                        runtime="edgetpu",
                        model_file=str(model_path),
                        device="pci",
                        model_cache_dir=str(cache_dir),
                    ),
                    RuntimeProfile("edgetpu", "edgetpu"),
                    ["person"],
                )

        self.assertTrue(tflite.path.endswith(".tflite"))
        self.assertNotIn("edgetpu", Path(tflite.path).name.lower())
        self.assertIn("edgetpu", Path(edgetpu.path).name.lower())
