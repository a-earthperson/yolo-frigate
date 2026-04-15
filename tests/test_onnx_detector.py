import asyncio
import sys
import types
import unittest
import unittest.mock

import numpy as np

from yolo_frigate.ultralytics_detector import UltralyticsDetector


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        self.cls = np.array([1.0], dtype=np.float32)
        self.conf = np.array([0.85], dtype=np.float32)


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.names = {0: "person", 1: "dog"}


class FakeYOLOE:
    instances = []

    def __init__(self, model_file):
        self.model_file = model_file
        self.predict_calls = []
        FakeYOLOE.instances.append(self)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [FakeResult()]


class TestUltralyticsDetector(unittest.TestCase):
    def tearDown(self):
        FakeYOLOE.instances.clear()

    def test_detect_maps_results_into_contract(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.engine",
                runtime="tensorrt",
                conf=0.4,
                iou=0.5,
                device="gpu:1",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
            )

        self.assertEqual(FakeYOLOE.instances[0].model_file, "model.engine")
        self.assertEqual(FakeYOLOE.instances[0].predict_calls[0]["device"], "1")
        self.assertAlmostEqual(FakeYOLOE.instances[0].predict_calls[0]["conf"], 0.4)
        self.assertAlmostEqual(FakeYOLOE.instances[0].predict_calls[0]["iou"], 0.5)
        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "dog")
        self.assertAlmostEqual(predictions.predictions[0].confidence, 0.85, places=6)
        self.assertEqual(predictions.predictions[0].y_min, 20.0)
        self.assertEqual(predictions.predictions[0].x_min, 10.0)
        self.assertEqual(predictions.predictions[0].y_max, 40.0)
        self.assertEqual(predictions.predictions[0].x_max, 30.0)
        self.assertTrue(predictions.success)

    def test_explicit_class_names_override_result_names(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.tflite",
                runtime="tflite",
                class_names=["person", "vehicle"],
                device="cpu",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))
            )

        self.assertNotIn("device", FakeYOLOE.instances[0].predict_calls[0])
        self.assertEqual(predictions.predictions[0].label, "vehicle")

    def test_openvino_runtime_normalizes_device(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model_openvino_model",
                runtime="openvino",
                device="gpu:2",
            )
            asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        self.assertEqual(FakeYOLOE.instances[0].predict_calls[0]["device"], "intel:gpu")

    def test_openvino_runtime_normalizes_cpu_device(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model_openvino_model",
                runtime="openvino",
                device="cpu",
            )
            asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        self.assertEqual(FakeYOLOE.instances[0].predict_calls[0]["device"], "intel:cpu")

    def test_tflite_runtime_does_not_force_predict_device(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.tflite",
                runtime="tflite",
                device="cpu",
            )
            asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        self.assertNotIn("device", FakeYOLOE.instances[0].predict_calls[0])

    def test_tensor_rt_engine_requires_gpu_device(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            with self.assertRaises(ValueError):
                UltralyticsDetector("model.engine", runtime="tensorrt", device="cpu")

    def test_onnx_runtime_normalizes_gpu_device(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.onnx",
                runtime="onnx",
                device="gpu:2",
            )
            asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        self.assertEqual(FakeYOLOE.instances[0].predict_calls[0]["device"], "2")

    def test_onnx_runtime_passes_cpu_device_to_predict(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.onnx",
                runtime="onnx",
                device="cpu",
            )
            asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        self.assertEqual(FakeYOLOE.instances[0].predict_calls[0]["device"], "cpu")

    def test_invalid_openvino_device_is_rejected(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model_openvino_model",
                runtime="openvino",
                device="usb",
            )
            with self.assertRaises(ValueError):
                asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

    def test_detect_offloads_blocking_predict_call(self):
        ultralytics_module = types.SimpleNamespace(YOLOE=FakeYOLOE)
        with unittest.mock.patch.dict(sys.modules, {"ultralytics": ultralytics_module}):
            detector = UltralyticsDetector(
                "model.engine",
                runtime="tensorrt",
                device="gpu:0",
            )

            async def fake_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)

            with unittest.mock.patch(
                "yolo_frigate.ultralytics_detector.asyncio.to_thread",
                side_effect=fake_to_thread,
            ) as to_thread:
                asyncio.run(detector.detect(np.zeros((4, 4, 3), dtype=np.uint8)))

        to_thread.assert_called_once()
