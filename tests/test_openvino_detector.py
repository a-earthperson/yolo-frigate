import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from yolo_frigate.openvino_detector import OpenVINOAsyncDetector


class FakeLayout:
    def __init__(self, empty=True):
        self._empty = empty

    def empty(self):
        return self._empty


class FakeInputPort:
    def __init__(self, shape):
        self.shape = shape
        self._layout = FakeLayout()

    def get_any_name(self):
        return "images"

    def get_layout(self):
        return self._layout

    def set_layout(self, layout):
        self._layout = layout


class FakeInputPortWithoutLayoutMethods:
    def __init__(self, shape):
        self.shape = shape
        self._layout = FakeLayout()

    def get_any_name(self):
        return "images"


class FakeDynamicDimension:
    def is_dynamic(self):
        return True

    def get_length(self):
        raise RuntimeError("dynamic dimension")


class FakeDynamicInputPort(FakeInputPort):
    def __init__(self, shape):
        self._shape = shape
        self._layout = FakeLayout()

    @property
    def shape(self):
        raise RuntimeError("to_shape was called on a dynamic shape.")

    def get_partial_shape(self):
        return [1, 3, FakeDynamicDimension(), FakeDynamicDimension()]


class FakeOVModel:
    def __init__(self, shape, input_port_cls=FakeInputPort):
        self._input = input_port_cls(shape)

    def input(self, index):
        assert index == 0
        return self._input


class FakeCompiledModel:
    def __init__(self, output, shape, input_port_cls=FakeInputPort):
        self.output = output
        self._input = input_port_cls(shape)

    def input(self, index=0):
        assert index == 0
        return self._input


class FakeRequest:
    def __init__(self, result):
        self.results = result if isinstance(result, dict) else {"output0": result}


class FakeAsyncInferQueue:
    def __init__(self, compiled_model, jobs=0):
        self.compiled_model = compiled_model
        self.jobs = jobs
        self.callback = None
        self.started_inputs = []

    def __len__(self):
        return 2

    def set_callback(self, callback):
        self.callback = callback

    def start_async(self, inputs, userdata):
        self.started_inputs.append(inputs)
        self.callback(FakeRequest(self.compiled_model.output), userdata)


class FakeCore:
    def __init__(self, output, shape, available_devices, input_port_cls=FakeInputPort):
        self.output = output
        self.shape = shape
        self.available_devices = available_devices
        self.input_port_cls = input_port_cls
        self.read_model_calls = []
        self.compile_model_calls = []

    def read_model(self, model, weights):
        self.read_model_calls.append((model, weights))
        return FakeOVModel(self.shape, self.input_port_cls)

    def compile_model(self, model, device_name, config):
        self.compile_model_calls.append((model, device_name, config))
        return FakeCompiledModel(self.output, self.shape, self.input_port_cls)


class FakeOpenVINO:
    def __init__(
        self,
        output,
        shape=(1, 3, 640, 640),
        available_devices=None,
        input_port_cls=FakeInputPort,
        layout_helpers=None,
    ):
        self._core = FakeCore(
            output,
            shape,
            available_devices or ["CPU", "GPU.0", "GPU.1"],
            input_port_cls=input_port_cls,
        )
        self.AsyncInferQueue = FakeAsyncInferQueue
        self.layout_helpers = layout_helpers

    def Core(self):
        return self._core

    class Layout:
        def __init__(self, value):
            self.value = value


class TestOpenVINOAsyncDetector(unittest.TestCase):
    def _write_model_dir(self, metadata):
        tempdir = tempfile.TemporaryDirectory()
        model_dir = Path(tempdir.name) / "model_openvino_model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.xml").write_text("<xml />", encoding="utf-8")
        (model_dir / "model.bin").write_bytes(b"\x00")
        (model_dir / "metadata.yaml").write_text(metadata, encoding="utf-8")
        return tempdir, model_dir

    def test_detect_decodes_raw_outputs_via_native_async_queue(self):
        raw_output = np.array(
            [
                [
                    [320.0, 325.0],
                    [320.0, 325.0],
                    [100.0, 110.0],
                    [100.0, 110.0],
                    [0.05, 0.10],
                    [0.90, 0.70],
                ]
            ],
            dtype=np.float32,
        )
        fake_ov = FakeOpenVINO(raw_output)
        tempdir, model_dir = self._write_model_dir(
            "names: ['person', 'vehicle']\nstride: 32\nimgsz: [640, 640]\n"
        )
        self.addCleanup(tempdir.cleanup)

        with patch(
            "yolo_frigate.openvino_detector._import_openvino", return_value=fake_ov
        ):
            detector = OpenVINOAsyncDetector(
                str(model_dir),
                conf=0.25,
                iou=0.45,
                device="gpu:1",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
            )

        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "vehicle")
        self.assertAlmostEqual(predictions.predictions[0].confidence, 0.9, places=6)
        self.assertEqual(predictions.predictions[0].x_min, 270.0)
        self.assertEqual(predictions.predictions[0].y_min, 270.0)
        self.assertEqual(predictions.predictions[0].x_max, 370.0)
        self.assertEqual(predictions.predictions[0].y_max, 370.0)

        compiled_call = fake_ov._core.compile_model_calls[0]
        self.assertEqual(compiled_call[1], "GPU.1")
        self.assertEqual(compiled_call[2], {"PERFORMANCE_HINT": "THROUGHPUT"})

    def test_detect_handles_end_to_end_outputs_and_label_override(self):
        end2end_output = np.array(
            [[[10.0, 20.0, 30.0, 40.0, 0.85, 1.0]]],
            dtype=np.float32,
        )
        fake_ov = FakeOpenVINO(
            end2end_output, shape=(1, 3, 320, 320), available_devices=["CPU"]
        )
        tempdir, model_dir = self._write_model_dir(
            "names: ['ignored', 'also-ignored']\nstride: 32\nimgsz: [320, 320]\n"
        )
        self.addCleanup(tempdir.cleanup)

        with patch(
            "yolo_frigate.openvino_detector._import_openvino", return_value=fake_ov
        ):
            detector = OpenVINOAsyncDetector(
                str(model_dir),
                class_names=["person", "package"],
                conf=0.25,
                iou=0.45,
                device="cpu",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            )

        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "package")
        self.assertAlmostEqual(predictions.predictions[0].confidence, 0.85, places=6)

        compiled_call = fake_ov._core.compile_model_calls[0]
        self.assertEqual(compiled_call[1], "CPU")
        self.assertEqual(compiled_call[2], {"PERFORMANCE_HINT": "LATENCY"})

    def test_detector_supports_output_ports_without_layout_methods(self):
        class FakeLayoutHelpers:
            @staticmethod
            def get_layout(port):
                return port._layout

            @staticmethod
            def set_layout(port, layout):
                port._layout = layout

        end2end_output = np.array(
            [[[10.0, 20.0, 30.0, 40.0, 0.85, 1.0]]],
            dtype=np.float32,
        )
        fake_ov = FakeOpenVINO(
            end2end_output,
            shape=(1, 3, 320, 320),
            available_devices=["CPU"],
            input_port_cls=FakeInputPortWithoutLayoutMethods,
            layout_helpers=FakeLayoutHelpers(),
        )
        tempdir, model_dir = self._write_model_dir(
            "names: ['person', 'package']\nstride: 32\nimgsz: [320, 320]\n"
        )
        self.addCleanup(tempdir.cleanup)

        with patch(
            "yolo_frigate.openvino_detector._import_openvino", return_value=fake_ov
        ):
            detector = OpenVINOAsyncDetector(
                str(model_dir),
                conf=0.25,
                iou=0.45,
                device="cpu",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            )

        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "package")

    def test_detect_prefers_end_to_end_detection_tensor_over_mask_prototypes(self):
        detection_output = np.array(
            [[[10.0, 20.0, 30.0, 40.0, 0.85, 1.0] + ([0.0] * 32)]],
            dtype=np.float32,
        )
        proto_output = np.zeros((1, 32, 80, 80), dtype=np.float32)
        fake_ov = FakeOpenVINO(
            {
                "output1": proto_output,
                "output0": detection_output,
            },
            shape=(1, 3, 320, 320),
            available_devices=["GPU"],
        )
        tempdir, model_dir = self._write_model_dir(
            "names: ['person', 'package']\nstride: 32\nimgsz: [320, 320]\n"
        )
        self.addCleanup(tempdir.cleanup)

        with patch(
            "yolo_frigate.openvino_detector._import_openvino", return_value=fake_ov
        ):
            detector = OpenVINOAsyncDetector(
                str(model_dir),
                conf=0.25,
                iou=0.45,
                device="gpu",
            )
            predictions = asyncio.run(
                detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            )

        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "package")
        self.assertAlmostEqual(predictions.predictions[0].confidence, 0.85, places=6)
        self.assertEqual(predictions.predictions[0].x_min, 10.0)
        self.assertEqual(predictions.predictions[0].y_min, 20.0)
        self.assertEqual(predictions.predictions[0].x_max, 30.0)
        self.assertEqual(predictions.predictions[0].y_max, 40.0)

    def test_dynamic_input_shape_falls_back_to_metadata_imgsz(self):
        end2end_output = np.array(
            [[[10.0, 20.0, 30.0, 40.0, 0.85, 1.0]]],
            dtype=np.float32,
        )
        fake_ov = FakeOpenVINO(
            end2end_output,
            shape=(1, 3, 640, 640),
            available_devices=["GPU"],
            input_port_cls=FakeDynamicInputPort,
        )
        tempdir, model_dir = self._write_model_dir(
            "names: ['person', 'package']\nstride: 32\nimgsz: [640, 640]\n"
        )
        self.addCleanup(tempdir.cleanup)

        with patch(
            "yolo_frigate.openvino_detector._import_openvino", return_value=fake_ov
        ):
            detector = OpenVINOAsyncDetector(
                str(model_dir),
                conf=0.25,
                iou=0.45,
                device="gpu",
            )
            self.assertIsNone(detector.input_hw)
            predictions = asyncio.run(
                detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
            )

        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "package")
