from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from yolo_frigate.prediction import Prediction, Predictions
from yolo_frigate.ultralytics_support import import_ultralytics_yoloe


class UltralyticsDetector:
    def __init__(
        self,
        model: str,
        runtime: str,
        class_names: list[str] | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model
        self.runtime = runtime
        self.class_names = class_names
        self.conf = conf
        self.iou = iou
        self.requested_device = device

        self._validate_runtime_device()
        self.model = import_ultralytics_yoloe()(model)
        self._predict_lock = asyncio.Lock()

    async def detect(self, img: np.ndarray) -> Predictions:
        predict_kwargs = {
            "source": img,
            "verbose": False,
            "conf": self.conf,
            "iou": self.iou,
        }
        predict_device = self._predict_device()
        if predict_device is not None:
            predict_kwargs["device"] = predict_device

        # Ultralytics inference is a blocking call; run it off the event loop and
        # serialize access to the shared model instance until native async backends land.
        async with self._predict_lock:
            results = await asyncio.to_thread(self.model.predict, **predict_kwargs)
        if not results:
            return Predictions(predictions=[], success=True)
        return self._result_to_predictions(results[0])

    def _validate_runtime_device(self) -> None:
        model_suffix = Path(self.model_path).suffix.lower()
        if (
            self.runtime == "tensorrt"
            and model_suffix == ".engine"
            and not self.requested_device.startswith("gpu")
        ):
            raise ValueError("TensorRT engine inference requires gpu or gpu:<index>.")

    def _predict_device(self) -> str | None:
        if self.runtime == "tensorrt":
            return _normalize_tensorrt_predict_device(self.requested_device)
        if self.runtime == "onnx":
            return _normalize_onnx_predict_device(self.requested_device)
        if self.runtime == "openvino":
            return _normalize_openvino_predict_device(self.requested_device)
        return None

    def _result_to_predictions(self, result: Any) -> Predictions:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return Predictions(predictions=[], success=True)

        xyxy = _to_list(getattr(boxes, "xyxy", []))
        class_ids = _to_list(getattr(boxes, "cls", []))
        confidences = _to_list(getattr(boxes, "conf", []))
        names = (
            _class_names_to_map(self.class_names) or getattr(result, "names", {}) or {}
        )

        predictions = []
        for coordinates, class_id, confidence in zip(xyxy, class_ids, confidences):
            x1, y1, x2, y2 = (float(value) for value in coordinates)
            class_index = int(class_id)
            predictions.append(
                Prediction(
                    label=names.get(class_index, str(class_index)),
                    confidence=float(confidence),
                    y_min=y1,
                    x_min=x1,
                    y_max=y2,
                    x_max=x2,
                )
            )

        return Predictions(predictions=predictions, success=True)


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _class_names_to_map(class_names: list[str] | None) -> dict[int, str]:
    if class_names is None:
        return {}
    return {index: name for index, name in enumerate(class_names)}


def _normalize_onnx_predict_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
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
        f"Invalid ONNX device '{device}'. Supported values are cpu, gpu, or gpu:<index>."
    )


def _normalize_tensorrt_predict_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
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
        f"Invalid TensorRT device '{device}'. Supported values are cpu, gpu, or gpu:<index>."
    )


def _normalize_openvino_predict_device(device: str) -> str:
    if device == "cpu":
        return "intel:cpu"
    if device in {"gpu", "gpu:0"} or device.startswith("gpu:"):
        return "intel:gpu"
    if device in {"npu", "npu:0"} or device.startswith("npu:"):
        return "intel:npu"
    raise ValueError(
        f"Invalid OpenVINO device '{device}'. Supported values are cpu, gpu, gpu:<index>, npu, or npu:<index>."
    )
