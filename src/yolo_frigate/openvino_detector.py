from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from yolo_frigate.prediction import Prediction, Predictions

logger = logging.getLogger(__name__)

_DEFAULT_STRIDE = 32
_LETTERBOX_PAD_VALUE = (114, 114, 114)


@dataclass(frozen=True)
class _ModelMetadata:
    class_names: tuple[str, ...]
    stride: int
    imgsz: tuple[int, int] | None


@dataclass(frozen=True)
class _PendingInference:
    future: asyncio.Future[Predictions]
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]


class OpenVINOAsyncDetector:
    def __init__(
        self,
        model: str,
        class_names: list[str] | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        ov = _import_openvino()

        self.model_path = Path(model).expanduser().resolve()
        self.conf = conf
        self.iou = iou
        self.requested_device = device
        self.metadata = _load_model_metadata(self.model_path, class_names)

        core = ov.Core()
        device_name = _resolve_device_name(core.available_devices, device)
        performance_hint = _resolve_performance_hint(device_name)

        xml_path = _resolve_openvino_xml_path(self.model_path)
        ov_model = core.read_model(
            model=str(xml_path), weights=str(xml_path.with_suffix(".bin"))
        )
        input_port = ov_model.input(0)
        _ensure_input_layout(ov, input_port, "NCHW")

        compile_config = {"PERFORMANCE_HINT": performance_hint}
        self.compiled_model = core.compile_model(
            ov_model,
            device_name=device_name,
            config=compile_config,
        )
        self.input_name = self.compiled_model.input(0).get_any_name()
        self.input_hw = _resolve_input_hw(self.compiled_model.input(0))
        self._queue = ov.AsyncInferQueue(self.compiled_model, 0)
        self._slots = asyncio.Semaphore(max(len(self._queue), 1))
        self._queue.set_callback(self._handle_completion)

        logger.info(
            "Initialized native OpenVINO backend on %s with hint=%s and %s infer request(s).",
            device_name,
            performance_hint,
            len(self._queue),
        )

    async def detect(self, img: np.ndarray) -> Predictions:
        input_tensor, input_shape = self._preprocess(img)
        await self._slots.acquire()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Predictions] = loop.create_future()
        pending = _PendingInference(
            future=future,
            original_shape=img.shape[:2],
            input_shape=input_shape,
        )

        try:
            self._queue.start_async(
                inputs={self.input_name: input_tensor}, userdata=pending
            )
        except Exception:
            self._slots.release()
            raise

        return await future

    def _handle_completion(self, request: Any, pending: _PendingInference) -> None:
        loop = pending.future.get_loop()
        try:
            predictions = self._postprocess(
                _extract_outputs(request),
                pending.original_shape,
                pending.input_shape,
            )
        except Exception as exc:
            loop.call_soon_threadsafe(_set_future_exception, pending.future, exc)
        else:
            loop.call_soon_threadsafe(_set_future_result, pending.future, predictions)
        finally:
            loop.call_soon_threadsafe(self._slots.release)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        target_shape = self._target_shape(img.shape[:2])
        resized = _letterbox(img, target_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        nchw = (
            np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)
            / 255.0
        )
        return nchw, target_shape

    def _target_shape(self, original_shape: tuple[int, int]) -> tuple[int, int]:
        if self.input_hw is not None:
            return self.input_hw
        if self.metadata.imgsz is not None:
            return self.metadata.imgsz

        stride = max(self.metadata.stride, 1)
        return (
            _make_divisible(original_shape[0], stride),
            _make_divisible(original_shape[1], stride),
        )

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        original_shape: tuple[int, int],
        input_shape: tuple[int, int],
    ) -> Predictions:
        prediction = _select_primary_output(outputs)
        detections = _decode_detections(prediction, self.conf, self.iou)
        if detections.size == 0:
            return Predictions(predictions=[], success=True)

        boxes = detections[:, :4].copy()
        _scale_boxes(input_shape, boxes, original_shape)

        predictions = []
        for row, box in zip(detections, boxes):
            class_index = int(row[5])
            predictions.append(
                Prediction(
                    label=_label_for_index(self.metadata.class_names, class_index),
                    confidence=float(row[4]),
                    y_min=float(box[1]),
                    x_min=float(box[0]),
                    y_max=float(box[3]),
                    x_max=float(box[2]),
                )
            )
        return Predictions(predictions=predictions, success=True)


def _import_openvino():
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO runtime is required for the native OpenVINO detector backend."
        ) from exc
    return ov


def _set_future_result(future: asyncio.Future[Predictions], value: Predictions) -> None:
    if not future.done():
        future.set_result(value)


def _set_future_exception(future: asyncio.Future[Predictions], exc: Exception) -> None:
    if not future.done():
        future.set_exception(exc)


def _resolve_openvino_xml_path(model_path: Path) -> Path:
    if model_path.is_file():
        if model_path.suffix.lower() != ".xml":
            raise ValueError(
                f"OpenVINO model path must be a model directory or .xml file, received '{model_path}'."
            )
        return model_path

    matches = sorted(model_path.glob("*.xml"))
    if not matches:
        raise FileNotFoundError(
            f"No OpenVINO .xml model file found under '{model_path}'."
        )
    return matches[0]


def _load_model_metadata(
    model_path: Path, class_names: list[str] | None
) -> _ModelMetadata:
    if class_names:
        override = tuple(class_names)
    else:
        override = ()

    metadata_path = (
        model_path / "metadata.yaml"
        if model_path.is_dir()
        else model_path.parent / "metadata.yaml"
    )
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}

    metadata_class_names = _parse_class_names(metadata.get("names"))
    resolved_class_names = override or metadata_class_names
    stride = _coerce_stride(metadata.get("stride"))
    imgsz = _parse_imgsz(metadata.get("imgsz"))
    return _ModelMetadata(
        class_names=resolved_class_names,
        stride=stride,
        imgsz=imgsz,
    )


def _parse_class_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(entry) for entry in value)
    if isinstance(value, dict):
        ordered = sorted(value.items(), key=lambda item: int(item[0]))
        return tuple(str(name) for _, name in ordered)
    return ()


def _coerce_stride(value: Any) -> int:
    try:
        stride = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_STRIDE
    return stride if stride > 0 else _DEFAULT_STRIDE


def _parse_imgsz(value: Any) -> tuple[int, int] | None:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            height = int(value[0])
            width = int(value[1])
        except (TypeError, ValueError):
            return None
        if height > 0 and width > 0:
            return height, width
    return None


def _resolve_device_name(available_devices: list[str], device: str) -> str:
    normalized = device.lower()
    if normalized == "cpu":
        return _select_indexed_device(available_devices, "CPU", None)
    if normalized == "gpu":
        return _select_indexed_device(available_devices, "GPU", 0)
    if normalized.startswith("gpu:"):
        return _select_indexed_device(
            available_devices, "GPU", _parse_device_index(normalized, "gpu")
        )
    if normalized == "npu":
        return _select_indexed_device(available_devices, "NPU", 0)
    if normalized.startswith("npu:"):
        return _select_indexed_device(
            available_devices, "NPU", _parse_device_index(normalized, "npu")
        )
    raise ValueError(
        f"Invalid OpenVINO device '{device}'. Supported values are cpu, gpu, gpu:<index>, npu, or npu:<index>."
    )


def _parse_device_index(device: str, prefix: str) -> int:
    index = device.split(":", maxsplit=1)[1]
    if not index.isdigit():
        raise ValueError(
            f"Invalid OpenVINO device '{device}'. Expected {prefix} or {prefix}:<index>."
        )
    return int(index)


def _select_indexed_device(
    available_devices: list[str], prefix: str, index: int | None
) -> str:
    candidates = [
        device
        for device in available_devices
        if device == prefix or device.startswith(f"{prefix}.")
    ]
    if not candidates:
        fallback = "CPU" if "CPU" in available_devices else "AUTO"
        logger.warning(
            "OpenVINO device prefix '%s' is unavailable. Falling back to '%s'.",
            prefix,
            fallback,
        )
        return fallback

    if index is None:
        return prefix if prefix in candidates else candidates[0]
    if index >= len(candidates):
        logger.warning(
            "Requested %s:%s but only %s device(s) are available. Using '%s'.",
            prefix.lower(),
            index,
            len(candidates),
            candidates[0],
        )
        return candidates[0]
    return candidates[index]


def _resolve_performance_hint(device_name: str) -> str:
    return "LATENCY" if device_name.startswith("CPU") else "THROUGHPUT"


def _ensure_input_layout(ov: Any, input_port: Any, layout_name: str) -> None:
    if not _layout_is_empty(ov, input_port):
        return

    layout = ov.Layout(layout_name)

    setter = getattr(input_port, "set_layout", None)
    if callable(setter):
        setter(layout)
        return

    layout_helpers = getattr(ov, "layout_helpers", None)
    helper_setter = getattr(layout_helpers, "set_layout", None)
    if callable(helper_setter):
        helper_setter(input_port, layout)
        return

    node_getter = getattr(input_port, "get_node", None)
    if callable(node_getter):
        node = node_getter()
        node_setter = getattr(node, "set_layout", None)
        if callable(node_setter):
            node_setter(layout)
            return

    logger.debug(
        "Unable to set OpenVINO input layout to %s; proceeding without explicit layout metadata.",
        layout_name,
    )


def _layout_is_empty(ov: Any, input_port: Any) -> bool:
    getter = getattr(input_port, "get_layout", None)
    if callable(getter):
        layout = getter()
        empty = getattr(layout, "empty", None)
        return bool(empty()) if callable(empty) else False

    layout_helpers = getattr(ov, "layout_helpers", None)
    helper_getter = getattr(layout_helpers, "get_layout", None)
    if not callable(helper_getter):
        return False

    layout = helper_getter(input_port)
    empty = getattr(layout, "empty", None)
    return bool(empty()) if callable(empty) else False


def _resolve_input_hw(input_port: Any) -> tuple[int, int] | None:
    shape = _safe_get_partial_shape(input_port)
    if shape is None:
        shape = _safe_get_shape(input_port)
    if shape is None:
        return None

    dims: list[int] = []
    for dim in shape:
        value = _resolve_dimension_length(dim)
        if value is None:
            return None
        if value <= 0:
            return None
        dims.append(value)

    if len(dims) != 4:
        return None
    return dims[2], dims[3]


def _safe_get_partial_shape(input_port: Any) -> Any | None:
    getter = getattr(input_port, "get_partial_shape", None)
    if callable(getter):
        try:
            return getter()
        except RuntimeError:
            return None

    try:
        return getattr(input_port, "partial_shape", None)
    except RuntimeError:
        return None


def _safe_get_shape(input_port: Any) -> Any | None:
    getter = getattr(input_port, "get_shape", None)
    if callable(getter):
        try:
            return getter()
        except RuntimeError:
            return None

    try:
        return getattr(input_port, "shape", None)
    except RuntimeError:
        return None


def _resolve_dimension_length(dim: Any) -> int | None:
    is_dynamic = getattr(dim, "is_dynamic", None)
    if callable(is_dynamic):
        try:
            if is_dynamic():
                return None
        except RuntimeError:
            return None
    elif isinstance(is_dynamic, bool) and is_dynamic:
        return None

    length_getter = getattr(dim, "get_length", None)
    if callable(length_getter):
        try:
            return int(length_getter())
        except RuntimeError:
            return None

    try:
        return int(dim)
    except (TypeError, ValueError):
        return None


def _extract_outputs(request: Any) -> list[np.ndarray]:
    results = getattr(request, "results", None)
    if not isinstance(results, dict):
        raise RuntimeError("OpenVINO async callback did not provide a results mapping.")
    return [np.asarray(value) for value in results.values()]


def _select_primary_output(outputs: list[np.ndarray]) -> np.ndarray:
    candidates: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    output_shapes = [tuple(np.asarray(output).shape) for output in outputs]

    for output in outputs:
        array = np.asarray(output)
        if array.ndim not in {2, 3}:
            continue
        try:
            normalized = _normalize_prediction_shape(array)
        except RuntimeError:
            continue
        if normalized.shape[-1] < 6:
            continue
        candidates.append((array, normalized.shape))

    if not candidates:
        raise RuntimeError(
            "OpenVINO detector produced no detection-like tensor outputs. "
            f"Received shapes: {output_shapes}."
        )

    def score(item: tuple[np.ndarray, tuple[int, int, int]]) -> tuple[int, int, int]:
        array, shape = item
        _, boxes, features = shape
        is_end2end = int(_is_end2end_tensor(array))
        return (is_end2end, boxes, -features)

    return max(candidates, key=score)[0]


def _decode_detections(
    prediction: np.ndarray, conf_thres: float, iou_thres: float
) -> np.ndarray:
    is_end2end = _is_end2end_tensor(np.asarray(prediction))
    normalized = _normalize_prediction_shape(prediction)
    batch = normalized[0]
    if is_end2end:
        detections = batch[batch[:, 4] > conf_thres]
        return detections[:, :6].astype(np.float32, copy=False)

    boxes_xywh = batch[:, :4]
    class_scores = batch[:, 4:]
    if class_scores.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = confidences > conf_thres
    if not np.any(keep):
        return np.empty((0, 6), dtype=np.float32)

    boxes = _xywh2xyxy(boxes_xywh[keep])
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    keep_indices = _batched_nms(boxes, confidences, class_ids, iou_thres)
    if keep_indices.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    selected_boxes = boxes[keep_indices]
    selected_confidences = confidences[keep_indices, None]
    selected_classes = class_ids[keep_indices, None].astype(np.float32)
    return np.concatenate(
        [selected_boxes, selected_confidences, selected_classes],
        axis=1,
        dtype=np.float32,
    )


def _normalize_prediction_shape(prediction: np.ndarray) -> np.ndarray:
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise RuntimeError(
            f"Unsupported OpenVINO output shape {array.shape}; expected a 2D or 3D detection tensor."
        )

    candidates = [array, array.transpose(0, 2, 1)]
    scored_candidates = []
    for candidate in candidates:
        _, boxes, features = candidate.shape
        if not 6 <= features <= 512:
            continue
        scored_candidates.append(
            (
                int(boxes >= features),
                boxes,
                -features,
                candidate,
            )
        )

    if scored_candidates:
        return max(scored_candidates, key=lambda item: item[:3])[3]

    return array


def _is_end2end_tensor(prediction: np.ndarray) -> bool:
    if prediction.ndim != 3:
        return False
    _, boxes, features = prediction.shape
    return boxes <= 300 and 6 <= features <= 512


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thres: float
) -> np.ndarray:
    keep: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        if class_indices.size == 0:
            continue
        class_keep = _nms(boxes[class_indices], scores[class_indices], iou_thres)
        keep.extend(class_indices[class_keep].tolist())

    if not keep:
        return np.empty((0,), dtype=np.int64)

    ordered = np.array(keep, dtype=np.int64)
    scores_view = scores[ordered]
    return ordered[np.argsort(scores_view)[::-1]]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= iou_thres]

    return np.array(keep, dtype=np.int64)


def _scale_boxes(
    input_shape: tuple[int, int],
    boxes: np.ndarray,
    original_shape: tuple[int, int],
) -> np.ndarray:
    gain = min(
        input_shape[0] / original_shape[0],
        input_shape[1] / original_shape[1],
    )
    pad_x = round((input_shape[1] - round(original_shape[1] * gain)) / 2 - 0.1)
    pad_y = round((input_shape[0] - round(original_shape[0] * gain)) / 2 - 0.1)

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
    return boxes


def _letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    shape = img.shape[:2]
    if shape == new_shape:
        return img

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=_LETTERBOX_PAD_VALUE,
    )


def _make_divisible(value: int, divisor: int) -> int:
    return int(np.ceil(value / divisor) * divisor)


def _label_for_index(class_names: tuple[str, ...], class_index: int) -> str:
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return str(class_index)
