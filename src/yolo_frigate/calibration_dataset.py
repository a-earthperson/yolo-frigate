from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import logging
import random
import shutil
import urllib.request
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is optional outside export runtimes.
    Image = None

logger = logging.getLogger(__name__)

_ANNOTATIONS_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
_BOXABLE_CLASS_URL = (
    "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
)
_CALIBRATION_MAX_SAMPLES = 512
_CALIBRATION_SEED = 0
_DATASET_DIRNAME = "open-images-v7-validation-yolo-v5"
_IMAGE_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
)
_IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 30
_IMAGE_DOWNLOAD_WORKERS = 8
_USER_AGENT = "yolo-frigate-open-images-bootstrap/1.0"


@dataclass(frozen=True)
class ClassIndex:
    display_lookup: dict[str, str]
    display_to_label: dict[str, str]
    label_to_display: dict[str, str]


@dataclass(frozen=True)
class Detection:
    label_name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    primary_url: str
    fallback_url: str | None
    rotation: int


def ensure_open_images_v7_validation_dataset(
    cache_root: Path,
    requested_classes: list[str] | tuple[str, ...] | None = None,
) -> Path:
    base_root = cache_root.expanduser().resolve() / "datasets" / _DATASET_DIRNAME
    metadata_root = base_root / "metadata"
    class_index = _load_open_images_boxable_class_index(metadata_root)
    selected_classes = _resolve_selected_classes(requested_classes, class_index)
    dataset_root = base_root / _selection_key(selected_classes)
    export_root = dataset_root / "yolo"
    data_yaml = export_root / "data.yaml"
    if data_yaml.is_file():
        return data_yaml

    dataset_root.mkdir(parents=True, exist_ok=True)
    with _locked_file(dataset_root / ".lock"):
        if data_yaml.is_file():
            return data_yaml

        staging_root = dataset_root / "yolo.tmp"
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        try:
            classes = _build_open_images_subset(
                metadata_root=metadata_root,
                export_root=staging_root,
                class_index=class_index,
                selected_classes=selected_classes,
            )
            _write_dataset_yaml(
                path=staging_root / "data.yaml",
                dataset_root=export_root,
                classes=classes,
            )
            if export_root.exists():
                shutil.rmtree(export_root)
            staging_root.rename(export_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise

    return export_root / "data.yaml"


def _build_open_images_subset(
    metadata_root: Path,
    export_root: Path,
    class_index: ClassIndex,
    selected_classes: list[str] | None,
) -> list[str]:
    logger.info(
        "Bootstrapping Open Images V7 validation detections for INT8 calibration under %s using classes=%s max_samples=%s.",
        export_root,
        selected_classes or "all boxable classes",
        _CALIBRATION_MAX_SAMPLES,
    )

    selected_label_ids = None
    if selected_classes:
        selected_label_ids = {
            class_index.display_to_label[class_name] for class_name in selected_classes
        }

    detections_by_image = _load_detections(metadata_root, selected_label_ids)
    candidate_image_ids = _shuffled_image_ids(detections_by_image)
    sampled_image_ids = candidate_image_ids[:_CALIBRATION_MAX_SAMPLES]
    image_records = _load_image_records(metadata_root, candidate_image_ids)
    classes = _resolve_dataset_classes(
        selected_classes=selected_classes,
        sampled_image_ids=sampled_image_ids,
        detections_by_image=detections_by_image,
        label_to_display=class_index.label_to_display,
    )
    _materialize_dataset(
        export_root=export_root,
        image_ids=candidate_image_ids,
        image_records=image_records,
        detections_by_image=detections_by_image,
        label_to_display=class_index.label_to_display,
        classes=classes,
        target_count=len(sampled_image_ids),
    )
    return classes


def _load_open_images_boxable_class_index(metadata_root: Path) -> ClassIndex:
    class_file = _ensure_downloaded(
        metadata_root / "oidv7-class-descriptions-boxable.csv",
        _BOXABLE_CLASS_URL,
    )

    display_lookup: dict[str, str] = {}
    display_to_label: dict[str, str] = {}
    label_to_display: dict[str, str] = {}
    with class_file.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            label_name = row.get("LabelName", "").strip()
            display_name = row.get("DisplayName", "").strip()
            if not label_name or not display_name:
                continue
            display_lookup[display_name.casefold()] = display_name
            display_to_label[display_name] = label_name
            label_to_display[label_name] = display_name
    return ClassIndex(
        display_lookup=display_lookup,
        display_to_label=display_to_label,
        label_to_display=label_to_display,
    )


def _resolve_selected_classes(
    requested_classes: list[str] | tuple[str, ...] | None,
    class_index: ClassIndex,
) -> list[str] | None:
    if not requested_classes:
        return None

    selected: list[str] = []
    unsupported: list[str] = []
    seen: set[str] = set()
    for raw_class in requested_classes:
        normalized = raw_class.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        resolved = class_index.display_lookup.get(key)
        if resolved is None:
            unsupported.append(normalized)
            continue
        selected.append(resolved)

    if unsupported:
        logger.warning(
            "Ignoring labelmap classes that are not available in Open Images V7 boxable detections: %s",
            ", ".join(unsupported),
        )
    if selected:
        return selected

    logger.warning(
        "No labelmap classes matched Open Images V7 boxable detections. Falling back to an unfiltered validation subset for INT8 calibration."
    )
    return None


def _load_detections(
    metadata_root: Path,
    selected_label_ids: set[str] | None,
) -> dict[str, list[Detection]]:
    annotation_file = _ensure_downloaded(
        metadata_root / "validation-annotations-bbox.csv",
        _ANNOTATIONS_URL,
    )

    detections_by_image: dict[str, list[Detection]] = defaultdict(list)
    with annotation_file.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            image_id = row.get("ImageID", "").strip()
            label_name = row.get("LabelName", "").strip()
            if not image_id or not label_name:
                continue
            if selected_label_ids is not None and label_name not in selected_label_ids:
                continue

            x_min = float(row["XMin"])
            x_max = float(row["XMax"])
            y_min = float(row["YMin"])
            y_max = float(row["YMax"])
            if x_max <= x_min or y_max <= y_min:
                continue

            detections_by_image[image_id].append(
                Detection(
                    label_name=label_name,
                    x_min=x_min,
                    x_max=x_max,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

    if not detections_by_image:
        raise ValueError("Open Images calibration bootstrap found no matching detections.")
    return dict(detections_by_image)


def _sample_image_ids(detections_by_image: dict[str, list[Detection]]) -> list[str]:
    return _shuffled_image_ids(detections_by_image)[:_CALIBRATION_MAX_SAMPLES]


def _shuffled_image_ids(detections_by_image: dict[str, list[Detection]]) -> list[str]:
    image_ids = sorted(detections_by_image)
    rng = random.Random(_CALIBRATION_SEED)
    rng.shuffle(image_ids)
    return image_ids


def _load_image_records(
    metadata_root: Path,
    image_ids: list[str],
) -> dict[str, ImageRecord]:
    image_metadata_file = _ensure_downloaded(
        metadata_root / "validation-images-with-rotation.csv",
        _IMAGE_METADATA_URL,
    )
    wanted = set(image_ids)
    records: dict[str, ImageRecord] = {}
    with image_metadata_file.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            image_id = row.get("ImageID", "").strip()
            if image_id not in wanted:
                continue
            thumbnail_url = row.get("Thumbnail300KURL", "").strip()
            original_url = row.get("OriginalURL", "").strip()
            primary_url = thumbnail_url or original_url
            if not primary_url:
                continue
            records[image_id] = ImageRecord(
                image_id=image_id,
                primary_url=primary_url,
                fallback_url=original_url if thumbnail_url and original_url else None,
                rotation=_parse_rotation(row.get("Rotation")),
            )
    return records


def _resolve_dataset_classes(
    selected_classes: list[str] | None,
    sampled_image_ids: list[str],
    detections_by_image: dict[str, list[Detection]],
    label_to_display: dict[str, str],
) -> list[str]:
    if selected_classes:
        return list(selected_classes)

    observed = sorted(
        {
            label_to_display[detection.label_name]
            for image_id in sampled_image_ids
            for detection in detections_by_image.get(image_id, [])
            if detection.label_name in label_to_display
        }
    )
    if not observed:
        raise ValueError("No Open Images classes were observed in the sampled calibration subset.")
    return observed


def _materialize_dataset(
    export_root: Path,
    image_ids: list[str],
    image_records: dict[str, ImageRecord],
    detections_by_image: dict[str, list[Detection]],
    label_to_display: dict[str, str],
    classes: list[str],
    target_count: int | None = None,
) -> None:
    images_dir = export_root / "images" / "val"
    labels_dir = export_root / "labels" / "val"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    target_count = min(target_count or _CALIBRATION_MAX_SAMPLES, len(image_ids))
    written = 0
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=_IMAGE_DOWNLOAD_WORKERS) as executor:
        pending_ids = iter(image_ids)
        futures = {}

        def submit_next() -> bool:
            try:
                image_id = next(pending_ids)
            except StopIteration:
                return False
            futures[
                executor.submit(
                    _write_sample,
                    image_id=image_id,
                    record=image_records.get(image_id),
                    detections=detections_by_image.get(image_id, []),
                    label_to_display=label_to_display,
                    class_to_index=class_to_index,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                )
            ] = image_id
            return True

        for _ in range(min(_IMAGE_DOWNLOAD_WORKERS, len(image_ids))):
            submit_next()

        while futures and written < target_count:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                image_id = futures.pop(future)
                try:
                    if future.result():
                        written += 1
                except Exception as exc:
                    failures.append((image_id, str(exc)))
                    logger.debug(
                        "Skipping Open Images calibration sample %s after download failure: %s",
                        image_id,
                        exc,
                    )
                if written < target_count:
                    submit_next()

    if failures:
        preview = "; ".join(
            f"{image_id}: {reason}" for image_id, reason in failures[:3]
        )
        if len(failures) > 3:
            preview += f"; plus {len(failures) - 3} more"
        logger.warning(
            "Skipped %s Open Images calibration samples while materializing %s/%s images. Examples: %s",
            len(failures),
            written,
            target_count,
            preview,
        )

    if written < 1:
        raise ValueError("Failed to materialize any Open Images calibration samples.")


def _write_sample(
    image_id: str,
    record: ImageRecord | None,
    detections: list[Detection],
    label_to_display: dict[str, str],
    class_to_index: dict[str, int],
    images_dir: Path,
    labels_dir: Path,
) -> bool:
    if record is None:
        logger.warning(
            "Skipping Open Images calibration sample %s because image metadata is missing.",
            image_id,
        )
        return False

    image = _download_image(record)
    image_path = images_dir / f"{image_id}.jpg"
    label_path = labels_dir / f"{image_id}.txt"

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("OpenCV failed to encode the downloaded image as JPEG.")
    image_path.write_bytes(encoded.tobytes())

    lines: list[str] = []
    for detection in detections:
        label_line = _format_label_line(
            detection=detection,
            rotation=record.rotation,
            label_to_display=label_to_display,
            class_to_index=class_to_index,
        )
        if label_line is not None:
            lines.append(label_line)

    label_contents = "\n".join(lines)
    if label_contents:
        label_contents += "\n"
    label_path.write_text(label_contents, encoding="utf-8")
    return True


def _format_label_line(
    detection: Detection,
    rotation: int,
    label_to_display: dict[str, str],
    class_to_index: dict[str, int],
) -> str | None:
    class_name = label_to_display.get(detection.label_name)
    if class_name not in class_to_index:
        return None

    polygon = _rotate_polygon(
        [
            (detection.x_min, detection.y_min),
            (detection.x_max, detection.y_min),
            (detection.x_max, detection.y_max),
            (detection.x_min, detection.y_max),
        ],
        rotation,
    )
    clipped = [_clip_point(x, y) for x, y in polygon]
    xs = [x for x, _ in clipped]
    ys = [y for _, y in clipped]
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        return None

    coordinates = " ".join(f"{value:.6f}" for point in clipped for value in point)
    return f"{class_to_index[class_name]} {coordinates}"


def _download_image(record: ImageRecord) -> np.ndarray:
    failures: list[str] = []
    for url in [record.primary_url, record.fallback_url]:
        if not url:
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(
                request, timeout=_IMAGE_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                payload = response.read()
            image = _decode_image_payload(payload)
            if image is None:
                failures.append(f"{url} returned an unsupported image payload")
                continue
            return _rotate_image(image, record.rotation)
        except Exception as exc:
            failures.append(f"{url}: {exc}")
            continue
    preview = "; ".join(failures[:2]) if failures else "no image URLs were available"
    if len(failures) > 2:
        preview += f"; plus {len(failures) - 2} more"
    raise ValueError(f"Failed to download/decode Open Images sample: {preview}")


def _decode_image_payload(payload: bytes) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is not None or Image is None:
        return image

    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            rgb = decoded.convert("RGB")
            array = np.asarray(rgb)
    except Exception:
        return None
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _rotate_box(detection: Detection, rotation: int) -> tuple[float, float, float, float]:
    points = [
        (detection.x_min, detection.y_min),
        (detection.x_min, detection.y_max),
        (detection.x_max, detection.y_min),
        (detection.x_max, detection.y_max),
    ]
    rotated_points = [_rotate_point(x, y, rotation) for x, y in points]
    xs = [point[0] for point in rotated_points]
    ys = [point[1] for point in rotated_points]
    return min(xs), max(xs), min(ys), max(ys)


def _rotate_polygon(
    points: list[tuple[float, float]],
    rotation: int,
) -> list[tuple[float, float]]:
    return [_rotate_point(x, y, rotation) for x, y in points]


def _rotate_point(x: float, y: float, rotation: int) -> tuple[float, float]:
    if rotation == 90:
        return y, 1.0 - x
    if rotation == 180:
        return 1.0 - x, 1.0 - y
    if rotation == 270:
        return 1.0 - y, x
    return x, y


def _parse_rotation(value: str | None) -> int:
    if not value:
        return 0
    try:
        rotation = int(float(value))
    except ValueError:
        return 0
    return rotation if rotation in {0, 90, 180, 270} else 0


def _clip_point(x: float, y: float) -> tuple[float, float]:
    return (
        min(max(x, 0.0), 1.0),
        min(max(y, 0.0), 1.0),
    )


def _selection_key(selected_classes: list[str] | None) -> str:
    if not selected_classes:
        return "unfiltered"
    digest = hashlib.sha256("\n".join(selected_classes).encode("utf-8")).hexdigest()
    return digest[:16]


def _ensure_downloaded(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urllib.request.urlretrieve(url, path)
    return path


def _write_dataset_yaml(path: Path, dataset_root: Path, classes: list[str]) -> None:
    payload = {
        "path": str(dataset_root),
        "train": "images/val",
        "val": "images/val",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@contextlib.contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
