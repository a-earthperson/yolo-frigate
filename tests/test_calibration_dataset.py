import tempfile
import unittest
import unittest.mock
from pathlib import Path

import cv2
import numpy as np
import yaml

import yolo_frigate.calibration_dataset as calibration_dataset
from yolo_frigate.calibration_dataset import (
    ClassIndex,
    Detection,
    ImageRecord,
    _download_image,
    _format_label_line,
    _materialize_dataset,
    _resolve_selected_classes,
    ensure_open_images_v7_validation_dataset,
)


class TestCalibrationDataset(unittest.TestCase):
    def test_resolve_selected_classes_warns_and_ignores_unknown_labels(self):
        class_index = ClassIndex(
            display_lookup={
                "person": "Person",
                "cat": "Cat",
                "box": "Box",
            },
            display_to_label={
                "Person": "/m/person",
                "Cat": "/m/cat",
                "Box": "/m/box",
            },
            label_to_display={
                "/m/person": "Person",
                "/m/cat": "Cat",
                "/m/box": "Box",
            },
        )

        with self.assertLogs("yolo_frigate.calibration_dataset", level="WARNING") as logs:
            selected = _resolve_selected_classes(
                ["person", "package", "box"],
                class_index,
            )

        self.assertEqual(selected, ["Person", "Box"])
        self.assertIn("Ignoring labelmap classes", "\n".join(logs.output))

    def test_open_images_validation_dataset_uses_requested_labelmap_classes(self):
        class_index = ClassIndex(
            display_lookup={
                "person": "Person",
                "cat": "Cat",
                "box": "Box",
            },
            display_to_label={
                "Person": "/m/person",
                "Cat": "/m/cat",
                "Box": "/m/box",
            },
            label_to_display={
                "/m/person": "Person",
                "/m/cat": "Cat",
                "/m/box": "Box",
            },
        )

        detections = {
            "img-1": [
                Detection("/m/person", 0.1, 0.4, 0.2, 0.7),
                Detection("/m/box", 0.5, 0.8, 0.4, 0.9),
            ]
        }
        image_records = {
            "img-1": ImageRecord(
                image_id="img-1",
                primary_url="https://example.invalid/img-1.jpg",
                fallback_url=None,
                rotation=0,
            )
        }

        def fake_materialize(**kwargs):
            export_root = kwargs["export_root"]
            (export_root / "images" / "val").mkdir(parents=True, exist_ok=True)
            (export_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
            (export_root / "images" / "val" / "img-1.jpg").write_bytes(b"jpg")
            (export_root / "labels" / "val" / "img-1.txt").write_text(
                "0 0.250000 0.450000 0.300000 0.500000\n",
                encoding="utf-8",
            )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._load_open_images_boxable_class_index",
                return_value=class_index,
            ),
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._load_detections",
                return_value=detections,
            ) as load_detections,
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._shuffled_image_ids",
                return_value=["img-1"],
            ),
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._load_image_records",
                return_value=image_records,
            ),
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._materialize_dataset",
                side_effect=fake_materialize,
            ) as materialize,
        ):
            first = ensure_open_images_v7_validation_dataset(
                Path(tmpdir), ["person", "box"]
            )
            second = ensure_open_images_v7_validation_dataset(
                Path(tmpdir), ["person", "box"]
            )

            payload = yaml.safe_load(first.read_text(encoding="utf-8"))
            self.assertEqual(payload["train"], "images/val")
            self.assertEqual(payload["val"], "images/val")
            self.assertEqual(payload["nc"], 2)
            self.assertEqual(payload["names"][0], "Person")
            self.assertEqual(payload["names"][1], "Box")

        self.assertEqual(first, second)
        load_detections.assert_called_once_with(
            (
                Path(tmpdir).resolve()
                / "datasets"
                / "open-images-v7-validation-yolo-v5"
                / "metadata"
            ),
            {"/m/person", "/m/box"},
        )
        materialize.assert_called_once()

    def test_format_label_line_emits_segmentation_polygon_from_bbox(self):
        line = _format_label_line(
            detection=Detection("/m/box", 0.1, 0.4, 0.2, 0.8),
            rotation=0,
            label_to_display={"/m/box": "Box"},
            class_to_index={"Box": 2},
        )

        self.assertEqual(
            line,
            "2 0.100000 0.200000 0.400000 0.200000 0.400000 0.800000 0.100000 0.800000",
        )

    @unittest.skipIf(
        calibration_dataset.Image is None,
        "Pillow is not installed in this environment.",
    )
    def test_download_image_falls_back_to_pillow_when_opencv_decode_fails(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = (0, 255, 0)
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        payload = encoded.tobytes()
        record = ImageRecord(
            image_id="img-1",
            primary_url="https://example.invalid/img-1.png",
            fallback_url=None,
            rotation=0,
        )

        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = payload

        with (
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset.urllib.request.urlopen",
                return_value=response,
            ),
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset.cv2.imdecode",
                return_value=None,
            ),
        ):
            decoded = _download_image(record)

        self.assertEqual(decoded.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(decoded[0, 0], image[0, 0]))

    def test_materialize_dataset_backfills_failed_downloads(self):
        image_ids = ["img-1", "img-2", "img-3"]

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._IMAGE_DOWNLOAD_WORKERS", 1
            ),
            unittest.mock.patch(
                "yolo_frigate.calibration_dataset._write_sample",
                side_effect=[ValueError("bad payload"), True, True],
            ) as write_sample,
            self.assertLogs("yolo_frigate.calibration_dataset", level="WARNING") as logs,
        ):
            _materialize_dataset(
                export_root=Path(tmpdir),
                image_ids=image_ids,
                image_records={},
                detections_by_image={},
                label_to_display={},
                classes=[],
                target_count=2,
            )

        self.assertEqual(write_sample.call_count, 3)
        self.assertIn(
            "Skipped 1 Open Images calibration samples while materializing 2/2 images",
            "\n".join(logs.output),
        )
