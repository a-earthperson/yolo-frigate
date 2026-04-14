from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException

from yolo_frigate.detector_backend import DetectorBackend
from yolo_frigate.prediction import Predictions
from yolo_frigate.prediction_saver import PredictionItem, PredictionSaver

logger = logging.getLogger(__name__)


def create_app(detector: DetectorBackend, prediction_saver: PredictionSaver) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(prediction_saver.process())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan)
    app.state.force_save = False

    @app.get("/")
    def root():
        logger.debug("Root endpoint accessed.")
        return {"message": "Hello World"}

    @app.get("/health")
    def health():
        return ""

    @app.post("/force_save/{state}")
    def set_force_save(state: bool):
        app.state.force_save = state
        logger.info("Force save set to: %s", app.state.force_save)
        return {"force_save": app.state.force_save}

    @app.post("/detect", response_model=Predictions)
    async def detect_objects(image: Annotated[bytes, File()]) -> Predictions:
        logger.debug("Detect endpoint accessed.")
        binary_content = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(binary_content, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image.")
            raise HTTPException(status_code=400, detail="Invalid image format")

        predictions = detector.detect(img)
        detected_labels = (
            ", ".join(
                f"{prediction.label} ({prediction.confidence:.3f})"
                for prediction in predictions.predictions
            )
            or "none"
        )
        logger.debug(
            "Detection completed. Found %s objects: %s",
            len(predictions.predictions),
            detected_labels,
        )
        prediction_item = PredictionItem(
            image=image,
            predictions=predictions,
            forced=app.state.force_save,
        )
        await prediction_saver.add_prediction(prediction_item)
        return predictions

    return app
