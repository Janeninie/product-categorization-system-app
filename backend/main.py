"""FastAPI application for Smart Product Categorization System."""

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import torchvision.transforms as transforms
from database import (
    HumanFeedback,
    PredictionEvent,
    SessionLocal,
    init_db,
)
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ml_model import build_model
from orchestrator import run_orchestrator_from_db
from PIL import Image
from quality import analyze_quality
from sqlalchemy import text
from schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    PredictionResponse,
)

MODEL_PATH = Path(__file__).parent / "models" / "model.safetensors"
CLASS_LABELS = ["beverage", "snack"]
NUM_CLASSES = 2
MODEL_NAME = os.getenv("MODEL_NAME", "").strip().lower()

classifier: Any = None
logger = logging.getLogger(__name__)


def configure_app_logging() -> None:
    """Ensure app logs are always visible in terminal output."""
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


configure_app_logging()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def infer_model_name_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Infer primary model architecture hint from checkpoint parameter names."""
    keys = set(state_dict.keys())

    if any(k.startswith("conv_layers.") for k in keys):
        return "simple_cnn"
    if any(k.startswith("classifier.2.") for k in keys):
        return "efficientnet_b0"
    if any(k.startswith("_backbone.features.") and "layer_scale" in k for k in keys):
        return "convnext_tiny"
    if any(k.startswith("_backbone.fc.") for k in keys):
        return "resnet50"
    if any(k.startswith("_backbone.classifier.2.1.") for k in keys):
        return "convnext_tiny"
    if any(k.startswith("_backbone.classifier.3.") for k in keys):
        return "mobilenetv3_large"

    return "efficientnet_b0"


def get_model_candidates_from_state_dict(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Return ordered model candidates based on checkpoint key patterns."""
    keys = set(state_dict.keys())

    if any(k.startswith("_backbone.features.") and "layer_scale" in k for k in keys):
        return ["convnext_tiny", "convnext_small", "convnext_base"]
    if any(k.startswith("_backbone.classifier.2.1.") for k in keys):
        return ["convnext_tiny", "convnext_small", "convnext_base"]
    if any(k.startswith("_backbone.classifier.3.") for k in keys):
        return ["mobilenetv3_large"]
    if any(k.startswith("_backbone.fc.") for k in keys):
        return ["resnet50"]
    if any(k.startswith("conv_layers.") for k in keys):
        return ["simple_cnn"]
    if any(k.startswith("classifier.2.") for k in keys):
        return ["efficientnet_b0"]

    return [infer_model_name_from_state_dict(state_dict)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at system startup."""
    global classifier

    init_db()

    try:
        state_dict = None
        if MODEL_PATH.exists():
            state_dict = safetensors.torch.load_file(str(MODEL_PATH), device="cpu")

        selected_model_name = MODEL_NAME or "mobilenetv3_large"
        if state_dict is not None and not MODEL_NAME:
            candidates = get_model_candidates_from_state_dict(state_dict)
        elif MODEL_NAME:
            candidates = [MODEL_NAME]
        else:
            candidates = [selected_model_name]

        model = None
        last_error: Exception | None = None
        for candidate_name in candidates:
            try:
                candidate_model = build_model(
                    name=candidate_name,
                    num_classes=NUM_CLASSES,
                    freeze_backbone=False,
                    dropout=0.3,
                )
                if state_dict is not None:
                    candidate_model.load_state_dict(state_dict, strict=True)
                model = candidate_model
                selected_model_name = candidate_name
                break
            except Exception as e:
                last_error = e

        if model is None:
            raise RuntimeError(
                "Could not match checkpoint to any backend model candidates "
                f"{candidates}. Last error: {last_error}"
            )

        model_loaded = state_dict is not None
        if model_loaded:
            print(f"Model weights loaded from {MODEL_PATH}")

        model.eval()

        class LabeledClassifier:
            """Wrapper to add label mapping to the model."""

            def __init__(self, model, label_map, loaded):
                self.model = model
                self.label_map = label_map
                self.idx_to_class = {int(k): v for k, v in label_map.items()}
                self.loaded = loaded

            def predict(self, x):
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits, dim=-1)
                    conf, preds = torch.max(probs, dim=-1)
                pred_idx = preds.item()
                confidence = conf.item()
                predicted_class = self.idx_to_class.get(pred_idx, str(pred_idx))
                return predicted_class, confidence

        classifier = LabeledClassifier(
            model=model,
            label_map={str(i): label for i, label in enumerate(CLASS_LABELS)},
            loaded=model_loaded,
        )
        print(
            f"Model initialized successfully "
            f"(name={selected_model_name}, weights_loaded={model_loaded})"
        )
    except Exception as e:
        print(f"Error initializing model: {e}")
        classifier = None

    yield

    classifier = None


app = FastAPI(
    title="Smart Product Categorization System",
    description="ML-powered product categorization API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_image_format(file: UploadFile) -> None:
    """Validate that the uploaded file is JPG or PNG."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only JPG/PNG are allowed.",
        )


def run_orchestrator_background_task() -> None:
    """Run orchestrator in background and log lifecycle/errors."""
    started_at = time.perf_counter()
    logger.info("[orchestrator-bg] started")
    try:
        result = run_orchestrator_from_db()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "[orchestrator-bg] finished in %.2f ms with result=%s",
            elapsed_ms,
            result,
        )
    except Exception:
        logger.exception("[orchestrator-bg] failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Process an image and return the prediction."""
    global classifier

    validate_image_format(file)

    try:
        contents = await file.read()
        if not contents:
            raise ValueError("empty file")
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    quality_metrics = analyze_quality(pil_image)

    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()

    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        predicted_class, confidence = classifier.predict(input_tensor)

    latency_ms = (time.perf_counter() - start_time) * 1000

    low_confidence_flag = confidence < 0.6

    db = SessionLocal()
    try:
        image_data_url = (
            f"data:{file.content_type};base64,"
            f"{base64.b64encode(contents).decode('ascii')}"
        )

        prediction_event = PredictionEvent(
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            brightness=quality_metrics.brightness,
            blur_var=quality_metrics.blur_var,
            width=quality_metrics.width,
            height=quality_metrics.height,
            quality_warnings=json.dumps(quality_metrics.quality_warnings),
            image_data_url=image_data_url,
        )
        db.add(prediction_event)
        db.commit()
        db.refresh(prediction_event)
        prediction_id = prediction_event.id
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

    logger.info("[orchestrator-bg] queued for prediction_id=%s", prediction_id)
    background_tasks.add_task(run_orchestrator_background_task)

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        latency_ms=latency_ms,
        low_confidence_flag=low_confidence_flag,
        brightness=quality_metrics.brightness,
        blur_var=quality_metrics.blur_var,
        width=quality_metrics.width,
        height=quality_metrics.height,
        quality_warnings=quality_metrics.quality_warnings,
        prediction_id=prediction_id,
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Get the history of prediction events."""
    db = SessionLocal()
    try:
        total = db.query(PredictionEvent).count()
        predictions = (
            db.query(PredictionEvent)
            .order_by(PredictionEvent.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        history_items = []
        for p in predictions:
            warnings = json.loads(p.quality_warnings) if p.quality_warnings else []
            history_items.append(
                HistoryItem(
                    id=p.id,
                    timestamp=p.timestamp,
                    predicted_class=p.predicted_class,
                    confidence=p.confidence,
                    latency_ms=p.latency_ms,
                    brightness=p.brightness,
                    blur_var=p.blur_var,
                    width=p.width,
                    height=p.height,
                    quality_warnings=warnings,
                    image_data_url=p.image_data_url,
                )
            )

        return HistoryResponse(
            predictions=history_items,
            total=total,
            limit=limit,
            offset=offset,
        )
    finally:
        db.close()


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - must respond within 1 second."""
    db_connected = False
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        db_connected=db_connected,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit human feedback for a low-confidence prediction."""
    db = SessionLocal()
    try:
        normalized_true_label = {
            "beverages": "beverage",
            "snacks": "snack",
        }.get(request.true_label, request.true_label)

        prediction = (
            db.query(PredictionEvent)
            .filter(PredictionEvent.id == request.prediction_id)
            .first()
        )

        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with id {request.prediction_id} not found.",
            )

        feedback = HumanFeedback(
            prediction_id=request.prediction_id,
            true_label=normalized_true_label,
        )
        db.add(feedback)
        db.commit()

        return FeedbackResponse(saved=True)
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
