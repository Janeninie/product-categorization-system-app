"""FastAPI application for Smart Product Categorization System."""
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from database import (
    init_db,
    get_db,
    PredictionEvent,
    HumanFeedback,
    SessionLocal,
)
from ml_model import build_model, ProductClassifier
from quality import analyze_quality
from schemas import (
    PredictionResponse,
    HistoryResponse,
    HistoryItem,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)


MODEL_PATH = Path(__file__).parent / "models" / "model.safetensors"
CLASS_LABELS = ["beverage", "snack"]
NUM_CLASSES = 2

classifier: Optional[ProductClassifier] = None

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at system startup."""
    global classifier

    init_db()

    try:
        model = build_model(
            name="efficientnet_b0",
            num_classes=NUM_CLASSES,
            freeze_backbone=False,
            dropout=0.3,
        )

        model_loaded = False
        if MODEL_PATH.exists():
            try:
                state_dict = safetensors.torch.load_file(str(MODEL_PATH), device="cpu")
                model.load_state_dict(state_dict, strict=False)
                model_loaded = True
                print(f"Model weights loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
                print("Using untrained model with random weights")

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
        print(f"Model initialized successfully (weights loaded: {model_loaded})")
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Process an image and return the prediction."""
    global classifier

    validate_image_format(file)

    try:
        contents = await file.read()
        pil_image = Image.open(file.file if hasattr(file, "file") else None)
        if hasattr(file, "file") and file.file:
            file.file.seek(0)
            pil_image = Image.open(file.file)
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
        prediction_event = PredictionEvent(
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            brightness=quality_metrics.brightness,
            blur_var=quality_metrics.blur_var,
            width=quality_metrics.width,
            height=quality_metrics.height,
            quality_warnings=json.dumps(quality_metrics.quality_warnings),
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
        db.execute("SELECT 1")
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
        prediction = db.query(PredictionEvent).filter(
            PredictionEvent.id == request.prediction_id
        ).first()

        if not prediction:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction with id {request.prediction_id} not found.",
            )

        feedback = HumanFeedback(
            prediction_id=request.prediction_id,
            true_label=request.true_label,
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
