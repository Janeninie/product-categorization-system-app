"""Lightweight drift-computation utilities used by backend orchestrator."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

WINDOW_SIZE = 100
BATCH_SIZE = 32

# Tuned for the lightweight drift scores below.
EMBEDDING_THRESHOLD = 0.08
CONFIDENCE_THRESHOLD = 0.20
CLASS_THRESHOLD = 0.30


class _NoOpModel:
    def to(self, _device: Any) -> "_NoOpModel":
        return self


def build_model(num_classes: int) -> _NoOpModel:
    _ = num_classes
    return _NoOpModel()


def build_transform(input_size: int, mean: list[float], std: list[float]) -> dict[str, Any]:
    return {
        "input_size": int(input_size),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }


def load_reference_stats() -> dict[str, Any]:
    return {
        "model_info": {
            "classes": ["beverage", "snack"],
            "input_size": 224,
            "norm_mean": [0.485, 0.456, 0.406],
            "norm_std": [0.229, 0.224, 0.225],
        },
        "confidence_mean": 0.8,
        "confidence_std": 0.1,
        "class_distribution": {
            "beverage": 0.5,
            "snack": 0.5,
        },
        "embedding_mean": [0.5, 0.5, 0.5, 0.2, 0.2, 0.2],
    }


def load_reference_embedding_mean(ref_stats: dict[str, Any]) -> np.ndarray:
    raw = ref_stats.get("embedding_mean", [0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
    emb = np.asarray(raw, dtype=np.float32)
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    return emb


def _decode_data_url_image(image_data_url: str) -> Image.Image:
    if "," not in image_data_url:
        raise ValueError("invalid data URL")

    _, payload = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(payload)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def _image_to_embedding(image: Image.Image, input_size: int) -> np.ndarray:
    img = image.resize((input_size, input_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    channel_mean = arr.mean(axis=(0, 1))
    channel_std = arr.std(axis=(0, 1))
    embedding = np.concatenate([channel_mean, channel_std]).astype(np.float32)
    return embedding


def infer_recent_embeddings(
    model: _NoOpModel,
    image_data_urls: list[str],
    transform: dict[str, Any],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    _ = model, device, batch_size

    input_size = int(transform.get("input_size", 224))
    embeddings: list[np.ndarray] = []

    for data_url in image_data_urls:
        try:
            image = _decode_data_url_image(data_url)
            embeddings.append(_image_to_embedding(image, input_size))
        except Exception:
            embeddings.append(np.zeros(6, dtype=np.float32))

    if not embeddings:
        return np.zeros((0, 6), dtype=np.float32)

    return np.vstack(embeddings)


def compute_embedding_drift(ref_mean: np.ndarray, recent_embeddings: np.ndarray) -> float:
    if recent_embeddings.size == 0:
        return 0.0

    current_mean = recent_embeddings.mean(axis=0)
    return float(np.linalg.norm(current_mean - ref_mean))


def compute_confidence_drift(ref_stats: dict[str, Any], recent_confidences: np.ndarray) -> float:
    if recent_confidences.size == 0:
        return 0.0

    ref_mean = float(ref_stats.get("confidence_mean", 0.8))
    ref_std = max(float(ref_stats.get("confidence_std", 0.1)), 1e-6)
    z = abs(float(recent_confidences.mean()) - ref_mean) / ref_std
    return float(z)


def compute_class_ratio_drift(ref_stats: dict[str, Any], recent_classes: list[str]) -> float:
    if not recent_classes:
        return 0.0

    ref_dist = ref_stats.get("class_distribution", {})
    classes = sorted(set(list(ref_dist.keys()) + recent_classes))

    n = float(len(recent_classes))
    score = 0.0
    for name in classes:
        ref_p = float(ref_dist.get(name, 0.0))
        cur_p = float(sum(1 for c in recent_classes if c == name) / n)
        score += abs(cur_p - ref_p)

    return float(score)
