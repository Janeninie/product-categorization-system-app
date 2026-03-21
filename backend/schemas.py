"""Pydantic schemas for API request/response validation."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for POST /predict endpoint."""
    predicted_class: str
    confidence: float
    latency_ms: float
    low_confidence_flag: bool
    brightness: float
    blur_var: float
    width: int
    height: int
    quality_warnings: List[str]
    prediction_id: int


class HistoryItem(BaseModel):
    """Schema for a single history item."""
    id: int
    timestamp: datetime
    predicted_class: str
    confidence: float
    latency_ms: Optional[float]
    brightness: Optional[float]
    blur_var: Optional[float]
    width: Optional[int]
    height: Optional[int]
    quality_warnings: List[str]
    image_data_url: Optional[str]

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """Response schema for GET /history endpoint."""
    predictions: List[HistoryItem]
    total: int
    limit: int
    offset: int


class FeedbackRequest(BaseModel):
    """Request schema for POST /feedback endpoint."""
    prediction_id: int = Field(..., gt=0)
    true_label: str = Field(..., pattern="^(beverage|snack|beverages|snacks)$")


class FeedbackResponse(BaseModel):
    """Response schema for POST /feedback endpoint."""
    saved: bool


class HealthResponse(BaseModel):
    """Response schema for GET /healthz endpoint."""
    status: str
    model_loaded: bool
    db_connected: bool


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str
