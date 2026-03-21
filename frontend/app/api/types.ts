"use client";

export interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  latency_ms: number;
  low_confidence_flag: boolean;
  brightness: number;
  blur_var: number;
  width: number;
  height: number;
  quality_warnings: string[];
  prediction_id: number;
}

export interface HistoryItem {
  id: number;
  timestamp: string;
  predicted_class: string;
  confidence: number;
  latency_ms: number | null;
  brightness: number | null;
  blur_var: number | null;
  width: number | null;
  height: number | null;
  quality_warnings: string[];
  image_data_url: string | null;
}

export interface HistoryResponse {
  predictions: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface FeedbackRequest {
  prediction_id: number;
  true_label: "beverages" | "snacks";
}

export interface FeedbackResponse {
  saved: boolean;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  db_connected: boolean;
}

export interface ErrorResponse {
  detail: string;
}
