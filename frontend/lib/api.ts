import axios from "axios";
import {
  PredictionResponse,
  HistoryResponse,
  FeedbackRequest,
  FeedbackResponse,
  HealthResponse,
} from "@/app/api/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

export const predictionApi = {
  predict: async (file: File): Promise<PredictionResponse> => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await apiClient.post<PredictionResponse>("/predict", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  getHistory: async (limit = 20, offset = 0): Promise<HistoryResponse> => {
    const response = await apiClient.get<HistoryResponse>("/history", {
      params: { limit, offset },
    });
    return response.data;
  },

  submitFeedback: async (feedback: FeedbackRequest): Promise<FeedbackResponse> => {
    const response = await apiClient.post<FeedbackResponse>("/feedback", feedback);
    return response.data;
  },

  getHealth: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>("/healthz");
    return response.data;
  },
};
