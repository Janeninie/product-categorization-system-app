import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { predictionApi } from "./api";
import { FeedbackRequest, PredictionResponse } from "@/app/api/types";

export const usePrediction = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => predictionApi.predict(file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
    },
  });
};

export const useHistory = (limit = 20, offset = 0) => {
  return useQuery({
    queryKey: ["history", { limit, offset }],
    queryFn: () => predictionApi.getHistory(limit, offset),
    staleTime: 1000 * 30,
    refetchOnWindowFocus: false,
  });
};

export const useFeedback = () => {
  return useMutation({
    mutationFn: (feedback: FeedbackRequest) => predictionApi.submitFeedback(feedback),
  });
};

export const useHealth = () => {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => predictionApi.getHealth(),
    staleTime: 1000 * 60,
    refetchInterval: 1000 * 60,
  });
};
