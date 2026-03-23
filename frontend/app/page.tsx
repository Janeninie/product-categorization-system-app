"use client";

import { useState, useCallback } from "react";
import { usePrediction, useHistory, useFeedback } from "@/lib/hooks";
import { PredictionResponse, HistoryItem } from "./api/types";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<string>("");
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error";
  } | null>(null);

  const {
    data: historyData,
    refetch: refetchHistory,
    isLoading: isHistoryLoading,
  } = useHistory(20, 0);
  const feedbackMutation = useFeedback();

  const predictionMutation = usePrediction();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setSelectedLabel("");
      predictionMutation.reset();
      feedbackMutation.reset();
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    try {
      feedbackMutation.reset();
      const result = await predictionMutation.mutateAsync(file);
      showToast("Prediction completed successfully!", "success");
      refetchHistory();
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Prediction failed";
      showToast(message, "error");
    }
  };

  const handleFeedbackSubmit = async () => {
    const prediction = predictionMutation.data;
    if (!prediction || !selectedLabel) return;

    try {
      await feedbackMutation.mutateAsync({
        prediction_id: prediction.prediction_id,
        true_label: selectedLabel as "beverages" | "snacks",
      });
      showToast("Feedback submitted successfully!", "success");
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Failed to submit feedback";
      showToast(message, "error");
    }
  };

  const showToast = (message: string, type: "success" | "error") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const formatConfidence = (conf: number) => (conf * 100).toFixed(1);
  const formatTime = (timestamp: string) =>
    new Date(timestamp).toLocaleString();
  const formatLatency = (latency: number | null) =>
    latency == null ? "-" : `${latency.toFixed(2)} ms`;
  const formatImageSize = (width: number | null, height: number | null) =>
    width == null || height == null ? "-" : `${width} x ${height}`;
  const formatMetric = (value: number | null) =>
    value == null ? "-" : value.toFixed(2);

  const prediction = predictionMutation.data;
  const history = historyData?.predictions ?? [];
  const feedbackSaved = feedbackMutation.isSuccess;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-5xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Smart Product Categorization
          </h1>
          <p className="text-gray-600 mt-1">
            Classify products as beverages or snacks using ML
          </p>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8">
        {toast && (
          <div
            className={`fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 ${
              toast.type === "success"
                ? "bg-green-500 text-white"
                : "bg-red-500 text-white"
            }`}
          >
            {toast.message}
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
              <input
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Preview"
                    className="max-h-48 rounded-lg mb-4"
                  />
                ) : (
                  <svg
                    className="w-12 h-12 text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                )}
                <span className="text-gray-600">
                  {file ? file.name : "Click to select an image (JPG/PNG)"}
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={!file || predictionMutation.isPending}
              className={`w-full py-3 px-6 rounded-lg font-medium text-white transition-colors ${
                !file || predictionMutation.isPending
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700"
              }`}
            >
              {predictionMutation.isPending
                ? "Processing..."
                : "Classify Image"}
            </button>
          </form>
        </div>

        {predictionMutation.isError && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4">
            <p className="text-red-800 font-medium">Error</p>
            <p className="text-red-600">
              {predictionMutation.error?.message || "An error occurred"}
            </p>
          </div>
        )}

        {prediction && (
          <div className="space-y-4">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-semibold mb-4">Prediction Result</h2>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Predicted Class</p>
                  <p className="text-3xl font-bold text-blue-600 capitalize">
                    {prediction.predicted_class}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Confidence</p>
                  <p className="text-3xl font-bold text-green-600">
                    {formatConfidence(prediction.confidence)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Latency</p>
                  <p className="text-lg">
                    {prediction.latency_ms.toFixed(2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Image Size</p>
                  <p className="text-lg">
                    {prediction.width} x {prediction.height}
                  </p>
                </div>
              </div>

              {prediction.quality_warnings.length > 0 && (
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm text-gray-500 mb-2">
                    Quality Warnings:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {prediction.quality_warnings.map((warning) => (
                      <span
                        key={warning}
                        className="px-3 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm"
                      >
                        {warning.replace("_", " ")}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {prediction.low_confidence_flag && (
              <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
                <div className="flex items-center mb-4">
                  <svg
                    className="w-6 h-6 text-red-600 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                  <h3 className="text-lg font-bold text-red-800">
                    Human Verification Required
                  </h3>
                </div>
                <p className="text-red-700 mb-4">
                  Confidence is below 60%. Please verify the correct category.
                </p>

                {!feedbackSaved ? (
                  <div className="flex gap-4 items-center">
                    <select
                      value={selectedLabel}
                      onChange={(e) => setSelectedLabel(e.target.value)}
                      className="px-4 py-2 border border-red-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500"
                    >
                      <option value="">Select true label</option>
                      <option value="beverages">Beverages</option>
                      <option value="snacks">Snacks</option>
                    </select>
                    <button
                      onClick={handleFeedbackSubmit}
                      disabled={!selectedLabel || feedbackMutation.isPending}
                      className={`px-6 py-2 rounded-lg font-medium text-white transition-colors ${
                        !selectedLabel || feedbackMutation.isPending
                          ? "bg-gray-400 cursor-not-allowed"
                          : "bg-red-600 hover:bg-red-700"
                      }`}
                    >
                      {feedbackMutation.isPending
                        ? "Submitting..."
                        : "Submit Feedback"}
                    </button>
                  </div>
                ) : (
                  <p className="text-green-700 font-medium">
                    Feedback saved successfully!
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        <div className="bg-white rounded-xl shadow-sm p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Prediction History</h2>
            <button
              onClick={() => refetchHistory()}
              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              Refresh
            </button>
          </div>

          {isHistoryLoading ? (
            <p className="text-gray-500 text-center py-8">Loading history...</p>
          ) : history.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No predictions yet</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-semibold text-gray-600">
                      Image
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-600">
                      Time
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-600">
                      Class
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-600">
                      Confidence
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-600">
                      Latency
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-600">
                      Brightness
                    </th>
                    <th className="text-right py-3 px-4 font-semibold text-gray-600">
                      Blur Var
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-600">
                      Warnings
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((item: HistoryItem) => (
                    <tr
                      key={item.id}
                      className="border-b border-gray-100 hover:bg-gray-50"
                    >
                      <td className="py-3 px-4">
                        {item.image_data_url ? (
                          <img
                            src={item.image_data_url}
                            alt={`Prediction ${item.id}`}
                            className="h-14 w-14 rounded-lg object-cover border border-gray-200"
                          />
                        ) : (
                          <div className="h-14 w-14 rounded-lg border border-gray-200 bg-gray-100" />
                        )}
                      </td>
                      <td className="py-3 px-4 text-gray-600">
                        {formatTime(item.timestamp)}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium ${
                            item.predicted_class === "beverage"
                              ? "bg-blue-100 text-blue-800"
                              : "bg-green-100 text-green-800"
                          }`}
                        >
                          {item.predicted_class}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-medium">
                        {formatConfidence(item.confidence)}%
                      </td>
                      <td className="py-3 px-4 text-right text-gray-700">
                        {formatLatency(item.latency_ms)}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-700">
                        {formatMetric(item.brightness)}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-700">
                        {formatMetric(item.blur_var)}
                      </td>
                      <td className="py-3 px-4">
                        {item.quality_warnings.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {item.quality_warnings.map((warning) => (
                              <span
                                key={`${item.id}-${warning}`}
                                className="px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-full text-xs"
                              >
                                {warning.replace("_", " ")}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
