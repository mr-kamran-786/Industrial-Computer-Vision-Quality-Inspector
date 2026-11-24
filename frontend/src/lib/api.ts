import {
  AnalyticsSummary,
  AnalyticsTrend,
  InspectionDetail,
  InspectionSummary,
  ModelInfoResponse,
  PredictionResponse,
  ReviewRequest,
  ReviewResponse,
  Traceability,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

// Optional API key for protected endpoints (predict, review). When the backend
// runs with auth disabled this can be left unset.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

function authHeaders(): Record<string, string> {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

export function getImageUrl(path?: string): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE_URL}${path}`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${path}`);
  }
  return response.json();
}

export function fetchModelInfo(): Promise<ModelInfoResponse> {
  return getJson<ModelInfoResponse>("/api/model/info");
}

export function fetchAnalytics(): Promise<AnalyticsSummary> {
  return getJson<AnalyticsSummary>("/api/analytics/summary");
}

export function fetchAnalyticsTrend(days = 14): Promise<AnalyticsTrend> {
  return getJson<AnalyticsTrend>(`/api/analytics/trend?days=${days}`);
}

export function fetchInspections(limit = 20): Promise<InspectionSummary[]> {
  return getJson<InspectionSummary[]>(`/api/inspections?limit=${limit}`);
}

export function fetchInspectionDetail(id: string): Promise<InspectionDetail> {
  return getJson<InspectionDetail>(`/api/inspections/${id}`);
}

export async function uploadInspectionImage(
  file: File,
  modelKey: string,
  traceability?: Traceability
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_key", modelKey);

  if (traceability) {
    for (const [key, value] of Object.entries(traceability)) {
      if (value) formData.append(key, String(value));
    }
  }

  const response = await fetch(`${API_BASE_URL}/api/predict/image`, {
    method: "POST",
    body: formData,
    headers: authHeaders(),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Image inspection failed.");
  }

  return response.json();
}

export async function submitReview(
  inspectionId: string,
  review: ReviewRequest
): Promise<ReviewResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/inspections/${inspectionId}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(review),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Failed to submit review.");
  }

  return response.json();
}
