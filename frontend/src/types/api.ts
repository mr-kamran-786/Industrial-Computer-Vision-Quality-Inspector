export type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  severity?: string | null;
  threshold?: number | null;
  bbox: BoundingBox;
};

export type Traceability = {
  serial_number?: string | null;
  batch_lot?: string | null;
  production_line?: string | null;
  station_id?: string | null;
  operator_id?: string | null;
  shift?: string | null;
};

export type PredictionResponse = {
  inspection_id?: string;
  filename: string;
  model_key: string;
  model_name: string;
  model_version: string;
  total_detections: number;
  is_defective: boolean;
  quality_status: string;
  highest_severity?: string | null;
  decision_reason?: string | null;
  inference_ms?: number | null;
  detections: Detection[];
  traceability?: Traceability;
  result_image_url?: string;
};

export type DefectCount = {
  class_name: string;
  count: number;
};

export type SeverityCount = {
  severity: string;
  count: number;
};

export type RecentFailure = {
  inspection_id: string;
  original_filename: string;
  total_detections: number;
  highest_severity?: string | null;
  result_image_url: string;
  created_at: string;
};

export type AnalyticsSummary = {
  total_inspections: number;
  pass_count: number;
  fail_count: number;
  defect_rate_percent: number;
  yield_percent: number;
  dppm: number;
  total_detected_defects: number;
  average_confidence: number | null;
  most_common_defect: string | null;
  defect_distribution: DefectCount[];
  severity_distribution: SeverityCount[];
  reviewed_count: number;
  escape_rate_percent: number | null;
  false_call_rate_percent: number | null;
  agreement_rate_percent: number | null;
  recent_failures: RecentFailure[];
};

export type TrendPoint = {
  date: string;
  total: number;
  pass_count: number;
  fail_count: number;
  defect_rate_percent: number;
  dppm: number;
};

export type AnalyticsTrend = {
  days: number;
  points: TrendPoint[];
};

export type InspectionSummary = {
  inspection_id: string;
  original_filename: string;
  quality_status: string;
  is_defective: boolean;
  total_detections: number;
  highest_severity?: string | null;
  serial_number?: string | null;
  production_line?: string | null;
  reviewed: boolean;
  review_is_correct?: boolean | null;
  result_image_url: string;
  created_at: string;
};

export type InspectionDetail = {
  inspection_id: string;
  original_filename: string;
  saved_filename: string;
  result_filename: string;
  result_image_url: string;
  quality_status: string;
  is_defective: boolean;
  total_detections: number;
  highest_severity?: string | null;
  decision_reason?: string | null;
  model_key: string;
  model_name: string;
  model_version: string;
  inference_ms?: number | null;
  traceability?: Traceability;
  reviewed: boolean;
  review_verdict?: string | null;
  review_is_correct?: boolean | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  review_notes?: string | null;
  created_at: string;
  detections: Detection[];
};

export type ReviewRequest = {
  verdict: "PASS" | "FAIL";
  reviewed_by: string;
  notes?: string;
};

export type ReviewResponse = {
  inspection_id: string;
  quality_status: string;
  review_verdict: string;
  review_is_correct: boolean;
  reviewed_by: string;
  reviewed_at: string;
  review_notes?: string | null;
};

export type ModelOption = {
  key: string;
  name: string;
  mode: string;
  version?: string;
  description: string;
  available: boolean;
  path: string;
};

export type ModelInfoResponse = {
  model_type: string;
  task: string;
  default_model_key: string;
  classes: string[];
  class_severity?: Record<string, string>;
  per_class_thresholds?: Record<string, number>;
  fail_severities?: string[];
  models: ModelOption[];
};
