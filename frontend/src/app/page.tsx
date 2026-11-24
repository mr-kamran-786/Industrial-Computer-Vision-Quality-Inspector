"use client";

import { useEffect, useMemo, useState } from "react";
import {
  fetchAnalytics,
  fetchAnalyticsTrend,
  fetchInspections,
  fetchModelInfo,
  getImageUrl,
  uploadInspectionImage,
} from "@/lib/api";
import { SeverityBreakdown, TrendChart, severityColor } from "@/components/charts";
import { ReviewControls } from "@/components/ReviewControls";
import {
  AnalyticsSummary,
  AnalyticsTrend,
  InspectionSummary,
  ModelInfoResponse,
  PredictionResponse,
  Traceability,
} from "@/types/api";

const EMPTY_TRACE: Traceability = {
  serial_number: "",
  batch_lot: "",
  production_line: "",
  station_id: "",
  operator_id: "",
  shift: "",
};

export default function Home() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [selectedModelKey, setSelectedModelKey] = useState("nano");
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [trend, setTrend] = useState<AnalyticsTrend | null>(null);
  const [inspections, setInspections] = useState<InspectionSummary[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [trace, setTrace] = useState<Traceability>(EMPTY_TRACE);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string>("");

  async function loadDashboardData() {
    try {
      setRefreshing(true);
      const [analyticsData, trendData, inspectionsData, modelData] =
        await Promise.all([
          fetchAnalytics(),
          fetchAnalyticsTrend(14),
          fetchInspections(),
          fetchModelInfo(),
        ]);
      setAnalytics(analyticsData);
      setTrend(trendData);
      setInspections(inspectionsData);
      setModelInfo(modelData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard.");
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadDashboardData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setPrediction(null);
    setError("");
    setPreviewUrl(URL.createObjectURL(file));
  }

  function updateTrace(key: keyof Traceability, value: string) {
    setTrace((prev) => ({ ...prev, [key]: value }));
  }

  async function handleUpload() {
    if (!selectedFile) {
      setError("Please select a PCB image first.");
      return;
    }
    try {
      setLoading(true);
      setError("");
      const result = await uploadInspectionImage(selectedFile, selectedModelKey, trace);
      setPrediction(result);
      await loadDashboardData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  const resultImageUrl = prediction?.result_image_url
    ? getImageUrl(prediction.result_image_url)
    : "";

  const lineStatus = useMemo(() => {
    if (!analytics) {
      return { label: "INITIALIZING", className: "statusNeutral", description: "Waiting for inspection data." };
    }
    if (analytics.defect_rate_percent >= 60) {
      return { label: "HIGH DEFECT RATE", className: "statusDanger", description: "Quality line requires attention." };
    }
    if (analytics.defect_rate_percent >= 25) {
      return { label: "MONITORING", className: "statusWarning", description: "Defects detected, monitor production quality." };
    }
    return { label: "STABLE", className: "statusSuccess", description: "Inspection line is operating normally." };
  }, [analytics]);

  const pct = (v: number | null | undefined) =>
    v === null || v === undefined ? "N/A" : `${v}%`;

  return (
    <main className="page">
      <div className="backgroundGrid" />

      <section className="topBar">
        <div className="brand">
          <div className="brandMark">N</div>
          <div>
            <strong>Neurals Vision</strong>
            <span>Industrial AI Quality System</span>
          </div>
        </div>
        <div className="topActions">
          <div className="apiPill">
            <span className="pulseDot" />
            API Connected
          </div>
          <button className="secondaryButton" onClick={loadDashboardData}>
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </section>

      <section className="hero">
        <div className="heroContent">
          <p className="eyebrow">AI-Powered PCB Inspection</p>
          <h1>Industrial Computer Vision Quality Inspector</h1>
          <p className="heroText">
            Detect PCB manufacturing defects with YOLO, apply an auditable
            severity-based PASS/FAIL policy, trace every unit, capture operator
            feedback, and monitor SPC-style quality trends.
          </p>
          <div className="heroBadges">
            <span>Severity Policy</span>
            <span>Full Traceability</span>
            <span>Operator Feedback</span>
            <span>SPC Analytics</span>
          </div>
        </div>

        <div className="lineStatusCard">
          <div className="cardGlow" />
          <p>Production Line Status</p>
          <strong className={lineStatus.className}>{lineStatus.label}</strong>
          <span>{lineStatus.description}</span>
          <div className="lineStatMini">
            <div>
              <small>Yield</small>
              <b>{pct(analytics?.yield_percent)}</b>
            </div>
            <div>
              <small>DPPM</small>
              <b>{analytics?.dppm ?? 0}</b>
            </div>
            <div>
              <small>Reviewed</small>
              <b>{analytics?.reviewed_count ?? 0}</b>
            </div>
          </div>
        </div>
      </section>

      {error && <div className="errorBox">{error}</div>}

      {/* Production KPIs */}
      <section className="cardsGrid">
        <MetricCard label="Total Inspections" value={analytics?.total_inspections ?? 0} hint="All scanned boards" />
        <MetricCard label="PASS" value={analytics?.pass_count ?? 0} hint="Accepted products" tone="success" />
        <MetricCard label="FAIL" value={analytics?.fail_count ?? 0} hint="Rejected products" tone="danger" />
        <MetricCard
          label="Defect Rate"
          value={pct(analytics?.defect_rate_percent)}
          hint="Failure percentage"
          tone={analytics && analytics.defect_rate_percent > 50 ? "danger" : "neutral"}
        />
        <MetricCard label="Yield" value={pct(analytics?.yield_percent)} hint="Pass percentage" tone="success" />
        <MetricCard label="DPPM" value={analytics?.dppm ?? 0} hint="Defective parts / million" />
      </section>

      {/* Feedback-quality KPIs (from operator review loop) */}
      <section className="cardsGrid feedbackGrid">
        <MetricCard label="Reviewed" value={analytics?.reviewed_count ?? 0} hint="Operator-verified inspections" />
        <MetricCard
          label="Escape Rate"
          value={pct(analytics?.escape_rate_percent)}
          hint="AI PASS but human FAIL"
          tone={analytics?.escape_rate_percent ? "danger" : "neutral"}
        />
        <MetricCard
          label="False-Call Rate"
          value={pct(analytics?.false_call_rate_percent)}
          hint="AI FAIL but human PASS"
          tone={analytics?.false_call_rate_percent ? "danger" : "neutral"}
        />
        <MetricCard
          label="Agreement"
          value={pct(analytics?.agreement_rate_percent)}
          hint="AI matches operator"
          tone="success"
        />
      </section>

      <section className="mainGrid">
        <div className="panel uploadPanel">
          <PanelHeader
            title="Run New Inspection"
            subtitle="Upload a PCB image, tag it for traceability, and classify PASS/FAIL."
            tag="Input Station"
          />
          <div className="uploadBox">
            <div className="modelSelector">
              <div>
                <strong>Inspection Model</strong>
                <span>Select speed or accuracy mode</span>
              </div>
              <select value={selectedModelKey} onChange={(e) => setSelectedModelKey(e.target.value)}>
                {modelInfo?.models.map((model) => (
                  <option key={model.key} value={model.key} disabled={!model.available}>
                    {model.name} — {model.mode}
                    {!model.available ? " (missing file)" : ""}
                  </option>
                ))}
              </select>
            </div>

            <div className="traceGrid">
              <TraceInput label="Serial Number" value={trace.serial_number} onChange={(v) => updateTrace("serial_number", v)} />
              <TraceInput label="Batch / Lot" value={trace.batch_lot} onChange={(v) => updateTrace("batch_lot", v)} />
              <TraceInput label="Production Line" value={trace.production_line} onChange={(v) => updateTrace("production_line", v)} />
              <TraceInput label="Station ID" value={trace.station_id} onChange={(v) => updateTrace("station_id", v)} />
              <TraceInput label="Operator ID" value={trace.operator_id} onChange={(v) => updateTrace("operator_id", v)} />
              <TraceInput label="Shift" value={trace.shift} onChange={(v) => updateTrace("shift", v)} />
            </div>

            <label className="fileDrop">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <span className="fileIcon">⬡</span>
              <strong>{selectedFile ? selectedFile.name : "Choose PCB image"}</strong>
              <small>JPG, PNG, BMP, or WEBP supported</small>
            </label>

            {previewUrl && (
              <div className="imageBlock">
                <div className="imageHeader">
                  <p>Input Image</p>
                  <span>Ready for inspection</span>
                </div>
                <img src={previewUrl} alt="Selected PCB" />
              </div>
            )}

            <button className="primaryButton" onClick={handleUpload} disabled={loading}>
              {loading ? "Running AI Inspection..." : "Inspect Image"}
            </button>
          </div>
        </div>

        <div className="panel resultPanel">
          <PanelHeader
            title="Inspection Result"
            subtitle="Verdict, decision rationale, defect severity, and visual evidence."
            tag="Output Station"
          />
          {!prediction ? (
            <div className="emptyState">
              <div className="emptyIcon">◎</div>
              <strong>No inspection result yet</strong>
              <span>Upload an image to generate a quality decision.</span>
            </div>
          ) : (
            <div className="resultContent">
              <div className="decisionRow">
                <div className={prediction.quality_status === "PASS" ? "decisionPass" : "decisionFail"}>
                  {prediction.quality_status}
                </div>
                <div className="decisionMeta">
                  <span>Total defects</span>
                  <strong>{prediction.total_detections}</strong>
                  <small>{prediction.model_name} · {prediction.model_version}</small>
                </div>
              </div>

              <div className="resultMetaRow">
                {prediction.highest_severity && (
                  <span className="sevChip" style={{ borderColor: severityColor(prediction.highest_severity) }}>
                    <i className="dot" style={{ background: severityColor(prediction.highest_severity) }} />
                    {prediction.highest_severity}
                  </span>
                )}
                {prediction.inference_ms != null && (
                  <span className="metaChip">{prediction.inference_ms} ms</span>
                )}
                {prediction.inspection_id && (
                  <span className="metaChip mono">{prediction.inspection_id.slice(0, 10)}</span>
                )}
              </div>

              {prediction.decision_reason && (
                <div className="decisionReason">{prediction.decision_reason}</div>
              )}

              {prediction.inspection_id && (
                <ReviewControls
                  inspectionId={prediction.inspection_id}
                  aiVerdict={prediction.quality_status}
                  onReviewed={loadDashboardData}
                />
              )}

              {resultImageUrl && (
                <div className="imageBlock">
                  <div className="imageHeader">
                    <p>Annotated Result</p>
                    <a href={resultImageUrl} target="_blank" rel="noreferrer">Open Full Image</a>
                  </div>
                  <img src={resultImageUrl} alt="Annotated inspection result" />
                </div>
              )}

              <div className="detectionsList">
                {prediction.detections.length ? (
                  prediction.detections.map((detection, index) => (
                    <div
                      className="detectionItem"
                      key={index}
                      style={{ borderLeft: `3px solid ${severityColor(detection.severity)}` }}
                    >
                      <div className="detectionTop">
                        <strong>
                          {detection.class_name}
                          {detection.severity ? ` · ${detection.severity}` : ""}
                        </strong>
                        <span>{Math.round(detection.confidence * 100)}%</span>
                      </div>
                      <div className="confidenceTrack">
                        <div className="confidenceFill" style={{ width: `${Math.round(detection.confidence * 100)}%` }} />
                      </div>
                      <small>
                        Box: x1 {detection.bbox.x1}, y1 {detection.bbox.y1}, x2 {detection.bbox.x2}, y2 {detection.bbox.y2}
                      </small>
                    </div>
                  ))
                ) : (
                  <div className="emptyState small">No defects detected. Product marked as PASS.</div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* SPC trend + severity mix */}
      <section className="spcGrid">
        <div className="panel">
          <PanelHeader
            title="Defect Rate Control Chart"
            subtitle="Daily defect rate with mean center line and ±2σ control band (last 14 days)."
            tag="SPC / Trend"
          />
          <TrendChart points={trend?.points ?? []} />
        </div>
        <div className="panel">
          <PanelHeader
            title="Defect Severity Mix"
            subtitle="Share of detected defects by severity class."
            tag="Risk Profile"
          />
          <SeverityBreakdown items={analytics?.severity_distribution ?? []} />
        </div>
      </section>

      <section className="bottomGrid">
        <div className="panel">
          <PanelHeader
            title="Defect Distribution"
            subtitle={`Most common defect: ${analytics?.most_common_defect ?? "N/A"}`}
            tag="Quality Analytics"
          />
          <div className="barList">
            {analytics?.defect_distribution.length ? (
              analytics.defect_distribution.map((item) => (
                <div className="barRow" key={item.class_name}>
                  <div>
                    <strong>{item.class_name}</strong>
                    <span>{item.count} occurrence(s)</span>
                  </div>
                  <div className="barCount">{item.count}</div>
                </div>
              ))
            ) : (
              <div className="emptyState small">No defects recorded yet.</div>
            )}
          </div>
        </div>

        <div className="panel">
          <PanelHeader
            title="Recent Inspection History"
            subtitle="Latest inspections with traceability and operator review."
            tag="Traceability"
          />
          <div className="historyList">
            {inspections.length ? (
              inspections.map((inspection) => (
                <div className="historyItem" key={inspection.inspection_id}>
                  <img src={getImageUrl(inspection.result_image_url)} alt="Inspection thumbnail" />
                  <div className="historyText">
                    <div className="historyStatusRow">
                      <strong className={inspection.quality_status === "PASS" ? "textSuccess" : "textDanger"}>
                        {inspection.quality_status}
                      </strong>
                      {inspection.highest_severity && (
                        <i className="dot" style={{ background: severityColor(inspection.highest_severity) }} />
                      )}
                      <span>{inspection.total_detections} detection(s)</span>
                    </div>
                    <p>{inspection.original_filename}</p>
                    <div className="traceChips">
                      {inspection.serial_number && <span>SN {inspection.serial_number}</span>}
                      {inspection.production_line && <span>{inspection.production_line}</span>}
                    </div>
                    <ReviewControls
                      inspectionId={inspection.inspection_id}
                      aiVerdict={inspection.quality_status}
                      reviewed={inspection.reviewed}
                      reviewVerdict={
                        inspection.reviewed
                          ? inspection.review_is_correct
                            ? inspection.quality_status
                            : inspection.quality_status === "PASS"
                            ? "FAIL"
                            : "PASS"
                          : null
                      }
                      onReviewed={loadDashboardData}
                    />
                  </div>
                  <a href={getImageUrl(inspection.result_image_url)} target="_blank" rel="noreferrer">
                    View
                  </a>
                </div>
              ))
            ) : (
              <div className="emptyState small">No inspections saved yet.</div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

function TraceInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value?: string | null;
  onChange: (value: string) => void;
}) {
  return (
    <label className="traceInput">
      <span>{label}</span>
      <input type="text" value={value ?? ""} onChange={(e) => onChange(e.target.value)} placeholder="—" />
    </label>
  );
}

function MetricCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  hint: string;
  tone?: "neutral" | "success" | "danger";
}) {
  return (
    <div className={`metricCard ${tone}`}>
      <div className="metricOrb" />
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{hint}</span>
    </div>
  );
}

function PanelHeader({ title, subtitle, tag }: { title: string; subtitle: string; tag: string }) {
  return (
    <div className="panelHeader">
      <div>
        <span className="panelTag">{tag}</span>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}
