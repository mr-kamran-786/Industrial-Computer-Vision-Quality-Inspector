"use client";

import { SeverityCount, TrendPoint } from "@/types/api";

const SEVERITY_COLORS: Record<string, string> = {
  critical: "var(--danger)",
  major: "var(--warning)",
  minor: "var(--accent)",
};

/**
 * SPC-style control chart for the daily defect rate: plots each day's rate with
 * a center line (mean) and a ±2σ control band. Points breaching the upper
 * control limit are highlighted — the classic "out of control" signal.
 */
export function TrendChart({ points }: { points: TrendPoint[] }) {
  if (!points.length) {
    return (
      <div className="chartEmpty">
        No trend data yet. Run inspections over several days to populate the
        control chart.
      </div>
    );
  }

  const W = 680;
  const H = 240;
  const padL = 40;
  const padR = 16;
  const padT = 18;
  const padB = 34;

  const rates = points.map((p) => p.defect_rate_percent);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const variance =
    rates.reduce((a, b) => a + (b - mean) ** 2, 0) / rates.length;
  const sigma = Math.sqrt(variance);
  const ucl = Math.min(100, mean + 2 * sigma);
  const lcl = Math.max(0, mean - 2 * sigma);

  const maxY = Math.max(10, Math.ceil(Math.max(ucl, ...rates) / 10) * 10);

  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const x = (i: number) =>
    padL + (points.length === 1 ? innerW / 2 : (i * innerW) / (points.length - 1));
  const y = (v: number) => padT + innerH - (v / maxY) * innerH;

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(p.defect_rate_percent)}`)
    .join(" ");

  return (
    <div className="chartWrap">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Defect rate control chart"
        className="chartSvg"
      >
        {/* control band */}
        <rect
          x={padL}
          y={y(ucl)}
          width={innerW}
          height={Math.max(0, y(lcl) - y(ucl))}
          fill="rgba(79,140,255,0.10)"
        />
        {/* center line (mean) */}
        <line
          x1={padL}
          x2={W - padR}
          y1={y(mean)}
          y2={y(mean)}
          stroke="var(--accent-2)"
          strokeDasharray="5 5"
          strokeWidth={1}
          opacity={0.8}
        />
        {/* UCL */}
        <line
          x1={padL}
          x2={W - padR}
          y1={y(ucl)}
          y2={y(ucl)}
          stroke="var(--danger)"
          strokeDasharray="2 4"
          strokeWidth={1}
          opacity={0.55}
        />
        {/* axes labels */}
        <text x={padL - 6} y={y(maxY) + 4} textAnchor="end" className="axisText">
          {maxY}%
        </text>
        <text x={padL - 6} y={y(0) + 4} textAnchor="end" className="axisText">
          0%
        </text>
        <text x={W - padR} y={y(mean) - 5} textAnchor="end" className="axisText">
          CL {mean.toFixed(1)}%
        </text>

        {/* data line */}
        {points.length > 1 && (
          <path d={linePath} fill="none" stroke="var(--accent-2)" strokeWidth={2} />
        )}
        {/* points */}
        {points.map((p, i) => {
          const breach = p.defect_rate_percent > ucl + 1e-9;
          return (
            <g key={p.date}>
              <circle
                cx={x(i)}
                cy={y(p.defect_rate_percent)}
                r={breach ? 5 : 3.5}
                fill={breach ? "var(--danger)" : "var(--accent)"}
                stroke="var(--background)"
                strokeWidth={1.5}
              />
              {points.length <= 10 && (
                <text x={x(i)} y={H - 12} textAnchor="middle" className="axisText">
                  {p.date.slice(5)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="chartLegend">
        <span><i className="dot" style={{ background: "var(--accent-2)" }} /> Center line (mean)</span>
        <span><i className="dot" style={{ background: "var(--danger)" }} /> Upper control limit (2σ)</span>
      </div>
    </div>
  );
}

/** Segmented horizontal bar of defects by severity, with a legend. */
export function SeverityBreakdown({ items }: { items: SeverityCount[] }) {
  const total = items.reduce((a, b) => a + b.count, 0);
  if (!total) {
    return <div className="emptyState small">No severity data recorded yet.</div>;
  }

  const order = ["critical", "major", "minor"];
  const sorted = [...items].sort(
    (a, b) => order.indexOf(a.severity) - order.indexOf(b.severity)
  );

  return (
    <div className="severityWrap">
      <div className="severityBar">
        {sorted.map((item) => (
          <div
            key={item.severity}
            className="severitySeg"
            style={{
              width: `${(item.count / total) * 100}%`,
              background: SEVERITY_COLORS[item.severity] || "var(--muted)",
            }}
            title={`${item.severity}: ${item.count}`}
          />
        ))}
      </div>
      <div className="severityLegend">
        {sorted.map((item) => (
          <span key={item.severity}>
            <i
              className="dot"
              style={{ background: SEVERITY_COLORS[item.severity] || "var(--muted)" }}
            />
            {item.severity} · <strong>{item.count}</strong>
            <small>{((item.count / total) * 100).toFixed(0)}%</small>
          </span>
        ))}
      </div>
    </div>
  );
}

export function severityColor(severity?: string | null): string {
  if (!severity) return "var(--muted)";
  return SEVERITY_COLORS[severity] || "var(--muted)";
}
