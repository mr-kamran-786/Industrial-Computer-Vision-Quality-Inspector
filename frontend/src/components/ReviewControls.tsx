"use client";

import { useState } from "react";
import { submitReview } from "@/lib/api";

/**
 * Operator ground-truth review. The inspector confirms or overrides the AI
 * verdict; the result feeds the backend's escape-rate / false-call-rate metrics.
 */
export function ReviewControls({
  inspectionId,
  aiVerdict,
  reviewed,
  reviewVerdict,
  onReviewed,
}: {
  inspectionId: string;
  aiVerdict: string;
  reviewed?: boolean;
  reviewVerdict?: string | null;
  onReviewed?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [reviewer, setReviewer] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState<string | null>(reviewed ? reviewVerdict ?? null : null);

  async function send(verdict: "PASS" | "FAIL") {
    if (!reviewer.trim()) {
      setError("Enter your operator ID / name first.");
      return;
    }
    try {
      setBusy(true);
      setError("");
      const result = await submitReview(inspectionId, {
        verdict,
        reviewed_by: reviewer.trim(),
        notes: notes.trim() || undefined,
      });
      setDone(result.review_verdict);
      setOpen(false);
      onReviewed?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review failed.");
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    const agreed = done === aiVerdict;
    return (
      <div className={`reviewBadge ${agreed ? "reviewAgree" : "reviewDisagree"}`}>
        Reviewed: {done} {agreed ? "· confirms AI" : "· overrides AI"}
      </div>
    );
  }

  if (!open) {
    return (
      <button className="reviewOpenBtn" onClick={() => setOpen(true)}>
        Review verdict
      </button>
    );
  }

  return (
    <div className="reviewPanel">
      <div className="reviewFields">
        <input
          type="text"
          placeholder="Operator ID / name"
          value={reviewer}
          onChange={(e) => setReviewer(e.target.value)}
        />
        <input
          type="text"
          placeholder="Notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>
      {error && <div className="reviewError">{error}</div>}
      <div className="reviewActions">
        <button
          className="reviewPass"
          disabled={busy}
          onClick={() => send("PASS")}
        >
          Confirm PASS
        </button>
        <button
          className="reviewFail"
          disabled={busy}
          onClick={() => send("FAIL")}
        >
          Confirm FAIL
        </button>
        <button className="reviewCancel" disabled={busy} onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
}
