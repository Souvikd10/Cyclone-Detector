"use client";

import React, { useState } from "react";
import { CheckCircle2, PenLine, Send, X, RotateCcw, BarChart3 } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import type { Prediction } from "@/types/prediction";
import type { FeedbackStats } from "@/types/feedback";

interface FeedbackModalProps {
  prediction: Prediction;
  stats: FeedbackStats | undefined;
  onSubmit: (data: {
    prediction_id: string;
    action: "accept" | "override";
    corrected_value?: Record<string, unknown>;
    forecaster_note?: string;
    forecaster_id?: string;
  }) => void;
  isSubmitting: boolean;
}

export default function FeedbackModal({
  prediction,
  stats,
  onSubmit,
  isSubmitting,
}: FeedbackModalProps) {
  const { feedbackModalOpen, setFeedbackModalOpen } = useDashboardStore();
  const [action, setAction] = useState<"accept" | "override">("accept");
  const [note, setNote] = useState("");
  const [correctedWind, setCorrectedWind] = useState("");
  const [correctedPressure, setCorrectedPressure] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    const corrected: Record<string, unknown> = {};
    if (correctedWind) corrected.max_wind_knots = parseFloat(correctedWind);
    if (correctedPressure) corrected.min_pressure_hpa = parseFloat(correctedPressure);

    onSubmit({
      prediction_id: prediction.id,
      action,
      corrected_value: action === "override" ? corrected : undefined,
      forecaster_note: note || undefined,
      forecaster_id: "Demo Forecaster",
    });
    setSubmitted(true);
    setTimeout(() => {
      setSubmitted(false);
      setFeedbackModalOpen(false);
      setNote("");
      setCorrectedWind("");
      setCorrectedPressure("");
      setAction("accept");
    }, 1500);
  };

  const overrideRate = stats ? stats.override_rate * 100 : 0;
  const retrainWarning = overrideRate > 30;

  return (
    <>
      {/* Action Bar (always visible) */}
      <div className="glass-card p-4 animate-slide-in">
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-blue-400" />
          <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
            Forecaster Review (Stage 17)
          </h3>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-2 mb-3">
            <div className="text-center p-2 rounded-lg bg-[rgba(16,185,129,0.06)] border border-[rgba(16,185,129,0.15)]">
              <div className="text-lg font-bold text-emerald-400">{stats.accepts}</div>
              <div className="text-[0.6rem] text-[var(--text-muted)] uppercase">Accepts</div>
            </div>
            <div className="text-center p-2 rounded-lg bg-[rgba(239,68,68,0.06)] border border-[rgba(239,68,68,0.15)]">
              <div className="text-lg font-bold text-red-400">{stats.overrides}</div>
              <div className="text-[0.6rem] text-[var(--text-muted)] uppercase">Overrides</div>
            </div>
            <div
              className={`text-center p-2 rounded-lg border ${
                retrainWarning
                  ? "bg-[rgba(245,158,11,0.1)] border-[rgba(245,158,11,0.3)]"
                  : "bg-[rgba(100,116,139,0.06)] border-[rgba(100,116,139,0.15)]"
              }`}
            >
              <div className={`text-lg font-bold ${retrainWarning ? "text-amber-400" : "text-[var(--text-secondary)]"}`}>
                {overrideRate.toFixed(0)}%
              </div>
              <div className="text-[0.6rem] text-[var(--text-muted)] uppercase">Override Rate</div>
            </div>
          </div>
        )}

        {retrainWarning && (
          <div className="flex items-center gap-2 mb-3 p-2 rounded-lg bg-[rgba(245,158,11,0.08)] border border-[rgba(245,158,11,0.2)]">
            <RotateCcw className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
            <span className="text-[0.72rem] text-amber-300">
              Override rate &gt;30% — retraining check triggered
            </span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-2">
          <button
            onClick={() => {
              setAction("accept");
              setFeedbackModalOpen(true);
            }}
            className="btn-primary flex-1 flex items-center justify-center gap-1.5"
          >
            <CheckCircle2 className="w-4 h-4" />
            Accept
          </button>
          <button
            onClick={() => {
              setAction("override");
              setFeedbackModalOpen(true);
            }}
            className="btn-danger flex-1 flex items-center justify-center gap-1.5"
          >
            <PenLine className="w-4 h-4" />
            Override
          </button>
        </div>
      </div>

      {/* Modal */}
      {feedbackModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="glass-card w-full max-w-md m-4">
            <div className="flex items-center justify-between p-4 border-b border-[var(--bg-glass-border)]">
              <div className="flex items-center gap-2">
                {action === "accept" ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                ) : (
                  <PenLine className="w-5 h-5 text-red-400" />
                )}
                <h4 className="text-sm font-bold">
                  {action === "accept" ? "Accept Prediction" : "Override Prediction"}
                </h4>
              </div>
              <button
                onClick={() => setFeedbackModalOpen(false)}
                className="btn-ghost p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Current prediction summary */}
              <div className="p-3 rounded-lg bg-[rgba(100,116,139,0.06)] border border-[rgba(100,116,139,0.15)]">
                <div className="text-[0.68rem] text-[var(--text-muted)] uppercase mb-1">Current Prediction</div>
                <div className="flex gap-4 text-sm">
                  <span>
                    <span className="text-[var(--text-muted)]">Wind:</span>{" "}
                    <span className="font-semibold text-sky-700 dark:text-cyan-300">
                      {prediction.current_state?.max_wind_knots ?? "—"} kt
                    </span>
                  </span>
                  <span>
                    <span className="text-[var(--text-muted)]">Pressure:</span>{" "}
                    <span className="font-semibold text-amber-600 dark:text-amber-300">
                      {prediction.current_state?.min_pressure_hpa ?? "—"} hPa
                    </span>
                  </span>
                </div>
              </div>

              {/* Override fields */}
              {action === "override" && (
                <div className="space-y-3">
                  <div>
                    <label className="text-[0.72rem] text-[var(--text-secondary)] block mb-1">
                      Corrected Wind Speed (kt)
                    </label>
                    <input
                      type="number"
                      value={correctedWind}
                      onChange={(e) => setCorrectedWind(e.target.value)}
                      placeholder={String(prediction.current_state?.max_wind_knots ?? "")}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-blue-500/50 transition-colors"
                    />
                  </div>
                  <div>
                    <label className="text-[0.72rem] text-[var(--text-secondary)] block mb-1">
                      Corrected Pressure (hPa)
                    </label>
                    <input
                      type="number"
                      value={correctedPressure}
                      onChange={(e) => setCorrectedPressure(e.target.value)}
                      placeholder={String(prediction.current_state?.min_pressure_hpa ?? "")}
                      className="w-full px-3 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-blue-500/50 transition-colors"
                    />
                  </div>
                </div>
              )}

              {/* Note */}
              <div>
                <label className="text-[0.72rem] text-[var(--text-secondary)] block mb-1">
                  Forecaster Note (optional)
                </label>
                <textarea
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  rows={3}
                  placeholder="Add observations or reasoning..."
                  className="w-full px-3 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] focus:outline-none focus:border-blue-500/50 transition-colors resize-none"
                />
              </div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={isSubmitting || submitted}
                className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-semibold text-sm transition-all ${
                  submitted
                    ? "bg-emerald-600 text-white"
                    : action === "accept"
                    ? "btn-primary"
                    : "btn-danger"
                }`}
              >
                {submitted ? (
                  <>
                    <CheckCircle2 className="w-4 h-4" />
                    Submitted
                  </>
                ) : isSubmitting ? (
                  "Submitting..."
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Submit {action === "accept" ? "Acceptance" : "Override"}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
