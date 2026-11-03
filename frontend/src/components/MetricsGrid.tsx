"use client";

import React from "react";
import { Wind, Gauge, Navigation, MapPin, Clock, Target } from "lucide-react";
import { getCategoryColor, getCategoryLabel, formatDateTime, hoursUntil } from "@/lib/utils";
import type { Prediction } from "@/types/prediction";

interface MetricsGridProps {
  prediction: Prediction;
}

export default function MetricsGrid({ prediction }: MetricsGridProps) {
  const cs = prediction.current_state;
  const ri = prediction.ri;
  const impact = prediction.impact;

  if (!cs) return null;

  const catColor = getCategoryColor(cs.category);
  const etaHours = impact?.landfall_eta ? hoursUntil(impact.landfall_eta) : null;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 xl:grid-cols-6 gap-2 animate-fade-in">
      {/* Wind Speed */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <Wind className="w-3 h-3 text-blue-400" />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">Wind</span>
        </div>
        <span className="text-lg font-bold" style={{ color: catColor }}>
          {cs.max_wind_knots}
          <span className="text-[0.65rem] font-normal text-[var(--text-muted)] ml-1">kt</span>
        </span>
      </div>

      {/* Pressure */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <Gauge className="w-3 h-3 text-[var(--accent-cyan)]" />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">Pressure</span>
        </div>
        <span className="text-lg font-bold text-sky-700 dark:text-cyan-300">
          {cs.min_pressure_hpa}
          <span className="text-[0.65rem] font-normal text-[var(--text-muted)] ml-1">hPa</span>
        </span>
      </div>

      {/* Category */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <Target className="w-3 h-3" style={{ color: catColor }} />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">Category</span>
        </div>
        <span className="text-sm font-bold" style={{ color: catColor }}>
          {cs.category}
        </span>
        <span className="text-[0.6rem] text-[var(--text-muted)]">
          {getCategoryLabel(cs.category)}
        </span>
      </div>

      {/* Motion */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <Navigation className="w-3 h-3 text-[var(--accent-purple)]" />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">Motion</span>
        </div>
        <span className="text-sm font-bold text-purple-700 dark:text-purple-300">
          {cs.motion_speed_knots?.toFixed(0) ?? "—"} kt
        </span>
        <span className="text-[0.6rem] text-[var(--text-muted)]">
          {cs.motion_direction_deg?.toFixed(0) ?? "—"}°
        </span>
      </div>

      {/* Landfall ETA */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <Clock className="w-3 h-3 text-[var(--accent-amber)]" />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">Landfall ETA</span>
        </div>
        {etaHours !== null ? (
          <>
            <span className={`text-lg font-bold ${etaHours < 24 ? "text-red-600 dark:text-red-400" : "text-amber-600 dark:text-amber-300"}`}>
              {etaHours}h
            </span>
            <span className="text-[0.6rem] text-[var(--text-muted)]">
              {impact?.landfall_eta ? formatDateTime(impact.landfall_eta) : "—"}
            </span>
          </>
        ) : (
          <span className="text-sm text-[var(--text-muted)]">N/A</span>
        )}
      </div>

      {/* RI Probability */}
      <div className="metric-pill flex flex-col">
        <div className="flex items-center gap-1.5 mb-1">
          <MapPin className="w-3 h-3 text-[var(--accent-red)]" />
          <span className="text-[0.65rem] text-[var(--text-muted)] uppercase tracking-wider">RI Prob</span>
        </div>
        {ri ? (
          <>
            <span
              className={`text-lg font-bold ${
                ri.probability > 0.3
                  ? "text-red-600 dark:text-red-400"
                  : ri.probability > 0.15
                  ? "text-amber-600 dark:text-amber-400"
                  : "text-emerald-600 dark:text-emerald-400"
              }`}
            >
              {(ri.probability * 100).toFixed(0)}%
            </span>
            <span className="text-[0.6rem] text-[var(--text-muted)]">
              +{ri.expected_increase_knots?.toFixed(0) ?? "?"} kt / {ri.timeframe_hours}h
            </span>
          </>
        ) : (
          <span className="text-sm text-[var(--text-muted)]">—</span>
        )}
      </div>
    </div>
  );
}
