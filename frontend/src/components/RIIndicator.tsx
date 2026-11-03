"use client";

import React from "react";
import { TrendingUp, AlertTriangle, Zap } from "lucide-react";
import type { RIData } from "@/types/prediction";

interface RIIndicatorProps {
  ri: RIData;
}

export default function RIIndicator({ ri }: RIIndicatorProps) {
  const pct = ri.probability * 100;
  const isHigh = pct > 30;
  const isMedium = pct > 15;

  const color = isHigh ? "#ef4444" : isMedium ? "#f59e0b" : "#10b981";
  const label = isHigh ? "CRITICAL" : isMedium ? "ELEVATED" : "LOW";
  const bgGlow = isHigh
    ? "rgba(239,68,68,0.08)"
    : isMedium
    ? "rgba(245,158,11,0.08)"
    : "rgba(16,185,129,0.08)";

  // SVG gauge arc
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (circumference * Math.min(pct, 100)) / 100;

  return (
    <div className="glass-card p-4" style={{ background: bgGlow }}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
          Rapid Intensification
        </h3>
        {isHigh && (
          <span className="badge badge-danger">
            <AlertTriangle className="w-3 h-3" />
            Warning
          </span>
        )}
      </div>

      <div className="flex items-center gap-5">
        {/* Gauge */}
        <div className="relative flex-shrink-0">
          <svg width="120" height="120" viewBox="0 0 120 120">
            {/* Background circle */}
            <circle
              cx="60"
              cy="60"
              r={radius}
              fill="none"
              stroke="rgba(100,116,139,0.15)"
              strokeWidth="8"
            />
            {/* Progress arc */}
            <circle
              cx="60"
              cy="60"
              r={radius}
              fill="none"
              stroke={color}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              transform="rotate(-90 60 60)"
              style={{
                filter: `drop-shadow(0 0 6px ${color}80)`,
                transition: "stroke-dashoffset 1s ease",
              }}
            />
            {/* Center text */}
            <text
              x="60"
              y="54"
              textAnchor="middle"
              fill={color}
              fontSize="24"
              fontWeight="bold"
              fontFamily="var(--font-geist-sans)"
            >
              {pct.toFixed(0)}%
            </text>
            <text
              x="60"
              y="72"
              textAnchor="middle"
              fill="#64748b"
              fontSize="10"
              fontWeight="500"
            >
              {ri.timeframe_hours}h window
            </text>
          </svg>
        </div>

        {/* Details */}
        <div className="flex flex-col gap-2">
          <div
            className="text-sm font-bold uppercase tracking-wider"
            style={{ color }}
          >
            <Zap className="w-3.5 h-3.5 inline mr-1" style={{ color }} />
            {label} RISK
          </div>

          <div className="flex items-center gap-1.5 text-[0.75rem] text-[var(--text-secondary)]">
            <TrendingUp className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            Expected intensification:
          </div>
          <div className="text-lg font-bold" style={{ color }}>
            +{ri.expected_increase_knots?.toFixed(0) ?? "?"}{" "}
            <span className="text-[0.7rem] font-normal text-[var(--text-muted)]">
              knots / {ri.timeframe_hours}h
            </span>
          </div>

          <div className="text-[0.65rem] text-[var(--text-muted)] mt-1">
            RI threshold: ≥30 kt increase in 24h
          </div>
        </div>
      </div>
    </div>
  );
}
