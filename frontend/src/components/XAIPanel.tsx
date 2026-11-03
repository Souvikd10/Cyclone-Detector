"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { BrainCircuit, MessageSquareText } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import type { XAIData } from "@/types/prediction";

interface XAIPanelProps {
  xai: XAIData;
}

const FEATURE_LABELS: Record<string, string> = {
  sea_surface_temp: "Sea Surface Temperature",
  vertical_wind_shear: "Vertical Wind Shear",
  ocean_heat_content: "Ocean Heat Content",
  upper_divergence: "Upper-Level Divergence",
  relative_humidity: "Relative Humidity",
};

const FEATURE_COLORS = ["#0284c7", "#2563eb", "#7c3aed", "#d97706", "#059669"];

export default function XAIPanel({ xai }: XAIPanelProps) {
  const theme = useDashboardStore((s) => s.theme);
  const isDark = theme === "dark";

  const chartOption = useMemo(() => {
    if (!xai.feature_importance) return null;

    const entries = Object.entries(xai.feature_importance)
      .sort(([, a], [, b]) => b - a);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: isDark ? "#111827" : "#ffffff",
        borderColor: isDark ? "rgba(100,116,139,0.3)" : "rgba(203,213,225,0.8)",
        textStyle: { color: isDark ? "#f1f5f9" : "#0f172a", fontSize: 11 },
        formatter: (params: { name: string; value: number }[]) => {
          const p = params[0];
          return `${p.name}<br/>Importance: <b>${(p.value * 100).toFixed(1)}%</b>`;
        },
      },
      grid: { top: 8, right: 24, bottom: 4, left: 140 },
      xAxis: {
        type: "value",
        max: 0.4,
        axisLabel: {
          color: isDark ? "#64748b" : "#475569",
          fontSize: 9,
          formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: isDark ? "#1e293b" : "#f1f5f9" } },
      },
      yAxis: {
        type: "category",
        data: entries.map(([key]) => FEATURE_LABELS[key] || key),
        axisLabel: { color: isDark ? "#94a3b8" : "#334155", fontSize: 10 },
        axisLine: { show: false },
        axisTick: { show: false },
        inverse: true,
      },
      series: [
        {
          type: "bar",
          data: entries.map(([, val], i) => ({
            value: val,
            itemStyle: {
              color: FEATURE_COLORS[i % FEATURE_COLORS.length],
              borderRadius: [0, 4, 4, 0],
            },
          })),
          barWidth: 14,
        },
      ],
    };
  }, [xai.feature_importance]);

  return (
    <div className="glass-card p-4 animate-slide-in">
      <div className="flex items-center gap-2 mb-3">
        <BrainCircuit className="w-4 h-4 text-purple-400" />
        <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
          Explainability (XAI)
        </h3>
      </div>

      {/* Feature importance chart */}
      {chartOption && (
        <ReactECharts
          option={chartOption}
          style={{ height: 160 }}
          notMerge
          opts={{ renderer: "canvas" }}
        />
      )}

      {/* Reasoning narrative */}
      {xai.reasoning && (
        <div className="mt-3 p-3 rounded-lg bg-[rgba(139,92,246,0.06)] border border-[rgba(139,92,246,0.15)]">
          <div className="flex items-start gap-2">
            <MessageSquareText className="w-3.5 h-3.5 text-purple-400 flex-shrink-0 mt-0.5" />
            <p className="text-[0.75rem] text-[var(--text-secondary)] leading-relaxed">
              {xai.reasoning}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
