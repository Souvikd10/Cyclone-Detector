"use client";

import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { CATEGORY_THRESHOLDS } from "@/lib/utils";
import { useDashboardStore } from "@/store/useDashboardStore";
import type { Prediction } from "@/types/prediction";

interface IntensityChartProps {
  prediction: Prediction;
}

export default function IntensityChart({ prediction }: IntensityChartProps) {
  const intensity = prediction.forecast?.intensity || [];
  const theme = useDashboardStore((s) => s.theme);
  const isDark = theme === "dark";

  const option = useMemo(() => {
    if (!intensity.length) return {};

    const timestamps = intensity.map((pt) =>
      new Date(pt.timestamp).toLocaleString("en-IN", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: "Asia/Kolkata",
      })
    );
    const windSpeeds = intensity.map((pt) => pt.wind_speed);
    const pressures = intensity.map((pt) => pt.pressure);

    // Simulated uncertainty bands (±10% envelope)
    const windUpper = windSpeeds.map((w) => Math.round(w * 1.1));
    const windLower = windSpeeds.map((w) => Math.round(w * 0.9));

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: isDark ? "#111827" : "#ffffff",
        borderColor: isDark ? "rgba(100,116,139,0.3)" : "rgba(203,213,225,0.8)",
        textStyle: { color: isDark ? "#f1f5f9" : "#0f172a", fontSize: 12 },
        axisPointer: { type: "cross", crossStyle: { color: isDark ? "#64748b" : "#94a3b8" } },
      },
      legend: {
        data: ["Wind Speed", "Pressure", "Uncertainty"],
        textStyle: { color: isDark ? "#94a3b8" : "#475569", fontSize: 11 },
        top: 6,
      },
      grid: { top: 50, right: 60, bottom: 40, left: 60 },
      xAxis: {
        type: "category",
        data: timestamps,
        axisLabel: {
          color: isDark ? "#64748b" : "#475569",
          fontSize: 9,
          rotate: 30,
        },
        axisLine: { lineStyle: { color: isDark ? "#334155" : "#cbd5e1" } },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: "value",
          name: "Wind (kt)",
          nameTextStyle: { color: isDark ? "#94a3b8" : "#475569", fontSize: 10 },
          axisLabel: { color: isDark ? "#94a3b8" : "#475569", fontSize: 10 },
          axisLine: { lineStyle: { color: isDark ? "#334155" : "#cbd5e1" } },
          splitLine: { lineStyle: { color: isDark ? "#1e293b" : "#f1f5f9" } },
        },
        {
          type: "value",
          name: "Pressure (hPa)",
          nameTextStyle: { color: isDark ? "#94a3b8" : "#475569", fontSize: 10 },
          axisLabel: { color: isDark ? "#94a3b8" : "#475569", fontSize: 10 },
          axisLine: { lineStyle: { color: isDark ? "#334155" : "#cbd5e1" } },
          splitLine: { show: false },
          inverse: true, // Lower pressure = more intense
        },
      ],
      series: [
        // Uncertainty band (area between upper and lower)
        {
          name: "Uncertainty",
          type: "line",
          data: windUpper,
          lineStyle: { opacity: 0 },
          areaStyle: { opacity: 0 },
          stack: "confidence",
          symbol: "none",
          silent: true,
        },
        {
          name: "Uncertainty",
          type: "line",
          data: windLower,
          lineStyle: { opacity: 0 },
          areaStyle: {
            color: "rgba(59, 130, 246, 0.08)",
          },
          stack: "confidence",
          symbol: "none",
          silent: true,
        },
        // Wind speed
        {
          name: "Wind Speed",
          type: "line",
          data: windSpeeds,
          yAxisIndex: 0,
          smooth: true,
          lineStyle: { color: "#06b6d4", width: 2.5 },
          itemStyle: { color: "#06b6d4" },
          symbol: "circle",
          symbolSize: 4,
          areaStyle: {
            color: {
              type: "linear",
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: "rgba(6,182,212,0.15)" },
                { offset: 1, color: "rgba(6,182,212,0)" },
              ],
            },
          },
        },
        // Pressure
        {
          name: "Pressure",
          type: "line",
          data: pressures,
          yAxisIndex: 1,
          smooth: true,
          lineStyle: { color: "#f59e0b", width: 2, type: "dashed" },
          itemStyle: { color: "#f59e0b" },
          symbol: "diamond",
          symbolSize: 4,
        },
        // Category threshold marklines
        ...CATEGORY_THRESHOLDS.map((thresh) => ({
          name: thresh.label,
          type: "line" as const,
          data: [] as number[],
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: {
              color: thresh.color,
              type: "dotted" as const,
              width: 1,
              opacity: 0.4,
            },
            label: {
              formatter: thresh.label,
              color: thresh.color,
              fontSize: 9,
              opacity: 0.7,
            },
            data: [{ yAxis: thresh.value }],
          },
        })),
      ],
    };
  }, [intensity]);

  return (
    <div className="glass-card p-4">
      <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-2">
        Intensity & Pressure Forecast
      </h3>
      {intensity.length > 0 ? (
        <ReactECharts
          option={option}
          style={{ height: 280 }}
          notMerge
          opts={{ renderer: "canvas" }}
        />
      ) : (
        <div className="h-[280px] flex items-center justify-center text-[var(--text-muted)] text-sm">
          No forecast data available
        </div>
      )}
    </div>
  );
}
