"use client";

import React from "react";
import { Eye, CircleDot, Layers, Waves, BarChart3 } from "lucide-react";
import type { StructureData, CurrentStateData } from "@/types/prediction";

interface StructurePanelProps {
  structure: StructureData;
  currentState?: CurrentStateData;
}

export default function StructurePanel({ structure, currentState }: StructurePanelProps) {
  return (
    <div className="glass-card p-4 animate-slide-in">
      <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)] mb-3">
        Eye & Structure Analysis
      </h3>

      <div className="space-y-3">
        {/* Eye Detection */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm">
            <Eye className="w-4 h-4 text-cyan-400" />
            <span className="text-[var(--text-secondary)]">Eye Detected</span>
          </div>
          <span
            className={`badge ${
              structure.eye_detected ? "badge-active" : "badge-historical"
            }`}
          >
            {structure.eye_detected ? "YES" : "NO"}
          </span>
        </div>

        {/* Eye Radius */}
        {structure.eye_radius_km != null && (
          <MetricRow
            icon={<CircleDot className="w-4 h-4 text-blue-400" />}
            label="Eye Diameter"
            value={`${(structure.eye_radius_km * 2).toFixed(0)} km`}
            subValue={`Radius: ${structure.eye_radius_km.toFixed(1)} km`}
          />
        )}

        {/* Eyewall Completeness */}
        {structure.eyewall_completeness != null && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2 text-sm">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span className="text-[var(--text-secondary)]">Eyewall</span>
              </div>
              <span className="text-sm font-semibold text-indigo-600 dark:text-indigo-300">
                {(structure.eyewall_completeness * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full h-1.5 bg-[rgba(100,116,139,0.15)] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${structure.eyewall_completeness * 100}%`,
                  background: "linear-gradient(90deg, #818cf8, #6366f1)",
                }}
              />
            </div>
          </div>
        )}

        {/* Rain Bands */}
        {structure.rain_bands != null && (
          <MetricRow
            icon={<Waves className="w-4 h-4 text-sky-400" />}
            label="Rain Bands"
            value={`${structure.rain_bands}`}
          />
        )}

        {/* Symmetry Score */}
        {structure.symmetry_score != null && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2 text-sm">
                <BarChart3 className="w-4 h-4 text-emerald-400" />
                <span className="text-[var(--text-secondary)]">Symmetry</span>
              </div>
              <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-300">
                {(structure.symmetry_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="w-full h-1.5 bg-[rgba(100,116,139,0.15)] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${structure.symmetry_score * 100}%`,
                  background: "linear-gradient(90deg, #34d399, #10b981)",
                }}
              />
            </div>
          </div>
        )}

        {/* Dvorak CI */}
        {currentState?.dvorak_ci != null && (
          <MetricRow
            icon={<BarChart3 className="w-4 h-4 text-amber-400" />}
            label="Dvorak C.I."
            value={currentState.dvorak_ci.toFixed(1)}
          />
        )}
      </div>
    </div>
  );
}

function MetricRow({
  icon,
  label,
  value,
  subValue,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2 text-sm">
        {icon}
        <span className="text-[var(--text-secondary)]">{label}</span>
      </div>
      <div className="text-right">
        <span className="text-sm font-semibold text-[var(--text-primary)]">{value}</span>
        {subValue && (
          <div className="text-[0.6rem] text-[var(--text-muted)]">{subValue}</div>
        )}
      </div>
    </div>
  );
}
