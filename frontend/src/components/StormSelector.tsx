"use client";

import React, { useState, useMemo } from "react";
import { CloudLightning, Wind, Gauge, MapPin, Radio, Waves, Filter, Search } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import { getCategoryColor, getCategoryLabel } from "@/lib/utils";
import type { Storm } from "@/types/storm";

interface StormSelectorProps {
  storms: Storm[];
}

type BasinFilter = "ALL" | "BOB" | "AS";

export default function StormSelector({ storms }: StormSelectorProps) {
  const { selectedStormId, setSelectedStormId } = useDashboardStore();
  const [basinFilter, setBasinFilter] = useState<BasinFilter>("ALL");
  const [searchTerm, setSearchTerm] = useState("");

  // Helper to normalize basin identification
  const getBasinKey = (storm: Storm): "BOB" | "AS" => {
    const b = (storm.basin || "").toUpperCase();
    const n = storm.name.toLowerCase();
    if (b === "BOB" || b === "BAY OF BENGAL" || ["amphan", "remal", "fani", "mocha", "michaung", "dana"].some((s) => n.includes(s))) {
      return "BOB";
    }
    return "AS";
  };

  const filteredStorms = useMemo(() => {
    return storms.filter((storm) => {
      const basin = getBasinKey(storm);
      const matchesBasin =
        basinFilter === "ALL" ||
        (basinFilter === "BOB" && basin === "BOB") ||
        (basinFilter === "AS" && basin === "AS");

      const matchesSearch =
        searchTerm.trim() === "" ||
        storm.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        storm.category.toLowerCase().includes(searchTerm.toLowerCase());

      return matchesBasin && matchesSearch;
    });
  }, [storms, basinFilter, searchTerm]);

  const active = filteredStorms.filter((s) => s.status === "active");
  const historical = filteredStorms.filter((s) => s.status === "historical");

  const bobCount = storms.filter((s) => getBasinKey(s) === "BOB").length;
  const asCount = storms.filter((s) => getBasinKey(s) === "AS").length;

  return (
    <div className="glass-card p-3.5 custom-scrollbar overflow-y-auto flex flex-col gap-3" style={{ maxHeight: "calc(100vh - 120px)" }}>
      {/* Title & Basin Filter Pill Header */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <Waves className="w-4 h-4 text-cyan-400" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
              Cyclone Systems
            </h2>
          </div>
          <span className="text-[0.62rem] font-bold px-2 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/30 text-cyan-300">
            {storms.length} Total
          </span>
        </div>

        {/* Ocean Basin Tabs */}
        <div className="grid grid-cols-3 gap-1 p-1 bg-slate-900/60 rounded-lg border border-white/10 text-[0.68rem]">
          <button
            onClick={() => setBasinFilter("ALL")}
            className={`py-1 px-1.5 rounded-md font-semibold text-center transition-all ${
              basinFilter === "ALL"
                ? "bg-cyan-500 text-slate-950 font-bold shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            All ({storms.length})
          </button>
          <button
            onClick={() => setBasinFilter("BOB")}
            className={`py-1 px-1.5 rounded-md font-semibold text-center transition-all ${
              basinFilter === "BOB"
                ? "bg-blue-500 text-white font-bold shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
            title="Bay of Bengal Cyclones"
          >
            BOB ({bobCount})
          </button>
          <button
            onClick={() => setBasinFilter("AS")}
            className={`py-1 px-1.5 rounded-md font-semibold text-center transition-all ${
              basinFilter === "AS"
                ? "bg-emerald-500 text-slate-950 font-bold shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
            title="Arabian Sea Cyclones"
          >
            Arabian ({asCount})
          </button>
        </div>

        {/* Search Bar */}
        <div className="relative mt-2">
          <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search cyclone name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-2.5 py-1.5 rounded-lg bg-slate-900/70 border border-white/10 text-[0.72rem] text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>
      </div>

      {/* Standby Empty State when 0 storms match */}
      {filteredStorms.length === 0 && (
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-center">
          <div className="w-9 h-9 mx-auto mb-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Radio className="w-4 h-4 animate-pulse" />
          </div>
          <div className="text-xs font-semibold text-emerald-400 mb-1">
            No Cyclones in this View
          </div>
          <p className="text-[0.66rem] text-[var(--text-muted)] leading-relaxed">
            Select another basin or clear search query to inspect tracked systems.
          </p>
        </div>
      )}

      {/* Active storms */}
      {active.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5 px-1">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[0.66rem] font-bold uppercase tracking-wider text-emerald-400">
                Active ({active.length})
              </span>
            </div>
          </div>
          <div className="space-y-1.5">
            {active.map((storm) => (
              <StormCard
                key={storm.id}
                storm={storm}
                basinName={getBasinKey(storm) === "BOB" ? "Bay of Bengal" : "Arabian Sea"}
                isSelected={storm.id === selectedStormId}
                onSelect={() => setSelectedStormId(storm.id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Historical storms */}
      {historical.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-1.5 px-1 pt-2 border-t border-white/5">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-slate-500" />
              <span className="text-[0.66rem] font-bold uppercase tracking-wider text-[var(--text-muted)]">
                Historical Replay ({historical.length})
              </span>
            </div>
          </div>
          <div className="space-y-1.5">
            {historical.map((storm) => (
              <StormCard
                key={storm.id}
                storm={storm}
                basinName={getBasinKey(storm) === "BOB" ? "Bay of Bengal" : "Arabian Sea"}
                isSelected={storm.id === selectedStormId}
                onSelect={() => setSelectedStormId(storm.id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StormCard({
  storm,
  basinName,
  isSelected,
  onSelect,
}: {
  storm: Storm;
  basinName: string;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const catColor = getCategoryColor(storm.category);
  const isBOB = basinName === "Bay of Bengal";

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left rounded-xl p-2.5 transition-all duration-200 group relative ${
        isSelected
          ? "storm-card-selected ring-1 ring-cyan-400/80 shadow-[0_0_15px_rgba(6,182,212,0.25)]"
          : "bg-[var(--bg-card)] border border-white/5 hover:border-cyan-500/30 hover:bg-[var(--bg-card-hover)]"
      }`}
    >
      <div className="flex items-start justify-between mb-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <CloudLightning
            className="w-4 h-4 flex-shrink-0 animate-pulse"
            style={{ color: catColor }}
          />
          <span className="text-xs font-bold text-[var(--text-primary)] truncate">
            {storm.name}
          </span>
        </div>
        <span
          className="text-[0.58rem] font-extrabold px-1.5 py-0.5 rounded shrink-0 shadow-sm"
          style={{
            color: catColor,
            background: `${catColor}20`,
            border: `1px solid ${catColor}40`,
          }}
        >
          {storm.category}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[0.66rem] text-[var(--text-secondary)]">
        <div className="flex items-center gap-1">
          <Wind className="w-3 h-3 text-cyan-400" />
          <span className="font-medium text-slate-200">{storm.max_wind_speed} kt</span>
        </div>
        <div className="flex items-center gap-1">
          <Gauge className="w-3 h-3 text-amber-400" />
          <span>{storm.min_pressure} hPa</span>
        </div>
        <div className="flex items-center gap-1 col-span-2 text-[0.62rem] text-slate-400">
          <MapPin className="w-3 h-3 text-red-400" />
          <span>
            {storm.current_lat.toFixed(1)}°N, {storm.current_lon.toFixed(1)}°E
          </span>
        </div>
      </div>

      <div className="mt-2 flex items-center justify-between pt-1.5 border-t border-white/5 text-[0.58rem]">
        <span className="text-slate-400 truncate max-w-[110px]">
          {getCategoryLabel(storm.category)}
        </span>
        <span
          className={`px-1.5 py-0.2 rounded font-semibold ${
            isBOB
              ? "bg-blue-500/15 text-blue-300 border border-blue-500/30"
              : "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
          }`}
        >
          🌊 {isBOB ? "Bay of Bengal" : "Arabian Sea"}
        </span>
      </div>
    </button>
  );
}

