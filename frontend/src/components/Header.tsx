"use client";

import React, { useMemo } from "react";
import { Activity, Satellite, Radio, History, Zap, Sun, Moon, MapPin, AlertTriangle } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import type { Storm } from "@/types/storm";

interface HeaderProps {
  storms: Storm[];
  selectedStorm: Storm | undefined;
}

function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return Math.round(R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
}

export default function Header({ storms, selectedStorm }: HeaderProps) {
  const { mode, setMode, theme, toggleTheme, userLocation, setLocationModalOpen } =
    useDashboardStore();
  const activeCount = storms.filter((s) => s.status === "active").length;

  // Proximity to selected storm
  const stormDistanceKm = useMemo(() => {
    if (!userLocation || !selectedStorm?.current_lat || !selectedStorm?.current_lon) {
      return null;
    }
    return haversineKm(
      userLocation.lat,
      userLocation.lon,
      selectedStorm.current_lat,
      selectedStorm.current_lon
    );
  }, [userLocation, selectedStorm]);

  return (
    <header className="header-bar sticky top-0 z-50 px-4 py-3">
      <div className="flex items-center justify-between max-w-[1920px] mx-auto">
        {/* ── Logo & Title ── */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg animate-pulse-glow">
              <Satellite className="w-5 h-5 text-white" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#0f172a]" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              CycloneAI Dashboard
            </h1>
            <p className="text-[0.65rem] text-[var(--text-muted)] tracking-wider uppercase">
              North Indian Ocean • SIH-2K26
            </p>
          </div>
        </div>

        {/* ── Active Storm Badge ── */}
        <div className="hidden md:flex items-center gap-3">
          <div className="flex items-center gap-2 metric-pill">
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-[var(--text-secondary)]">Active:</span>
            <span className="text-emerald-400 font-bold">{activeCount}</span>
          </div>
          {selectedStorm && (
            <div className="flex items-center gap-2 metric-pill">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span className="font-semibold text-[var(--text-primary)]">
                {selectedStorm.name}
              </span>
              <span
                className={`badge ${
                  selectedStorm.status === "active" ? "badge-active" : "badge-historical"
                }`}
              >
                {selectedStorm.status}
              </span>
            </div>
          )}
        </div>

        {/* ── Location Indicator & Mode & Theme Controls ── */}
        <div className="flex items-center gap-2.5">
          {/* Location Verification Pill */}
          <button
            onClick={() => setLocationModalOpen(true)}
            title={userLocation ? "Click to change your location" : "Enable location services for local cyclone alerts"}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all shadow-sm ${
              userLocation
                ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 hover:border-cyan-400"
                : "border-amber-500/50 bg-amber-500/15 text-amber-300 hover:bg-amber-500/25 animate-pulse-glow"
            }`}
          >
            <MapPin className={`w-3.5 h-3.5 ${userLocation ? "text-cyan-400" : "text-amber-400 animate-bounce"}`} />
            {userLocation ? (
              <div className="flex items-center gap-1.5">
                <span className="truncate max-w-[120px] sm:max-w-[160px]">
                  {userLocation.city || `${userLocation.lat.toFixed(1)}°N, ${userLocation.lon.toFixed(1)}°E`}
                </span>
                {stormDistanceKm !== null && (
                  <span
                    className={`text-[0.68rem] px-1.5 py-0.2 rounded font-bold ${
                      stormDistanceKm < 150
                        ? "bg-rose-500/30 text-rose-300 border border-rose-500/50"
                        : stormDistanceKm < 350
                        ? "bg-amber-500/30 text-amber-300 border border-amber-500/50"
                        : "bg-blue-500/20 text-blue-300"
                    }`}
                  >
                    {stormDistanceKm < 350 && <AlertTriangle className="w-2.5 h-2.5 inline mr-0.5" />}
                    {stormDistanceKm} km
                  </span>
                )}
              </div>
            ) : (
              <span className="font-bold">Enable Location</span>
            )}
          </button>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            title={theme === "light" ? "Switch to Dark Mode" : "Switch to Light Mode"}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-[var(--bg-glass-border)] bg-[var(--bg-card)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-blue)] transition-all shadow-sm"
          >
            {theme === "light" ? (
              <>
                <Sun className="w-3.5 h-3.5 text-amber-500" />
                <span className="hidden sm:inline text-[0.72rem]">Light</span>
              </>
            ) : (
              <>
                <Moon className="w-3.5 h-3.5 text-blue-400" />
                <span className="hidden sm:inline text-[0.72rem]">Dark</span>
              </>
            )}
          </button>

          {/* Mode Toggle */}
          <div className="flex rounded-lg overflow-hidden border border-[var(--bg-glass-border)]">
            <button
              onClick={() => setMode("live")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold transition-all ${
                mode === "live"
                  ? "bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-sm"
                  : "bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <Radio className="w-3 h-3" />
              Live
            </button>
            <button
              onClick={() => setMode("replay")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold transition-all ${
                mode === "replay"
                  ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-sm"
                  : "bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              <History className="w-3 h-3" />
              Replay
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
