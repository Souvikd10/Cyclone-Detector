"use client";

import React, { useEffect, useMemo } from "react";
import { Loader2 } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import { useStorms } from "@/hooks/useStorms";
import { useStormLatest } from "@/hooks/useStormLatest";
import { useStormHistory } from "@/hooks/useStormHistory";
import { useStormAlert } from "@/hooks/useStormAlert";
import { useStormFeedback } from "@/hooks/useStormFeedback";
import { useStormWebSocket } from "@/hooks/useStormWebSocket";

import Header from "@/components/Header";
import StormSelector from "@/components/StormSelector";
import MetricsGrid from "@/components/MetricsGrid";
import MapView from "@/components/MapView";
import IntensityChart from "@/components/IntensityChart";
import RIIndicator from "@/components/RIIndicator";
import StructurePanel from "@/components/StructurePanel";
import XAIPanel from "@/components/XAIPanel";
import OODWarningBanner from "@/components/OODWarningBanner";
import AlertPanel from "@/components/AlertPanel";
import FeedbackModal from "@/components/FeedbackModal";
import ReplayControls from "@/components/ReplayControls";
import VoiceAssistantWidget from "@/components/VoiceAssistantWidget";
import LocationModal from "@/components/LocationModal";

export default function DashboardPage() {
  const { selectedStormId, setSelectedStormId, mode, userLocation, setLocationModalOpen } =
    useDashboardStore();
  const replayIndex = useDashboardStore((s) => s.replayIndex);

  // ── Auto-prompt location modal on initial site visit ──
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("cyclone_user_location");
      if (!saved && !userLocation) {
        const timer = setTimeout(() => {
          setLocationModalOpen(true);
        }, 600);
        return () => clearTimeout(timer);
      }
    }
  }, [userLocation, setLocationModalOpen]);

  // ── Data Hooks ──
  const { data: storms = [], isLoading: stormsLoading } = useStorms();
  const { data: latestPrediction } = useStormLatest(selectedStormId);
  const { data: history = [] } = useStormHistory(selectedStormId);
  const { data: alertData } = useStormAlert(selectedStormId);
  const { stats, submit } = useStormFeedback(selectedStormId);
  useStormWebSocket(selectedStormId);

  // ── Auto-select first storm ──
  useEffect(() => {
    if (!selectedStormId && storms.length > 0) {
      setSelectedStormId(storms[0].id);
    }
  }, [storms, selectedStormId, setSelectedStormId]);

  const selectedStorm = storms.find((s) => s.id === selectedStormId);

  // ── Active prediction based on mode ──
  const activePrediction = useMemo(() => {
    if (mode === "replay" && history.length > 0) {
      const idx = Math.min(replayIndex, history.length - 1);
      return history[idx];
    }
    return latestPrediction ?? null;
  }, [mode, latestPrediction, history, replayIndex]);

  // ── OOD detection heuristic ──
  const showOOD = useMemo(() => {
    if (!activePrediction?.uncertainty) return false;
    const spread = activePrediction.uncertainty.track_spread_km ?? 0;
    return spread > 200;
  }, [activePrediction]);

  // ── Loading state ──
  if (stormsLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--bg-primary)]">
        <div className="flex flex-col items-center gap-4 animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center animate-pulse-glow">
            <Loader2 className="w-8 h-8 text-white animate-spin" />
          </div>
          <div className="text-center">
            <h2 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
              CycloneAI Dashboard
            </h2>
            <p className="text-sm text-[var(--text-muted)] mt-1">
              Initializing storm data...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex flex-col">
      {/* ── Header ── */}
      <Header storms={storms} selectedStorm={selectedStorm} />

      {/* ── Main Content ── */}
      <main className="flex-1 flex gap-3 p-3 max-w-[1920px] mx-auto w-full">
        {/* ── Left Sidebar: Storm Selector ── */}
        <aside className="hidden lg:block w-[260px] flex-shrink-0">
          <StormSelector storms={storms} />
        </aside>

        {/* ── Center: Map + Charts ── */}
        <div className="flex-1 flex flex-col gap-3 min-w-0">
          {/* Live Standby Monitoring Strip when no active prediction */}
          {!activePrediction && (
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 rounded-xl border border-emerald-500/25 bg-emerald-500/5 backdrop-blur-md">
              <div className="flex items-center gap-3">
                <div className="relative flex items-center justify-center">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
                  <span className="absolute w-4 h-4 rounded-full bg-emerald-400/50 animate-ping"></span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">
                      Basin Monitoring Active
                    </span>
                    <span className="text-[0.68rem] px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-semibold">
                      0 Active Cyclones
                    </span>
                  </div>
                  <p className="text-[0.7rem] text-[var(--text-muted)] mt-0.5">
                    North Indian Ocean (Bay of Bengal & Arabian Sea) is calm • System is ready to ingest AI model dataset predictions.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[0.7rem] text-slate-300 font-mono bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700">
                  Ready for Model Feed
                </span>
              </div>
            </div>
          )}

          {/* Metrics */}
          {activePrediction && <MetricsGrid prediction={activePrediction} />}

          {/* OOD Warning */}
          <OODWarningBanner
            visible={showOOD}
            uncertaintySpread={activePrediction?.uncertainty?.track_spread_km}
          />

          {/* Replay Controls (visible in replay mode) */}
          {mode === "replay" && <ReplayControls predictions={history} />}

          {/* Map */}
          <div className="flex-1" style={{ minHeight: 420 }}>
            <MapView prediction={activePrediction} />
          </div>

          {/* Intensity Chart */}
          {activePrediction && <IntensityChart prediction={activePrediction} />}

          {/* ── Responsive Analysis Panels (visible on screens below XL) ── */}
          <div className="xl:hidden flex flex-col gap-3 mt-1">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {activePrediction?.ri && <RIIndicator ri={activePrediction.ri} />}
              {activePrediction?.structure && (
                <StructurePanel
                  structure={activePrediction.structure}
                  currentState={activePrediction.current_state}
                />
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {activePrediction?.xai && <XAIPanel xai={activePrediction.xai} />}
              {alertData && <AlertPanel alert={alertData} />}
            </div>
            {activePrediction && (
              <FeedbackModal
                prediction={activePrediction}
                stats={stats.data ?? undefined}
                onSubmit={(data) => submit.mutate(data)}
                isSubmitting={submit.isPending}
              />
            )}
          </div>
        </div>

        {/* ── Right Sidebar: Analysis & Controls (XL screens and above) ── */}
        <aside className="hidden xl:flex flex-col gap-3 w-[340px] flex-shrink-0 custom-scrollbar overflow-y-auto" style={{ maxHeight: "calc(100vh - 80px)" }}>
          {/* AI Model Ingestion Standby Panel when 0 active predictions */}
          {!activePrediction && (
            <div className="glass-card p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-[var(--text-secondary)]">
                  AI Model Pipeline
                </h3>
                <span className="text-[0.65rem] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-semibold flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  Ready
                </span>
              </div>

              <div className="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-xs text-[var(--text-secondary)] space-y-2">
                <div className="font-semibold text-[var(--text-primary)] flex items-center gap-1.5">
                  <span className="text-cyan-400">📡</span> Live Telemetry Standby
                </div>
                <p className="text-[0.72rem] text-[var(--text-muted)] leading-relaxed">
                  Mock cyclone data has been cleared. The system is listening and ready for predictions from your AI model pipeline.
                </p>
              </div>

              <div className="space-y-2.5 text-xs text-[var(--text-secondary)]">
                <div className="flex justify-between items-center py-1.5 border-b border-[var(--bg-glass-border)]">
                  <span className="text-[var(--text-muted)]">Tracked Cyclones</span>
                  <span className="font-semibold text-emerald-400">0 Active</span>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-[var(--bg-glass-border)]">
                  <span className="text-[var(--text-muted)]">Bay of Bengal</span>
                  <span className="font-semibold text-slate-300">Calm (0)</span>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-[var(--bg-glass-border)]">
                  <span className="text-[var(--text-muted)]">Arabian Sea</span>
                  <span className="font-semibold text-slate-300">Calm (0)</span>
                </div>
                <div className="flex justify-between items-center py-1.5 border-b border-[var(--bg-glass-border)]">
                  <span className="text-[var(--text-muted)]">Your Area</span>
                  <span className="font-semibold text-cyan-300">
                    {userLocation?.city || "Safe (Set in Header)"}
                  </span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20 text-[0.72rem] text-blue-300 leading-relaxed">
                💡 <span className="font-semibold">Automatic Activation:</span> When your AI model pushes predictions via the API/WebSocket, the interactive map, forecast cone, wind swaths, and early warning alerts will instantly illuminate.
              </div>
            </div>
          )}

          {/* RI Indicator */}
          {activePrediction?.ri && <RIIndicator ri={activePrediction.ri} />}

          {/* Structure Panel */}
          {activePrediction?.structure && (
            <StructurePanel
              structure={activePrediction.structure}
              currentState={activePrediction.current_state}
            />
          )}

          {/* XAI Panel */}
          {activePrediction?.xai && <XAIPanel xai={activePrediction.xai} />}

          {/* Alert Panel */}
          {alertData && (
            <div id="alert-panel-anchor">
              <AlertPanel alert={alertData} />
            </div>
          )}

          {/* Feedback / HITL */}
          {activePrediction && (
            <FeedbackModal
              prediction={activePrediction}
              stats={stats.data ?? undefined}
              onSubmit={(data) => submit.mutate(data)}
              isSubmitting={submit.isPending}
            />
          )}
        </aside>
      </main>

      {/* ── Mobile Storm Selector (bottom sheet alternative) ── */}
      <div className="lg:hidden fixed bottom-0 left-0 right-0 z-40 p-2">
        <div className="glass-card p-2">
          <select
            value={selectedStormId || ""}
            onChange={(e) => setSelectedStormId(e.target.value || null)}
            className="w-full px-3 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-sm text-[var(--text-primary)] focus:outline-none"
          >
            <option value="">Select Storm</option>
            {storms.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.category}) — {s.status}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Multilingual Voice + Text Agent Floating Widget ── */}
      <VoiceAssistantWidget storms={storms} />

      {/* ── Exact Location Onboarding & Verification Modal ── */}
      <LocationModal />
    </div>
  );
}
