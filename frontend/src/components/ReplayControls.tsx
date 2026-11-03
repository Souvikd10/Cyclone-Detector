"use client";

import React, { useEffect, useRef } from "react";
import { Play, Pause, SkipBack, SkipForward, FastForward, Clock } from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import { formatDateTime } from "@/lib/utils";
import type { Prediction } from "@/types/prediction";

interface ReplayControlsProps {
  predictions: Prediction[];
}

export default function ReplayControls({ predictions }: ReplayControlsProps) {
  const {
    replayIndex,
    replayPlaying,
    replaySpeed,
    setReplayIndex,
    setReplayLength,
    setReplayPlaying,
    setReplaySpeed,
    stepReplay,
  } = useDashboardStore();

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sync length with predictions
  useEffect(() => {
    setReplayLength(predictions.length);
  }, [predictions.length, setReplayLength]);

  // Auto-play timer
  useEffect(() => {
    if (replayPlaying && predictions.length > 0) {
      intervalRef.current = setInterval(() => {
        useDashboardStore.setState((s) => {
          const next = s.replayIndex + 1;
          if (next >= predictions.length) {
            return { replayPlaying: false, replayIndex: predictions.length - 1 };
          }
          return { replayIndex: next };
        });
      }, 2000 / replaySpeed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [replayPlaying, replaySpeed, predictions.length]);

  if (predictions.length === 0) {
    return (
      <div className="glass-card p-4 flex items-center justify-center text-[var(--text-muted)] text-sm">
        No historical data for replay
      </div>
    );
  }

  const current = predictions[replayIndex];
  const speedOptions = [1, 2, 4];

  return (
    <div className="glass-card p-4 animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-purple-400" />
        <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
          Historical Replay
        </h3>
        <span className="ml-auto text-[0.68rem] text-purple-700 dark:text-purple-300 font-semibold font-mono">
          {replayIndex + 1} / {predictions.length}
        </span>
      </div>

      {/* Timeline */}
      <div className="mb-3">
        <input
          type="range"
          min={0}
          max={predictions.length - 1}
          value={replayIndex}
          onChange={(e) => setReplayIndex(parseInt(e.target.value))}
          className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
          style={{
            background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${
              (replayIndex / Math.max(1, predictions.length - 1)) * 100
            }%, rgba(100,116,139,0.2) ${
              (replayIndex / Math.max(1, predictions.length - 1)) * 100
            }%, rgba(100,116,139,0.2) 100%)`,
          }}
        />
        <div className="flex justify-between mt-1 text-[0.6rem] text-[var(--text-muted)]">
          <span>{formatDateTime(predictions[0].timestamp)}</span>
          <span>{formatDateTime(predictions[predictions.length - 1].timestamp)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => { setReplayPlaying(false); setReplayIndex(0); }}
            className="btn-ghost p-1.5"
            title="Reset"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => { setReplayPlaying(false); stepReplay(-1); }}
            className="btn-ghost p-1.5"
            title="Step Back (−6h)"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setReplayPlaying(!replayPlaying)}
            className="btn-primary px-3 py-1.5 flex items-center gap-1"
          >
            {replayPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {replayPlaying ? "Pause" : "Play"}
          </button>
          <button
            onClick={() => { setReplayPlaying(false); stepReplay(1); }}
            className="btn-ghost p-1.5"
            title="Step Forward (+6h)"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Speed */}
        <div className="flex items-center gap-1">
          <FastForward className="w-3 h-3 text-[var(--text-muted)]" />
          {speedOptions.map((s) => (
            <button
              key={s}
              onClick={() => setReplaySpeed(s)}
              className={`text-[0.68rem] font-semibold px-2 py-0.5 rounded transition-all ${
                replaySpeed === s
                  ? "bg-purple-100 text-purple-700 border border-purple-300 dark:bg-purple-600/30 dark:text-purple-300 dark:border-purple-500/30"
                  : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Current timestamp */}
      {current && (
        <div className="mt-2 text-center text-[0.72rem] text-purple-700 dark:text-purple-300 font-semibold font-mono">
          📍 {formatDateTime(current.timestamp)}
        </div>
      )}
    </div>
  );
}
