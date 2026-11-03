"use client";

import React from "react";
import { AlertOctagon, ShieldAlert, UserCheck } from "lucide-react";

interface OODWarningBannerProps {
  visible: boolean;
  uncertaintySpread?: number;
}

export default function OODWarningBanner({ visible, uncertaintySpread }: OODWarningBannerProps) {
  if (!visible) return null;

  return (
    <div className="ood-banner flex items-start gap-3 animate-fade-in">
      <div className="flex-shrink-0 mt-0.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-500/20 to-red-500/20 flex items-center justify-center">
          <AlertOctagon className="w-4.5 h-4.5 text-amber-400" />
        </div>
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-sm font-bold text-amber-300 uppercase tracking-wider">
            Out-of-Distribution / Novelty Warning
          </span>
        </div>
        <p className="text-[0.78rem] text-[var(--text-secondary)] leading-relaxed">
          The current storm pattern shows characteristics that differ significantly from the
          training distribution. Model predictions may have reduced reliability.
          {uncertaintySpread && (
            <span className="text-amber-400 font-medium">
              {" "}Track spread: {uncertaintySpread} km.
            </span>
          )}
        </p>
        <div className="flex items-center gap-1.5 mt-2 text-[0.7rem] text-amber-400/80">
          <UserCheck className="w-3 h-3" />
          <span className="font-semibold">Forecaster review recommended before dissemination</span>
        </div>
      </div>
    </div>
  );
}
