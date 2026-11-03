"use client";

import React from "react";
import { Bell, Shield, Clock, FileCode2, X, Copy, Check } from "lucide-react";
import { SEVERITY_COLORS, formatDateTime } from "@/lib/utils";
import { useDashboardStore } from "@/store/useDashboardStore";
import type { AlertResponse } from "@/types/alert";

interface AlertPanelProps {
  alert: AlertResponse;
}

export default function AlertPanel({ alert }: AlertPanelProps) {
  const { xmlModalOpen, setXmlModalOpen } = useDashboardStore();
  const [copied, setCopied] = React.useState(false);
  const info = alert.info;
  const sevColor = SEVERITY_COLORS[info.severity] || "#94a3b8";

  const handleCopy = async () => {
    await navigator.clipboard.writeText(alert.cap_xml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="glass-card p-4 animate-slide-in">
      {/* Prototype Disclaimer */}
      <div className="prototype-banner mb-3">
        ⚠ PROTOTYPE ALERT — FOR DEMONSTRATION ONLY — NOT A GOVERNMENT ADVISORY
      </div>

      <div className="flex items-center gap-2 mb-3">
        <Bell className="w-4 h-4" style={{ color: sevColor }} />
        <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">
          CAP 1.2 Alert
        </h3>
        <span
          className="badge ml-auto"
          style={{
            color: sevColor,
            background: `${sevColor}18`,
            border: `1px solid ${sevColor}40`,
          }}
        >
          {info.severity}
        </span>
      </div>

      {/* Headline */}
      <div
        className="text-sm font-semibold mb-3 p-2.5 rounded-lg"
        style={{
          color: sevColor,
          background: `${sevColor}0a`,
          borderLeft: `3px solid ${sevColor}`,
        }}
      >
        {info.headline}
      </div>

      {/* Description */}
      <p className="text-[0.78rem] text-[var(--text-secondary)] leading-relaxed mb-3">
        {info.description}
      </p>

      {/* Metadata Grid */}
      <div className="grid grid-cols-2 gap-2 text-[0.72rem] mb-3">
        <MetaItem icon={<Shield className="w-3 h-3" />} label="Urgency" value={info.urgency} />
        <MetaItem icon={<Shield className="w-3 h-3" />} label="Certainty" value={info.certainty} />
        <MetaItem
          icon={<Clock className="w-3 h-3" />}
          label="Effective"
          value={formatDateTime(info.effective)}
        />
        <MetaItem
          icon={<Clock className="w-3 h-3" />}
          label="Expires"
          value={info.expires ? formatDateTime(info.expires) : "—"}
        />
      </div>

      <div className="text-[0.65rem] text-[var(--text-muted)] mb-3">
        Sender: {info.sender}
      </div>

      {/* View XML Button */}
      <button
        onClick={() => setXmlModalOpen(true)}
        className="btn-ghost flex items-center gap-1.5 w-full justify-center"
      >
        <FileCode2 className="w-3.5 h-3.5" />
        View CAP XML
      </button>

      {/* XML Modal */}
      {xmlModalOpen && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className="glass-card w-full max-w-2xl max-h-[80vh] m-4 flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-[var(--bg-glass-border)]">
              <div className="flex items-center gap-2">
                <FileCode2 className="w-4 h-4 text-purple-400" />
                <h4 className="text-sm font-bold">CAP 1.2 XML Document</h4>
              </div>
              <div className="flex items-center gap-2">
                <button onClick={handleCopy} className="btn-ghost flex items-center gap-1 text-[0.7rem]">
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? "Copied" : "Copy"}
                </button>
                <button onClick={() => setXmlModalOpen(false)} className="btn-ghost p-1">
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4 custom-scrollbar">
              <pre className="text-[0.72rem] text-sky-800 dark:text-cyan-300/90 bg-[var(--bg-secondary)] p-3 rounded-lg border border-[var(--bg-glass-border)] leading-relaxed whitespace-pre-wrap font-mono">
                {alert.cap_xml}
              </pre>
            </div>
            <div className="prototype-banner m-4 mt-0">
              PROTOTYPE OUTPUT — STATUS: TEST
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MetaItem({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[var(--text-muted)]">{icon}</span>
      <span className="text-[var(--text-muted)]">{label}:</span>
      <span className="text-[var(--text-primary)] font-medium">{value}</span>
    </div>
  );
}
