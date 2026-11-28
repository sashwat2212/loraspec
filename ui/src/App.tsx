import React, { Component, useEffect, useState } from "react";
import { fetchHealth } from "./api/client";
import type { HealthStatus } from "./types/telemetry";
import { KnOTSAlignmentLab } from "./views/KnOTSAlignmentLab";
import { LiveMoLoRAArena } from "./views/LiveMoLoRAArena";
import { RooflineAblationView } from "./views/RooflineAblationView";
import { SubspaceObservatory } from "./views/SubspaceObservatory";

type TabId = "observatory" | "knots" | "arena" | "roofline";

class StudioErrorBoundary extends Component<
  { children: React.ReactNode },
  { hasError: boolean; message: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, message: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="m-6 rounded-xl border border-rose-500/40 bg-rose-500/10 p-6 text-sm text-rose-200">
          Studio recovered from an error: {this.state.message}
        </div>
      );
    }
    return this.props.children;
  }
}

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>("observatory");
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    let active = true;
    fetchHealth()
      .then((h) => {
        if (active) setHealth(h);
      })
      .catch(() => {
        // Handled per-view
      });
    return () => {
      active = false;
    };
  }, []);

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: "observatory", label: "1. Subspace Observatory" },
    { id: "knots", label: "2. KnOTS Alignment Lab" },
    { id: "arena", label: "3. Live MoLoRA Arena" },
    { id: "roofline", label: "4. Roofline & Ablation" },
  ];

  return (
    <div className="min-h-screen bg-[#0A0D12] text-slate-100">
      <header className="border-b border-slate-800 bg-[#0D1118]/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/15 font-mono text-sm font-bold text-cyan-300">
              LS
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-wide text-slate-100">
                loraspec Studio
              </h1>
              <p className="font-mono text-[11px] text-slate-400">
                LoRA Subspace Auditor (KnOTS) &amp; Top-k MoLoRA Triton Telemetry
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <span
              data-testid="health-badge"
              className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-emerald-300"
            >
              {health
                ? `${health.demo_replay ? "DEMO_REPLAY" : "LIVE_GPU"} | ${health.backend} | v${health.version}`
                : "connecting..."}
            </span>
          </div>
        </div>

        <nav className="mx-auto flex max-w-7xl gap-1 overflow-x-auto px-4 sm:px-6">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              data-testid={`nav-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`border-b-2 px-4 py-2.5 text-xs font-medium whitespace-nowrap transition ${
                activeTab === tab.id
                  ? "border-cyan-400 text-cyan-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <StudioErrorBoundary>
          {activeTab === "observatory" && <SubspaceObservatory />}
          {activeTab === "knots" && <KnOTSAlignmentLab />}
          {activeTab === "arena" && <LiveMoLoRAArena />}
          {activeTab === "roofline" && <RooflineAblationView />}
        </StudioErrorBoundary>
      </main>
    </div>
  );
};

export default App;

