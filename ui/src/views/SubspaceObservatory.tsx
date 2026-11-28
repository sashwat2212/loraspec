import React, { useEffect, useMemo, useState } from "react";
import { fetchDefaultAudit, runAudit } from "../api/client";
import type { AuditFinding, AuditReport, PairwiseCollisionMetrics } from "../types/telemetry";

const SEVERITY_BADGE: Record<string, string> = {
  critical: "bg-rose-500/20 text-rose-300 border-rose-500/40",
  high: "bg-orange-500/20 text-orange-300 border-orange-500/40",
  medium: "bg-amber-500/20 text-amber-300 border-amber-500/40",
  low: "bg-sky-500/20 text-sky-300 border-sky-500/40",
  info: "bg-slate-700/40 text-slate-300 border-slate-600",
};

const GRADE_STYLE: Record<string, string> = {
  A: "text-emerald-400 border-emerald-500/40 bg-emerald-500/10",
  B: "text-cyan-400 border-cyan-500/40 bg-cyan-500/10",
  C: "text-amber-400 border-amber-500/40 bg-amber-500/10",
  D: "text-orange-400 border-orange-500/40 bg-orange-500/10",
  F: "text-rose-400 border-rose-500/40 bg-rose-500/10",
};

const SCREE_PALETTE = ["#10B981", "#F97316", "#06B6D4", "#8B5CF6", "#F59E0B"];

export const SubspaceObservatory: React.FC = () => {
  const [report, setReport] = useState<AuditReport | null>(null);
  const [selectedPair, setSelectedPair] = useState<PairwiseCollisionMetrics | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>("all");
  const [customPaths, setCustomPaths] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchDefaultAudit()
      .then((rep) => {
        if (!active) return;
        setReport(rep);
        if (rep.pairwise_collisions.length > 0) {
          setSelectedPair(rep.pairwise_collisions[0]);
        }
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setErrorMsg(err instanceof Error ? err.message : "Failed to load audit bundle.");
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const collisionLookup = useMemo(() => {
    const map = new Map<string, PairwiseCollisionMetrics>();
    if (!report) return map;
    for (const item of report.pairwise_collisions) {
      map.set(`${item.adapter_a}::${item.adapter_b}`, item);
      map.set(`${item.adapter_b}::${item.adapter_a}`, item);
    }
    return map;
  }, [report]);

  const filteredFindings: AuditFinding[] = useMemo(() => {
    if (!report) return [];
    if (severityFilter === "all") return report.findings;
    return report.findings.filter((f) => f.severity === severityFilter);
  }, [report, severityFilter]);

  const handleRunCustomAudit = async (e: React.FormEvent) => {
    e.preventDefault();
    const parts = customPaths
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (parts.length === 0) return;
    setErrorMsg(null);
    setLoading(true);
    try {
      const res = await runAudit(parts);
      setReport(res);
      setSelectedPair(res.pairwise_collisions[0] ?? null);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Audit failed.");
    } finally {
      setLoading(false);
    }
  };

  if (loading && !report) {
    return (
      <div className="rounded-xl border border-slate-800 bg-[#111622] p-8 text-sm text-slate-400">
        Loading Subspace Observatory telemetry...
      </div>
    );
  }

  const adapters = report?.adapters ?? [];
  const layerMetrics = report?.layer_metrics ?? [];
  const maxSigma = Math.max(
    1.0,
    ...layerMetrics.flatMap((m) => m.singular_values)
  );

  return (
    <div className="space-y-6">
      {errorMsg && (
        <div
          role="alert"
          className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200"
        >
          {errorMsg}
        </div>
      )}

      {report && (
        <>
          {/* Summary Strip */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
              <div className="text-xs uppercase tracking-wider text-slate-400">Fleet Grade</div>
              <div className="mt-2 flex items-center gap-3">
                <span
                  data-testid="observatory-grade"
                  className={`inline-flex h-10 w-10 items-center justify-center rounded-lg border font-mono text-xl font-bold ${
                    GRADE_STYLE[report.grade] ?? "border-slate-700 text-slate-100"
                  }`}
                >
                  {report.grade}
                </span>
                <div className="text-xs text-slate-400">
                  Policy:{" "}
                  <span className="font-mono text-slate-200">
                    {report.recommendations.join(", ") || "none"}
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
              <div className="text-xs uppercase tracking-wider text-slate-400">MergeRiskScore</div>
              <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-amber-300">
                {report.merge_risk_score.toFixed(1)}
                <span className="text-sm font-normal text-slate-500"> / 100.0</span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500"
                  style={{ width: `${Math.min(100, Math.max(0, report.merge_risk_score))}%` }}
                />
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
              <div className="text-xs uppercase tracking-wider text-slate-400">
                Loaded LoRA Fleet
              </div>
              <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-slate-100">
                {adapters.length}
                <span className="ml-2 text-xs font-normal text-slate-400">
                  ({adapters.map((a) => `r=${a.rank}`).join(", ")})
                </span>
              </div>
              <div className="mt-1 truncate font-mono text-xs text-slate-500">
                {adapters[0]?.base_model_name_or_path ?? "Qwen/Qwen2.5-3B-Instruct"}
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
              <div className="text-xs uppercase tracking-wider text-slate-400">
                Spectral Rules Flagged
              </div>
              <div className="mt-2 font-mono text-2xl font-bold tabular-nums text-cyan-300">
                {report.findings.length}
                <span className="ml-2 text-xs font-normal text-slate-400">
                  Tier 0–1 probes
                </span>
              </div>
              <form onSubmit={handleRunCustomAudit} className="mt-2 flex gap-2">
                <input
                  type="text"
                  value={customPaths}
                  onChange={(e) => setCustomPaths(e.target.value)}
                  placeholder="/path/a, /path/b (or @replay)"
                  className="w-full rounded border border-slate-700 bg-[#0A0D12] px-2 py-1 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                />
                <button
                  type="submit"
                  className="rounded border border-cyan-500/40 bg-cyan-500/15 px-2.5 py-1 text-xs font-medium text-cyan-200 hover:bg-cyan-500/25"
                >
                  Audit
                </button>
              </form>
            </div>
          </div>

          {/* Grassmann Chordal Matrix + Drilldown */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-7">
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-100">
                    Grassmann Chordal Similarity Matrix
                  </h2>
                  <p className="text-xs text-slate-400">
                    Click any off-diagonal cell to inspect principal angles theta_1..theta_r and sign
                    conflict.
                  </p>
                </div>
                <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">
                  sim_chord in [0, 1]
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse font-mono text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-slate-500">Adapter</th>
                      {adapters.map((a) => (
                        <th key={a.adapter_id} className="p-2 text-center text-slate-300">
                          {a.adapter_id}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {adapters.map((rowA) => (
                      <tr key={rowA.adapter_id}>
                        <td className="p-2 font-semibold text-slate-200">{rowA.adapter_id}</td>
                        {adapters.map((colB) => {
                          if (rowA.adapter_id === colB.adapter_id) {
                            return (
                              <td key={colB.adapter_id} className="p-1 text-center">
                                <div className="rounded border border-slate-800 bg-slate-900/70 py-2 text-slate-500">
                                  1.00
                                </div>
                              </td>
                            );
                          }
                          const metric = collisionLookup.get(
                            `${rowA.adapter_id}::${colB.adapter_id}`
                          );
                          const sim = metric?.grassmann_chordal_similarity ?? 0;
                          const isSelected =
                            selectedPair &&
                            ((selectedPair.adapter_a === rowA.adapter_id &&
                              selectedPair.adapter_b === colB.adapter_id) ||
                              (selectedPair.adapter_b === rowA.adapter_id &&
                                selectedPair.adapter_a === colB.adapter_id));
                          const heatColor =
                            sim >= 0.55
                              ? "rgba(244, 63, 94, 0.34)"
                              : sim >= 0.35
                              ? "rgba(245, 158, 11, 0.25)"
                              : "rgba(16, 185, 129, 0.18)";
                          return (
                            <td key={colB.adapter_id} className="p-1 text-center">
                              <button
                                type="button"
                                data-testid={`grassmann-cell-${rowA.adapter_id}__${colB.adapter_id}`}
                                onClick={() => metric && setSelectedPair(metric)}
                                style={{ backgroundColor: heatColor }}
                                className={`w-full rounded border py-2 tabular-nums transition ${
                                  isSelected
                                    ? "border-cyan-400 ring-1 ring-cyan-400/60"
                                    : "border-slate-700/70 hover:border-slate-400"
                                }`}
                              >
                                {sim.toFixed(2)}
                              </button>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Drilldown Inspector */}
            <div
              data-testid="grassmann-drilldown"
              className="rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-5"
            >
              <h2 className="text-sm font-semibold text-slate-100">Pairwise Subspace Drilldown</h2>
              {selectedPair ? (
                <div className="mt-3 space-y-4">
                  <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-[#0A0D12] px-3 py-2 font-mono text-xs">
                    <span className="text-cyan-300">{selectedPair.adapter_a}</span>
                    <span className="text-slate-500">vs</span>
                    <span className="text-orange-300">{selectedPair.adapter_b}</span>
                  </div>
                  <div className="truncate font-mono text-[11px] text-slate-400">
                    Layer: {selectedPair.layer_name}
                  </div>
                  <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                    <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3">
                      <div className="text-slate-400">Chordal Similarity</div>
                      <div className="mt-1 text-lg font-bold text-slate-100 tabular-nums">
                        {selectedPair.grassmann_chordal_similarity.toFixed(4)}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3">
                      <div className="text-slate-400">Geodesic Distance</div>
                      <div className="mt-1 text-lg font-bold text-slate-100 tabular-nums">
                        {selectedPair.grassmann_geodesic_dist.toFixed(4)}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3">
                      <div className="text-slate-400">Row Similarity</div>
                      <div className="mt-1 text-lg font-bold text-slate-100 tabular-nums">
                        {selectedPair.row_chordal_similarity.toFixed(4)}
                      </div>
                    </div>
                    <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3">
                      <div className="text-slate-400">Sign Conflict Rate</div>
                      <div className="mt-1 text-lg font-bold text-rose-300 tabular-nums">
                        {(selectedPair.sign_conflict_rate * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-2 text-xs text-slate-400">
                      Principal Angles (deg):{" "}
                      <span className="font-mono text-slate-200">
                        {selectedPair.principal_angles_deg.map((d) => `${d.toFixed(1)}°`).join(", ")}
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      {selectedPair.principal_angles_deg.map((deg, idx) => (
                        <div key={idx} className="flex items-center gap-2 font-mono text-[11px]">
                          <span className="w-10 text-slate-400">theta_{idx + 1}</span>
                          <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                            <div
                              className="h-full rounded-full bg-cyan-400"
                              style={{ width: `${Math.min(100, (deg / 90.0) * 100)}%` }}
                            />
                          </div>
                          <span className="w-12 text-right tabular-nums text-slate-200">
                            {deg.toFixed(1)}°
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-4 text-xs text-slate-400">
                  Select a matrix cell to inspect pairwise principal angles.
                </div>
              )}
            </div>
          </div>

          {/* Rank Utilization + Scree Curves */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-[#111622] p-5">
              <h2 className="text-sm font-semibold text-slate-100">
                Effective Rank Utilization (r_eff / r)
              </h2>
              <div className="mt-4 space-y-3">
                {layerMetrics.map((m) => (
                  <div key={`${m.adapter_id}-${m.layer_name}`} className="space-y-1">
                    <div className="flex justify-between font-mono text-xs">
                      <span className="text-slate-200">{m.adapter_id}</span>
                      <span className="tabular-nums text-slate-400">
                        r_eff={m.effective_stable_rank.toFixed(1)} / {m.rank} (
                        {(m.rank_utilization * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-emerald-400"
                        style={{ width: `${Math.min(100, m.rank_utilization * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#111622] p-5">
              <h2 className="text-sm font-semibold text-slate-100">Singular Value Scree Curves</h2>
              <svg
                viewBox="0 0 360 150"
                className="mt-3 h-40 w-full rounded border border-slate-800/80 bg-[#0A0D12] p-2"
              >
                <line x1="28" y1="125" x2="345" y2="125" stroke="#334155" strokeWidth="1" />
                <line x1="28" y1="15" x2="28" y2="125" stroke="#334155" strokeWidth="1" />
                {layerMetrics.map((m, idx) => {
                  const pts = m.singular_values
                    .map((sv, sIdx) => {
                      const x = 35 + sIdx * (300 / Math.max(1, m.singular_values.length - 1));
                      const y = 122 - (sv / maxSigma) * 100;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(" ");
                  const color = SCREE_PALETTE[idx % SCREE_PALETTE.length];
                  return (
                    <polyline
                      key={m.adapter_id}
                      fill="none"
                      stroke={color}
                      strokeWidth="2"
                      points={pts}
                    />
                  );
                })}
              </svg>
              <div className="mt-2 flex flex-wrap gap-3 font-mono text-[11px] text-slate-300">
                {layerMetrics.map((m, idx) => (
                  <span key={m.adapter_id} className="inline-flex items-center gap-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: SCREE_PALETTE[idx % SCREE_PALETTE.length] }}
                    />
                    {m.adapter_id}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* RULE-* Findings Drawer */}
          <div className="rounded-xl border border-slate-800 bg-[#111622] p-5">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-slate-100">
                SARIF & Spectral Rule Findings
              </h2>
              <div className="flex gap-1.5 font-mono text-xs">
                {["all", "critical", "high", "medium", "low"].map((sev) => (
                  <button
                    key={sev}
                    type="button"
                    onClick={() => setSeverityFilter(sev)}
                    className={`rounded border px-2.5 py-1 uppercase ${
                      severityFilter === sev
                        ? "border-cyan-500 bg-cyan-500/20 text-cyan-200"
                        : "border-slate-700 text-slate-400 hover:border-slate-500"
                    }`}
                  >
                    {sev}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2.5">
              {filteredFindings.length === 0 ? (
                <div className="text-xs text-slate-400">No findings match this filter.</div>
              ) : (
                filteredFindings.map((f, idx) => (
                  <div
                    key={`${f.rule_id}-${idx}`}
                    className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3 text-xs"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono font-bold text-slate-100">{f.rule_id}</span>
                      <span
                        className={`rounded border px-2 py-0.5 font-mono text-[10px] uppercase ${
                          SEVERITY_BADGE[f.severity] ?? SEVERITY_BADGE.info
                        }`}
                      >
                        {f.severity}
                      </span>
                      <span className="font-medium text-slate-200">{f.title}</span>
                    </div>
                    <p className="mt-1.5 font-mono text-[11px] text-slate-400">{f.evidence}</p>
                    <p className="mt-1 text-[11px] text-emerald-300/90">
                      Remediation: {f.remediation}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

