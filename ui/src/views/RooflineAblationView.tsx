import React, { useEffect, useMemo, useState } from "react";
import { fetchBenchmarks } from "../api/client";
import type { BenchmarkSuiteSummary } from "../types/telemetry";

const ARM_COLORS: Record<string, string> = {
  peft_eager_arrow: "#F43F5E",
  vllm_static: "#94A3B8",
  knots_ties: "#F59E0B",
  triton_arrow_sgmv: "#10B981",
};

const ARM_LABELS: Record<string, string> = {
  peft_eager_arrow: "PEFT Eager Arrow (K-loop)",
  vllm_static: "vLLM Static Single-LoRA",
  knots_ties: "KnOTS Merged LoRA",
  triton_arrow_sgmv: "Triton Top-k Arrow-SGMV",
};

export const RooflineAblationView: React.FC = () => {
  const [hwFilter, setHwFilter] = useState<"all" | "l4" | "a30" | "a100">("all");
  const [suite, setSuite] = useState<BenchmarkSuiteSummary | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchBenchmarks(hwFilter)
      .then((res) => {
        if (!active) return;
        setSuite(res);
        setErrorMsg(null);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setErrorMsg(err instanceof Error ? err.message : "Failed to load benchmarks.");
      });
    return () => {
      active = false;
    };
  }, [hwFilter]);

  const seriesByArm = useMemo(() => {
    if (!suite) return new Map<string, Array<{ batch: number; latency: number }>>();
    const grouped = new Map<string, Map<number, number[]>>();
    for (const pt of suite.points) {
      if (!grouped.has(pt.arm_name)) grouped.set(pt.arm_name, new Map());
      const byB = grouped.get(pt.arm_name)!;
      if (!byB.has(pt.batch_size)) byB.set(pt.batch_size, []);
      byB.get(pt.batch_size)!.push(pt.latency_ms);
    }
    const out = new Map<string, Array<{ batch: number; latency: number }>>();
    for (const [arm, byB] of grouped.entries()) {
      const pts = Array.from(byB.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([batch, lats]) => ({
          batch,
          latency: lats.reduce((s, v) => s + v, 0) / lats.length,
        }));
      out.set(arm, pts);
    }
    return out;
  }, [suite]);

  const maxLat = useMemo(() => {
    let mx = 1.0;
    for (const arr of seriesByArm.values()) {
      for (const p of arr) mx = Math.max(mx, p.latency);
    }
    return mx;
  }, [seriesByArm]);

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

      {/* Filter Bar + KPI Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-[#111622] p-4">
        <div className="flex flex-wrap gap-2 font-mono text-xs">
          {(
            [
              ["all", "All GPUs (L4 + A30 + A100)"],
              ["l4", "L4 (300 GB/s)"],
              ["a30", "A30 Gate (933 GB/s)"],
              ["a100", "A100 (2039 GB/s)"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setHwFilter(id)}
              className={`rounded-lg border px-3 py-1.5 transition ${
                hwFilter === id
                  ? "border-cyan-400 bg-cyan-500/20 text-cyan-200"
                  : "border-slate-700 text-slate-400 hover:border-slate-500"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {suite && (
          <div className="flex flex-wrap gap-4 font-mono text-xs">
            <span className="text-emerald-400">
              Speedup: <strong>{suite.speedup_vs_peft_eager.toFixed(2)}x</strong>
            </span>
            <span className="text-cyan-300">
              VRAM Saved: <strong>{suite.vram_savings_pct.toFixed(1)}%</strong>
            </span>
            <span className="text-amber-300">
              Ledger Cost: <strong>₹{suite.manifest.total_gpu_cost_inr.toFixed(2)}</strong> / ₹3,000
            </span>
          </div>
        )}
      </div>

      {suite && (
        <>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            {/* Latency Scaling SVG Chart */}
            <div className="rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-7">
              <h2 className="text-sm font-semibold text-slate-100">
                Multi-LoRA Latency vs Batch Size (B = 1..64, seq_len = 64)
              </h2>
              <svg
                data-testid="roofline-latency-svg"
                viewBox="0 0 420 190"
                className="mt-3 h-52 w-full rounded border border-slate-800 bg-[#0A0D12] p-2"
              >
                <line x1="40" y1="160" x2="395" y2="160" stroke="#334155" strokeWidth="1" />
                <line x1="40" y1="18" x2="40" y2="160" stroke="#334155" strokeWidth="1" />
                {Array.from(seriesByArm.entries()).map(([arm, pts]) => {
                  const color = ARM_COLORS[arm] ?? "#38BDF8";
                  const polyPts = pts
                    .map((p, idx) => {
                      const x = 50 + idx * (330 / Math.max(1, pts.length - 1));
                      const y = 156 - (p.latency / maxLat) * 130;
                      return `${x.toFixed(1)},${y.toFixed(1)}`;
                    })
                    .join(" ");
                  return (
                    <polyline
                      key={arm}
                      data-arm={arm}
                      fill="none"
                      stroke={color}
                      strokeWidth="2.5"
                      points={polyPts}
                    />
                  );
                })}
              </svg>
              <div className="mt-3 flex flex-wrap gap-4 font-mono text-xs">
                {Array.from(seriesByArm.keys()).map((arm) => (
                  <span key={arm} className="inline-flex items-center gap-1.5 text-slate-300">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: ARM_COLORS[arm] ?? "#38BDF8" }}
                    />
                    {ARM_LABELS[arm] ?? arm}
                  </span>
                ))}
              </div>
            </div>

            {/* 4-Arm Accuracy Pass@1 */}
            <div className="rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-5">
              <h2 className="text-sm font-semibold text-slate-100">
                4-Arm Compositional Quality (pass@1)
              </h2>
              <div className="mt-4 space-y-3 font-mono text-xs">
                {Object.entries(suite.arm_accuracy_pass_at_1).map(([arm, acc]) => (
                  <div key={arm} className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-slate-200">{ARM_LABELS[arm] ?? arm}</span>
                      <span className="font-bold text-slate-100 tabular-nums">
                        {(acc * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400"
                        style={{ width: `${Math.min(100, acc * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Roofline Detail Table */}
          <div className="rounded-xl border border-slate-800 bg-[#111622] p-5">
            <h2 className="mb-3 text-sm font-semibold text-slate-100">
              Hardware Roofline Telemetry Points
            </h2>
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="sticky top-0 bg-[#111622] text-slate-400">
                  <tr className="border-b border-slate-800">
                    <th className="py-2 pr-3">Hardware</th>
                    <th className="py-2 pr-3">Arm</th>
                    <th className="py-2 pr-3">Batch</th>
                    <th className="py-2 pr-3">Dtype</th>
                    <th className="py-2 pr-3 text-right">Latency</th>
                    <th className="py-2 pr-3 text-right">HBM GB/s</th>
                    <th className="py-2 pr-3 text-right">VRAM MB</th>
                    <th className="py-2 text-right">Max Err</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {suite.points.map((pt, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/30">
                      <td className="py-1.5 pr-3 text-slate-300">{pt.hardware_profile}</td>
                      <td className="py-1.5 pr-3 font-medium text-slate-100">{pt.arm_name}</td>
                      <td className="py-1.5 pr-3 tabular-nums text-slate-400">
                        {pt.batch_size}x{pt.seq_len}
                      </td>
                      <td className="py-1.5 pr-3 text-slate-300">{pt.dtype}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums text-cyan-300">
                        {pt.latency_ms.toFixed(4)} ms
                      </td>
                      <td className="py-1.5 pr-3 text-right tabular-nums text-slate-300">
                        {pt.hbm_bandwidth_gb_s.toFixed(1)}
                      </td>
                      <td className="py-1.5 pr-3 text-right tabular-nums text-slate-300">
                        {pt.peak_vram_mb.toFixed(1)}
                      </td>
                      <td className="py-1.5 text-right tabular-nums text-emerald-300">
                        {pt.max_abs_err_vs_ref.toExponential(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

