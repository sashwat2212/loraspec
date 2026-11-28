import React, { useEffect, useState } from "react";
import { previewKnots } from "../api/client";
import type { KnOTSPreviewResponse, KnotsMethod } from "../types/telemetry";

const REPLAY_ADAPTERS = [
  "replay:sql-lora",
  "replay:rust-lora",
  "replay:json-lora",
  "replay:math-lora",
  "replay:python-lora",
];

export const KnOTSAlignmentLab: React.FC = () => {
  const [targetRank, setTargetRank] = useState<number>(16);
  const [method, setMethod] = useState<KnotsMethod>("knots_ties");
  const [tiesDensity, setTiesDensity] = useState<number>(0.5);
  const [dareDropRate, setDareDropRate] = useState<number>(0.2);
  const [useGks, setUseGks] = useState<boolean>(false);
  const [preview, setPreview] = useState<KnOTSPreviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setErrorMsg(null);
    previewKnots({
      adapter_paths: REPLAY_ADAPTERS,
      method,
      target_rank: targetRank,
      ties_density: tiesDensity,
      dare_drop_rate: dareDropRate,
      use_gks: useGks,
      rng_seed: 42,
    })
      .then((res) => {
        if (!active) return;
        setPreview(res);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setErrorMsg(err instanceof Error ? err.message : "KnOTS preview failed.");
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [targetRank, method, tiesDensity, dareDropRate, useGks]);

  const cliSnippet = `loraspec merge ./adapters/sql-lora ./adapters/rust-lora ./adapters/json-lora --method ${method} --target-rank ${targetRank} --ties-density ${tiesDensity.toFixed(
    2
  )}${useGks ? " --gks-adapter ./adapters/general-lora" : ""} --out-dir ./merged_knots_lora`;

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

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Controls */}
        <div className="space-y-5 rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-5">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">KnOTS Joint-SVD Controls</h2>
            <p className="text-xs text-slate-400">
              Factorized subspace alignment in O((d_out + d_in) r) — zero dense weights materialized.
            </p>
          </div>

          <div>
            <div className="mb-1.5 flex justify-between font-mono text-xs">
              <span className="text-slate-300">Target Output Rank (r)</span>
              <span data-testid="knots-rank-label" className="font-bold text-cyan-300 tabular-nums">
                r = {targetRank}
              </span>
            </div>
            <input
              data-testid="knots-rank-slider"
              type="range"
              min={4}
              max={64}
              step={4}
              value={targetRank}
              onChange={(e) => setTargetRank(Number(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-500">
              <span>4</span>
              <span>16</span>
              <span>32</span>
              <span>48</span>
              <span>64</span>
            </div>
          </div>

          <div>
            <label className="mb-1.5 block font-mono text-xs text-slate-300">Merge Operator</label>
            <div className="grid grid-cols-3 gap-2 font-mono text-xs">
              {(["knots_ties", "knots_dare_ties", "knots_linear"] as KnotsMethod[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMethod(m)}
                  className={`rounded border px-2 py-2 text-center transition ${
                    method === m
                      ? "border-cyan-400 bg-cyan-500/15 text-cyan-200"
                      : "border-slate-700 text-slate-400 hover:border-slate-500"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="mb-1 flex justify-between font-mono text-xs">
              <span className="text-slate-300">TIES Trim Density</span>
              <span className="text-slate-200 tabular-nums">{(tiesDensity * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.05}
              value={tiesDensity}
              onChange={(e) => setTiesDensity(Number(e.target.value))}
              className="w-full accent-emerald-400"
            />
          </div>

          {method === "knots_dare_ties" && (
            <div>
              <div className="mb-1 flex justify-between font-mono text-xs">
                <span className="text-slate-300">DARE Drop Rate</span>
                <span className="text-slate-200 tabular-nums">
                  {(dareDropRate * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min={0.0}
                max={0.6}
                step={0.05}
                value={dareDropRate}
                onChange={(e) => setDareDropRate(Number(e.target.value))}
                className="w-full accent-orange-400"
              />
            </div>
          )}

          <label className="flex cursor-pointer items-center justify-between rounded-lg border border-slate-800 bg-[#0A0D12] px-3 py-2.5 text-xs">
            <div>
              <div className="font-medium text-slate-200">GenKnowSub Subtraction</div>
              <div className="text-[11px] text-slate-400">
                Subtract shared general-knowledge LoRA before SVD alignment
              </div>
            </div>
            <input
              type="checkbox"
              checked={useGks}
              onChange={(e) => setUseGks(e.target.checked)}
              className="h-4 w-4 accent-cyan-400"
            />
          </label>
        </div>

        {/* Live Telemetry Comparison */}
        <div className="space-y-5 rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-7">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-100">
              Naive vs. KnOTS Alignment Diagnostics
            </h2>
            {loading && <span className="font-mono text-xs text-cyan-300">recomputing...</span>}
          </div>

          {preview && (
            <>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-4">
                  <div className="text-xs text-slate-400">Energy Retention (eta)</div>
                  <div
                    data-testid="knots-energy-value"
                    className="mt-1 font-mono text-2xl font-bold tabular-nums text-emerald-400"
                  >
                    {(preview.summary.energy_retained * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-4">
                  <div className="text-xs text-slate-400">Sign Conflict Drop</div>
                  <div className="mt-1 font-mono text-2xl font-bold tabular-nums text-cyan-300">
                    -{preview.summary.sign_conflict_reduction_pct.toFixed(1)}%
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-4">
                  <div className="text-xs text-slate-400">Factorized RMSE</div>
                  <div className="mt-1 font-mono text-2xl font-bold tabular-nums text-slate-100">
                    {preview.summary.mean_reconstruction_rmse.toFixed(5)}
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-lg border border-slate-800 bg-[#0A0D12] p-4">
                <div className="text-xs font-medium text-slate-200">
                  Parameter Sign-Conflict Rate (%)
                </div>
                <div className="space-y-2 font-mono text-xs">
                  <div>
                    <div className="mb-1 flex justify-between text-slate-400">
                      <span>Naive Parameter-Space TIES / Linear</span>
                      <span className="text-rose-300 tabular-nums">
                        {(preview.pre_knots_sign_conflict_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-rose-500"
                        style={{
                          width: `${Math.min(100, preview.pre_knots_sign_conflict_rate * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="mb-1 flex justify-between text-slate-400">
                      <span>KnOTS Joint-SVD Aligned ({preview.summary.method})</span>
                      <span className="text-emerald-300 tabular-nums">
                        {(preview.post_knots_sign_conflict_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className="h-full rounded-full bg-emerald-400 transition-all"
                        style={{
                          width: `${Math.min(100, preview.post_knots_sign_conflict_rate * 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {preview.findings.length > 0 && (
                <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-200">
                  <span className="font-mono font-bold">{preview.findings[0].rule_id}: </span>
                  {preview.findings[0].evidence}
                </div>
              )}

              <div>
                <div className="mb-1.5 text-xs text-slate-400">PEFT Merge CLI Recipe</div>
                <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-[#0A0D12] p-3 font-mono text-[11px] text-cyan-200">
                  {cliSnippet}
                </pre>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

