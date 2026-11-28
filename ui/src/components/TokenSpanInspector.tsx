import React from "react";
import type { MoLoRARouteTelemetry } from "../types/telemetry";

export const ADAPTER_COLOR_MAP: Record<string, string> = {
  "sql-lora": "#10B981",
  "rust-lora": "#F97316",
  "json-lora": "#06B6D4",
  "math-lora": "#8B5CF6",
  "python-lora": "#F59E0B",
  base: "#94A3B8",
  "general-lora": "#94A3B8",
};

interface TokenSpanInspectorProps {
  token: MoLoRARouteTelemetry | null;
  temperature: number;
}

export const TokenSpanInspector: React.FC<TokenSpanInspectorProps> = ({ token, temperature }) => {
  if (!token) {
    return (
      <div
        data-testid="token-span-inspector"
        className="rounded-xl border border-slate-800 bg-[#111622] p-5 text-xs text-slate-400"
      >
        Click or hover any streamed token to inspect its Top-k adapter weights and router entropy.
      </div>
    );
  }

  return (
    <div
      data-testid="token-span-inspector"
      className="space-y-4 rounded-xl border border-slate-800 bg-[#111622] p-5"
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100">Token Router Inspector</h3>
        <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-300 tabular-nums">
          token #{token.token_index}
        </span>
      </div>

      <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-3">
        <div className="text-[11px] text-slate-400">Decoded Token Text</div>
        <div
          data-testid="inspector-token-text"
          className="mt-1 break-all font-mono text-sm font-semibold text-slate-100"
        >
          {JSON.stringify(token.token_text)}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 font-mono text-xs">
        <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-2.5">
          <div className="text-[10px] text-slate-400">Entropy H_t</div>
          <div className="mt-0.5 text-sm font-bold text-cyan-300 tabular-nums">
            {token.router_entropy.toFixed(4)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-2.5">
          <div className="text-[10px] text-slate-400">Token ITL</div>
          <div className="mt-0.5 text-sm font-bold text-emerald-300 tabular-nums">
            {token.itl_ms.toFixed(1)} ms
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#0A0D12] p-2.5">
          <div className="text-[10px] text-slate-400">Temp tau</div>
          <div className="mt-0.5 text-sm font-bold text-slate-200 tabular-nums">
            {temperature.toFixed(2)}
          </div>
        </div>
      </div>

      <div className="space-y-2.5">
        <div className="text-xs font-medium text-slate-300">Active Top-k Adapter Weights</div>
        {token.top_k_adapters.map((adapterId, idx) => {
          const weight = token.top_k_weights[idx] ?? 0;
          const color = ADAPTER_COLOR_MAP[adapterId] ?? "#94A3B8";
          return (
            <div key={`${adapterId}-${idx}`} className="space-y-1 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1.5 text-slate-200">
                  <span
                    className="inline-block h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  {adapterId}
                </span>
                <span className="tabular-nums text-slate-300">{(weight * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(100, Math.max(0, weight * 100))}%`,
                    backgroundColor: color,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

