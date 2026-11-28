import React, { useEffect, useRef, useState } from "react";
import {
  connectTelemetryWs,
  fetchReplayScenarios,
  streamChatCompletions,
} from "../api/client";
import { ADAPTER_COLOR_MAP, TokenSpanInspector } from "../components/TokenSpanInspector";
import type { MoLoRARouteTelemetry, ReplayScenarioSummary } from "../types/telemetry";

function hexToRgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const r = parseInt(clean.slice(0, 2), 16) || 148;
  const g = parseInt(clean.slice(2, 4), 16) || 163;
  const b = parseInt(clean.slice(4, 6), 16) || 184;
  return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

export const LiveMoLoRAArena: React.FC = () => {
  const [scenarios, setScenarios] = useState<ReplayScenarioSummary[]>([]);
  const [selectedReplayId, setSelectedReplayId] = useState<string>("rust_sql_json_etl");
  const [prompt, setPrompt] = useState<string>(
    "Write a Rust parser emitting strict JSON over an analytical SQL query."
  );
  const [topK, setTopK] = useState<number>(2);
  const [temperature, setTemperature] = useState<number>(1.0);
  const [speed, setSpeed] = useState<string>("instant");
  const [tokens, setTokens] = useState<MoLoRARouteTelemetry[]>([]);
  const [selectedToken, setSelectedToken] = useState<MoLoRARouteTelemetry | null>(null);
  const [ttftMs, setTtftMs] = useState<number>(28.4);
  const [meanItlMs, setMeanItlMs] = useState<number>(11.0);
  const [vramMb, setVramMb] = useState<number>(6480.0);
  const [streaming, setStreaming] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let active = true;
    fetchReplayScenarios()
      .then((list) => {
        if (!active || list.length === 0) return;
        setScenarios(list);
        setSelectedReplayId(list[0].replay_id);
        setPrompt(list[0].prompt);
      })
      .catch(() => {
        // Fallback when DEMO_REPLAY is off
      });
    return () => {
      active = false;
      wsRef.current?.close();
    };
  }, []);

  const runWsReplay = (replayId: string) => {
    wsRef.current?.close();
    setErrorMsg(null);
    setTokens([]);
    setSelectedToken(null);
    setStreaming(true);

    const acc: MoLoRARouteTelemetry[] = [];
    wsRef.current = connectTelemetryWs(
      replayId,
      speed,
      (ev) => {
        if (ev.type === "gpu_stats") {
          setTtftMs(ev.ttft_ms);
          setMeanItlMs(ev.itl_ms);
          setVramMb(ev.vram_mb);
        } else if (ev.type === "token") {
          acc.push(ev.routing);
          setTokens([...acc]);
          if (acc.length === 1) {
            setSelectedToken(ev.routing);
          }
        } else if (ev.type === "done") {
          setMeanItlMs(ev.mean_itl_ms);
          setVramMb(ev.peak_vram_mb);
          setStreaming(false);
        } else if (ev.type === "error") {
          setErrorMsg(ev.detail);
          setStreaming(false);
        }
      },
      () => {
        setStreaming(false);
      }
    );
  };

  const runSsePrompt = async (e: React.FormEvent) => {
    e.preventDefault();
    wsRef.current?.close();
    setErrorMsg(null);
    setTokens([]);
    setSelectedToken(null);
    setStreaming(true);
    try {
      const collected: MoLoRARouteTelemetry[] = [];
      await streamChatCompletions(prompt, {
        topK,
        temperatureRouter: temperature,
        onChunk: ({ routing }) => {
          if (!routing) return;
          collected.push(routing);
          setTokens([...collected]);
          if (collected.length === 1) setSelectedToken(routing);
        },
      });
      if (collected.length > 0) {
        const avgItl = collected.reduce((s, r) => s + r.itl_ms, 0) / collected.length;
        setMeanItlMs(avgItl);
      }
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "SSE generation failed.");
    } finally {
      setStreaming(false);
    }
  };

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

      {/* Telemetry Gauges */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
          <div className="text-xs text-slate-400">TTFT</div>
          <div className="mt-1 font-mono text-xl font-bold tabular-nums text-cyan-300">
            {ttftMs.toFixed(1)} ms
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
          <div className="text-xs text-slate-400">Mean ITL</div>
          <div className="mt-1 font-mono text-xl font-bold tabular-nums text-emerald-400">
            {meanItlMs.toFixed(2)} ms/tok
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
          <div className="text-xs text-slate-400">Peak VRAM</div>
          <div className="mt-1 font-mono text-xl font-bold tabular-nums text-slate-100">
            {(vramMb / 1024).toFixed(2)} GB
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#111622] p-4">
          <div className="text-xs text-slate-400">Streamed Tokens</div>
          <div className="mt-1 font-mono text-xl font-bold tabular-nums text-slate-100">
            {tokens.length}
          </div>
        </div>
      </div>

      {/* Controls + Color Legend */}
      <div className="rounded-xl border border-slate-800 bg-[#111622] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap gap-2">
            {scenarios.map((sc) => (
              <button
                key={sc.replay_id}
                type="button"
                data-testid={`preset-${sc.replay_id}`}
                onClick={() => {
                  setSelectedReplayId(sc.replay_id);
                  setPrompt(sc.prompt);
                  runWsReplay(sc.replay_id);
                }}
                className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
                  selectedReplayId === sc.replay_id
                    ? "border-cyan-400 bg-cyan-500/20 text-cyan-200"
                    : "border-slate-700 text-slate-300 hover:border-slate-500"
                }`}
              >
                {sc.title}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <label className="text-slate-400">Speed:</label>
            <select
              value={speed}
              onChange={(e) => setSpeed(e.target.value)}
              className="rounded border border-slate-700 bg-[#0A0D12] px-2 py-1 text-slate-200"
            >
              <option value="1.0">1.0x</option>
              <option value="2.0">2.0x</option>
              <option value="instant">instant</option>
            </select>
            <button
              type="button"
              data-testid="arena-replay-btn"
              onClick={() => runWsReplay(selectedReplayId)}
              disabled={streaming}
              className="rounded-lg border border-emerald-500/50 bg-emerald-500/20 px-3 py-1.5 font-semibold text-emerald-200 hover:bg-emerald-500/30 disabled:opacity-50"
            >
              {streaming ? "Streaming..." : "Replay WS Trace"}
            </button>
          </div>
        </div>

        <form onSubmit={runSsePrompt} className="grid grid-cols-1 gap-3 lg:grid-cols-12">
          <input
            type="text"
            data-testid="arena-prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter multi-domain prompt (e.g. Rust + SQL + JSON)..."
            className="rounded-lg border border-slate-700 bg-[#0A0D12] px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:border-cyan-400 focus:outline-none lg:col-span-7"
          />
          <div className="flex items-center gap-2 font-mono text-xs lg:col-span-3">
            <span className="text-slate-400">k={topK}</span>
            <input
              type="range"
              min={1}
              max={4}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-20 accent-cyan-400"
            />
            <span className="text-slate-400">tau={temperature.toFixed(1)}</span>
            <input
              type="range"
              min={0.3}
              max={2.0}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="w-20 accent-cyan-400"
            />
          </div>
          <button
            type="submit"
            data-testid="arena-stream-sse-btn"
            disabled={streaming}
            className="rounded-lg border border-cyan-500/50 bg-cyan-500/20 px-3 py-2 text-xs font-semibold text-cyan-200 hover:bg-cyan-500/30 disabled:opacity-50 lg:col-span-2"
          >
            Stream Prompt SSE
          </button>
        </form>

        {/* Adapter Color Legend */}
        <div className="flex flex-wrap gap-3 border-t border-slate-800 pt-3 font-mono text-[11px]">
          {Object.entries(ADAPTER_COLOR_MAP)
            .filter(([name]) => name !== "general-lora")
            .map(([name, color]) => (
              <span key={name} className="inline-flex items-center gap-1.5 text-slate-300">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-sm"
                  style={{ backgroundColor: color }}
                />
                {name}
              </span>
            ))}
        </div>
      </div>

      {/* Colorized Token Stream + Inspector */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="rounded-xl border border-slate-800 bg-[#111622] p-5 lg:col-span-7">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-slate-100">
              Per-Token Top-k Colorized Output
            </h2>
            <span className="font-mono text-[11px] text-slate-400">
              React DOM text nodes (XSS-safe)
            </span>
          </div>
          <div
            data-testid="arena-token-stream"
            className="min-h-[180px] rounded-lg border border-slate-800 bg-[#0A0D12] p-4 font-mono text-sm leading-relaxed"
          >
            {tokens.length === 0 ? (
              <span className="text-xs text-slate-500">
                Click &quot;Replay WS Trace&quot;, select a preset, or click &quot;Stream Prompt
                SSE&quot; to inspect per-token routing weights.
              </span>
            ) : (
              tokens.map((tok) => {
                const top1 = tok.top_k_adapters[0] ?? "base";
                const g1 = tok.top_k_weights[0] ?? 0.5;
                const hex = ADAPTER_COLOR_MAP[top1] ?? "#94A3B8";
                const isSel = selectedToken?.token_index === tok.token_index;
                return (
                  <button
                    key={tok.token_index}
                    type="button"
                    data-testid={`arena-token-${tok.token_index}`}
                    data-top-adapter={top1}
                    onClick={() => setSelectedToken(tok)}
                    onMouseEnter={() => setSelectedToken(tok)}
                    style={{
                      backgroundColor: hexToRgba(hex, Math.max(0.18, Math.min(0.45, g1 * 0.48))),
                      borderColor: hex,
                    }}
                    className={`mr-1 mb-1 inline whitespace-pre-wrap rounded border px-1 py-0.5 text-left text-slate-100 transition ${
                      isSel ? "ring-2 ring-white/80" : "hover:brightness-125"
                    }`}
                  >
                    {tok.token_text}
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="lg:col-span-5">
          <TokenSpanInspector token={selectedToken} temperature={temperature} />
        </div>
      </div>
    </div>
  );
};

