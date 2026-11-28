import type {
  AuditReport,
  BenchmarkSuiteSummary,
  HealthStatus,
  KnOTSPreviewRequest,
  KnOTSPreviewResponse,
  LoRAAdapterSpec,
  ReplayScenarioSummary,
  WsTelemetryEvent,
} from "../types/telemetry";

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody.detail) {
        detail = errBody.detail;
      }
    } catch {
      // Ignore parse failures on plain text errors
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch("/health");
  return parseJsonOrThrow<HealthStatus>(res);
}

export async function fetchAdapters(): Promise<LoRAAdapterSpec[]> {
  const res = await fetch("/api/v1/adapters");
  return parseJsonOrThrow<LoRAAdapterSpec[]>(res);
}

export async function fetchDefaultAudit(): Promise<AuditReport> {
  const res = await fetch("/api/v1/audit");
  return parseJsonOrThrow<AuditReport>(res);
}

export async function runAudit(
  adapterPaths: string[],
  failOn = "high",
  tiers = "0,1"
): Promise<AuditReport> {
  const res = await fetch("/api/v1/audit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      adapter_paths: adapterPaths,
      fail_on: failOn,
      tiers,
    }),
  });
  return parseJsonOrThrow<AuditReport>(res);
}

export async function previewKnots(req: KnOTSPreviewRequest): Promise<KnOTSPreviewResponse> {
  const res = await fetch("/api/v1/knots/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return parseJsonOrThrow<KnOTSPreviewResponse>(res);
}

export async function fetchBenchmarks(
  hardware: "all" | "l4" | "a30" | "a100" = "all"
): Promise<BenchmarkSuiteSummary> {
  const res = await fetch(`/api/v1/benchmarks?hardware=${encodeURIComponent(hardware)}`);
  return parseJsonOrThrow<BenchmarkSuiteSummary>(res);
}

export async function fetchReplayScenarios(): Promise<ReplayScenarioSummary[]> {
  const res = await fetch("/api/v1/replays");
  return parseJsonOrThrow<ReplayScenarioSummary[]>(res);
}

export function connectTelemetryWs(
  replayId: string,
  speed: string,
  onEvent: (ev: WsTelemetryEvent) => void,
  onClose?: () => void
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${proto}//${window.location.host}/ws/telemetry?replay_id=${encodeURIComponent(
    replayId
  )}&speed=${encodeURIComponent(speed)}`;
  const ws = new WebSocket(url);
  ws.onmessage = (evt) => {
    try {
      const parsed = JSON.parse(String(evt.data)) as WsTelemetryEvent;
      onEvent(parsed);
    } catch {
      onEvent({ type: "error", detail: "Malformed telemetry JSON frame." });
    }
  };
  ws.onerror = () => {
    onEvent({ type: "error", detail: "WebSocket telemetry stream encountered an error." });
  };
  ws.onclose = () => {
    if (onClose) onClose();
  };
  return ws;
}

export async function streamChatCompletions(
  prompt: string,
  options: {
    topK?: number;
    temperatureRouter?: number;
    replayId?: string;
    onChunk: (payload: {
      text: string;
      routing?: import("../types/telemetry").MoLoRARouteTelemetry;
    }) => void;
  }
): Promise<void> {
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "loraspec-molora",
      messages: [{ role: "user", content: prompt }],
      top_k: options.topK ?? 2,
      temperature_router: options.temperatureRouter ?? 1.0,
      stream: true,
      replay_id: options.replayId ?? null,
    }),
  });
  if (!res.ok || !res.body) {
    await parseJsonOrThrow<unknown>(res);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const bodyStr = trimmed.slice(5).trim();
      if (bodyStr === "[DONE]") return;
      try {
        const obj = JSON.parse(bodyStr) as {
          choices?: Array<{ delta?: { content?: string } }>;
          loraspec_routing?: import("../types/telemetry").MoLoRARouteTelemetry;
        };
        const deltaText = obj.choices?.[0]?.delta?.content ?? "";
        if (obj.loraspec_routing || deltaText) {
          options.onChunk({ text: deltaText, routing: obj.loraspec_routing });
        }
      } catch {
        // Ignore partial chunk
      }
    }
  }
}

