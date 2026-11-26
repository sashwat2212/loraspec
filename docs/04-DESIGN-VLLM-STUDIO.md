# Stage 4 / SDD-04: `vLLM V1` Zero-Fork Plugin, FastAPI Streaming Gateway & React 19 Studio — `loraspec`

> [!NOTE]
> **Subsystem**: `SDD-04` (`ARCH-COMP-08` `vLLM V1` Plugin + `ARCH-COMP-09` FastAPI/WS Gateway & `DEMO_REPLAY` + `ARCH-COMP-12` React 19 Studio)
> **Linked Features**: `FEAT-13`, `FEAT-14`, `FEAT-15`, `FEAT-16`, `FEAT-17`
> **Upstream Docs**: [02-FRD-PRD.md](./02-FRD-PRD.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/loraspec/serving/`, `ui/`, `Dockerfile`, `docker-compose.yml` (Target: Milestone **M3**)

---

## 1. Zero-Fork `vLLM V1` General Plugin (`ARCH-COMP-08`)

### 1.1 Entry-Point Registration (`pyproject.toml`)
```toml
[project.entry-points."vllm.general_plugins"]
loraspec_molora = "loraspec.serving.vllm_plugin:register"
```
Because `vLLM V1` propagates `vllm.general_plugins` across engine core and worker processes automatically (`load_plugins_by_group`), `loraspec` never forks or patches `vLLM` files on disk.

### 1.2 Runtime Hook & Non-Blocking Telemetry Ring Buffer
When `LORASPEC_MOLORA_ENABLE=1` (and `LORASPEC_ADAPTERS=/path/to/adapters.json`):
1. **`register()`**: Registers a post-load model hook wrapping target attention/MLP linear projections (`q/k/v/o/gate/up/down_proj`) with `MoLoRALinearAdapter`, loading `Arrow` prototypes (`P`) and packed `(A_packed, B_packed, ranks, scalings)` tensors directly onto the worker GPU device.
2. **Decode Step Execution**: Base weight `W_0 @ x` runs through native `vLLM` GEMM/cuBLAS; `ArrowMoLoRADispatcher` (`SDD-03`) adds per-token Top-$k$ `Arrow-SGMV` delta in-place.
3. **Lock-Free Telemetry Capture**: A lightweight CPU/pinned-memory ring buffer records `(step_idx, request_id, top_k_adapter_ids, top_k_weights, router_entropy, step_latency_ms)` from the designated telemetry layer (e.g., mid-stack + final decoder block) without stalling inner-layer CUDA streams.

---

## 2. FastAPI Gateway, `/ws/telemetry` & `DEMO_REPLAY=true` (`ARCH-COMP-09`)

### 2.1 Gateway Endpoints (`src/loraspec/serving/gateway.py`)
- `GET /health`: Reports service status, `demo_replay: bool`, active compute backend (`triton_cuda | torch_mps | torch_cpu | replay`), loaded adapter count, and package version.
- `GET /api/v1/adapters`: Returns loaded `LoRAAdapterSpec` descriptors + layer count + parameter budget.
- `POST /api/v1/audit`: Runs Tier 0–1 spectral audit over local adapter directory paths and returns full `AuditReport` JSON (reusing `SDD-01` directly).
- `POST /api/v1/knots/preview`: Computes factorized `KnOTS` preview summary (`energy_retained`, pre/post sign-conflict drop) for UI sliders.
- `GET /api/v1/benchmarks`: Returns `BenchmarkSuiteSummary` (4-arm L4/A100 ablation, speedup vs. `peft` eager Arrow, accuracy & GST-inclusive INR spend ledger).
- `POST /v1/chat/completions`: OpenAI-compatible SSE stream emitting `data: {...}` chunks augmented with `"loraspec_routing": MoLoRARouteTelemetry.model_dump()`.
- `WebSocket /ws/telemetry`: Streams real-time token-by-token routing frames + VRAM & ITL telemetry to connected Studio sessions.

### 2.2 `DEMO_REPLAY=true` Engine (`src/loraspec/serving/demo_replay.py`)
To satisfy **`NFR-08`** (`₹0/month` public live demo):
- Loads verified pre-recorded L4 & A100 multi-LoRA sessions from `assets/demo_replays/*.jsonl` (plus recorded `AuditReport` & roofline JSON).
- Replays authentic per-token Top-$k$ weights, router entropy, TTFT (`ms`), ITL (`ms/tok`), and VRAM traces over `/v1/chat/completions` and `/ws/telemetry` at configurable playback rate (`1.0x`, `2.0x`, `instant`).
- Zero GPU, `torch`, or `vLLM` installation needed when `DEMO_REPLAY=true`!

---

## 3. React 19 + TypeScript + Vite + Tailwind Web Studio (`ui/`) (`ARCH-COMP-12`)

Dark-mode engineering telemetry UI (`#0A0D12` obsidian canvas, monospace numeric tables, crisp high-contrast domain palette):

| Studio View | Key Interactive Visualizations & Components | Data Source |
|---|---|---|
| **1. Subspace Observatory** | Pairwise **Grassmann Chordal Similarity Matrix** (clickable cells showing principal angles $\theta_1 \dots \theta_r$), **Effective Rank Utilization** ($r_{\text{eff}}/r$) bar chart, **Singular Value Scree Curves**, and `RULE-*` SARIF findings drawer. | `/api/v1/audit` or preloaded L4 audit bundle |
| **2. `KnOTS` Alignment Lab** | Interactive side-by-side comparison of **Naive Linear/TIES vs. `KnOTS` Joint-SVD** sign-conflict %, target rank slider ($r \in [4..64]$) updating **Energy Retention $\eta$** & RMSE live, plus one-click PEFT merge config generator. | `/api/v1/knots/preview` |
| **3. Live Per-Token `MoLoRA` Arena** | Prompt playground + preset compositional scenarios (*"Rust parser emitting JSON over SQL query"*). Streams tokens in real time, **colorizing each token span** by dominant adapter:<br>• `sql-lora` → Emerald (`#10B981`)<br>• `rust-lora` → Orange (`#F97316`)<br>• `json-lora` → Cyan (`#06B6D4`)<br>• `math-lora` → Violet (`#8B5CF6`)<br>• `python-lora` → Amber (`#F59E0B`)<br>Clicking/hovering any token opens the **Token Router Inspector** (exact Top-$k$ weights, softmax temperature response, Shannon entropy $H_t$) paired with live **TTFT**, **ITL (ms/tok)**, and **VRAM** gauges. | `/ws/telemetry` + `/v1/chat/completions` (`live` or `DEMO_REPLAY=true`) |
| **4. Kernel Roofline & Ablation** | **L4 vs A100 Roofline & Latency Chart**: `peft` eager Arrow vs `vLLM` static vs `KnOTS` merge vs `@triton.jit` `Arrow-SGMV` across batch sizes $B \in [1..64]$, plus 4-arm accuracy & exact `₹` cost table. | `/api/v1/benchmarks` |

---

## 4. Single-Command Container Stack (`Dockerfile`, `docker-compose.yml`)

- Multi-stage `Dockerfile`:
  1. `node:22-slim` stage compiles `ui/` via `npm ci && npm run build` into `/app/ui/dist`.
  2. `python:3.12-slim` stage installs `loraspec[studio,spectral]` (CPU-light) and serves both `/api/*`, `/ws/*`, and static SPA assets via `uvicorn loraspec.serving.gateway:app`.
- `docker-compose.yml`: Defaults to `DEMO_REPLAY=true` on port `8000` (`< 30s` cold boot on Apple Silicon OrbStack & Linux), with optional `gpu` Compose profile mounting CUDA + local adapters.

---

## 5. Automated Verification Matrix (`SDD-04`)
1. FastAPI `TestClient` + WebSocket integration tests verifying `/health`, `/api/v1/audit`, `/v1/chat/completions` SSE chunks, and `/ws/telemetry` replay frames in `DEMO_REPLAY=true` mode (`tests/integration/test_gateway_replay.py`).
2. React/TypeScript `npm run build` + `tsc --noEmit` + Playwright headless DOM & XSS-escape smoke checks (ensuring `<script>` tokens render as literal text, never executed).
3. `vLLM` worker registration mock/GPU test verifying `entry_points(group="vllm.general_plugins")` resolves `loraspec.serving.vllm_plugin:register`.
