# Stage 0: Project Charter — `loraspec` (LoRA Subspace Auditor, `KnOTS` Aligner & `MoLoRA` Studio)

> [!NOTE]
> **Document Status**: Locked (Stage 0)
> **Companion Research Dossier**: [RESEARCH-DOSSIER.md](./RESEARCH-DOSSIER.md)
> **Stage 1 BRD**: [01-BRD.md](./01-BRD.md)
> **Stage 2 FRD / PRD**: [02-FRD-PRD.md](./02-FRD-PRD.md)
> **Stage 3 Architecture**: [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Master Implementation Plan**: [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md)

---

## 1. Project Identity, Rename Process & Mission

### 1.1 Explicit Naming & Rename Process
During initial blueprint selection, this initiative carried the working descriptive title *"LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio"* and provisional shorthand aliases (`PrismLoRA`, `VeLoRA`, `lorafleet`). Before initializing the repository, a formal **4-Gate Brand & Namespace Collision Process** was executed (detailed in [RESEARCH-DOSSIER.md §4](./RESEARCH-DOSSIER.md#4-brand--namespace-collision-audit-naming-gate)):

1. **Gate 1 — PyPI Package Uniqueness**: Verified candidate slugs against PyPI registry availability (`lsa`, `lorascope`, etc. rejected due to existing packages or unrelated utilities).
2. **Gate 2 — arXiv & Academic Literature Separation**: Screened against 2024–2026 PEFT/LoRA papers (`VeLoRA`, `PRISM-LoRA`, `SubLoRA`, `LoRASpace` [`arXiv:2610.27444`], `LSAR` rejected to avoid academic identity collisions).
3. **Gate 3 — GitHub Repository & Marketplace Slug Clarity**: Confirmed clean `sashwat2212/loraspec` namespace alignment for GitHub Marketplace Action tagging (`sashwat2212/loraspec@v1`).
4. **Gate 4 — Developer CLI Ergonomics & Semantic Clarity**: Locked **`loraspec`** (**LoRA Spec**tral Subspace Auditor, `KnOTS` Aligner & Per-Token Top-$k$ `MoLoRA` Triton Studio)—concise `uvx loraspec` / `pip install loraspec` CLI ergonomics directly signaling LoRA spectral inspection, specification auditing, and multi-adapter composition.

### 1.2 Locked Identity
- **Package / CLI / Repo Name**: **`loraspec`**
- **Full Product Title**: **`loraspec`** — *LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio*
- **Product Category**: Open-source PyPI SDK & CLI + `$0` CPU GitHub Marketplace SARIF 2.1.0 Action + Custom `@triton.jit` GPU Kernel + `vLLM V1` Zero-Fork Plugin + Interactive React 19 Telemetry Studio (`DEMO_REPLAY=true`) + Reproducible L4/A100 Empirical Study.
- **Mission Statement**: Eliminate silent destructive interference in multi-LoRA deployments by giving ML systems and post-training teams (1) a `$0` pre-flight Grassmann/SVD subspace collision & rank-collapse auditor in CI/CD, (2) a memory-safe `KnOTS` joint-SVD alignment merger, and (3) a production-ready custom `@triton.jit` per-token Top-$k$ `Arrow-SGMV` kernel + `vLLM V1` plugin + live token-colorized Web Studio.

---

## 2. Locked Stage 0 Strategic Parameters

| Parameter | Locked Decision | Engineering & Research Implication |
|---|---|---|
| **Flagship Direction** | **Option 1**: LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio | Bridges verified 2026 whitespace between `huggingface/peft` (static merge + unbatched eager `Arrow`) and `vLLM V1` (Punica SGMV locked to 1 `LoRARequest` per sequence). |
| **Five-Channel OSS Footprint** | 1. PyPI `loraspec`<br>2. GitHub Marketplace `sashwat2212/loraspec@v1`<br>3. `vLLM V1` entry-point plugin<br>4. React 19 + FastAPI Web Studio (`docker compose up` + `DEMO_REPLAY=true`)<br>5. Hugging Face Hub adapters & reproducible manifests | Every engineering surface is unified in one strictly typed monorepo (`src/loraspec/`, `ui/`, `action.yml`, `recipes/`). |
| **Total GPU Budget Envelope** | **₹2,000 – ₹3,000 all-in (incl. 18% GST)** across existing E2E credits + ~₹1,000 cash | Enforced via `GpuBudgetGuard` + mandatory `--smoke` (`< ₹10`) pre-flight check before any L4 (`₹57.82/hr`) or A100 (`₹223.02/hr`) sweep. |
| **Cross-Platform Parity Rule** | Zero-dependency / CPU-native Tier 0–1 on macOS Apple Silicon (`arm64`) & Linux (`amd64`); automatic PyTorch MPS/CPU fallback for Triton kernels | PR CI & GitHub Marketplace Action run at **`$0` on standard `ubuntu-latest` CPU runners** in `< 20s`; CUDA/Triton tests run on GPU nodes with strict numerical parity assertions against PyTorch reference math. |
| **Workspace Strategy** | Clean standalone repository at `/Users/sashwatsinha/projects/loraspec` | Strictly decoupled from `dasv`, inheriting its SDLC discipline (`Stages 0–7`, `uv`, `ruff`, strict `pyright`, `pytest`, deterministic exit codes, SARIF 2.1.0, offline HTML, `manifest.json`). |

---

## 3. Explicit Non-Goals (Scope Guardrails)

To keep execution razor-sharp and prevent unshippable scope creep, **`loraspec` will explicitly NOT**:
1. **Rebuild base LLM LoRA training frameworks**: Training individual domain LoRAs uses standard `peft` + `trl` / `unsloth` scripts in `recipes/`, not a custom training engine.
2. **Fork `vLLM` or `transformers`**: All `vLLM V1` multi-LoRA dynamic routing integrates strictly through `[project.entry-points."vllm.general_plugins"]` and clean runtime hooks.
3. **Require GPUs for CI/CD PR Gating**: Tier 0 (`inspect`, `.safetensors` header/config checks, Grassmann/SVD subspace collision audit, SARIF 2.1.0, offline HTML) and `KnOTS` CPU alignment **must run on standard CPU runners at `$0`** without `torch` CUDA or `triton` installed.
4. **Operate a hosted paid GPU SaaS cloud**: Live demo hosting uses `DEMO_REPLAY=true` (replaying authentic recorded L4/A100 GPU telemetry streams at `₹0/month` on Vercel / Hugging Face Spaces), while live interactive inference runs locally via `docker compose up` or user-supplied GPUs.
5. **Materialize dense $O(d_{\text{out}} d_{\text{in}})$ matrices during auditing**: All spectral SVD and Grassmann computations operate strictly in factorized rank-$r$ QR+SVD space ($O((d_{\text{out}}+d_{\text{in}})r)$ memory).

---

## 4. Frugal Budget Allocation Plan (`₹2,000–₹3,000` Incl. 18% GST)

| Spend Bucket | Target Allocation | Execution & Cost Guardrail Strategy |
|---|---|---|
| **Local Dev, Pure-Python/CPU/MPS Tests & PR CI** | **₹0** | Apple M5 (16 GB) + OrbStack + `uv` offline/CPU tests + GitHub Actions `ubuntu-latest` free minutes. |
| **E2E L4 Domain Adapter Prep & `--smoke` Checks** | **₹350 – ₹550** (`6–9.5 hrs @ ₹57.82/hr`) | Fine-tune/verify 6 domain LoRAs on `Qwen2.5-3B-Instruct` or `Qwen3-4B` (SQL, Rust, JSON, Python, Math, General) with `--smoke` cap `< ₹10` per script validation. |
| **E2E L4 Triton Kernel Parity, Autotune & 4-Arm Ablation** | **₹750 – ₹1,050** (`13–18 hrs @ ₹57.82/hr`) | Verify `@triton.jit` `Arrow-SGMV` equivalence across mixed ranks, benchmark `peft` eager vs `vLLM` static vs `KnOTS` vs Triton Top-$k$ `MoLoRA`, record studio `DEMO_REPLAY` sessions. |
| **E2E A100 Roofline & High-Concurrency `vLLM` Scaling** | **₹335 – ₹446** (`1.5–2.0 hrs @ ₹223.02/hr`) | One tightly scripted final pass generating the L4 vs A100 roofline & concurrent ITL/TTFT scaling curves. |
| **Contingency Reserve** | **₹350 – ₹550** | Seed stability checks or unexpected driver/vLLM debugging. |
| **Total Hard Ceiling** | **≤ ₹3,000** (Target **₹1,800 – ₹2,500**) | Enforced by `GpuBudgetGuard` runtime timers and explicit `--max-cost-inr` caps. |

---

## 5. SDLC Stage-Gate Governance Tracker

| Stage | Artifact | Status |
|---|---|---|
| **Stage 0** | [00-CHARTER.md](./00-CHARTER.md) (Charter, Rename Gate & Budget Plan) | **LOCKED** |
| **Stage 1** | [01-BRD.md](./01-BRD.md) (Business Requirements Document, KPIs & `BRD-REQ-01..22`) | **LOCKED** |
| **Stage 2** | [02-FRD-PRD.md](./02-FRD-PRD.md) (Functional Requirements Document `FRD-REQ-01..36`, CLI/API/SDK Contracts) | **LOCKED** |
| **Stage 3** | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md) (5-Plane System Architecture, Sequence Flows, Threat Model & `ADR-001..009`) | **LOCKED** |
| **Stage 4** | [SDD-01](./04-DESIGN-SPECTRAL-AUDITOR.md) · [SDD-02](./04-DESIGN-KNOTS-MERGER.md) · [SDD-03](./04-DESIGN-TRITON-MOLORA.md) · [SDD-04](./04-DESIGN-VLLM-STUDIO.md) · [SDD-05](./04-DESIGN-CLI-CI-BENCHMARKS.md) | **LOCKED** |
| **Stage 5** | [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md) + Repo Initialization & Walking Skeleton (`loraspec` v0.1.0-alpha) | **EXECUTING (Initial Commit)** |
| **Stage 6** | Phased Delivery (`M1` Auditor/KnOTS/SARIF Action → `M2` Triton `Arrow-SGMV` → `M3` `vLLM` + React Studio → `M4` L4/A100 Study & Launch) | Queued Post-Stage 5 |
| **Stage 7** | Release Hardening, PyPI OIDC Trusted Publishing, Marketplace Action `v1`, `DEMO_REPLAY` Deployment & Technical Report | Queued Post-M4 |
