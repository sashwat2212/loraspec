# Stage 0: Project Charter — `loraspec` (LoRA Subspace Auditor, `KnOTS` Aligner & `MoLoRA` Studio)

> [!NOTE]
> **Document Status**: Locked (`GATE-0` Passed)
> **SDLC Methodology**: **`SPEC-GATE`** 8-Stage SDLC Process (`Stage 0`–`Stage 7`)
> **Companion Research Dossier**: [RESEARCH-DOSSIER.md](./RESEARCH-DOSSIER.md)
> **Stage 1 BRD**: [01-BRD.md](./01-BRD.md)
> **Stage 2 FRD / PRD**: [02-FRD-PRD.md](./02-FRD-PRD.md)
> **Stage 3 Architecture**: [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Master Implementation Plan & RTM**: [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md)

---

## 1. Project Identity, Rename Process & Mission

### 1.1 Explicit Naming & Rename Process (4-Gate Brand & Namespace Collision Process)
During initial blueprint selection, this initiative carried the working descriptive title *"LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio"* and provisional shorthand aliases (`PrismLoRA`, `VeLoRA`, `lorafleet`). Before initializing the repository, a formal **4-Gate Brand & Namespace Collision Process** was executed (detailed in [RESEARCH-DOSSIER.md §4](./RESEARCH-DOSSIER.md#4-brand--namespace-collision-audit-naming-gate)):

1. **Gate 1 — PyPI Package Uniqueness**: Verified candidate slugs against PyPI registry availability (`lsa`, `LoraScope`, etc. rejected due to existing packages or unrelated utilities).
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
| **Workspace Strategy** | Clean standalone repository at `/Users/sashwatsinha/projects/loraspec` | Strictly decoupled from prior projects, enforcing full **`SPEC-GATE`** SDLC discipline (`Stages 0–7`, `uv`, `ruff`, strict `pyright`, `pytest`, deterministic exit codes `0–5`, SARIF 2.1.0, offline HTML, `manifest.json`). |

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

## 5. Named SDLC Methodology: The **`SPEC-GATE`** 8-Stage SDLC Process

All engineering, research, and release activity in `loraspec` follows the **`SPEC-GATE` 8-Stage SDLC Process** (*Specification-Driven, Contract-Locked, Gated, Audited, Tested, Evaluated*), spanning `Stage 0` through `Stage 7`.

### 5.1 Governance & Sign-Off Rules
1. **Sequential Upstream Gate Lock**: Work in Stage $N+1$ cannot be promoted until Stage $N$ satisfies every criterion of its formal Exit Gate (`GATE-N`).
2. **Contract & Schema Immutability**: CLI commands/flags, deterministic exit codes (`0–5`), `loraspec.toml` keys (`extra="forbid"`), and Pydantic v2 models (`src/loraspec/core/schemas.py`) lock at `GATE-2`; post-lock changes require a formal Architecture Decision Record (`ADR-*`) or documented spec amendment.
3. **Unified Multi-Dimensional Grading**: Every `BRD-REQ`, `FEAT`, `FRD-REQ`, and `NFR` is graded across all 6 rubric dimensions defined in [§6](#6-unified-multi-dimensional-grading-rubric) (`Priority`, `MoSCoW`, `Impact/Risk`, `Complexity`, `Compute Tier`, `Verification Gate`).
4. **Quantitative Evidence & Proof Rule**: No stage passes on prose claims alone. Every gate demands concrete, reproducible evidence (`ruff`, strict `pyright`, `pytest`, `uv build`, `uvx` smoke tests, CUDA parity assertions, or billing manifests).
5. **Frugal GPU Circuit Breaker**: `GATE-5` and `GATE-6 (M1)` must pass at `₹0` on CPU runners. Any `GATE-6 (M2–M4)` cloud GPU execution must first pass `--smoke` (`< ₹10`) via `GpuBudgetGuard`.
6. **Hard Milestone Solidification & Zero-Defect Gate Rule**: No transition to Milestone $k+1$ (e.g., `M2` Triton/SGMV) is permitted until Milestone $k$ passes 100% of unit, integration, scale (`KPI-ENG-01`/`NFR-02`), cross-backend parity (`numpy` vs `pure_python`), CLI exit-code, CI/Action, offline HTML/SARIF security, and documentation/AST-schema parity gates with zero open defects, silent fallbacks, or placeholders.

### 5.2 `SPEC-GATE` Stage-Gate Governance Tracker (`GATE-0` – `GATE-7`)

| Stage | Stage Name | Entry Criteria | Required Artifacts | Exit Gate & Quantitative Pass Criteria | Sign-Off Rule | Current Status |
|---|---|---|---|---|---|---|
| **Stage 0** | Charter, Naming Gate & Cost Envelope | Validated multi-LoRA systems research problem | [RESEARCH-DOSSIER.md](./RESEARCH-DOSSIER.md), [00-CHARTER.md](./00-CHARTER.md) | **`GATE-0`**: 4-Gate Brand & Namespace Audit passes (`loraspec` zero collisions); `<= ₹3,000` INR budget envelope, Explicit Non-Goals, `SPEC-GATE` process & §6 Rubric locked. | Lead Architect locks Stage 0 on `main`. | **PASSED (`GATE-0`)** |
| **Stage 1** | Business Requirements (`BRD`) | `GATE-0` PASSED | [01-BRD.md](./01-BRD.md) | **`GATE-1`**: All 22 `BRD-REQ-01..22` explicitly graded across all 6 rubric dimensions; 13 KPIs (`KPI-ENG/RES/SHW`) quantified; 6 risks (`RISK-01..06`) mitigated; Stakeholder Sign-Off Matrix complete. | Product Owner + ML Systems Lead sign-off (`01-BRD.md` §8). | **PASSED (`GATE-1`)** |
| **Stage 2** | Functional & Product Requirements (`FRD / PRD`) | `GATE-1` PASSED | [02-FRD-PRD.md](./02-FRD-PRD.md) | **`GATE-2`**: 7 Epics (`EPIC-01..07`), 22 graded Features (`FEAT-01..22`), 36 graded atomic `FRD-REQ-01..36` (with Inputs/Preconditions/Math/Outputs/Exceptions/Tests), frozen CLI/Schema/HTTP/WS contracts & graded `NFR-01..08` locked with 100% BRD coverage. | Engineering Lead + Security/QA Lead sign-off. | **PASSED (`GATE-2`)** |
| **Stage 3** | System Architecture & ADRs | `GATE-2` PASSED | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md) | **`GATE-3`**: 5-Plane C4 diagrams, `ARCH-COMP-01..12` covering 100% of `FEAT-01..22`, Threat Model `T1–T6` controls, E2E sequence flows, repo layout & `ADR-001..009` locked. | Systems Architect sign-off. | **PASSED (`GATE-3`)** |
| **Stage 4** | Subsystem Detailed Design (`SDD`) | `GATE-3` PASSED | [SDD-01](./04-DESIGN-SPECTRAL-AUDITOR.md), [SDD-02](./04-DESIGN-KNOTS-MERGER.md), [SDD-03](./04-DESIGN-TRITON-MOLORA.md), [SDD-04](./04-DESIGN-VLLM-STUDIO.md), [SDD-05](./04-DESIGN-CLI-CI-BENCHMARKS.md) | **`GATE-4`**: Exact factorized QR+SVD, Grassmann, `MergeRiskScore`, Gram-QR `KnOTS`, `Arrow` routing, two-stage `Arrow-SGMV` kernel layout, `vLLM`/Studio/SARIF specs & 4-arm L4/A100 study formulas locked. | Subsystem Leads sign-off. | **PASSED (`GATE-4`)** |
| **Stage 5** | Master RTM, Repo Init & Walking Skeleton (`v0.1.0-alpha`) | `GATE-4` PASSED | [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md), `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`, `src/loraspec/` core/inspect/auditor/reporting/cli, `tests/unit/*.py` | **`GATE-5`**: 100% 1-to-1 RTM (`36/36` FRDs, `22/22` Features, `0` wildcards/orphans); `0` unfinished specification markers or broken links; `ruff check`, `ruff format --check`, strict `pyright` (`0` errors), `pytest -v` (`100%` pass), `uv build` + out-of-tree `uvx` smoke pass; clean commit on `main`. | Automated CI (`ci.yml`) + Maintainer sign-off. | **PASSED (`GATE-5`)** |
| **Stage 6** | Phased Feature Delivery (`M1`–`M4`) | `GATE-5` PASSED | **M1**: `knots/`, HTML card, `planted_collisions/`, `action.yml`<br>**M2**: `kernels/`, CUDA parity, `loraspec bench`<br>**M3**: `vllm_plugin.py`, `gateway.py`, `demo_replay.py`, `ui/`, Docker<br>**M4**: `recipes/molora_knots_benchmark/`, `export_social_assets.py`, `TECHNICAL-REPORT.md` | **`GATE-6 (M1–M4)`**:<br>• `M1`: `KPI-RES-01` (100% recall, `<= 5%` FAR), `KnOTS` dense-SVD parity `atol < 1e-5`, `< 20s` CPU Action.<br>• `M2`: `KPI-ENG-02` (`atol <= 1e-3` FP16 / `<= 5e-3` BF16), `KPI-RES-03` (`>= 2.0x–3.0x` speedup).<br>• `M3`: `KPI-ENG-06`, Playwright XSS & WebSocket checks, `docker compose up < 30s`.<br>• `M4`: `KPI-RES-02`, `KPI-RES-04`, cumulative INR `<= ₹3,000`. | Milestone Gate Lead + Automated Unit/Integration/GPU CI. | **M1 PASSED (`GATE-6 [M1]` Solidified); M2 PASSED (`GATE-6 [M2]` Solidified on CPU/MPS + NVIDIA A30 CUDA); M3–M4 QUEUED** |
| **Stage 7** | Release Hardening & 5-Channel Launch (`v1.0.0`) | `GATE-6 (M1..M4)` PASSED | PyPI OIDC Trusted Publishing (`loraspec==1.0.0`), GitHub Marketplace `v1` release, live `DEMO_REPLAY` Studio URL, `assets/social/`, final security/license audit | **`GATE-7`**: `KPI-SHW-01..03` verified live; `pip install loraspec` & `uvx loraspec doctor` pass; Marketplace Action green on live PR; `0` open `G1-Critical` defects; final `manifest.json` cost ledger `<= ₹3,000`. | Final Release Sign-Off (Maintainer). | **QUEUED (Post-M4)** |

---

## 6. Unified Multi-Dimensional Grading Rubric

Every requirement, feature, and NFR across Stages 1–6 is graded along six orthogonal dimensions:

### 6.1 Priority Grade (`P0` / `P1` / `P2`)
| Grade | Label | Definition & Gate Policy |
|---|---|---|
| **`P0`** | **Foundation / Release Blocker** | Mandatory for Stage 5 Walking Skeleton viability, security (`pickle`-free parsing, header bounds, XSS), deterministic CLI/config contracts (`0–5`), `GpuBudgetGuard`, or PR SARIF CI gating. Failure blocks `GATE-5` and any release. |
| **`P1`** | **Core Differentiator** | Primary algorithmic and systems differentiation (`KnOTS` joint-SVD merger, `Arrow` prototypes, `@triton.jit` `Arrow-SGMV`, MPS/CPU parity, `loraspec bench`, Marketplace `action.yml`). Failure blocks `GATE-6 (M1/M2)`. |
| **`P2`** | **Serving, Studio & Empirical Study** | End-to-end production serving & showcase ecosystem (`vLLM V1` plugin, FastAPI SSE/WS gateway, `DEMO_REPLAY=true`, React 19 Studio, Docker Compose, 4-arm L4/A100 study & social exports). Failure blocks `GATE-6 (M3/M4)` or `GATE-7`. |

### 6.2 MoSCoW Grade (`Must` / `Should` / `Could`)
| Grade | Definition | Governance Rule |
|---|---|---|
| **`Must`** | Non-negotiable system invariant or core user commitment. | Cannot ship milestone or tag release if unverified. |
| **`Should`** | High-value capability committed in baseline milestone scope. | Expected in target milestone; deferral requires explicit ADR justification. |
| **`Could`** | Convenience, packaging, or polish enhancement (`Won't` items reside in §3 Non-Goals). | Delivered when schedule and INR budget remain comfortably inside envelope. |

### 6.3 Impact / Risk Grade (`G1-Critical` / `G2-High` / `G3-Standard`)
| Grade | Failure Blast Radius | Mandatory Safeguard |
|---|---|---|
| **`G1-Critical`** | Security/DoS vulnerability, silent numerical corruption, false-negative security/audit bypass, or runaway GPU billing (`> ₹3,000`). | Mandatory boundary/property/fuzz unit tests + automated CI enforcement + strict `ConfigDict` / bounds validation. |
| **`G2-High`** | Degrades multi-LoRA merge accuracy, kernel throughput, `vLLM` serving stability, or PR SARIF diagnostics. | End-to-end integration tests + numerical parity or recall checks (`KPI-RES-01..04`). |
| **`G3-Standard`** | Visual/UX presentation, container ergonomics, or empirical publication asset polish. | Automated build/DOM/smoke tests (`tsc`, `vite build`, Playwright XSS, Docker `/health`). |

### 6.4 Complexity Grade (`XS` / `S` / `M` / `L` / `XL`)
| Grade | Effort & Surface Scope | Typical Engineering Profile |
|---|---|---|
| **`XS`** | `< 0.5` dev-day; isolated helper or config schema (`<= 120` LOC). | Single module change + targeted unit tests. |
| **`S`** | `0.5–1.0` dev-day; single cohesive module (`~120–250` LOC). | Clear I/O contract + boundary/error-path unit suite. |
| **`M`** | `1–2` dev-days; multi-function module or mathematical operator (`~250–500` LOC). | Algorithmic invariants + multi-platform/determinism tests. |
| **`L`** | `2–4` dev-days; multi-module subsystem or cross-stack integration (`~500–1,000` LOC). | Cross-component integration + edge/stress tests. |
| **`XL`** | `4+` dev-days; custom `@triton.jit` GPU kernel or multi-GPU empirical study. | Property-based CUDA parity suite + L4/A100 roofline & `GpuBudgetGuard` ledger. |

### 6.5 Compute Tier (`T0` / `T1` / `T2` / `T3`) & Verification Gate (`GATE-0..GATE-7`)
- **`T0` (`$0` CPU Control & Inspection)**: Zero `torch`/`triton`; executes on macOS `arm64` & `ubuntu-latest` (`inspect`, config probes, `GpuBudgetGuard`, JSON/SARIF/HTML reporting, CLI).
- **`T1` (`$0` CPU Factorized Linear Algebra)**: Factorized thin-QR + SVD, Grassmann principal angles, and scalable `KnOTS` joint-SVD alignment (`O((d_out + d_in) * r)` RAM; NumPy fast-path + pure-Python fallback).
- **`T2` (GPU Kernel / Local MPS Parity)**: `Arrow` prototype extraction, `@triton.jit` `Arrow-SGMV` CUDA kernel (`L4` / `A100`), PyTorch MPS/CPU reference parity, and roofline `bench`.
- **`T3` (Serving & Telemetry Studio)**: `vLLM V1` entry-point plugin, FastAPI `/v1/chat/completions` + `/ws/telemetry`, `DEMO_REPLAY=true`, React 19 Studio (`ui/`), and Docker Compose stack.
- **`Verification Gate`**: Each requirement/feature cites the exact `SPEC-GATE` milestone gate (`GATE-5`, `GATE-6 (M1)`, `GATE-6 (M2)`, `GATE-6 (M3)`, `GATE-6 (M4)`, `GATE-7`) at which automated proof is required.
