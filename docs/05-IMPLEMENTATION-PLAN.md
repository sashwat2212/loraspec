# Master Feature-to-Implementation Plan & SDLC Execution Roadmap — `loraspec`

> [!NOTE]
> **Document Status**: Locked (Stage 5 Execution Blueprint)
> **Product**: **`loraspec`** — LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio
> **Purpose**: Maps every Business Requirement (`BRD-REQ-01..22`), Functional Requirement (`FRD-REQ-01..36`), and Feature (`FEAT-01..22`) **1-to-1 down to the exact milestone, source file, verification test file, exit gate, and INR budget allocation**.

---

## 1. End-to-End Requirement & Feature → Code Module Mapping

| Feature ID | BRD / FRD Coverage | Architecture & SDD | Implementation Milestone | Exact Source Module(s) | Exact Verification Test(s) |
|---|---|---|---|---|---|
| **`FEAT-01`**: Bounds-Safe `.safetensors` & PEFT Reader | `BRD-REQ-01` / `FRD-REQ-01,02` | `ARCH-COMP-03` (`SDD-01`) | **Stage 5** | `src/loraspec/inspect/safetensors_reader.py` | `tests/unit/test_safetensors_and_math.py` |
| **`FEAT-02`**: Config Parity, `rsLoRA` & Placement Audit | `BRD-REQ-02` / `FRD-REQ-03,04` | `ARCH-COMP-04` (`SDD-01`) | **Stage 5** | `src/loraspec/auditor/probes.py` (`RULE-CONF-001..004`) | `tests/unit/test_auditor_and_grading.py` |
| **`FEAT-03`**: Factorized QR+SVD Spectral Health Engine | `BRD-REQ-03` / `FRD-REQ-05,06` | `ARCH-COMP-04` (`SDD-01`) | **Stage 5** | `src/loraspec/core/spectral_math.py`, `src/loraspec/auditor/probes.py` (`RULE-SPEC-001..003`) | `tests/unit/test_safetensors_and_math.py`, `tests/unit/test_auditor_and_grading.py` |
| **`FEAT-04`**: Grassmann Principal Angles & Collision Analyzer | `BRD-REQ-04` / `FRD-REQ-07,08` | `ARCH-COMP-04` (`SDD-01`) | **Stage 5** | `src/loraspec/core/spectral_math.py`, `src/loraspec/auditor/probes.py` (`RULE-SPEC-004`) | `tests/unit/test_safetensors_and_math.py`, `tests/unit/test_auditor_and_grading.py` |
| **`FEAT-05`**: `MergeRiskScore`, Grade `A–F` & Policy Engine | `BRD-REQ-05` / `FRD-REQ-09` | `ARCH-COMP-10` (`SDD-01`) | **Stage 5** | `src/loraspec/auditor/grading.py` | `tests/unit/test_auditor_and_grading.py` |
| **`FEAT-06`**: Factorized `KnOTS` Joint-SVD Aligner | `BRD-REQ-06` / `FRD-REQ-10,11` | `ARCH-COMP-05` (`SDD-02`) | **M1** | `src/loraspec/knots/aligner.py` | `tests/unit/test_knots_aligner.py` |
| **`FEAT-07`**: `knots_ties` / `dare` / `linear` + `GenKnowSub` | `BRD-REQ-07` / `FRD-REQ-12,13` | `ARCH-COMP-05` (`SDD-02`) | **M1** | `src/loraspec/knots/operators.py` | `tests/unit/test_knots_operators.py` |
| **`FEAT-08`**: Energy Retention, RMSE & PEFT Serializer | `BRD-REQ-08` / `FRD-REQ-14,15` | `ARCH-COMP-05` (`SDD-02`) | **M1** | `src/loraspec/knots/serializer.py` | `tests/integration/test_knots_merge_e2e.py` |
| **`FEAT-09`**: `Arrow` Spectral Prototype & Top-$k$ Router | `BRD-REQ-09` / `FRD-REQ-16,17` | `ARCH-COMP-06` (`SDD-03`) | **M2** | `src/loraspec/kernels/router.py` | `tests/unit/test_arrow_router.py` |
| **`FEAT-10`**: `@triton.jit` Two-Stage Top-$k$ `Arrow-SGMV` | `BRD-REQ-10` / `FRD-REQ-18,19` | `ARCH-COMP-07` (`SDD-03`) | **M2** | `src/loraspec/kernels/arrow_sgmv.py` | `tests/gpu/test_triton_parity.py` |
| **`FEAT-11`**: PyTorch MPS/CPU Reference & Dispatcher | `BRD-REQ-11` / `FRD-REQ-20` | `ARCH-COMP-07` (`SDD-03`) | **M2** | `src/loraspec/kernels/reference.py`, `src/loraspec/kernels/dispatch.py` | `tests/unit/test_kernel_reference_parity.py` |
| **`FEAT-12`**: Roofline & Speedup Harness (`loraspec bench`) | `BRD-REQ-12` / `FRD-REQ-21` | `ARCH-COMP-07` (`SDD-03,05`) | **M2** | `src/loraspec/kernels/bench.py`, `src/loraspec/cli/app.py` | `tests/unit/test_bench_smoke.py`, `tests/gpu/test_bench_cuda.py` |
| **`FEAT-13`**: `vLLM V1` Zero-Fork `general_plugins` Hook | `BRD-REQ-13` / `FRD-REQ-22,23` | `ARCH-COMP-08` (`SDD-04`) | **M3** | `src/loraspec/serving/vllm_plugin.py`, `pyproject.toml` | `tests/integration/test_vllm_plugin.py` |
| **`FEAT-14`**: FastAPI SSE Gateway & `/ws/telemetry` | `BRD-REQ-14` / `FRD-REQ-24,25` | `ARCH-COMP-09` (`SDD-04`) | **M3** | `src/loraspec/serving/gateway.py` | `tests/integration/test_gateway_replay.py` |
| **`FEAT-15`**: `DEMO_REPLAY=true` Recorded Stream Engine | `BRD-REQ-15` / `FRD-REQ-26` | `ARCH-COMP-09` (`SDD-04`) | **M3** | `src/loraspec/serving/demo_replay.py`, `assets/demo_replays/` | `tests/integration/test_gateway_replay.py` |
| **`FEAT-16`**: React 19 Studio (4 Views + Token Arena) | `BRD-REQ-16` / `FRD-REQ-27,28,29` | `ARCH-COMP-12` (`SDD-04`) | **M3** | `ui/src/**` | `ui/package.json` (`tsc`, `vite build`, Playwright smoke) |
| **`FEAT-17`**: `Dockerfile` + `docker-compose.yml` | `BRD-REQ-17` / `FRD-REQ-30` | `ARCH-COMP-09,12` (`SDD-04`) | **M3** | `Dockerfile`, `docker-compose.yml` | Docker build & `/health` compose smoke |
| **`FEAT-18`**: JSON, SARIF 2.1.0, Offline HTML & Manifest | `BRD-REQ-18` / `FRD-REQ-31,32` | `ARCH-COMP-11` (`SDD-05`) | **Stage 5** (HTML **M1**) | `src/loraspec/reporting/json_writer.py`, `sarif_writer.py`, `html_renderer.py`, `core/manifest.py` | `tests/unit/test_reporting.py` |
| **`FEAT-19`**: Typer CLI, `loraspec.toml` & Exit `0–5` | `BRD-REQ-19` / `FRD-REQ-33` | `ARCH-COMP-01` (`SDD-05`) | **Stage 5** | `src/loraspec/cli/app.py`, `src/loraspec/core/config.py`, `errors.py`, `schemas.py` | `tests/unit/test_cli.py`, `tests/unit/test_schemas_and_config.py` |
| **`FEAT-20`**: `$0` CPU GitHub Marketplace Action | `BRD-REQ-20` / `FRD-REQ-34` | `ARCH-COMP-11` (`SDD-05`) | **M1** | `action.yml`, `.github/workflows/` | Live PR workflow & SARIF upload check |
| **`FEAT-21`**: Atomic `GpuBudgetGuard` & `--smoke < ₹10` | `BRD-REQ-21` / `FRD-REQ-35` | `ARCH-COMP-02` (`SDD-05`) | **Stage 5** | `src/loraspec/core/budget.py` | `tests/unit/test_budget.py` |
| **`FEAT-22`**: 4-Arm L4/A100 Study & Social Exporter | `BRD-REQ-22` / `FRD-REQ-36` | `ARCH-COMP-10` (`SDD-05`) | **M4** | `recipes/molora_knots_benchmark/`, `scripts/export_social_assets.py`, `docs/TECHNICAL-REPORT.md` | `tests/integration/test_recipe_smoke.py` |

---

## 2. Phase-by-Phase Execution Plan, Exit Gates & Budget Ledger

### Phase 0: Stage 5 — Repo Initialization, Complete SDLC Docs & Walking Skeleton *(Current Task — `₹0`)*
- **Deliverables**:
  - Locked SDLC corpus (`docs/RESEARCH-DOSSIER.md`, `00-CHARTER.md`, `01-BRD.md`, `02-FRD-PRD.md`, `03-ARCHITECTURE.md`, `04-DESIGN-*.md`, `05-IMPLEMENTATION-PLAN.md`).
  - Packaging & repo hygiene (`pyproject.toml`, `loraspec.toml.example`, `.gitignore`, `.editorconfig`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`, `.github/workflows/ci.yml`, `README.md`).
  - Functional Stage 5 Walking Skeleton (`loraspec` `0.1.0-alpha`): bounds-safe `.safetensors` binary reader/writer, factorized thin-QR/SVD & Grassmann principal angles, `RULE-CONF-001..004` & `RULE-SPEC-001..004`, `MergeRiskScore` & Grade `A–F`, `GpuBudgetGuard`, `RunManifest`, `report.json`, SARIF 2.1.0, and CLI (`doctor`, `inspect`, `audit`, `report`).
- **Exit Gate**:
  1. `uv sync --offline --python 3.12 --dev` (or clean `uv` env) passes cleanly.
  2. `uv run ruff check .` and `uv run ruff format --check .` pass with `0` errors.
  3. `uv run pyright` passes strict typecheck with `0` errors.
  4. `uv run pytest -v` passes 100% of unit tests (including real `.safetensors` binary fixtures).
  5. Initial git repository initialized and clean commit created on `main`.

---

### Milestone 1 (M1): Complete CPU `KnOTS` Merger, Offline HTML Report, Planted Corpus & Marketplace Action *(`₹0`)*
- **Target Duration**: Days 1–2
- **Deliverables**:
  1. `src/loraspec/knots/` (`aligner.py`, `operators.py`, `serializer.py`) implementing Gram-QR factorized `KnOTS`, `knots_ties`, `knots_dare_ties`, `knots_linear`, `GenKnowSub`, energy retention, RMSE, and PEFT export.
  2. `loraspec merge` CLI subcommand + `tests/unit/test_knots_*.py` verifying exact equivalence against dense SVD on small matrices (`atol < 1e-5`) and heterogeneous rank handling.
  3. `src/loraspec/reporting/html_renderer.py` + Jinja2 template (`report.html` zero-CDN, XSS-safe, inline SVG Grassmann heatmap & rank utilization bars).
  4. `benchmarks/planted_collisions/` corpus (dead/NaN, rank-collapsed, spectral spike, base-model mismatch, colliding sign-conflicting pair + hardened orthogonal controls) proving **100% recall, <=5% FAR (`KPI-RES-01`)**.
  5. Root `action.yml` + self-test workflow proving `< 20s` `$0` CPU PR annotation on GitHub.
- **M1 Exit Gate**: `loraspec audit` + `loraspec merge` end-to-end tests pass on CPU with `ruff`/`pyright`/`pytest` green; `action.yml` validates on `ubuntu-latest`.

---

### Milestone 2 (M2): `Arrow` Spectral Router & `@triton.jit` Top-$k$ `Arrow-SGMV` Kernel *(`₹400–₹700` L4)*
- **Target Duration**: Days 3–5
- **Deliverables**:
  1. Build and validate `Arrow` prototype extraction (`router.py`) + PyTorch MPS/CPU reference (`reference.py`) + dispatcher (`dispatch.py`) **first at `₹0` on Apple Silicon M5**.
  2. Implement `@triton.jit` `_arrow_sgmv_shrink_kernel` & `_arrow_sgmv_expand_kernel` (`arrow_sgmv.py`) with token-to-expert segment permutation and ragged rank masking ($r_m \in \{8, 16, 32, 64\}$).
  3. Run `--smoke` (`< ₹10`) on 1× E2E L4 (`₹57.82/hr`), then execute full property-based parity suite (`tests/gpu/test_triton_parity.py`) asserting `max_abs_err <= 1e-3` (FP16) / `<= 5e-3` (BF16).
  4. Autotune block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_R`, `BLOCK_K`) on L4 and run `loraspec bench` against `peft` eager Arrow (`>= 2.0x–3.0x` speedup target).
- **M2 Exit Gate**: 100% of Triton property tests pass on L4 CUDA; `peft` vs Triton `Arrow-SGMV` speedup & peak VRAM logged cleanly inside `₹700` cumulative GPU spend.

---

### Milestone 3 (M3): `vLLM V1` Zero-Fork Plugin, FastAPI Gateway, `DEMO_REPLAY` & React 19 Studio *(`₹350–₹500` L4)*
- **Target Duration**: Days 6–8
- **Deliverables**:
  1. `src/loraspec/serving/vllm_plugin.py` wired to `[project.entry-points."vllm.general_plugins"]` with lock-free ring telemetry capture.
  2. `src/loraspec/serving/gateway.py` (OpenAI SSE `/v1/chat/completions`, `/api/v1/*`, `/ws/telemetry`) + `demo_replay.py`.
  3. `ui/` React 19 + TypeScript + Vite + Tailwind dark-mode Studio (Observatory, `KnOTS` Lab, Live Token-Colorized `MoLoRA` Arena, Roofline Explorer) + `Dockerfile` + `docker-compose.yml`.
  4. Record authentic L4 multi-LoRA telemetry sessions into `assets/demo_replays/` and verify `DEMO_REPLAY=true` locally at `₹0`.
- **M3 Exit Gate**: `docker compose up` launches Studio cleanly; `test_gateway_replay.py` + Playwright DOM/XSS checks pass; live L4 `vLLM` multi-adapter mixed prompt produces domain-aligned token colors.

---

### Milestone 4 (M4) & Stage 7: Controlled 4-Arm L4/A100 Study, Public Launch & PyPI/Marketplace Release *(`₹750–₹1,200` L4+A100)*
- **Target Duration**: Days 9–12
- **Deliverables**:
  1. Prepare/standardize 6 domain LoRAs (`SQL`, `Rust`, `JSON`, `Python`, `Math`, `General`) on `Qwen2.5-3B-Instruct` or `Qwen3-4B-Instruct` (`recipes/molora_knots_benchmark/`).
  2. Run controlled 4-arm accuracy & latency ablation on L4 + strictly time-boxed **1.5–2.0 hr A100 80 GB pass** (`₹335–₹446`) generating L4 (`300 GB/s`) vs A100 (`2039 GB/s`) roofline and high-concurrency ITL/TTFT curves.
  3. Author `docs/TECHNICAL-REPORT.md` + export `2×`-DPI figures and 60fps UI `.mp4` via `scripts/export_social_assets.py`.
  4. Tag `v1.0.0`, publish `loraspec` to PyPI via OIDC Trusted Publishing, list `sashwat2212/loraspec@v1` on GitHub Marketplace, and deploy `DEMO_REPLAY=true` live Studio URL (`₹0/month`).
- **M4 Exit Gate**: All KPIs (`KPI-ENG-01..06`, `KPI-RES-01..04`, `KPI-SHW-01..03`) verified; total cumulative cloud GPU spend verified **`<= ₹3,000`** in `manifest.json`.
