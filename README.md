# `loraspec` — LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio

**Pre-flight Grassmann/SVD subspace collision & rank-collapse auditor, scalable factorized `KnOTS` joint-SVD alignment merger, and custom `@triton.jit` per-token Top-$k$ `Arrow-SGMV` multi-LoRA serving engine.**

`loraspec` bridges the 2026 production gap between **`huggingface/peft`** (static merges and unbatched eager `Arrow` routing) and **`vLLM V1`** (Punica SGMV kernels locked to one static `LoRARequest` per sequence).

---

## Current Status: Stage 5 Walking Skeleton (`v0.1.0-alpha`)

This repository contains the complete locked SDLC Stages 0–5 engineering specification corpus (`docs/`) and the verified Stage 5 Python foundation:

- **Bounds-Safe `.safetensors` & PEFT Inspector**: Zero-`pickle` binary header + IEEE-754 `F32`/`F16`/`BF16` tensor reader with `adapter_config.json` validation and `lora_A`/`lora_B` pairing (`src/loraspec/inspect/`).
- **Factorized Rank-$r$ QR+SVD & Grassmann Subspace Math**: Computes singular spectra, effective stable rank $r_{\text{eff}} = \|\Delta W\|_F^2 / \|\Delta W\|_2^2$, spectral norm spikes, and pairwise **Grassmann principal angles**, **chordal similarity**, **geodesic distance**, and **directional sign conflicts** without materializing $O(d_{\text{out}} d_{\text{in}})$ matrices—supporting both pure-Python offline CPU execution and optional NumPy BLAS/LAPACK fast paths.
- **Tier 0–1 Spectral Audit Rules & Deterministic Grading**: Evaluates `RULE-CONF-001..004` & `RULE-SPEC-001..004`, computing `MergeRiskScore (0–100)`, Grade `A–F`, and deployment policy (`SAFE_LINEAR_MERGE`, `USE_KNOTS_ALIGNMENT`, `USE_TOP_K_MOLORA`, `RETRAIN_COLLAPSED_LORA`).
- **Atomic `GpuBudgetGuard` & `RunManifest`**: Strict INR/hour budget circuit breaker (`<= ₹3,000` ceiling, `--smoke < ₹10`) and SHA-256 adapter provenance manifests.
- **Synchronized Reporting & Working CLI**: Emits `report.json`, `manifest.json`, and OASIS **SARIF 2.1.0** (with git-repo-relative normalized URIs) via `loraspec doctor`, `loraspec inspect`, `loraspec audit`, and `loraspec report`.

Stage 6 Milestones **M1–M4** deliver scalable `KnOTS` alignment (`merge`), offline HTML certification cards & the `$0` CPU GitHub Marketplace Action (`action.yml`), the `@triton.jit` Top-$k$ `Arrow-SGMV` kernel (`bench`), the `vLLM V1` plugin + FastAPI/React 19 Studio (`serve`, `replay`, `DEMO_REPLAY=true`), and the controlled 4-arm L4/A100 empirical benchmark.

---

## Quickstart

```bash
# Install core CLI & dev checks (zero GPU required)
uv sync

# 1. Check Python, CPU/MPS/CUDA backends, and optional extras
uv run loraspec doctor

# 2. Safely inspect a PEFT LoRA adapter directory (zero-pickle)
uv run loraspec inspect ./path/to/adapter

# 3. Audit >= 2 adapters for rank collapse & Grassmann collision -> JSON + SARIF 2.1.0
uv run loraspec audit ./adapters/sql_lora ./adapters/rust_lora \
  --out-dir ./artifacts/audit-01 \
  --sarif ./artifacts/audit-01/report.sarif \
  --fail-on high
```

### Development Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -v
```

---

## Documentation Index (SDLC Stages 0–5)

| Stage | Document | Description |
|---|---|---|
| Research | [docs/RESEARCH-DOSSIER.md](./docs/RESEARCH-DOSSIER.md) | 2025–2026 LoRA interference, Grassmann geometry, `KnOTS`, `Arrow`, `GenKnowSub` & `vLLM V1` dossier |
| Stage 0 | [docs/00-CHARTER.md](./docs/00-CHARTER.md) | Charter, explicit rename collision process (`loraspec`), 5-channel scope & `₹2,000–₹3,000` budget |
| Stage 1 | [docs/01-BRD.md](./docs/01-BRD.md) | Business Requirements (`BRD-REQ-01`–`22`), KPIs (`KPI-ENG/RES/SHW`) & risk register |
| Stage 2 | [docs/02-FRD-PRD.md](./docs/02-FRD-PRD.md) | Functional Requirements (`FRD-REQ-01`–`36`), `FEAT-01`–`22`, frozen CLI/API/SDK contracts |
| Stage 3 | [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) | 5-Plane Architecture (`ARCH-COMP-01`–`12`), sequence diagrams, threat model & `ADR-001`–`009` |
| Stage 4 | [docs/04-DESIGN-SPECTRAL-AUDITOR.md](./docs/04-DESIGN-SPECTRAL-AUDITOR.md) | `SDD-01`: Bounds-safe `.safetensors`, factorized QR+SVD, Grassmann angles & `MergeRiskScore` |
| Stage 4 | [docs/04-DESIGN-KNOTS-MERGER.md](./docs/04-DESIGN-KNOTS-MERGER.md) | `SDD-02`: Memory-safe Gram-QR `KnOTS` alignment, `TIES`/`DARE`, `GenKnowSub` & PEFT export |
| Stage 4 | [docs/04-DESIGN-TRITON-MOLORA.md](./docs/04-DESIGN-TRITON-MOLORA.md) | `SDD-03`: `Arrow` prototype router, `@triton.jit` Top-$k$ `Arrow-SGMV` & PyTorch parity spec |
| Stage 4 | [docs/04-DESIGN-VLLM-STUDIO.md](./docs/04-DESIGN-VLLM-STUDIO.md) | `SDD-04`: `vLLM V1` plugin, FastAPI SSE/WS gateway, `DEMO_REPLAY=true` & React 19 Studio |
| Stage 4 | [docs/04-DESIGN-CLI-CI-BENCHMARKS.md](./docs/04-DESIGN-CLI-CI-BENCHMARKS.md) | `SDD-05`: CLI, SARIF 2.1.0/HTML reports, `action.yml` & L4/A100 4-arm empirical study |
| Stage 5 | [docs/05-IMPLEMENTATION-PLAN.md](./docs/05-IMPLEMENTATION-PLAN.md) | Complete feature-to-file-to-test implementation plan, exit gates & INR budget ledger |

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
