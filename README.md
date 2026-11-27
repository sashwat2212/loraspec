# `loraspec` — LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio

**Pre-flight Grassmann/SVD subspace collision & rank-collapse auditor, scalable factorized `KnOTS` joint-SVD alignment merger, and custom `@triton.jit` per-token Top-$k$ `Arrow-SGMV` multi-LoRA serving engine.**

`loraspec` bridges the 2026 production gap between **`huggingface/peft`** (static merges and unbatched eager `Arrow` routing) and **`vLLM V1`** (Punica SGMV kernels locked to one static `LoRARequest` per sequence).

---

## Current Status: Stage 6 Milestone 1 (`M1` — `v0.1.0`, `$0` CPU Release)

This repository contains the complete **`SPEC-GATE`** 8-stage SDLC specification corpus (`docs/`) and the verified **Stage 5 + Milestone 1 (`M1`)** CPU-native engineering suite:

- **Bounds-Safe `.safetensors` & PEFT Inspector (`src/loraspec/inspect/`)**: Zero-`pickle` binary header + IEEE-754 `F32`/`F16`/`BF16` tensor reader with `adapter_config.json` validation and `lora_A`/`lora_B` pairing.
- **Factorized Rank-$r$ QR+SVD & Grassmann Subspace Math (`src/loraspec/core/`)**: Computes singular spectra, effective stable rank $r_{\text{eff}} = \|\Delta W\|_F^2 / \|\Delta W\|_2^2$, spectral norm spikes, and pairwise **Grassmann principal angles**, **chordal similarity**, **geodesic distance**, and **directional sign conflicts** without materializing $O(d_{\text{out}} d_{\text{in}})$ matrices—supporting both pure-Python offline CPU execution and optional `[spectral]` NumPy BLAS/LAPACK fast paths (`atol <= 1e-9`).
- **Scalable Factorized `KnOTS` Joint-SVD Alignment Merger (`src/loraspec/knots/`, `loraspec merge`)**: Conditioned thin-QR + compressed $R_{\text{sum}} \times R_{\text{sum}}$ core-SVD alignment across heterogeneous ranks ($r_m \in \{4..128\}$) and `union`/`intersection` layer policies, supporting `knots_ties`, `knots_dare_ties`, `knots_linear`, factorized **`GenKnowSub`** subtraction (`--gks-adapter`), trace-identity Frobenius RMSE, `RULE-KNOTS-001` energy retention checks, and PEFT export (`adapter_model.safetensors`, `adapter_config.json`, `knots_merge_manifest.json`).
- **Zero-CDN Offline HTML Certification Card (`--html`) & OASIS SARIF 2.1.0**: Renders self-contained `< 250 KB`, XSS-escaped (`autoescape=True`) HTML Certification Cards with inline SVG Grade/`MergeRiskScore` gauges, Grassmann chordal heatmaps, per-layer $r_{\text{eff}}/r$ rank bars, `KnOTSMergeSummary`, and `RunManifest` SHA-256 verification strips alongside repo-relative SARIF 2.1.0.
- **Planted Subspace Fault Corpus (`benchmarks/planted_collisions/`)**: Synthetic benchmark covering 7 planted fault classes (`RULE-CONF-001/003`, `RULE-SPEC-001..004`) and 3 orthogonal/`KnOTS` controls verifying **`100%` recall (`7/7`)** and **`0.0%` false-alarm rate (`0/3`)**.
- **`$0` CPU GitHub Marketplace Action (`action.yml`)**: Drop-in CI gate emitting live PR SARIF Code Scanning annotations and uploading HTML Certification artifacts in `< 20s` on `ubuntu-latest`.

Upcoming Stage 6 Milestones **M2–M4** deliver the `@triton.jit` Top-$k$ `Arrow-SGMV` kernel (`bench`), the `vLLM V1` plugin + FastAPI/React 19 Studio (`serve`, `replay`, `DEMO_REPLAY=true`), and the controlled 4-arm L4/A100 empirical benchmark.

---

## Quickstart

```bash
# Install core CLI & dev checks (zero GPU required; add --extra spectral for NumPy fast path)
uv sync --extra spectral

# 1. Check Python, CPU/MPS/CUDA backends, and optional extras
uv run loraspec doctor

# 2. Safely inspect a PEFT LoRA adapter directory (zero-pickle)
uv run loraspec inspect ./path/to/adapter

# 3. Audit >= 2 adapters for rank collapse & Grassmann collision -> JSON + SARIF 2.1.0 + HTML
uv run loraspec audit ./adapters/sql_lora ./adapters/rust_lora \
  --out-dir ./artifacts/audit-01 \
  --sarif ./artifacts/audit-01/report.sarif \
  --html ./artifacts/audit-01/report.html \
  --fail-on high

# 4. Align & merge heterogeneous LoRAs via factorized KnOTS joint-SVD
uv run loraspec merge ./adapters/sql_lora ./adapters/rust_lora \
  --method knots_ties \
  --target-rank 16 \
  --out-dir ./artifacts/merged_knots_lora
```

### GitHub Actions Gate (`action.yml`)

```yaml
permissions:
  contents: read
  actions: read
  security-events: write

steps:
  - uses: actions/checkout@v4
  - uses: sashwat2212/loraspec@v1
    with:
      paths: "./adapters/sql_lora ./adapters/rust_lora" # or adapter-paths
      fail-on: high
```

See [action/README.md](./action/README.md) for full inputs and outputs.

### Development Checks & Planted Corpus Benchmark

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -v
uv run python benchmarks/planted_collisions/generate_corpus.py --out-dir /tmp/loraspec-corpus --verify
```

---

## Documentation Index (`SPEC-GATE` SDLC Stages 0–5)

| Stage | Document | Description |
|---|---|---|
| Research | [docs/RESEARCH-DOSSIER.md](./docs/RESEARCH-DOSSIER.md) | 2025–2026 LoRA interference, Grassmann geometry, `KnOTS`, `Arrow`, `GenKnowSub` & `vLLM V1` dossier |
| Stage 0 | [docs/00-CHARTER.md](./docs/00-CHARTER.md) | Charter, 4-Gate rename collision process (`loraspec`), **`SPEC-GATE`** 8-Stage SDLC & Unified Grading Rubric, `₹2,000–₹3,000` budget |
| Stage 1 | [docs/01-BRD.md](./docs/01-BRD.md) | Graded Business Requirements (`BRD-REQ-01`–`22`), KPIs (`KPI-ENG/RES/SHW`), risk register & Stakeholder Sign-Off Matrix |
| Stage 2 | [docs/02-FRD-PRD.md](./docs/02-FRD-PRD.md) | Complete PRD (`EPIC-01`–`07`, Graded `FEAT-01`–`22`) & FRD (Graded `FRD-REQ-01`–`36`, CLI/API/Pydantic v2 contracts, Graded `NFR-01`–`08`) |
| Stage 3 | [docs/03-ARCHITECTURE.md](./docs/03-ARCHITECTURE.md) | 5-Plane Architecture (`ARCH-COMP-01`–`12`), sequence diagrams, threat model & `ADR-001`–`009` |
| Stage 4 | [docs/04-DESIGN-SPECTRAL-AUDITOR.md](./docs/04-DESIGN-SPECTRAL-AUDITOR.md) | `SDD-01`: Bounds-safe `.safetensors`, factorized QR+SVD, Grassmann angles & `MergeRiskScore` |
| Stage 4 | [docs/04-DESIGN-KNOTS-MERGER.md](./docs/04-DESIGN-KNOTS-MERGER.md) | `SDD-02`: Memory-safe Gram-QR `KnOTS` alignment, `TIES`/`DARE`, `GenKnowSub` & PEFT export |
| Stage 4 | [docs/04-DESIGN-TRITON-MOLORA.md](./docs/04-DESIGN-TRITON-MOLORA.md) | `SDD-03`: `Arrow` prototype router, `@triton.jit` Top-$k$ `Arrow-SGMV` & PyTorch parity spec |
| Stage 4 | [docs/04-DESIGN-VLLM-STUDIO.md](./docs/04-DESIGN-VLLM-STUDIO.md) | `SDD-04`: `vLLM V1` plugin, FastAPI SSE/WS gateway, `DEMO_REPLAY=true` & React 19 Studio |
| Stage 4 | [docs/04-DESIGN-CLI-CI-BENCHMARKS.md](./docs/04-DESIGN-CLI-CI-BENCHMARKS.md) | `SDD-05`: CLI, SARIF 2.1.0/HTML reports, `action.yml` & L4/A100 4-arm empirical study |
| Stage 5 | [docs/05-IMPLEMENTATION-PLAN.md](./docs/05-IMPLEMENTATION-PLAN.md) | Exhaustive 1-to-1 Bidirectional RTM (`36/36` FRDs, `22/22` Features, exact symbols/tests), file-by-file plan, `GATE-0..7` & INR ledger |

---

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
