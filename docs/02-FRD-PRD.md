# Stage 2: Complete PRD & Functional Requirements Document (FRD / PRD), Graded Feature Catalog & Frozen Contracts — `loraspec`

> [!NOTE]
> **Document Status**: Locked (`GATE-2` Passed)
> **Product**: **`loraspec`** — LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio
> **SDLC Methodology & Grading Rubric**: [`SPEC-GATE` 8-Stage Process & §6 Rubric](./00-CHARTER.md#5-named-sdlc-methodology-the-spec-gate-8-stage-sdlc-process)
> **Upstream References**: [00-CHARTER.md](./00-CHARTER.md) | [01-BRD.md](./01-BRD.md)
> **Downstream References**: [03-ARCHITECTURE.md](./03-ARCHITECTURE.md) | [SDD-01](./04-DESIGN-SPECTRAL-AUDITOR.md) · [SDD-02](./04-DESIGN-KNOTS-MERGER.md) · [SDD-03](./04-DESIGN-TRITON-MOLORA.md) · [SDD-04](./04-DESIGN-VLLM-STUDIO.md) · [SDD-05](./04-DESIGN-CLI-CI-BENCHMARKS.md) | [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md)

---

## 1. PRD: Release Cutlines, Delivery Milestones & Product Epics (`EPIC-01` – `EPIC-07`)

### 1.1 Release Cutlines & Delivery Milestones
| Milestone / Release | Scope & Channels | Budget Tier (Incl. 18% GST) | Shipped User & Engineering Value | Exit Gate |
|---|---|---|---|---|
| **Stage 5 Walking Skeleton (`v0.1.0-alpha`)** | `EPIC-01` (complete), `EPIC-06` (`JSON`/`SARIF`/`Manifest` + CLI core), `EPIC-07` (`GpuBudgetGuard`): `loraspec doctor`, `inspect`, `audit`, `report`. Bounds-safe `.safetensors` + PEFT reader, factorized thin-QR/SVD + Grassmann math (NumPy fast-path + pure-Python fallback), `RULE-CONF-001..004`, `RULE-SPEC-001..004`, `MergeRiskScore`, Grade `A–F`. | **₹0** | Immediately executable, offline-capable CLI & strict-typed SDK with full CI (`ruff`, strict `pyright`, `pytest`, `uv build` + `uvx` smoke). | **`GATE-5`** |
| **M1 (`v0.1.0` Auditor + `KnOTS` + SARIF Action)** | Completes `EPIC-02` (`KnOTS` joint-SVD merger: `knots_ties`, `knots_dare_ties`, `knots_linear`, `GenKnowSub`) and remaining `EPIC-06` (`report.html` zero-CDN Jinja2 card, `action.yml` `$0` CPU GitHub Action, `benchmarks/planted_collisions/`). | **₹0** | Full pre-flight PR gate + drop-in factorized `KnOTS` static alignment merger usable by any `peft` team on CPU. | **`GATE-6 (M1)`** |
| **M2 (`v0.2.0` Triton `Arrow-SGMV` Kernel)** | Completes `EPIC-03`: `Arrow` prototype extractor + `@triton.jit` two-stage Top-$k$ `Arrow-SGMV` CUDA kernel (mixed ranks $r \in \{8,16,32,64\}$) + PyTorch MPS/CPU fallback + property-based parity suite + `loraspec bench` L4 roofline harness. | **₹400–₹700** (L4) | Custom GPU kernel outperforming `peft` eager Arrow `>= 2.0x–3.0x` with verified FP16/BF16 numerical equivalence. | **`GATE-6 (M2)`** |
| **M3 (`v0.3.0` `vLLM V1` + React 19 Studio)** | Completes `EPIC-04`, `EPIC-05`: `[project.entry-points."vllm.general_plugins"]` worker hook, FastAPI `/v1/chat/completions` + `/api/v1/*` + `/ws/telemetry`, React 19 + TypeScript + Vite + Tailwind dark-mode Studio, `DEMO_REPLAY=true`, `Dockerfile` + `docker-compose.yml`. | **₹350–₹500** (L4) | End-to-end live token-colorized multi-LoRA serving + zero-cost hosted `DEMO_REPLAY=true` interactive demo. | **`GATE-6 (M3)`** |
| **M4 (`v1.0.0` L4/A100 Study & Stage 7 Launch)** | Completes `EPIC-07` (`recipes/molora_knots_benchmark/`): 6 domain LoRAs + 4-arm accuracy/throughput study across L4 & A100, `docs/TECHNICAL-REPORT.md`, `scripts/export_social_assets.py`, PyPI OIDC + Marketplace `v1`. | **₹750–₹1,200** (L4+A100) | Complete empirical proof-of-work + 5-channel OSS distribution strictly inside `₹2,000–₹3,000` ceiling. | **`GATE-6 (M4)`** / **`GATE-7`** |

### 1.2 Product Epics Catalog (`EPIC-01` – `EPIC-07`)
| Epic ID | Epic Title | Target Personas | Core Business Outcome | Linked BRDs | Linked Features | Target Phase | Budget Cap |
|---|---|---|---|---|---|---|---|
| **`EPIC-01`** | Safetensors Ingestion & Spectral Subspace Auditing | `P2`, `P3`, `P4` | Zero-pickle bounds-safe inspection, config/spectral health probing, Grassmann collision scoring & deterministic Grade `A–F`. | `BRD-REQ-01..05` | `FEAT-01..05` | `Stage 5` | `₹0` |
| **`EPIC-02`** | Scalable `KnOTS` Joint-SVD Subspace Aligner | `P2`, `P3` | Memory-safe $O((d_{\text{out}}+d_{\text{in}})R_{\text{sum}})$ joint-SVD alignment (`knots_ties`, `knots_dare_ties`, `knots_linear`, `GenKnowSub`) with PEFT export. | `BRD-REQ-06..08` | `FEAT-06..08` | `M1` | `₹0` |
| **`EPIC-03`** | Per-Token Top-$k$ `Arrow` Spectral Routing & `@triton.jit` `Arrow-SGMV` | `P1`, `P3` | Zero-training spectral prototypes, deterministic segment sorting, two-stage Triton `Arrow-SGMV` CUDA kernel, MPS/CPU parity fallback & roofline `bench`. | `BRD-REQ-09..12` | `FEAT-09..12` | `M2` | `₹400–₹700` |
| **`EPIC-04`** | `vLLM V1` Plugin, FastAPI Streaming Gateway & `DEMO_REPLAY` | `P1`, `P2` | Zero-fork `vLLM V1` worker plugin, non-blocking ring telemetry, OpenAI SSE + REST + `/ws/telemetry`, and `₹0/month` `DEMO_REPLAY=true`. | `BRD-REQ-13..15` | `FEAT-13..15` | `M3` | `₹350–₹500` |
| **`EPIC-05`** | Interactive React 19 Web Studio & Container Stack | `P1`, `P2`, `P3` | Four-view dark-mode Studio (Observatory, `KnOTS` Lab, Live Token-Colorized `MoLoRA` Arena, Roofline Explorer) & single-command Docker Compose. | `BRD-REQ-16..17` | `FEAT-16..17` | `M3` | `₹0` (included in M3) |
| **`EPIC-06`** | CLI, Synchronized Reporting, Config & `$0` GitHub Action | `P1`, `P4` | Deterministic Typer CLI (`0–5`), `loraspec.toml` (`extra="forbid"`), synchronized JSON/SARIF 2.1.0/zero-CDN HTML/manifest, and Marketplace `action.yml`. | `BRD-REQ-18..20` | `FEAT-18..20` | `Stage 5` (`M1` HTML/Action) | `₹0` |
| **`EPIC-07`** | Cost Guardrails & Controlled L4/A100 Empirical Study | `P1`, `P2`, `P3` | Atomic `GpuBudgetGuard` (`<= ₹3,000`, `--smoke < ₹10`), 6-LoRA 4-arm L4+A100 study, `TECHNICAL-REPORT.md`, and 2×-DPI/60fps social exports. | `BRD-REQ-21..22` | `FEAT-21..22` | `Stage 5` (`Guard`), `M4` (`Study`) | `₹750–₹1,200` (+`₹350–₹550` reserve) |

---

## 2. PRD: Graded Feature Catalog (`FEAT-01` – `FEAT-22`) & Master BRD Traceability

Every feature is graded under [00-CHARTER.md §6](./00-CHARTER.md#6-unified-multi-dimensional-grading-rubric).

| Feature ID & Name | Epic | Priority | MoSCoW | Impact Grade | Complexity | Tier | Target Phase | Linked BRD | Linked FRD(s) | Arch Component | Stage 4 SDD | Verification Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`FEAT-01`**: Bounds-Safe `.safetensors` & PEFT Config Reader | `EPIC-01` | `P0` | `Must` | `G1-Critical` | `M` | `T0` | `Stage 5` | `BRD-REQ-01` | `FRD-REQ-01`, `FRD-REQ-02` | `ARCH-COMP-03` | `SDD-01` | `GATE-5` |
| **`FEAT-02`**: Config Parity, `rsLoRA` Scaling & Placement Audit | `EPIC-01` | `P0` | `Must` | `G2-High` | `S` | `T0` | `Stage 5` | `BRD-REQ-02` | `FRD-REQ-03`, `FRD-REQ-04` | `ARCH-COMP-04` | `SDD-01` | `GATE-5` |
| **`FEAT-03`**: Factorized QR+SVD Spectral Health & Rank-Collapse Engine | `EPIC-01` | `P0` | `Must` | `G1-Critical` | `L` | `T1` | `Stage 5` | `BRD-REQ-03` | `FRD-REQ-05`, `FRD-REQ-06` | `ARCH-COMP-04` | `SDD-01` | `GATE-5` |
| **`FEAT-04`**: Grassmann Principal Angles & Pairwise Collision Analyzer | `EPIC-01` | `P0` | `Must` | `G2-High` | `L` | `T1` | `Stage 5` | `BRD-REQ-04` | `FRD-REQ-07`, `FRD-REQ-08` | `ARCH-COMP-04` | `SDD-01` | `GATE-5` |
| **`FEAT-05`**: `MergeRiskScore (0–100)`, Grade `A–F` & Policy Engine | `EPIC-01` | `P0` | `Must` | `G2-High` | `S` | `T1` | `Stage 5` | `BRD-REQ-05` | `FRD-REQ-09` | `ARCH-COMP-10` | `SDD-01` | `GATE-5` |
| **`FEAT-06`**: Factorized `KnOTS` Joint-SVD Aligner (Mixed Ranks) | `EPIC-02` | `P1` | `Must` | `G2-High` | `L` | `T1` | `M1` | `BRD-REQ-06` | `FRD-REQ-10`, `FRD-REQ-11` | `ARCH-COMP-05` | `SDD-02` | `GATE-6 (M1)` |
| **`FEAT-07`**: `knots_ties` / `knots_dare_ties` / `knots_linear` + `GenKnowSub` | `EPIC-02` | `P1` | `Must` | `G2-High` | `M` | `T1` | `M1` | `BRD-REQ-07` | `FRD-REQ-12`, `FRD-REQ-13` | `ARCH-COMP-05` | `SDD-02` | `GATE-6 (M1)` |
| **`FEAT-08`**: Energy Retention, Reconstruction RMSE & PEFT Serializer | `EPIC-02` | `P1` | `Should` | `G2-High` | `M` | `T1` | `M1` | `BRD-REQ-08` | `FRD-REQ-14`, `FRD-REQ-15` | `ARCH-COMP-05` | `SDD-02` | `GATE-6 (M1)` |
| **`FEAT-09`**: `Arrow` Spectral Prototype Extractor & Top-$k$ Router Gate | `EPIC-03` | `P1` | `Must` | `G2-High` | `M` | `T2` | `M2` | `BRD-REQ-09` | `FRD-REQ-16`, `FRD-REQ-17` | `ARCH-COMP-06` | `SDD-03` | `GATE-6 (M2)` |
| **`FEAT-10`**: `@triton.jit` Two-Stage Top-$k$ `Arrow-SGMV` CUDA Kernel | `EPIC-03` | `P1` | `Must` | `G1-Critical` | `XL` | `T2` | `M2` | `BRD-REQ-10` | `FRD-REQ-18`, `FRD-REQ-19` | `ARCH-COMP-07` | `SDD-03` | `GATE-6 (M2)` |
| **`FEAT-11`**: PyTorch MPS/CPU Reference Backend & Auto-Dispatcher | `EPIC-03` | `P1` | `Must` | `G1-Critical` | `M` | `T2` | `M2` | `BRD-REQ-11` | `FRD-REQ-20` | `ARCH-COMP-07` | `SDD-03` | `GATE-6 (M2)` |
| **`FEAT-12`**: Kernel Equivalence Suite & Roofline `loraspec bench` | `EPIC-03` | `P1` | `Should` | `G2-High` | `M` | `T2` | `M2` | `BRD-REQ-12` | `FRD-REQ-21` | `ARCH-COMP-07` | `SDD-03`, `SDD-05` | `GATE-6 (M2)` |
| **`FEAT-13`**: `vLLM V1` Zero-Fork `general_plugins` Worker Integration | `EPIC-04` | `P2` | `Should` | `G2-High` | `L` | `T3` | `M3` | `BRD-REQ-13` | `FRD-REQ-22`, `FRD-REQ-23` | `ARCH-COMP-08` | `SDD-04` | `GATE-6 (M3)` |
| **`FEAT-14`**: FastAPI `/v1/chat/completions`, `/api/v1/*` & `/ws/telemetry` | `EPIC-04` | `P2` | `Should` | `G2-High` | `L` | `T3` | `M3` | `BRD-REQ-14` | `FRD-REQ-24`, `FRD-REQ-25` | `ARCH-COMP-09` | `SDD-04` | `GATE-6 (M3)` |
| **`FEAT-15`**: `DEMO_REPLAY=true` Zero-Cost Recorded Stream Engine | `EPIC-04` | `P2` | `Should` | `G2-High` | `M` | `T3` | `M3` | `BRD-REQ-15` | `FRD-REQ-26` | `ARCH-COMP-09` | `SDD-04` | `GATE-6 (M3)` |
| **`FEAT-16`**: React 19 Studio (Observatory, `KnOTS` Lab, Token Arena, Roofline) | `EPIC-05` | `P2` | `Should` | `G3-Standard` | `L` | `T3` | `M3` | `BRD-REQ-16` | `FRD-REQ-27`, `FRD-REQ-28`, `FRD-REQ-29` | `ARCH-COMP-12` | `SDD-04` | `GATE-6 (M3)` |
| **`FEAT-17`**: Single-Command `Dockerfile` + `docker-compose.yml` Stack | `EPIC-05` | `P2` | `Could` | `G3-Standard` | `S` | `T3` | `M3` | `BRD-REQ-17` | `FRD-REQ-30` | `ARCH-COMP-09`, `ARCH-COMP-12` | `SDD-04` | `GATE-6 (M3)` |
| **`FEAT-18`**: JSON, SARIF 2.1.0, Offline HTML (`Jinja2`) & `manifest.json` | `EPIC-06` | `P0` | `Must` | `G1-Critical` | `M` | `T0` | `Stage 5` (HTML `M1`) | `BRD-REQ-18` | `FRD-REQ-31`, `FRD-REQ-32` | `ARCH-COMP-11` | `SDD-05` | `GATE-5` / `GATE-6 (M1)` |
| **`FEAT-19`**: Typer CLI Surface, `loraspec.toml` (`extra="forbid"`) & Exit `0–5` | `EPIC-06` | `P0` | `Must` | `G1-Critical` | `M` | `T0` | `Stage 5` | `BRD-REQ-19` | `FRD-REQ-33` | `ARCH-COMP-01` | `SDD-05` | `GATE-5` |
| **`FEAT-20`**: `$0` CPU GitHub Marketplace Action (`action.yml`) | `EPIC-06` | `P1` | `Should` | `G2-High` | `S` | `T0` | `M1` | `BRD-REQ-20` | `FRD-REQ-34` | `ARCH-COMP-11` | `SDD-05` | `GATE-6 (M1) / GATE-7` |
| **`FEAT-21`**: Atomic `GpuBudgetGuard` & `--smoke < ₹10` Circuit Breaker | `EPIC-07` | `P0` | `Must` | `G1-Critical` | `M` | `T0..T3` | `Stage 5` | `BRD-REQ-21` | `FRD-REQ-35` | `ARCH-COMP-02` | `SDD-05` | `GATE-5` |
| **`FEAT-22`**: 4-Arm L4/A100 Empirical Study & Social Asset Exporter | `EPIC-07` | `P2` | `Should` | `G3-Standard` | `XL` | `T2..T3` | `M4` | `BRD-REQ-22` | `FRD-REQ-36` | `ARCH-COMP-10` | `SDD-05` | `GATE-6 (M4) / GATE-7` |

---

## 3. FRD: Graded Atomic Functional Requirements (`FRD-REQ-01` – `FRD-REQ-36`)

Every atomic functional requirement specifies its exact **Grades**, **Inputs**, **Preconditions**, **Processing & Mathematical/Algorithmic Rules**, **Outputs**, **Failure / Edge-Case Handling**, and **Concrete Acceptance Test Criteria**.

---

### `EPIC-01`: Safetensors Ingestion & Spectral Subspace Auditing

#### `FRD-REQ-01` — Safe `.safetensors` Header Parsing & Bounds Validation
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T0` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-01` / `FEAT-01`
- **Inputs**: `path: Path` pointing to a PEFT adapter directory (containing `adapter_model.safetensors` or single `*.safetensors` + optional `adapter_config.json`) or a direct `.safetensors` file.
- **Preconditions**: Input path resolves on local filesystem; unsafe `.bin` pickle files are disallowed.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Verify `file_size >= 8`. Read 8-byte little-endian `uint64` header length `N`.
  2. Enforce `0 < N <= 104_857_600` (`MAX_HEADER_BYTES = 100 MiB`) and `8 + N <= file_size`.
  3. Decode header slice strictly as UTF-8 JSON dictionary; validate optional `"__metadata__"` is a dict/null.
  4. For every tensor entry, verify `dtype` is in `DTYPE_WIDTH_BYTES` (`F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL`), `shape` is a list of strict non-bool ints `>= 0`, `data_offsets = [begin, end]` satisfies `0 <= begin <= end <= payload_length`, and `end - begin == prod(shape) * DTYPE_WIDTH_BYTES[dtype]`.
  5. Sort non-empty `[begin, end]` spans and assert strict non-overlap (`cur_begin >= prev_end`).
  6. Decode 2D float tensor payloads (`F32`, `F16`, `BF16`, `F64`) via NumPy `frombuffer` fast-path or pure-Python `struct`/`array` (`uint16 << 16 -> float32` for `BF16`). Compute exact SHA-256 digest of the `.safetensors` file.
- **Outputs**: `LoadedSafetensors` header index + decoded `list[list[float]]` matrices + `weights_sha256`.
- **Failure & Edge-Case Handling**: Rejects missing files, `.bin` pickle checkpoints, size `< 8`, `N == 0`, `N > 100 MiB`, truncated headers/payloads, invalid JSON, overlapping byte offsets, non-2D tensors, unsupported dtypes, or boolean/negative shapes by raising `AdapterParseError` (`EXIT_CONFIG_ERROR = 2`) with zero arbitrary code execution.
- **Acceptance Test Criteria**: `F32`/`F16`/`BF16` roundtrip decoded values match within `1e-6` (`F32`) and `2e-2` (`F16`/`BF16`); oversized `header_len = 2^50`, zero-length header, out-of-bounds offsets, overlapping offsets, malformed `__metadata__`, 1D tensors, and `.bin` checkpoints all raise `AdapterParseError`.

#### `FRD-REQ-02` — `lora_A` / `lora_B` Layer Pairing & Shape Normalization
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: S` | `Tier: T0` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-01` / `FEAT-01`
- **Inputs**: `LoadedSafetensors`, optional `adapter_config.json`, optional `adapter_id: str | None`, optional `adapter_name: str | None`.
- **Preconditions**: `FRD-REQ-01` header & payload bounds passed.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Match tensor names via `^(?P<prefix>.*?)(?P<side>lora_A|lora_B)(?:\.(?P<adapter>[^.]+))?\.weight$`. Group strictly by `(adapter_tag, layer_prefix)` (using `"default"` when un-suffixed and `"__root__"` when unprefixed).
  2. Resolve namespace unambiguously: explicit `adapter_name` if supplied, sole namespace if 1, `"default"` if present, else reject ambiguous multi-adapter files.
  3. Assert both `"lora_A"` and `"lora_B"` exist for each layer, both are 2D, `rank_a == rank_b > 0`, `in_features > 0`, and `out_features > 0`.
  4. Validate PEFT config scalars strictly (`r` strict positive int non-bool; `lora_alpha` finite non-bool float; `lora_dropout` finite `>= 0.0`). Compute scaling $s = \alpha / \sqrt{r}$ if `use_rslora` else $\alpha / r$.
- **Outputs**: `LoadedLoRAWeights` containing canonical `LoRAAdapterSpec` (with sorted `list[LoRALayerSpec]`, `total_lora_params`, `metadata`) and `layer_factors: dict[str, tuple[Matrix, Matrix]]`.
- **Failure & Edge-Case Handling**: Empty safetensors, unpaired `lora_A`/`lora_B`, mismatched inner ranks, zero feature dims (`<= 0`), ambiguous multi-adapter files without `adapter_name`, malformed `adapter_config.json`, or boolean `r`/`lora_alpha` raise `AdapterParseError` (`EXIT_CONFIG_ERROR = 2`).
- **Acceptance Test Criteria**: Unprefixed `lora_A.weight`/`lora_B.weight` pairs cleanly; multi-adapter files isolate `adapter_name="beta"` without cross-pairing; unpaired, rank-mismatched, empty-dim, and boolean `r` checkpoints raise `AdapterParseError`.

#### `FRD-REQ-03` — Multi-Adapter Config & Base-Model Drift Verification
- **Grades & Traceability**: `P0` | `Must` | `G2-High` | `Complexity: S` | `Tier: T0` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-02` / `FEAT-02`
- **Inputs**: `adapters: list[LoRAAdapterSpec]`, `cfg: AuditConfig`.
- **Preconditions**: Each adapter successfully loaded into `LoRAAdapterSpec`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Per adapter: check `config_missing` (`RULE-CONF-002`, `HIGH`, tier 0), `peft_type.upper() != "LORA"` (`RULE-CONF-002`, `HIGH`, tier 0), missing/non-finite/non-positive `lora_alpha <= 0.0` (`RULE-CONF-002`, `HIGH`, tier 0), and `config_rank_mismatch` when `rank_pattern` is absent (`RULE-CONF-002`, `HIGH`, tier 0).
  2. Across `>= 2` adapters: compare non-null `base_model_name_or_path` values (`RULE-CONF-001`, `CRITICAL`, tier 0 if `> 1` distinct base models, pointing `file_path` to the conflicting adapter's config); compare `(in_features, out_features)` on shared layer names (`RULE-CONF-001`, `CRITICAL`, tier 0 on mismatch); compare fleet `rank` and `use_rslora` sets (`RULE-CONF-004`, `MEDIUM`, tier 0 if mixed).
- **Outputs**: Ordered `list[AuditFinding]` with exact `rule_id`, `severity`, `tier=0`, `source="config_probe"`, `file_path`, `line_number`, `evidence`, `remediation`.
- **Failure & Edge-Case Handling**: Raw `.safetensors` adapters lacking `adapter_config.json` do not crash fleet parity checks; base-model comparison cleanly skips `None` raw adapters while still pinpointing conflicting config files.
- **Acceptance Test Criteria**: Base-model mismatch emits `RULE-CONF-001` pointing to the mismatched config; unsupported `peft_type="IA3"`, missing/zero `lora_alpha`, missing config, and rank disagreement emit `RULE-CONF-002`; mixed ranks/rsLoRA emit `RULE-CONF-004`.

#### `FRD-REQ-04` — All-Linear Placement Audit (`RULE-CONF-003`)
- **Grades & Traceability**: `P0` | `Must` | `G2-High` | `Complexity: XS` | `Tier: T0` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-02` / `FEAT-02`
- **Inputs**: `adapters: list[LoRAAdapterSpec]`, `cfg: AuditConfig` (`require_all_linear: bool = True`).
- **Preconditions**: `LoRAAdapterSpec.layers` and `target_modules` populated.
- **Processing & Mathematical/Algorithmic Rules**:
  1. When `cfg.require_all_linear` is `True`, form `modules_present = {layer.module_type for layer in spec.layers} | set(spec.target_modules)`.
  2. Check intersection with `MLP_MODULE_MARKERS = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}`.
  3. If intersection is empty, emit `RULE-CONF-003` (`HIGH`, tier 0) citing attention-only placement omission of MLP projections (*LoRA Without Regret*).
- **Outputs**: `AuditFinding` (`RULE-CONF-003`, `HIGH`, `tier=0`) appended when MLP markers are missing.
- **Failure & Edge-Case Handling**: Disabling `require_all_linear = false` in `loraspec.toml` suppresses `RULE-CONF-003` cleanly without affecting other probes.
- **Acceptance Test Criteria**: Attention-only (`q_proj`) adapter triggers `RULE-CONF-003`; all-linear (`q_proj, gate_proj, up_proj, down_proj`) adapter yields zero `RULE-CONF-003` findings.

#### `FRD-REQ-05` — Factorized Thin-QR + SVD Computation
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: L` | `Tier: T1` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-03` / `FEAT-03`
- **Inputs**: `adapter_id: str`, `layer_spec: LoRALayerSpec`, `a_mat: Matrix` ($r \times d_{\text{in}}$), `b_mat: Matrix` ($d_{\text{out}} \times r$), `scaling: float`, `backend: MathBackend = "auto"`.
- **Preconditions**: Dimensions match `layer_spec.a_shape` and `layer_spec.b_shape`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Detect non-finite entries (`NaN`/`Inf`) first; return zeroed `has_non_finite=True` metrics/factors cleanly without failing QR/SVD.
  2. Never materialize dense $\Delta W = s B A \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Compute thin QR $s B = Q_B R_B$ ($Q_B \in \mathbb{R}^{d_{\text{out}} \times r}$) and $A^T = Q_A R_A$ ($Q_A \in \mathbb{R}^{d_{\text{in}} \times r}$).
  3. Compute SVD on tiny $r \times r$ core $C = R_B R_A^T = U_c \Sigma V_c^T$. Form $U = Q_B U_c$, $V = Q_A V_c$.
  4. Retain active singular directions with $\sigma_i > \max(10^{-9}, 10^{-6}\sigma_1)$ and synchronize column orthonormality (`_sync_orthonormal_bases`) so `len(active_sigmas) == ncol(u_active) == ncol(v_active)` even when $d < r$.
  5. Use NumPy BLAS/LAPACK fast-path when available; pure-Python 2-pass Modified Gram-Schmidt `thin_qr` + Hestenes one-sided Jacobi `small_svd` fallback otherwise.
- **Outputs**: Tuple `(LayerSpectralMetrics, LayerSubspaceFactors)`.
- **Failure & Edge-Case Handling**: Zero matrices ($\sigma_1 \le 10^{-12}$), rank-deficient layers ($d < r$), and non-finite weights return finite bounded schemas (`effective_stable_rank=0.0`, `u_active=[]`) without raising `ZeroDivisionError` or `LinAlgError`.
- **Acceptance Test Criteria**: `thin_qr` yields $Q^T Q = I$ (`abs <= 1e-10`) and $QR = M$ (`abs <= 1e-10`); `small_svd` reconstructs rectangular matrices (`abs <= 1e-9`); rank-4 orthogonal factors yield `effective_stable_rank == 4.0`, `rsLoRA` scales norm by $\sqrt{r}$, gauge transform $(2.5B, A/2.5)$ leaves singular values invariant, and $d < r$ keeps active bases aligned (`len == 2`).

#### `FRD-REQ-06` — Spectral Health & Rank-Collapse Rules (`RULE-SPEC-001..003`)
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T1` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-03` / `FEAT-03`
- **Inputs**: Per-adapter `list[LayerSpectralMetrics]`, `cfg: AuditConfig`.
- **Preconditions**: `FRD-REQ-05` computed for all layers (`run_tier1=True`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute exact metrics per layer: $\|\Delta W\|_2 = \sigma_1$, $\|\Delta W\|_F = \sqrt{\sum \sigma_i^2}$, $r_{\text{eff}} = \|\Delta W\|_F^2 / \sigma_1^2$, utilization $= r_{\text{eff}}/r$, `top1_energy_ratio` $= \sigma_1^2 / \sum \sigma_i^2$, `condition_number` $= \sigma_1 / \max(\sigma_r, 10^{-12})$. Preserve `12`-decimal rounding on norm fields so tiny non-zero weights ($4\times 10^{-9}$) do not underflow to `0.0`.
  2. Evaluate `RULE-SPEC-001` (`CRITICAL`, tier 1): triggers if any layer has `has_non_finite=True` OR `frobenius_norm <= 1e-12`.
  3. Evaluate `RULE-SPEC-002` (`HIGH`, tier 1): across non-dead finite layers with `rank >= 4`, triggers if mean `rank_utilization < cfg.rank_collapse_ratio` (`0.20`) OR fraction of layers with `top1_energy_ratio > cfg.top1_energy_threshold` (`0.85`) is `>= 0.30`.
  4. Evaluate `RULE-SPEC-003` (`HIGH`, tier 1): across layers with `spectral_norm > 0.0`, compute `spike_limit = max(10.0, cfg.spectral_spike_multiplier * median(norms))`; trigger if any layer `spectral_norm > spike_limit`.
- **Outputs**: Populated `LayerSpectralMetrics` list + `RULE-SPEC-001..003` `AuditFinding` objects.
- **Failure & Edge-Case Handling**: Dead zero layers already flagged by `RULE-SPEC-001` are excluded from `RULE-SPEC-002` denominators so single dead layers do not cause spurious collapse false alarms; tiny non-zero weights (`4e-9 > 1e-12`) do not falsely trigger `RULE-SPEC-001`.
- **Acceptance Test Criteria**: `NaN` or zero weights trigger `RULE-SPEC-001`; rank-1 collapsed `r=8` adapter triggers `RULE-SPEC-002` (`r_eff == 1.0`); `25x` spiked layer triggers `RULE-SPEC-003`; `4e-9` tiny non-zero layer passes without `RULE-SPEC-001`.

#### `FRD-REQ-07` — Pairwise Grassmann Principal Angles & Similarity
- **Grades & Traceability**: `P0` | `Must` | `G2-High` | `Complexity: M` | `Tier: T1` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-04` / `FEAT-04`
- **Inputs**: Active subspace factors `fa: LayerSubspaceFactors`, `fb: LayerSubspaceFactors` for each shared layer across every pair `(adapter_a, adapter_b)`.
- **Preconditions**: Shared layer has matching `(in_features, out_features)` (mismatched layers already caught by `RULE-CONF-001` are skipped safely).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute cross-Gram matrices $M_U = U_a^T U_b \in \mathbb{R}^{k_a \times k_b}$ and $M_V = V_a^T V_b \in \mathbb{R}^{k_{va} \times k_{vb}}$.
  2. Compute singular values of $M_U$ ($s_{U,i}$) and $M_V$ ($s_{V,i}$); clip top $p = \min(k_a, k_b)$ principal cosines $c_i = \text{clip}(s_{U,i}, 0.0, 1.0)$.
  3. Compute principal angles $\theta_i = \arccos(c_i)$ (in degrees), `cos_principal_angle_max = c_1`, column `grassmann_chordal_similarity` $= \frac{1}{p}\sum_{i=1}^p c_i^2$, `grassmann_geodesic_dist` $= \sqrt{\sum_{i=1}^p \theta_{i,\text{rad}}^2}$, and `row_chordal_similarity` $= \frac{1}{p_{\text{row}}}\sum c_{V,i}^2$.
- **Outputs**: `PairwiseCollisionMetrics` entry per shared layer and adapter pair.
- **Failure & Edge-Case Handling**: Disjoint layer sets across two adapters emit `0` pairwise collision objects cleanly; non-finite or zero-rank factors emit zeroed `PairwiseCollisionMetrics` (`chordal=0.0`, `angles=[]`).
- **Acceptance Test Criteria**: Identical subspaces yield `angles == 0.0°`, `chordal == 1.0` (`abs <= 1e-5`); orthogonal subspaces yield `angles == 90.0°`, `chordal == 0.0`; disjoint layers complete cleanly with `[]` collisions.

#### `FRD-REQ-08` — Sign-Conflict & Subspace Collision Diagnostics (`RULE-SPEC-004`)
- **Grades & Traceability**: `P0` | `Must` | `G2-High` | `Complexity: M` | `Tier: T1` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-04` / `FEAT-04`
- **Inputs**: `fa`, `fb` (including `active_sigmas`), cross-Gram matrices $M_U, M_V$, and `cfg: AuditConfig`.
- **Preconditions**: `FRD-REQ-07` active column/row factors computed.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute gauge-invariant Frobenius cancellation ratio over active singular triplets:
     $$\text{signed\_overlap}_{i,j} = \sigma_{a,i}\,\sigma_{b,j}\,(M_U)_{i,j}\,(M_V)_{i,j}, \qquad \text{sign\_conflict\_rate} = \frac{\sum_{\text{signed\_overlap}<0} |\text{signed\_overlap}_{i,j}|}{\sum_{i,j} |\text{signed\_overlap}_{i,j}|}$$
     (`0.0` if denominator `<= 1e-12`). This metric is strictly invariant under $(B, A) \mapsto (-B, -A)$ sign-gauge flips.
  2. Aggregate per adapter pair across `pair_metrics`:
     - `crit_hits`: layers with `grassmann_chordal_similarity >= cfg.collision_crit_chordal` (`0.70`) AND `sign_conflict_rate >= cfg.sign_conflict_threshold` (`0.35`).
     - `high_hits`: layers with `grassmann_chordal_similarity >= cfg.collision_high_chordal` (`0.45`) OR (`cos_principal_angle_max >= 0.85` AND `sign_conflict_rate >= 0.25`).
  3. Emit `RULE-SPEC-004`: `CRITICAL` if `len(crit_hits) / len(pair_metrics) >= 0.20`; else `HIGH` if `high_hits` non-empty.
- **Outputs**: `sign_conflict_rate` on `PairwiseCollisionMetrics` + `RULE-SPEC-004` `AuditFinding`.
- **Failure & Edge-Case Handling**: Identical weights duplicated across paths yield `chordal=1.0`, `sign_conflict=0.0` -> `HIGH RULE-SPEC-004` (not `CRITICAL`), allowing `--fail-on critical` policies to distinguish destructive cancellation from duplicate alignment.
- **Acceptance Test Criteria**: Opposite-sign colliding pair emits `chordal == 1.0`, `sign_conflict == 1.0`, `CRITICAL RULE-SPEC-004`; duplicate identical pair emits `HIGH RULE-SPEC-004` and passes `--fail-on critical`.

#### `FRD-REQ-09` — Deterministic `MergeRiskScore`, Grade (`A–F`) & Recommendations
- **Grades & Traceability**: `P0` | `Must` | `G2-High` | `Complexity: S` | `Tier: T1` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-05` / `FEAT-05`
- **Inputs**: `findings: Sequence[AuditFinding]`, `layer_metrics: Sequence[LayerSpectralMetrics]`, `collisions: Sequence[PairwiseCollisionMetrics]`.
- **Preconditions**: Config and Tier 1 probes evaluated (or Tier 0 if `--tiers 0`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute components in `[0.0, 1.0]`:
     - $I_{\text{crit\_conf\_or\_nan}} = 1.0$ if `CRITICAL RULE-CONF-001` or `CRITICAL RULE-SPEC-001`, else `0.0`.
     - $C_{\text{collision}} = \min(1.0, 0.7 \cdot \max_p w_p + 0.3 \cdot \text{mean}_p w_p)$ where $w_p = \text{chordal}_p (0.5 + 0.5 \cdot \text{conflict}_p)$ (floored at `0.9` if `CRITICAL RULE-SPEC-004`).
     - $R_{\text{collapse}} = \min(1.0, \max(0.5, 1.0 - \min_{r\ge 4} r_{\text{eff}}/r))$ when `RULE-SPEC-002` present, else `0.0`.
     - $S_{\text{spike\_or\_config}} = \min(1.0, 0.5 \cdot \text{count}(\text{RULE-CONF-002,003,004 or RULE-SPEC-003}))$.
  2. $\text{MergeRiskScore} = \text{round}(\text{clamp}(35 I + 30 C + 20 R + 15 S, 0, 100), 2)$.
  3. Assign Grade: `F` if any `CRITICAL` or `score > 60.0`; `D` if any `HIGH` or `score > 35.0`; `C` if `score > 15.0`; `B` if any findings or `score > 0.0`; `A` if `0` findings and `score == 0.0`.
  4. Synthesize ordered `list[DeploymentRecommendation]`: `QUARANTINE_CORRUPT_ADAPTER` (`RULE-CONF-001`/`RULE-SPEC-001`), `RETRAIN_COLLAPSED_LORA` (`RULE-SPEC-002`), `USE_TOP_K_MOLORA` + `USE_KNOTS_ALIGNMENT` (`CRITICAL RULE-SPEC-004`), `USE_KNOTS_ALIGNMENT` (`HIGH RULE-SPEC-004` or `RULE-CONF-004`, plus `USE_TOP_K_MOLORA` if max chordal `>= 0.45`), else `SAFE_LINEAR_MERGE`.
- **Outputs**: `merge_risk_score: float`, `grade: Literal["A","B","C","D","F"]`, `recommendations: list[DeploymentRecommendation]`.
- **Failure & Edge-Case Handling**: Empty findings/collisions cleanly produce `score=0.0`, `grade="A"`, `[SAFE_LINEAR_MERGE]`.
- **Acceptance Test Criteria**: Boundary tests assert exact transitions across `(0.0 -> A)`, `(7.5+med -> B)`, `(20.0+med -> C)`, `(40.0+high -> D)`, `(65.0+high -> F)`, plus `F` on `CRITICAL RULE-SPEC-004` and `RULE-SPEC-001`.

---

### `EPIC-02`: Scalable `KnOTS` Joint-SVD Subspace Aligner

#### `FRD-REQ-10` — Factorized Joint-SVD Shared Basis Extraction (`KnOTS`)
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: L` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-06` / `FEAT-06`
- **Inputs**: $K \ge 2$ `LoadedLoRAWeights` with matching base architecture (`base_model_name_or_path` and `(d_out, d_in)` per layer).
- **Preconditions**: Pre-merge `RULE-CONF-001` and `RULE-SPEC-001` check passes (zero non-finite or incompatible base adapters).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Let $R_{\text{sum}} = \sum_{m=1}^K r_m$. Never instantiate $\Delta W_{\text{stack}} \in \mathbb{R}^{d_{\text{out}} \times (K d_{\text{in}})}$.
  2. Form stacked scaled left factors $B_{\text{cat}} = [s_1 B_1, \dots, s_K B_K] \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}$ and compute thin QR $B_{\text{cat}} = Q_U R_U$ with slices $R_{U,m} \in \mathbb{R}^{R_{\text{sum}} \times r_m}$.
  3. Compute per-adapter thin QR $A_m^T = Q_{A,m} R_{A,m}$ ($Q_{A,m} \in \mathbb{R}^{d_{\text{in}} \times r_m}$), assemble tiny stacked core $M = [R_{U,1} R_{A,1}^T, \dots, R_{U,K} R_{A,K}^T] \in \mathbb{R}^{R_{\text{sum}} \times R_{\text{sum}}}$, and run SVD $M = W_c \Sigma \tilde{V}_{\text{cat}}^T$ (numerically conditioned per `SDD-02 §1.1` avoiding squared condition number $\kappa^2$).
  4. Recover exact shared orthogonal left basis $U = Q_U W_c \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}$ and aligned task representations $V_m^T = \tilde{V}_{\text{cat}, m}^T Q_{A,m}^T \in \mathbb{R}^{R_{\text{sum}} \times d_{\text{in}}}$.
- **Outputs**: `FactorizedKnOTSResult` per layer containing `(U, sigmas, {V_m_T})`.
- **Failure & Edge-Case Handling**: Base-model/shape mismatches or non-finite tensors abort cleanly before merging (`AuditFailedError` / `AdapterParseError`, exit code `1` or `2`).
- **Acceptance Test Criteria**: On synthetic dense matrices, reconstructed $U \Sigma [V_1^T, \dots, V_K^T]$ matches brute-force dense SVD of $\Delta W_{\text{stack}}$ within `atol < 1e-5` with `0` $d_{\text{out}} \times (K d_{\text{in}})$ allocations.

#### `FRD-REQ-11` — Heterogeneous Rank Harmonization
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: M` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-06` / `FEAT-06`
- **Inputs**: Adapters having differing ranks $r_m \in \{4, 8, 16, 32, 64, 128\}$ and/or mixed `use_rslora` flags, plus `--target-rank r_out`.
- **Preconditions**: `FRD-REQ-10` factorized basis computed; `r_out >= 1`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Apply each adapter's exact scaling ($s_m = \alpha_m / r_m$ or $\alpha_m / \sqrt{r_m}$) when constructing $B_{\text{cat}}$ so mixed scaling conventions align consistently.
  2. Index block slices $R_{U,m}$ according to exact column offsets `[sum_{j<m} r_j : sum_{j<=m} r_j]` inside $R_{\text{sum}}$.
  3. Project every adapter into the common $R_{\text{sum}}$ shared basis prior to merging and compress final merged factors to `min(target_rank, R_sum, d_in, d_out)`.
- **Outputs**: Uniform-rank `target_rank` aligned representations ready for merging.
- **Failure & Edge-Case Handling**: Disjoint optional layers present in a subset of adapters are either copied with scaled task weight or excluded cleanly per configurable `--layer-policy` (`union` default, `intersection` optional).
- **Acceptance Test Criteria**: Merging mixed $r \in \{8, 16, 32\}$ adapters yields valid `(r_out, d_in)` and `(d_out, r_out)` tensors with zero shape or slice-index mismatch errors.

#### `FRD-REQ-12` — Aligned `knots_ties`, `knots_dare_ties`, `knots_linear` Operators
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: M` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-07` / `FEAT-07`
- **Inputs**: Aligned `{V_m_T}`, normalized task weights `{w_m}` (default uniform $1/K$), `method`, `ties_density` (default `0.50`), `dare_drop_rate` (default `0.20`), `rng_seed` (default `42`).
- **Preconditions**: Weights sum > 0 and finite; `0 < ties_density <= 1.0`; `0.0 <= dare_drop_rate < 1.0`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. `knots_linear`: $V_{\text{merged}}^T = \sum_{m=1}^K w_m V_m^T$.
  2. `knots_ties`: trim each $V_m^T$ keeping top `ties_density` entries by $|v|$; compute elected sign $\gamma_{i,j} = \text{sign}(\sum_m w_m \tilde{V}_{m,i,j}^T)$; compute disjoint weighted average across entries matching $\gamma_{i,j}$.
  3. `knots_dare_ties`: draw seeded Bernoulli mask $M_m \sim \text{Bernoulli}(1 - p_{\text{dare}})$ using `rng_seed`, rescale $\tilde{V}_m^T = (M_m \odot V_m^T) / (1 - p_{\text{dare}})$, then apply `knots_ties`.
- **Outputs**: Aligned merged right factor $V_{\text{merged}}^T \in \mathbb{R}^{R_{\text{sum}} \times d_{\text{in}}}$.
- **Failure & Edge-Case Handling**: Invalid method name or out-of-range `ties_density`/`dare_drop_rate` raises `ConfigError` (`EXIT_CONFIG_ERROR = 2`); all-zero elected entries resolve cleanly to `0.0` without `0/0` `NaN`.
- **Acceptance Test Criteria**: Seeded `knots_dare_ties` is bitwise reproducible; `knots_ties` achieves lower sign-conflict cancellation than unaligned TIES on rotated conflicting adapter pairs.

#### `FRD-REQ-13` — `GenKnowSub` Residual Subtraction
- **Grades & Traceability**: `P1` | `Should` | `G2-High` | `Complexity: M` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-07` / `FEAT-07`
- **Inputs**: Task adapters `{(B_m, A_m, s_m)}`, optional `--gks-adapter` `(B_0, A_0, s_0)`, subtraction coefficient `--gks-lambda` (default `1.0`).
- **Preconditions**: General-knowledge adapter matches base model and layer dimensions of target task adapters.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Subtract general-knowledge direction $\lambda \Delta W_{\text{gen}}$ cleanly in factorized space for each task $m$: augment $B_m^{\text{gks}} = [s_m B_m,\; -\lambda s_0 B_0]$ and $A_m^{\text{gks}} = [A_m^T,\; A_0^T]^T$ (rank $r_m + r_0$) without materializing dense $d_{\text{out}} \times d_{\text{in}}$ matrices (`arXiv:2505.10939`).
  2. Pass purified factors into either `KnOTS` joint-SVD alignment (`FRD-REQ-10`) or `Arrow` prototype extraction (`FRD-REQ-16`).
- **Outputs**: Purified factorized task updates (or purified unit prototypes).
- **Failure & Edge-Case Handling**: Dimension/base-model mismatch between `--gks-adapter` and target adapters raises `AdapterParseError` (`EXIT_CONFIG_ERROR = 2`).
- **Acceptance Test Criteria**: Synthetic task adapter equal to `task_pure + gen_adapter` recovers `task_pure` within `atol < 1e-5` after `apply_genknowsub(..., lambda=1.0)` with zero dense $d^2$ allocations.

#### `FRD-REQ-14` — Singular Energy Retention & Factorized Reconstruction Diagnostics
- **Grades & Traceability**: `P1` | `Should` | `G2-High` | `Complexity: S` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-08` / `FEAT-08`
- **Inputs**: Aligned $U, \Sigma, V_{\text{merged}}^T$, target rank `r_out`, source task factors `{(B_m, A_m, s_m)}`, `min_energy_retention: float = 0.90`.
- **Preconditions**: `FRD-REQ-12` merged representation produced.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Run thin QR $V_{\text{merged}} = Q_V R_V$ and thin SVD on tiny $R_{\text{sum}} \times R_{\text{sum}}$ core $\Sigma R_V^T = \tilde{U} \tilde{\Sigma} \tilde{V}^T$.
  2. Compute retained singular energy $\eta_{r_{\text{out}}} = \sum_{i=1}^{r_{\text{out}}} \tilde{\sigma}_i^2 / \sum_{i} \tilde{\sigma}_i^2$.
  3. Compute factorized Frobenius RMSE vs. each weighted source task adapter via inner traces $\text{tr}(\Delta W_{\text{out}}^T \Delta W_m)$ in rank-$r$ space (zero dense matrices) and compute pre- vs. post-alignment sign-conflict reduction percentage.
  4. Emit `RULE-KNOTS-001` (`MEDIUM`, tier 1) if mean `energy_retained < min_energy_retention` (`0.90`).
- **Outputs**: `KnOTSMergeSummary` Pydantic model + optional `RULE-KNOTS-001` finding.
- **Failure & Edge-Case Handling**: Zero merged energy sets `energy_retained = 1.0` cleanly without division by zero and warns via `AuditFinding`.
- **Acceptance Test Criteria**: Full-rank retention (`r_out >= R_sum`) yields `energy_retained == 1.0`; low-rank truncation (`eta < 0.90`) reliably triggers `RULE-KNOTS-001`.

#### `FRD-REQ-15` — PEFT-Compatible Export & Merge Manifest Serialization
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: S` | `Tier: T1` | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-08` / `FEAT-08`
- **Inputs**: Truncated top-$r_{\text{out}}$ factors $B_{\text{out}} = U \tilde{U}[:, :r_{\text{out}}] \sqrt{\tilde{\Sigma}[:r_{\text{out}}]}$ and $A_{\text{out}} = \sqrt{\tilde{\Sigma}[:r_{\text{out}}]} \tilde{V}^T[:r_{\text{out}}, :] Q_V^T$, output directory `out_dir: Path`, dtype (`F16` / `BF16` / `F32`), source specs.
- **Preconditions**: Merged factors finite and validated.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Write `adapter_model.safetensors` via bounds-safe `write_safetensors` with exact PEFT key naming (`base_model.model...lora_A.weight`, `...lora_B.weight`).
  2. Write PEFT `adapter_config.json` with `peft_type="LORA"`, `r=r_out`, `lora_alpha=float(r_out)` (scaling `1.0`), `base_model_name_or_path`, `target_modules`, `use_rslora=False`, `bias="none"`.
  3. Compute `output_sha256` and write `knots_merge_manifest.json` (`KnOTSMergeSummary` + `RunManifest`).
- **Outputs**: Complete PEFT directory (`adapter_model.safetensors`, `adapter_config.json`, `knots_merge_manifest.json`).
- **Failure & Edge-Case Handling**: Unwritable `out_dir` exits with `EXIT_CONFIG_ERROR (2)`; emitted adapter round-trips through `loraspec inspect` and `loraspec audit` cleanly.
- **Acceptance Test Criteria**: Merged directory loads cleanly via `load_peft_adapter` and passes `peft.PeftModel.from_pretrained()` schema validation.

---

### `EPIC-03`: Per-Token Top-$k$ `Arrow` Spectral Routing & `@triton.jit` `Arrow-SGMV` Kernel

#### `FRD-REQ-16` — `Arrow` Spectral Prototype Extraction
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: M` | `Tier: T2` | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-09` / `FEAT-09`
- **Inputs**: List of $K$ `LoadedLoRAWeights`, optional `gks_adapter: LoadedLoRAWeights | None`, `gks_lambda: float = 1.0`.
- **Preconditions**: Shared target layer dimensions align across adapters (`RULE-CONF-001` passes).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Optionally subtract `--gks-adapter` via `FRD-REQ-13` in factorized space.
  2. For each adapter $m$ and target linear layer $\ell$, obtain top right singular vector $v_1^{(m,\ell)} \in \mathbb{R}^{d_{\text{in}}}$ from factorized QR+SVD (`FRD-REQ-05`).
  3. Normalize $p_m^{(\ell)} = v_1^{(m,\ell)} / \max(\|v_1^{(m,\ell)}\|_2, 10^{-12})$ and stack into layer prototype bank $P^{(\ell)} \in \mathbb{R}^{K \times d_{\text{in}}}$.
- **Outputs**: `ArrowPrototypeBank` mapping layer names to normalized `[K, d_in]` prototype tensors + packed `(A_packed, B_packed, ranks, scalings)`.
- **Failure & Edge-Case Handling**: Zero-norm layers emit safe zero prototype vectors with explicit warning rather than `NaN`.
- **Acceptance Test Criteria**: Extracted prototypes match direct dense SVD top right singular vector within cosine similarity `>= 0.9999` and have unit L2 norm.

#### `FRD-REQ-17` — Temperature-Scaled Sign-Invariant Top-$k$ Router
- **Grades & Traceability**: `P1` | `Must` | `G2-High` | `Complexity: M` | `Tier: T2` | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-09` / `FEAT-09`
- **Inputs**: Token activations $X \in \mathbb{R}^{T \times d_{\text{in}}}$, prototype bank $P \in \mathbb{R}^{K \times d_{\text{in}}}$, `top_k: int` ($1 \le k \le K$), `temperature: float > 0.0`.
- **Preconditions**: `X.shape[-1] == P.shape[1]`, `1 <= top_k <= K`, `temperature > 0.0`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute L2-normalized activations $\hat{X} = X / \max(\|X\|_2, \epsilon)$ ($\epsilon = 10^{-8}$) in FP32.
  2. Compute sign-invariant alignment scores $Z = |\hat{X} P^T| \in \mathbb{R}^{T \times K}$.
  3. Select top-$k$ expert indices $E \in \mathbb{Z}^{T \times k}$ and top-$k$ scores $Z_{\text{top}} \in \mathbb{R}^{T \times k}$; apply $G = \text{softmax}(Z_{\text{top}} / \tau)$ in FP32.
  4. Compute per-token router Shannon entropy $H_t = -\sum_{j=0}^{k-1} G_{t,j} \log(G_{t,j} + \epsilon)$.
- **Outputs**: `(expert_ids [T, k] int32, routing_weights [T, k] float32, router_entropy [T] float32)`.
- **Failure & Edge-Case Handling**: Zero-vector activations clamp denominator to $\epsilon$ avoiding `NaN`; `top_k > K` or `temperature <= 0` raises `ConfigError` (`EXIT_CONFIG_ERROR = 2`).
- **Acceptance Test Criteria**: Flipping sign of any adapter $(B_m, A_m) \to (-B_m, -A_m)$ produces identical `expert_ids` and `routing_weights` (`abs diff <= 1e-6`); weights sum to `1.0` per token.

#### `FRD-REQ-18` — Deterministic Token-to-Expert Segment Sorting
- **Grades & Traceability**: `P1` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T2` | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-10` / `FEAT-10`
- **Inputs**: `expert_ids: int32[T, k]`, `routing_weights: float32[T, k]`, `num_experts: int` ($K$).
- **Preconditions**: Total assignments $N = T \cdot k \ge 0$, all `expert_ids` in `[0, K)`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Flatten `flat_experts = expert_ids.reshape(-1)` and `flat_weights = routing_weights.reshape(-1)`.
  2. Compute deterministic stable permutation `perm = torch.argsort(flat_experts, stable=True)`.
  3. Construct contiguous metadata buffers without Python loops: `sorted_token_ids = perm // k`, `sorted_slots = perm % k`, `sorted_weights = flat_weights[perm]`, `expert_counts = bincount(flat_experts, minlength=K)`, and inclusive/exclusive prefix sum `expert_offsets` of length `K + 1`.
- **Outputs**: Contiguous segment layout descriptor consumed directly by `@triton.jit` Stage 1/2 kernels.
- **Failure & Edge-Case Handling**: Skewed routing where one or more experts receive `0` tokens (`expert_counts[e] == 0`) sets `expert_offsets[e] == expert_offsets[e+1]` cleanly; $T=0$ returns early cleanly.
- **Acceptance Test Criteria**: Reconstructing scattered `(token_id, slot_idx)` assignments restores exact original `(T, k)` weights; handles `0`-token empty expert buckets without out-of-bounds indexing.

#### `FRD-REQ-19` — Two-Stage Fused `@triton.jit` `Arrow-SGMV` CUDA Kernel
- **Grades & Traceability**: `P1` | `Must` | `G1-Critical` | `Complexity: XL` | `Tier: T2` | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-10` / `FEAT-10`
- **Inputs**: `X [T, d_in]`, `base_out [T, d_out]`, `A_packed [K, r_max, d_in]`, `B_packed [K, d_out, r_max]`, `ranks [K]`, `scalings [K]`, segment layout descriptors (`sorted_token_ids`, `sorted_slots`, `sorted_weights`, `expert_offsets`).
- **Preconditions**: CUDA device (`>= SM 80`), `triton` available, matching dtypes (`float16` or `bfloat16` or `float32`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. **Stage 1 (`_arrow_sgmv_shrink_kernel`)**: Grid `(K, cdiv(max_seg_len, BLOCK_M), cdiv(r_max, BLOCK_R))`. Early exit if `seg_len == 0`. Gather `X[token_id, offs_k]`, load `A_e[offs_r, offs_k]` masked by **`offs_r < rank_e`**, accumulate `acc += tl.dot(x_tile, tl.trans(a_tile))` strictly in **`tl.float32`**, write `H_sorted [N, r_max]`.
  2. **Stage 2 (`_arrow_sgmv_expand_kernel`) + Deterministic Slot Reduction**: Grid `(K, cdiv(max_seg_len, BLOCK_M), cdiv(d_out, BLOCK_N))`. Multiply `H_sorted` tile by `B_e^T` masked by `offs_r < rank_e` & `offs_n < d_out` in `tl.float32`, scale by `sorted_weights[n] * scalings[e]`, write into slot-indexed `Y_slots [T, k, d_out]`, then compute deterministic slot reduction `Y_out = base_out + Y_slots.sum(dim=1).to(base_out.dtype)` (eliminating cross-expert `atomic_add` non-determinism).
- **Outputs**: Updated tensor `Y_out [T, d_out]` preserving `base_out.dtype` and `MoLoRARouteTelemetryBatch`.
- **Failure & Edge-Case Handling**: Calling CUDA Triton kernel on host lacking CUDA/Triton raises `RuntimeExecutionError` (`EXIT_RUNTIME_ERROR = 4`) when `backend="triton_cuda"` is forced explicitly.
- **Acceptance Test Criteria**: Matches PyTorch FP32 reference within `atol <= 1e-3, rtol <= 1e-3` (FP16) and `atol <= 5e-3` (BF16) across $T \in \{1, 3, 17, 128, 1025, 2048\}$, $K \in \{2..6\}$, mixed $r \in \{8, 16, 32, 64\}$, $k \in \{1..4\}$, and skewed empty-bucket routing.

#### `FRD-REQ-20` — Automatic PyTorch CPU/MPS Reference Backend & Auto-Dispatcher
- **Grades & Traceability**: `P1` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T2` (`₹0` local) | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-11` / `FEAT-11`
- **Inputs**: Identical tensor signature to `ArrowMoLoRADispatcher.forward(..., backend="auto" | "triton_cuda" | "torch_reference")`.
- **Preconditions**: PyTorch installed (`[kernels]` extra) or pure-Python diagnostic call.
- **Processing & Mathematical/Algorithmic Rules**:
  1. `has_triton_cuda()` verifies `torch.cuda.is_available()` and importable `triton`.
  2. When `backend="auto"`, dispatch to `triton_cuda` if CUDA+Triton ready, else `reference.arrow_molora_reference()` on `mps` or `cpu` (with identical sign-invariant routing, ragged `rank_e` slicing, FP32 intermediate accumulation, and deterministic slot sum).
  3. Expose `pure_python_arrow_reference()` for zero-`torch` unit diagnostics.
- **Outputs**: `(Y_out, MoLoRARouteTelemetryBatch)` identical in schema and semantics across CUDA, MPS, and CPU.
- **Failure & Edge-Case Handling**: Explicit `backend="triton_cuda"` on macOS/CPU raises `RuntimeExecutionError` (`EXIT_RUNTIME_ERROR = 4`); `backend="auto"` falls back cleanly with zero errors.
- **Acceptance Test Criteria**: On Apple M5 MPS and Linux CPU, `ArrowMoLoRADispatcher(backend="auto")` reproduces reference `Y_out` within `atol <= 1e-6` (FP32) and identical `expert_ids` / `routing_weights`.

#### `FRD-REQ-21` — Property-Based Equivalence Suite & Roofline `loraspec bench`
- **Grades & Traceability**: `P1` | `Should` | `G2-High` | `Complexity: M` | `Tier: T2` | `Milestone: M2` | `Gate: GATE-6 (M2)` | `BRD-REQ-12` / `FEAT-12`
- **Inputs**: Adapter paths or synthetic mixed-rank configurations, batch sizes `B`, token lengths `T`, `top_k`, dtypes (`fp16`, `bf16`), `--hardware` profile, `--smoke` / `--max-cost-inr`.
- **Preconditions**: `GpuBudgetGuard` initialized before any cloud sweep; `--smoke` enforces `< ₹10`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Execute property-based numerical parity check comparing `@triton.jit` `Arrow-SGMV` against PyTorch FP32 reference; abort with `KernelParityError` (`EXIT_PARITY_ERROR = 5`) if `max_abs_err` exceeds `1e-3` (FP16) or `5e-3` (BF16).
  2. Time `Arrow-SGMV` vs. `peft` eager Arrow and static `vLLM`/cuBLAS baselines using CUDA Events / `time.perf_counter`, tracking peak VRAM (`torch.cuda.max_memory_allocated`), moved HBM bytes vs. hardware ceiling (L4 `300 GB/s`, A100 `2039 GB/s`), and achieved TFLOP/s.
  3. Reserve and reconcile elapsed GPU time through `GpuBudgetGuard`.
- **Outputs**: `BenchmarkSuiteSummary` JSON (`report_bench.json`) + `RunManifest` (`manifest.json`) with exact `total_gpu_cost_inr`.
- **Failure & Edge-Case Handling**: Parity violation exits `EXIT_PARITY_ERROR (5)`; budget breach exits `EXIT_BUDGET_EXCEEDED (3)` while saving completed partial points.
- **Acceptance Test Criteria**: Smoke benchmark passes locally on CPU/MPS at `₹0`; L4 CUDA bench confirms `>= 2.0x–3.0x` speedup and `>= 30%` scratchpad VRAM reduction vs. `peft` eager Arrow at `B >= 16`.

---

### `EPIC-04`: `vLLM V1` Plugin, Streaming Telemetry Gateway & `DEMO_REPLAY`

#### `FRD-REQ-22` — `vLLM V1` General Plugin Entry-Point Hook
- **Grades & Traceability**: `P2` | `Should` | `G2-High` | `Complexity: L` | `Tier: T3` | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-13` / `FEAT-13`
- **Inputs**: `[project.entry-points."vllm.general_plugins"]` declaration in `pyproject.toml`, env flags `LORASPEC_MOLORA_ENABLE=1` and `LORASPEC_ADAPTERS=<manifest_or_paths>`.
- **Preconditions**: `vLLM V1` environment active (or mocked worker hook in CPU integration tests); adapters pass pre-flight `loraspec audit` base-model parity.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Expose `loraspec.serving.vllm_plugin:register()`. When `LORASPEC_MOLORA_ENABLE != "1"`, return as a no-op so standard `vLLM` behavior is completely unaffected.
  2. When enabled, extract `Arrow` prototypes and pack `(A_packed, B_packed, ranks, scalings)` onto the worker GPU device without modifying any `vLLM` source file on disk (`ADR-005`).
  3. Wrap target attention/MLP linear modules (`q/k/v/o/gate/up/down_proj`) with `MoLoRALinearAdapter` calling `W_0 x` via native `vLLM` GEMM plus `ArrowMoLoRADispatcher` delta.
- **Outputs**: Registered `vLLM V1` worker plugin executing per-token Top-$k$ `MoLoRA` during continuous batching.
- **Failure & Edge-Case Handling**: Adapter config/base-model mismatch fails plugin initialization deterministically (`AdapterParseError` / `RuntimeExecutionError`, exit `2`/`4`) before serving corrupt weights.
- **Acceptance Test Criteria**: `importlib.metadata.entry_points(group="vllm.general_plugins")` resolves `loraspec_molora`; mock/L4 `vLLM` integration test routes SQL, Rust, and JSON tokens within a single prompt to distinct adapters.

#### `FRD-REQ-23` — Non-Blocking Token Routing Telemetry Capture
- **Grades & Traceability**: `P2` | `Should` | `G2-High` | `Complexity: M` | `Tier: T3` | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-13` / `FEAT-13`
- **Inputs**: `MoLoRARouteTelemetryBatch` emitted from designated probe decoder layers (e.g., mid-stack + final block) during each decode step.
- **Preconditions**: `MoLoRALinearAdapter` active; `TelemetryRingBuffer` pre-allocated with bounded capacity (e.g., `4096` frames).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Record `(step_idx, request_id, top_k_adapter_ids, top_k_weights, router_entropy, itl_ms)` into a lock-free/bounded `TelemetryRingBuffer`.
  2. Perform zero blocking `.item()` / `.cpu()` GPU-CPU synchronization inside intermediate transformer layers; synchronize/copy asynchronously once per decode step boundary.
- **Outputs**: Ordered stream of `MoLoRARouteTelemetry` records available to FastAPI SSE and WebSocket consumers.
- **Failure & Edge-Case Handling**: Ring buffer overflow drops oldest consumed frames cleanly with an overflow counter without blocking model forward passes or leaking memory.
- **Acceptance Test Criteria**: Multi-layer mock decode loop records exact `(token_index, top_k_adapters, top_k_weights, router_entropy, itl_ms)` bounded by fixed ring size.

#### `FRD-REQ-24` — FastAPI OpenAI-Compatible `/v1/chat/completions` & Studio REST API
- **Grades & Traceability**: `P2` | `Should` | `G2-High` | `Complexity: L` | `Tier: T3` | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-14` / `FEAT-14`
- **Inputs**: HTTP requests per §5.2 API contract (`/health`, `/api/v1/adapters`, `/api/v1/audit`, `/api/v1/knots/preview`, `/api/v1/benchmarks`, `/v1/chat/completions`).
- **Preconditions**: FastAPI app booted in either `live` GPU mode or `DEMO_REPLAY=true` mode.
- **Processing & Mathematical/Algorithmic Rules**:
  1. `/api/v1/audit` invokes `SDD-01` (`load_peft_adapter` + `run_spectral_and_collision_audit` + `grading`) directly and returns `AuditReport`.
  2. `/api/v1/knots/preview` invokes factorized `KnOTS` (`SDD-02`) for target rank & method sliders and returns `KnOTSPreviewResponse` (`KnOTSMergeSummary` + pre/post sign-conflict metrics).
  3. `/api/v1/benchmarks` returns `BenchmarkSuiteSummary` (4-arm L4/A100 ablation + roofline curves).
  4. `/v1/chat/completions` streams OpenAI-compatible `text/event-stream` (`data: {...}\n\n`) chunks augmented with `"loraspec_routing": MoLoRARouteTelemetry.model_dump()`.
- **Outputs**: Strict Pydantic-validated JSON payloads and SSE frames (`200 OK`).
- **Failure & Edge-Case Handling**: Invalid adapter paths or malformed payloads return structured HTTP `400` / `422`; missing GPU backend when `DEMO_REPLAY=false` returns HTTP `503 Service Unavailable` with clear remediation.
- **Acceptance Test Criteria**: FastAPI `TestClient` suite verifies all 6 HTTP endpoints, OpenAI SSE format, and schema validation on every chunk.

#### `FRD-REQ-25` — Real-Time `/ws/telemetry` WebSocket Bus
- **Grades & Traceability**: `P2` | `Should` | `G2-High` | `Complexity: M` | `Tier: T3` | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-14` / `FEAT-14`
- **Inputs**: WebSocket connection `/ws/telemetry?replay_id=...&speed=1.0` (or live `request_id`).
- **Preconditions**: Gateway running; telemetry queue or replay trace available.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Accept WebSocket connection; validate `speed` (`> 0` or `"instant"`).
  2. Multiplex typed JSON frames (`{"type": "token", "routing": MoLoRARouteTelemetry}`, `{"type": "gpu_stats", "vram_mb": ..., "itl_ms": ..., "ttft_ms": ...}`, `{"type": "done", ...}`) in strict monotonic `token_index` order.
  3. Cleanly unsubscribe disconnected clients without impacting concurrent Studio viewers.
- **Outputs**: Real-time WebSocket JSON telemetry stream.
- **Failure & Edge-Case Handling**: Unknown `replay_id` closes WebSocket gracefully with policy code `1008` and error JSON frame; client disconnect cleans up asyncio tasks with zero dangling coroutines.
- **Acceptance Test Criteria**: `TestClient.websocket_connect("/ws/telemetry?speed=instant")` receives deterministic sequence of `token`/`gpu_stats`/`done` frames validating against `MoLoRARouteTelemetry`.

#### `FRD-REQ-26` — `DEMO_REPLAY=true` Zero-Cost Recorded Telemetry Mode
- **Grades & Traceability**: `P2` | `Should` | `G2-High` | `Complexity: M` | `Tier: T3` (`₹0`) | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-15` / `FEAT-15`
- **Inputs**: `DEMO_REPLAY=true` env var (or `loraspec serve --demo-replay` / `loraspec replay <trace.jsonl> --speed <float>`), JSONL trace bundles in `assets/demo_replays/`.
- **Preconditions**: Recorded JSONL session files and precomputed audit/roofline JSON bundles exist in `assets/demo_replays/`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Boot FastAPI gateway with **zero `torch`, `triton`, or `vLLM` imports required**.
  2. Stream recorded L4/A100 `MoLoRARouteTelemetry` traces over both `/v1/chat/completions` SSE and `/ws/telemetry` paced by recorded `itl_ms / speed` (or zero-delay when `speed <= 0.0` in CLI or `speed="instant"` on `/ws/telemetry`).
  3. Serve pre-recorded L4/A100 `AuditReport`, `KnOTSPreviewResponse`, and `BenchmarkSuiteSummary` over `/api/v1/*`.
- **Outputs**: Authentic GPU telemetry experience at `₹0/month` hosting cost.
- **Failure & Edge-Case Handling**: Missing or corrupt JSONL replay lines raise deterministic `ConfigError` at startup rather than silent partial crashes.
- **Acceptance Test Criteria**: Gateway passes full integration test suite (`test_gateway_replay.py`) in CPU-only `uv` env without CUDA or `vLLM` installed.

---

### `EPIC-05`: Interactive React 19 Web Studio & Container Stack

#### `FRD-REQ-27` — Subspace Observatory & `KnOTS` Alignment Lab Views
- **Grades & Traceability**: `P2` | `Should` | `G3-Standard` | `Complexity: L` | `Tier: T3` (`₹0`) | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-16` / `FEAT-16`
- **Inputs**: JSON responses from `/api/v1/adapters`, `/api/v1/audit`, `/api/v1/knots/preview`.
- **Preconditions**: Gateway reachable (live or `DEMO_REPLAY=true`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. **View 1 (`SubspaceObservatory.tsx`)**: Render dark-mode (`#0A0D12`) interactive **Grassmann Chordal Similarity Matrix** (clicking any cell displays principal angles $\theta_1 \dots \theta_r$, geodesic distance, and sign-conflict rate), per-layer **Effective Rank Utilization** ($r_{\text{eff}}/r$) bar chart, **Singular Value Scree Curves**, and filterable `RULE-*` findings drawer.
  2. **View 2 (`KnOTSAlignmentLab.tsx`)**: Render interactive **Naive Linear/TIES vs. `KnOTS` Joint-SVD** sign-conflict % delta, live target rank slider (`4..64`) + method selector (`knots_ties`, `knots_dare_ties`, `knots_linear`, `use_gks`) fetching `/api/v1/knots/preview` to update **Energy Retention $\eta$** & RMSE, plus exportable CLI command snippet.
- **Outputs**: Interactive SVG/Canvas React 19 views updating reactively in `< 100 ms` on replay data.
- **Failure & Edge-Case Handling**: API `4xx/5xx` displays inline typed error banner without crashing React error boundary; all adapter identifiers escape via React DOM text nodes (`0` `dangerouslySetInnerHTML`).
- **Acceptance Test Criteria**: `tsc --noEmit` & `vite build` pass with `0` errors; Playwright verifies Grassmann matrix click-drilldown and `KnOTS` slider API update.

#### `FRD-REQ-28` — Live Per-Token `MoLoRA` Generation Arena
- **Grades & Traceability**: `P2` | `Should` | `G3-Standard` | `Complexity: L` | `Tier: T3` (`₹0`) | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-16` / `FEAT-16`
- **Inputs**: User prompt or compositional preset (e.g., *"Write a Rust parser emitting strict JSON over a SQL query"*), `top_k`, `temperature_router`, and `/ws/telemetry` + `/v1/chat/completions` streams.
- **Preconditions**: Active WebSocket / SSE connection to FastAPI gateway.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Render each streamed token span dynamically colorized by its Top-1 dominant adapter:
     - `sql-lora` → Emerald (`#10B981`)
     - `rust-lora` → Orange (`#F97316`)
     - `json-lora` → Cyan (`#06B6D4`)
     - `math-lora` → Violet (`#8B5CF6`)
     - `python-lora` → Amber (`#F59E0B`)
     - `base` / `general-lora` → Slate (`#94A3B8`)
     with background alpha proportional to $g_{t,1}$.
  2. Clicking or hovering any token opens `TokenSpanInspector.tsx` showing full Top-$k$ weights, router Shannon entropy $H_t$, and active layer summary alongside live **TTFT**, **ITL (ms/tok)**, and **VRAM** gauges.
- **Outputs**: Live token-colorized generation view with per-token forensic inspector.
- **Failure & Edge-Case Handling**: Adversarial tokens containing `<script>alert(1)</script>` or HTML tags render strictly as escaped plain text (`Threat T5`); WebSocket disconnect shows reconnect button cleanly.
- **Acceptance Test Criteria**: Playwright test verifies distinct adapter colors render across mixed-domain stream, clicking a token displays exact Top-$k$ weights, and `<script>` payload executes `0` JS dialogs.

#### `FRD-REQ-29` — L4 vs. A100 Kernel Roofline & 4-Arm Ablation Explorer
- **Grades & Traceability**: `P2` | `Should` | `G3-Standard` | `Complexity: M` | `Tier: T3` (`₹0`) | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-16` / `FEAT-16`
- **Inputs**: `BenchmarkSuiteSummary` JSON fetched from `GET /api/v1/benchmarks`.
- **Preconditions**: Recorded M2/M4 or live `loraspec bench` data available.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Render L4 (`300 GB/s`) vs. A100 (`2039 GB/s`) roofline throughput/latency curves comparing **`peft` eager Arrow**, **`vLLM` static LoRA**, **`KnOTS` merged**, and **`@triton.jit` Top-$k$ `Arrow-SGMV`** across batch sizes $B \in [1..64]$.
  2. Display 4-arm compositional accuracy (`pass@1`), ITL (`ms/tok`), peak scratchpad VRAM (`MB`), and exact GST-inclusive INR sweep cost ledger (`<= ₹3,000`).
- **Outputs**: Interactive Roofline & 4-Arm Ablation view (`RooflineAblationView.tsx`).
- **Failure & Edge-Case Handling**: Empty benchmark state renders clear instruction to run `loraspec bench` or enable `DEMO_REPLAY=true`.
- **Acceptance Test Criteria**: Renders all 4 arms and L4/A100 series cleanly in Chromium headless check with `0` console errors.

#### `FRD-REQ-30` — Single-Command `Dockerfile` + `docker-compose.yml` Stack
- **Grades & Traceability**: `P2` | `Could` | `G3-Standard` | `Complexity: S` | `Tier: T3` (`₹0`) | `Milestone: M3` | `Gate: GATE-6 (M3)` | `BRD-REQ-17` / `FEAT-17`
- **Inputs**: `Dockerfile`, `docker-compose.yml`, local `.env` / `DEMO_REPLAY` flag.
- **Preconditions**: Docker / OrbStack available.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Multi-stage `Dockerfile`: Stage 1 (`node:22-slim`) runs `npm ci && npm run build` producing `/app/ui/dist`; Stage 2 (`python:3.12-slim`) installs `loraspec[studio,spectral]` and serves static SPA + API/WS via `uvicorn loraspec.serving.gateway:app`.
  2. `docker-compose.yml` defaults to `DEMO_REPLAY=true` on port `8000` with healthcheck on `GET /health` and an optional `gpu` Compose profile (`deploy.resources.reservations.devices`) for local CUDA adapters.
- **Outputs**: Production container image & 1-command `docker compose up` runtime.
- **Failure & Edge-Case Handling**: Container healthcheck fails fast if `/health` does not return `200 OK` within `30s`.
- **Acceptance Test Criteria**: `docker compose up -d --wait` reaches `healthy` status in `< 30s` and `curl http://127.0.0.1:8000/health` returns `{"status":"ok","demo_replay":true,...}`.

---

### `EPIC-06`: CLI, Reporting, Config & `$0` CPU GitHub Marketplace Action

#### `FRD-REQ-31` — Synchronized `report.json`, `report.sarif` (SARIF 2.1.0) & `manifest.json`
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T0` (`₹0`) | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-18` / `FEAT-18`
- **Inputs**: `AuditReport`, `RunManifest`, output paths (`--out-dir`, `--sarif`).
- **Preconditions**: `AuditReport` schema valid.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Write `report.json` (`AuditReport.model_dump_json(indent=2) + "\n"`) and `manifest.json` (`RunManifest.model_dump_json(indent=2) + "\n"`).
  2. Build OASIS **SARIF 2.1.0** (`$schema = "https://json.schemastore.org/sarif-2.1.0.json"`, `version = "2.1.0"`), mapping `CRITICAL`/`HIGH` → `"error"`, `MEDIUM` → `"warning"`, `LOW`/`INFO` → `"note"`, deduplicating `driver.rules`, and stamping `run.properties` (`run_id`, `grade`, `merge_risk_score`, `recommendations`).
  3. Normalize every `artifactLocation.uri` via `_normalize_uri()`: walk upward for `.git` so subdirectory audits emit repo-relative POSIX paths (`adapters/sql_lora/adapter_config.json`), preserving dotfiles (e.g., `.env`) while stripping `..` / `./` traversal segments (`Threat T6`).
- **Outputs**: Synchronized `report.json`, `manifest.json`, and `report.sarif`.
- **Failure & Edge-Case Handling**: Unwritable target path (e.g., pointing `--out-dir` at an existing regular file) fails cleanly with `EXIT_CONFIG_ERROR (2)` rather than raw traceback.
- **Acceptance Test Criteria**: JSON round-trips through `AuditReport.model_validate_json`; SARIF validates schema fields, severity levels, repo-relative git URIs, dotfile preservation, and mixed internal/external adapter path normalization.

#### `FRD-REQ-32` — Zero-CDN Offline HTML Certification Card (`report.html`)
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T0` (`₹0`) | `Milestone: M1` | `Gate: GATE-6 (M1)` | `BRD-REQ-18` / `FEAT-18`
- **Inputs**: `AuditReport`, target `--html` output path.
- **Preconditions**: `Jinja2` environment initialized strictly with `autoescape=select_autoescape(["html", "xml"])` (`autoescape=True`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Render self-contained single-file HTML (`< 250 KB`) embedding all CSS and inline SVG charts (Grade `A–F` badge, `MergeRiskScore` arc gauge, pairwise Grassmann Chordal heatmap, per-layer $r_{\text{eff}}/r$ utilization bars, findings table, and `RunManifest` SHA-256 verification strip).
  2. Prohibit all external `<script src=...>`, `<link href=...>`, `@import url(...)`, or external font CDN calls so report renders identically offline (`NFR-06`).
- **Outputs**: Self-contained XSS-safe `report.html`.
- **Failure & Edge-Case Handling**: Adversarial metadata (`<script>alert(1)</script>` inside `base_model_name_or_path` or `adapter_id`) is HTML-entity escaped cleanly.
- **Acceptance Test Criteria**: Regex audit confirms `0` external `http(s)://` asset fetches; adversarial script payload is escaped as `&lt;script&gt;`; output size `< 250 KB`.

#### `FRD-REQ-33` — Typer CLI Surface, `loraspec.toml` (`extra="forbid"`) & Exit Codes `0–5`
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T0` (`₹0`) | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-19` / `FEAT-19`
- **Inputs**: CLI arguments matching §4 matrix and optional `loraspec.toml` file.
- **Preconditions**: Python `>= 3.11`; `loraspec.toml` validated via Pydantic v2 `ConfigDict(extra="forbid", allow_inf_nan=False)`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Validate `loraspec.toml` rejecting unknown keys, bad TOML syntax, `collision_crit_chordal < collision_high_chordal`, or out-of-range caps (`max_cost_inr > 3000.0`).
  2. Expose all 8 commands (`doctor`, `inspect`, `audit`, `merge`, `bench`, `serve`, `replay`, `report`) + `--version/-V`.
  3. Deduplicate repeated adapter directory basenames deterministically (`foo`, `foo_1`, ...) during `audit` so duplicate CLI targets never silently overwrite SHA-256 or subspace entries; escape untrusted strings via `rich.markup.escape`.
  4. Map exceptions strictly to exit codes `0–5` (`LoraspecError.exit_code`).
- **Outputs**: Formatted Rich terminal tables/summaries + written JSON/SARIF/HTML/PEFT files + deterministic exit code `0–5`.
- **Failure & Edge-Case Handling**: Stage 6 stubs (`merge` [M1], `bench` [M2], `serve`/`replay` [M3]) print milestone guidance and exit `EXIT_CONFIG_ERROR (2)` cleanly until implemented; invalid `--tiers`, missing files, or unwritable paths exit `2`; findings `>= --fail-on` exit `1`.
- **Acceptance Test Criteria**: `test_cli.py` and `test_schemas_and_config.py` pass across `--version`, `doctor`, `inspect`, `audit` (pass/fail/duplicate/markup/tiers/bad-out-dir), `report`, unknown TOML keys, inverted thresholds, and wheel-installed `uvx` smoke tests.

#### `FRD-REQ-34` — `$0` CPU GitHub Marketplace Composite Action (`action.yml`)
- **Grades & Traceability**: `P1` | `Should` | `G2-High` | `Complexity: S` | `Tier: T0` (`₹0`) | `Milestone: M1` | `Gate: GATE-6 (M1) / GATE-7` | `BRD-REQ-20` / `FEAT-20`
- **Inputs**: Action inputs (`adapter-paths`, `config`, `fail-on`, `sarif-file`, `html-file`, `out-dir`, `loraspec-version`).
- **Preconditions**: Runs on `ubuntu-latest` CPU runner with zero GPU or `torch` required.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Bootstrap `uv` via `astral-sh/setup-uv` and run `uvx --from loraspec` (or local workspace during repo self-test) `loraspec audit` emitting `report.json`, `manifest.json`, `loraspec.sarif`, and `loraspec.html`.
  2. Export action outputs (`grade`, `merge-risk-score`, `sarif-file`, `html-file`, `report-json`, `exit-code`).
  3. Upload `loraspec.sarif` via `github/codeql-action/upload-sarif@v3` (`if: always()`) and HTML + JSON via `actions/upload-artifact@v4`, propagating audit exit code cleanly.
- **Outputs**: PR inline Code Scanning alerts + HTML Certification Card artifact + step outputs.
- **Failure & Edge-Case Handling**: Uploads SARIF and HTML even when `loraspec audit` exits `1` (`EXIT_AUDIT_FAIL`) so reviewers see exact failing diagnostics on the PR tab.
- **Acceptance Test Criteria**: End-to-end `ubuntu-latest` workflow completes in `< 20s` at `$0`, failing a sample PR with colliding/collapsed adapter and posting valid SARIF 2.1.0 annotations.

---

### `EPIC-07`: Cost Controls & Controlled Empirical GPU Study

#### `FRD-REQ-35` — Atomic `GpuBudgetGuard` & `--smoke < ₹10` Circuit Breaker
- **Grades & Traceability**: `P0` | `Must` | `G1-Critical` | `Complexity: M` | `Tier: T0..T3` | `Milestone: Stage 5` | `Gate: GATE-5` | `BRD-REQ-21` / `FEAT-21`
- **Inputs**: `max_cost_inr` (default `2500.0`, ceiling `<= 3000.0`), `smoke: bool`, `smoke_cap_inr` (default `10.0`), `rate_table_inr_per_hr` (`local_cpu=0.0`, `local_mps=0.0`, `demo_replay=0.0`, `e2e_l4_24gb=57.82`, `e2e_a100_80gb=223.02`), `hardware`, `estimated_seconds`, `actual_seconds`.
- **Preconditions**: Finite non-negative caps and rates (`smoke_cap_inr > 0` when `smoke=True`).
- **Processing & Mathematical/Algorithmic Rules**:
  1. Compute `effective_cap_inr = min(max_cost_inr, smoke_cap_inr)` when `smoke=True`, else `max_cost_inr`.
  2. Acquire `asyncio.Lock` inside `reserve_or_raise(hardware, estimated_seconds)`; check `committed_cost + reserved_cost + projected <= effective_cap_inr + 1e-9` before issuing reservation.
  3. On step completion, `reconcile(reservation, actual_seconds)` releases reserved budget and commits actual elapsed `rate * (actual_seconds / 3600.0)` and GPU hours; `release()` frees unused reservations on cancellation.
- **Outputs**: `BudgetReservation`, accurate `spent_cost_inr`, `total_gpu_hours`, stamped into `RunManifest.total_gpu_cost_inr`.
- **Failure & Edge-Case Handling**: Exceeding cap raises `BudgetExceededError` (`EXIT_BUDGET_EXCEEDED = 3`); unknown hardware profiles, negative/NaN durations, or double-reconciliation raise `ConfigError` (`EXIT_CONFIG_ERROR = 2`).
- **Acceptance Test Criteria**: Concurrent `asyncio.gather` test enforces hard ceiling under race conditions; `30m` L4 under `--smoke` raises `BudgetExceededError` with `exit_code == 3`; `local_cpu` records `0.0` INR; double-reconcile raises `ConfigError`.

#### `FRD-REQ-36` — Controlled 4-Arm L4/A100 Empirical Study & Social Asset Generator
- **Grades & Traceability**: `P2` | `Should` | `G3-Standard` | `Complexity: XL` | `Tier: T2..T3` | `Milestone: M4` | `Gate: GATE-6 (M4) / GATE-7` | `BRD-REQ-22` / `FEAT-22`
- **Inputs**: Base model (`Qwen/Qwen2.5-3B-Instruct` or `Qwen/Qwen3-4B-Instruct`), 6 heterogeneous domain LoRAs (`sql-lora` $r=16$, `rust-lora` $r=32$, `json-lora` $r=8$, `python-lora` $r=32$, `math-lora` $r=16$, `general-lora` $r=16$), single-domain + compositional prompts, `--smoke` / `--max-cost-inr`.
- **Preconditions**: `M1`, `M2`, `M3` gates passed; every script passes `--smoke < ₹10` locally/L4 before full sweep; cumulative GPU spend remains `<= ₹3,000`.
- **Processing & Mathematical/Algorithmic Rules**:
  1. Execute controlled 4-arm evaluation: **Arm 0** (Base), **Arm 1** (`peft` static `linear` & unaligned `ties`), **Arm 2** (`loraspec merge` `KnOTS` + `TIES`, $r_{\text{out}}=32$), **Arm 3** (Per-Token Top-$k$ `MoLoRA` `Arrow-SGMV` via `vLLM V1`).
  2. Execute 1.5–2.0 hr A100 pass measuring L4 (`300 GB/s`) vs. A100 (`2039 GB/s`) roofline and concurrent ITL/TTFT scaling.
  3. Record verified replay traces to `assets/demo_replays/*.jsonl`, author `docs/TECHNICAL-REPORT.md`, and run `scripts/export_social_assets.py` (producing 2×-DPI PNG/SVG figures + 60fps Playwright `.mp4` arena clip).
- **Outputs**: `docs/TECHNICAL-REPORT.md`, verified `assets/demo_replays/`, `assets/social/` visual exports, and complete `manifest.json` INR billing ledger.
- **Failure & Edge-Case Handling**: All evaluation loops run inside `GpuBudgetGuard` saving intermediate JSONL checkpoints on every task batch so ceiling stops never lose completed measurements.
- **Acceptance Test Criteria**: Integration smoke (`test_recipe_smoke.py`) passes on CPU/mock at `₹0`; final L4+A100 ledger confirms `total_gpu_cost_inr <= 3000.0` alongside verified KPIs (`KPI-RES-02`, `KPI-RES-03`, `KPI-RES-04`, `KPI-SHW-03`).

---

## 4. Frozen CLI Contract (`src/loraspec/cli/app.py`)

### 4.1 Complete CLI Subcommands, Options & Flags Matrix

```text
loraspec [GLOBAL OPTIONS] <COMMAND> [ARGS] [OPTIONS]
```

| Command / Scope | Stage Status | Positional Arguments | Options & Flags (Exact CLI Flag, Type, Default & Semantics) |
|---|---|---|---|
| **Global** | **Stage 5** | — | `--version`, `-V` (`bool`, prints `loraspec <version>` and exits `0`); `--help` (Typer help). |
| **`doctor`** | **Stage 5** | *None* | *None*. Inspects Python `>= 3.11`, OS/arch, active backend (`detect_accelerator_backend`), built-in safetensors parser, and optional extras (`spectral`, `kernels`, `serving`, `studio`). Exits `2` if Python `< 3.11`. |
| **`inspect`** | **Stage 5** | `PATH` (`Path`, adapter dir or `.safetensors`) | `--json` (`bool`, default `False`): prints canonical `LoRAAdapterSpec` JSON instead of Rich table. |
| **`audit`** | **Stage 5** (`--html` **M1**) | `PATHS...` (`list[Path]`, `>= 1` adapter dirs/files) | • `--config PATH` (`Path \| None`, default `None` -> auto-discovers `./loraspec.toml`)<br>• `--tiers TEXT` (`str`, default `"0,1"`, supported `"0"` or `"0,1"`)<br>• `--fail-on [critical\|high\|medium\|low\|info]` (`Severity \| None`, overrides `config.project.fail_on`)<br>• `--out-dir PATH` (`Path \| None`, writes `report.json` + `manifest.json`)<br>• `--sarif PATH` (`Path \| None`, writes OASIS SARIF 2.1.0)<br>• `--seed INT` (`int`, default `42`, stamped into `RunManifest`)<br>• `[Stage 6 M1]` `--html PATH` (`Path \| None`, writes zero-CDN offline HTML Certification Card). |
| **`report`** | **Stage 5** (`--html` **M1**) | `REPORT_JSON` (`Path`, saved `report.json`) | • `--sarif PATH` (`Path \| None`, re-renders SARIF 2.1.0)<br>• `--json-out PATH` (`Path \| None`, validates & re-serializes `AuditReport` JSON)<br>• `[Stage 6 M1]` `--html PATH` (`Path \| None`, renders zero-CDN HTML from saved `report.json`). |
| **`merge`** | **Stage 5 Stub → M1** | `PATHS...` (`list[Path]`, `>= 2` adapters) | **Stage 5 Stub Flags**:<br>• `--out-dir PATH` (default `./merged_knots_lora`)<br>• `--method TEXT` (default `knots_ties`: `knots_ties \| knots_dare_ties \| knots_linear`)<br>• `--target-rank INT` (default `16`)<br>**Stage 6 M1 Planned Extensions**:<br>• `--weights TEXT` (comma-separated floats, default uniform)<br>• `--ties-density FLOAT` (default `0.50`), `--dare-drop-rate FLOAT` (default `0.20`)<br>• `--layer-policy [union\|intersection]` (default `union`, disjoint-layer handling)<br>• `--gks-adapter PATH`, `--gks-lambda FLOAT` (default `1.0`)<br>• `--dtype [f16\|bf16\|f32]`, `--seed INT` (default `42`), `--config PATH`. |
| **`bench`** | **Stage 5 Stub → M2** | `PATHS...` (`list[Path]`, adapters to benchmark) | **Stage 5 Stub Flags**:<br>• `--top-k INT` (default `2`)<br>• `--smoke / --no-smoke` (`bool`, default `True`, `< ₹10` cap)<br>**Stage 6 M2 Planned Extensions**:<br>• `--hardware [auto\|local_cpu\|local_mps\|e2e_l4_24gb\|e2e_a100_80gb]`<br>• `--max-cost-inr FLOAT` (default `2500.0`, ceiling `<= 3000.0`)<br>• `--batch-sizes TEXT` (`"1,4,16,32,64"`), `--dtypes TEXT` (`"fp16,bf16"`), `--out-dir PATH`, `--config PATH`. |
| **`serve`** | **Stage 5 Stub → M3** | *None* | **Stage 5 Stub Flags**:<br>• `--host TEXT` (default `127.0.0.1`), `--port INT` (default `8000`)<br>• `--demo-replay / --no-demo-replay` (`bool`, default `True`)<br>**Stage 6 M3 Planned Extensions**:<br>• `--model TEXT`, `--adapters PATH...`, `--top-k INT`, `--temperature FLOAT`, `--replay-dir PATH`, `--config PATH`. |
| **`replay`** | **Stage 5 Stub → M3** | `TRACE_PATH` (`Path`, telemetry JSONL) | **Stage 5 Stub Flags**:<br>• `--speed FLOAT` (default `1.0`; `<= 0.0` replays with zero delay)<br>**Stage 6 M3 Planned Extensions**:<br>• `--ws-url TEXT` (optional live WS target), `--json-stream` (`bool`, default `False`). |

### 4.2 Deterministic Exit Codes & Domain Exceptions (`src/loraspec/core/errors.py`)
| Code | Constant | Domain Exception(s) | Meaning & Trigger Conditions |
|---|---|---|---|
| `0` | `EXIT_PASS` | — | Command succeeded cleanly with zero audit findings meeting `--fail-on` threshold. |
| `1` | `EXIT_AUDIT_FAIL` | `AuditFailedError` | Completed audit/evaluation flagged findings $\ge$ `--fail-on` severity. |
| `2` | `EXIT_CONFIG_ERROR` | `ConfigError`, `AdapterParseError` | Invalid CLI args/tiers, malformed/unknown-field `loraspec.toml`, missing/corrupt `.safetensors` or `.bin` pickle checkpoint, unwritable path, or Stage 6 stub invocation. |
| `3` | `EXIT_BUDGET_EXCEEDED` | `BudgetExceededError` | Hard `--max-cost-inr` / `BudgetConfig.max_cost_inr` / `--smoke` ceiling triggered by `GpuBudgetGuard`. |
| `4` | `EXIT_RUNTIME_ERROR` | `RuntimeExecutionError` | Missing optional CUDA/Triton/`vLLM` runtime or unrecoverable kernel/serving failure. |
| `5` | `EXIT_PARITY_ERROR` | `KernelParityError` | Triton `Arrow-SGMV` vs. PyTorch reference check violated numerical tolerance (`atol`/`rtol`). |

### 4.3 Declarative Configuration (`loraspec.toml`, `extra="forbid"`, `allow_inf_nan=False`)
Validated via `LoraspecConfig` (`src/loraspec/core/config.py`) with cross-field validator asserting `audit.collision_crit_chordal >= audit.collision_high_chordal`:

```toml
[project]
fail_on = "high"                     # critical | high | medium | low | info
default_backend = "auto"             # auto | triton_cuda | torch_mps | torch_cpu | pure_python

[audit]
rank_collapse_ratio = 0.20           # (0.0, 1.0] Flag HIGH if mean r_eff / r < 0.20
top1_energy_threshold = 0.85         # (0.0, 1.0] Flag HIGH if >=30% layers exceed 85% energy in sigma_1
spectral_spike_multiplier = 5.0      # (> 1.0)    Flag HIGH if ||dW||_2 > max(10, 5x layer median)
collision_high_chordal = 0.45        # [0.0, 1.0] Grassmann chordal similarity threshold (HIGH)
collision_crit_chordal = 0.70        # [0.0, 1.0] Grassmann chordal similarity threshold (CRITICAL >= high)
sign_conflict_threshold = 0.35       # [0.0, 1.0] Directional sign conflict trigger
require_all_linear = true            # Enforce LoRA Without Regret attention+MLP warning

[knots]
default_method = "knots_ties"        # knots_ties | knots_dare_ties | knots_linear
ties_density = 0.50                  # (0.0, 1.0]
dare_drop_rate = 0.20                # [0.0, 1.0)
min_energy_retention = 0.90          # (0.0, 1.0]
rng_seed = 42

[molora]
top_k = 2                            # [1, 16]
router_temperature = 1.0             # (> 0.0)
use_gks = false

[budget]
max_cost_inr = 2500.0                # [0.0, 3000.0] Hard INR cap incl. 18% GST
smoke_cap_inr = 10.0                 # (0.0, 50.0] Mandatory --smoke dry-run cap
l4_rate_inr_per_hr = 57.82           # >= 0.0 (Rs 49/hr + 18% GST)
a100_rate_inr_per_hr = 223.02        # >= 0.0 (Rs 189/hr + 18% GST)

[studio]
host = "127.0.0.1"
port = 8000                          # [1, 65535]
demo_replay = true
```

---

## 5. Frozen Python SDK, Pydantic v2 Schemas & Studio HTTP/WS API Contracts

### 5.1 Complete Frozen Pydantic v2 Domain Schemas (`src/loraspec/core/schemas.py`)

#### A. Stage 5 Implemented & Frozen Core Schemas (exact match to `src/loraspec/core/schemas.py`)
```python
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def meets_fail_threshold(findings: Sequence[AuditFinding], fail_on: Severity) -> bool:
    """Return True if any finding meets or exceeds the configured fail_on severity."""
    threshold = SEVERITY_RANK[fail_on]
    return any(SEVERITY_RANK[f.severity] >= threshold for f in findings)


class AdapterFormat(StrEnum):
    PEFT_SAFETENSORS = "peft_safetensors"
    RAW_SAFETENSORS = "raw_safetensors"


class DeploymentRecommendation(StrEnum):
    SAFE_LINEAR_MERGE = "safe_linear_merge"
    USE_KNOTS_ALIGNMENT = "use_knots_alignment"
    USE_TOP_K_MOLORA = "use_top_k_molora"
    RETRAIN_COLLAPSED_LORA = "retrain_collapsed_lora"
    QUARANTINE_CORRUPT_ADAPTER = "quarantine_corrupt_adapter"


class LoRALayerSpec(BaseModel):
    layer_name: str
    module_type: str
    in_features: int
    out_features: int
    rank: int
    dtype: str
    a_key: str
    b_key: str
    a_shape: tuple[int, int]
    b_shape: tuple[int, int]
    param_count: int


class LoRAAdapterSpec(BaseModel):
    adapter_id: str
    adapter_format: AdapterFormat
    root_path: Path
    weights_path: Path
    config_path: Path | None = None
    base_model_name_or_path: str | None = None
    peft_type: str = "LORA"
    rank: int
    lora_alpha: float
    scaling: float
    lora_dropout: float = 0.0
    use_rslora: bool = False
    use_dora: bool = False
    bias: str = "none"
    target_modules: list[str] = Field(default_factory=list[str])
    total_lora_params: int
    weights_sha256: str
    layers: list[LoRALayerSpec] = Field(default_factory=list[LoRALayerSpec])
    metadata: dict[str, Any] = Field(default_factory=dict)


class LayerSpectralMetrics(BaseModel):
    adapter_id: str
    layer_name: str
    module_type: str
    rank: int
    singular_values: list[float]
    spectral_norm: float
    frobenius_norm: float
    effective_stable_rank: float
    rank_utilization: float
    top1_energy_ratio: float
    condition_number: float
    has_non_finite: bool = False


class PairwiseCollisionMetrics(BaseModel):
    adapter_a: str
    adapter_b: str
    layer_name: str
    active_rank_a: int
    active_rank_b: int
    principal_angles_deg: list[float]
    cos_principal_angle_max: float
    grassmann_chordal_similarity: float
    grassmann_geodesic_dist: float
    row_chordal_similarity: float
    sign_conflict_rate: float


class AuditFinding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    tier: Literal[0, 1, 2, 3]
    source: Literal["loraspec", "config_probe"]
    file_path: str | None = None
    line_number: int | None = None
    evidence: str
    remediation: str


class KnOTSMergeSummary(BaseModel):
    method: Literal["knots_ties", "knots_dare_ties", "knots_linear"]
    adapters_merged: list[str]
    target_rank: int
    energy_retained: float
    mean_reconstruction_rmse: float
    sign_conflict_reduction_pct: float
    output_sha256: str


class MoLoRARouteTelemetry(BaseModel):
    token_index: int
    token_text: str
    top_k_adapters: list[str]
    top_k_weights: list[float]
    router_entropy: float
    itl_ms: float


class RunManifest(BaseModel):
    run_id: str
    loraspec_version: str
    timestamp_utc: str
    git_sha: str | None
    host_os: str
    host_arch: str
    accelerator_backend: str
    adapter_sha256s: dict[str, str]
    rng_seed: int
    total_gpu_cost_inr: float = 0.0
    cli_args: list[str]


class AuditReport(BaseModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    manifest: RunManifest
    adapters: list[LoRAAdapterSpec] = Field(default_factory=list[LoRAAdapterSpec])
    grade: Literal["A", "B", "C", "D", "F"]
    merge_risk_score: float = 0.0
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
    layer_metrics: list[LayerSpectralMetrics] = Field(default_factory=list[LayerSpectralMetrics])
    pairwise_collisions: list[PairwiseCollisionMetrics] = Field(
        default_factory=list[PairwiseCollisionMetrics]
    )
    knots_summary: KnOTSMergeSummary | None = None
    recommendations: list[DeploymentRecommendation] = Field(
        default_factory=list[DeploymentRecommendation]
    )
```

#### B. Stage 6 M2–M3 Planned Extension Schemas *(Scheduled for M2/M3 addition in `src/loraspec/core/schemas.py`)*
```python
class MoLoRARouteTelemetryBatch(BaseModel):  # Planned: M2 (SDD-03)
    layer_name: str
    num_tokens: int
    top_k: int
    expert_indices: list[list[int]]
    routing_weights: list[list[float]]
    router_entropy: list[float]
    backend: Literal["triton_cuda", "torch_mps", "torch_cpu", "pure_python"]


class KernelBenchmarkPoint(BaseModel):  # Planned: M2 (SDD-03 / SDD-05)
    hardware_profile: str
    arm_name: Literal[
        "base",
        "peft_linear",
        "peft_ties",
        "peft_eager_arrow",
        "vllm_static",
        "knots_ties",
        "triton_arrow_sgmv",
    ]
    batch_size: int
    seq_len: int
    dtype: Literal["fp16", "bf16", "fp32"]
    latency_ms: float
    achieved_tflops: float
    hbm_bandwidth_gb_s: float
    peak_vram_mb: float
    max_abs_err_vs_ref: float = 0.0


class BenchmarkSuiteSummary(BaseModel):  # Planned: M2/M4 (SDD-04 / SDD-05)
    manifest: RunManifest
    points: list[KernelBenchmarkPoint]
    speedup_vs_peft_eager: float
    vram_savings_pct: float
    arm_accuracy_pass_at_1: dict[str, float] = Field(default_factory=dict)


class KnOTSPreviewRequest(BaseModel):  # Planned: M3 (SDD-04)
    adapter_paths: list[str]
    method: Literal["knots_ties", "knots_dare_ties", "knots_linear"] = "knots_ties"
    target_rank: int = Field(default=16, ge=1, le=256)
    ties_density: float = Field(default=0.50, gt=0.0, le=1.0)
    dare_drop_rate: float = Field(default=0.20, ge=0.0, lt=1.0)
    use_gks: bool = False
    gks_adapter_path: str | None = None
    rng_seed: int = 42


class KnOTSPreviewResponse(BaseModel):  # Planned: M3 (SDD-04)
    summary: KnOTSMergeSummary
    pre_knots_sign_conflict_rate: float
    post_knots_sign_conflict_rate: float
    findings: list[AuditFinding] = Field(default_factory=list[AuditFinding])
```

### 5.2 FastAPI Studio REST, SSE & WebSocket API Contract (M3 / `SDD-04`)

| Endpoint | Method / Protocol | Request / Query Contract | Response Payload & Error Statuses |
|---|---|---|---|
| `/health` | `GET` | *None* | `200 OK`: `{"status":"ok","demo_replay":bool,"backend":str,"loaded_adapters":int,"version":str}` |
| `/api/v1/adapters` | `GET` | *None* | `200 OK`: `list[LoRAAdapterSpec]` of currently mounted or replay adapters. |
| `/api/v1/audit` | `POST` | `{"adapter_paths":list[str],"fail_on":Severity="high","tiers":str="0,1"}` | `200 OK`: complete `AuditReport` JSON. `400`/`422` on invalid path or malformed `.safetensors` (`AdapterParseError`). |
| `/api/v1/knots/preview` | `POST` | `KnOTSPreviewRequest` | `200 OK`: `KnOTSPreviewResponse` (`KnOTSMergeSummary` + pre/post conflict comparison). `400`/`422` on invalid params or mismatched base models. |
| `/api/v1/benchmarks` | `GET` | Optional `?hardware=all\|l4\|a100` | `200 OK`: `BenchmarkSuiteSummary` (4-arm L4/A100 roofline, speedup vs `peft` eager Arrow, accuracy & INR ledger; aligns 100% with `SDD-04 §3`). `503` if no replay/bench artifact loaded. |
| `/v1/chat/completions` | `POST` (SSE) | OpenAI `ChatCompletionRequest` + `{"top_k":2,"temperature_router":1.0,"stream":true}` | `200 OK` SSE `data: {...}` OpenAI chunks augmented with `"loraspec_routing": MoLoRARouteTelemetry`. `503` if GPU unavailable and `DEMO_REPLAY=false`. |
| `/ws/telemetry` | `WebSocket` | `?replay_id=<id>&speed=1.0\|2.0\|instant` | Bidirectional WS streaming JSON frames `{"type":"token"\|"gpu_stats"\|"done", ...}` containing `MoLoRARouteTelemetry`, VRAM MB, TTFT, and ITL. Closes `1008` on invalid `replay_id`. |

### 5.3 Interactive React 19 Studio Views Functional Specification (`ui/`, M3 / `SDD-04`)
Dark-mode telemetry interface (`#0A0D12` obsidian canvas, monospace numeric tables, strictly React-escaped DOM nodes without `dangerouslySetInnerHTML`):
1. **View 1 — Subspace Observatory (`SubspaceObservatory.tsx`)**: Consumes `/api/v1/audit` + `/api/v1/adapters`; renders interactive **Grassmann Chordal Similarity Matrix** (click cell for $\theta_1 \dots \theta_r$, geodesic distance, sign conflict), per-layer **Effective Rank Utilization** ($r_{\text{eff}}/r$) bars, **Singular Value Scree Curves**, and `RULE-*` SARIF drawer.
2. **View 2 — `KnOTS` Alignment Lab (`KnOTSAlignmentLab.tsx`)**: Consumes `/api/v1/knots/preview`; renders side-by-side Naive Linear/TIES vs. `KnOTS` conflict %, live rank slider ($r \in [4..64]$) + method toggle updating **Energy Retention $\eta$** and RMSE, plus copyable `loraspec merge` command.
3. **View 3 — Live Per-Token `MoLoRA` Arena (`LiveMoLoRAArena.tsx` + `TokenSpanInspector.tsx`)**: Consumes `/ws/telemetry` + `/v1/chat/completions`; renders compositional presets and prompt input, colorizing streamed tokens dynamically (`sql-lora` `#10B981`, `rust-lora` `#F97316`, `json-lora` `#06B6D4`, `math-lora` `#8B5CF6`, `python-lora` `#F59E0B`, `base` `#94A3B8`) with click/hover Top-$k$ probability bar, Shannon entropy $H_t$, TTFT, ITL, and VRAM gauges.
4. **View 4 — Kernel Roofline & Ablation Explorer (`RooflineAblationView.tsx`)**: Consumes `/api/v1/benchmarks`; plots L4 (`300 GB/s`) vs. A100 (`2039 GB/s`) roofline and latency across batch sizes $B \in [1..64]$ (`peft` eager Arrow vs. `vLLM` static vs. `KnOTS` vs. Triton `Arrow-SGMV`) plus 4-arm `pass@1` accuracy & exact GST-inclusive INR spend ledger.

---

## 6. Graded Non-Functional Requirements (`NFR-01` – `NFR-08`)

| NFR ID | Category | Priority | MoSCoW | Impact / Risk Grade | Target Threshold / Requirement | Verification Method | Verification Gate |
|---|---|---|---|---|---|---|---|
| **NFR-01** | **Memory Safety** | `P0` | `Must` | `G1-Critical` | Spectral auditing & `KnOTS` alignment never materialize dense $d_{\text{out}} \times d_{\text{in}}$ updates ($O((d_{\text{out}}+d_{\text{in}})\sum r_m)$ bound; `< 250 MB` peak RAM on CI). | Peak RSS / `tracemalloc` test on 4096-dim synthetic layers. | `GATE-5`, `GATE-6 (M1)` |
| **NFR-02** | **Audit Latency** | `P0` | `Must` | `G2-High` | Inspecting + auditing 4 `Qwen3-4B` adapters on 4-vCPU CI completes in `< 10.0s`; `action.yml` end-to-end `< 20.0s`. | Timed benchmark in CI workflow. | `GATE-6 (M1)` |
| **NFR-03** | **Kernel Numerical Equivalence** | `P1` | `Must` | `G1-Critical` | Triton `Arrow-SGMV` matches PyTorch FP32 reference within `atol <= 1e-3, rtol <= 1e-3` (FP16) and `atol <= 5e-3` (BF16). | Randomized property-based CUDA parity suite (`EXIT_PARITY_ERROR=5` on breach). | `GATE-6 (M2)` |
| **NFR-04** | **Deterministic Reproducibility** | `P0` | `Must` | `G1-Critical` | Identical weights + `loraspec.toml` + `rng_seed` yield bitwise-identical `MergeRiskScore`, Grade `A–F`, SARIF, and seeded `KnOTS` factors. | Multi-run determinism unit tests across OSes. | `GATE-5`, `GATE-6 (M1)` |
| **NFR-05** | **Cross-OS Compatibility** | `P0` | `Must` | `G2-High` | Core CLI (`doctor`, `inspect`, `audit`, `report`, `merge`) & unit suite pass on macOS `arm64` (without CUDA/Triton) and Linux `x86_64` (`ubuntu-latest`). | Dual OS CI matrix + out-of-tree `uvx` wheel smoke test. | `GATE-5` |
| **NFR-06** | **Security & XSS Hygiene** | `P0` | `Must` | `G1-Critical` | Bounds-checked `.safetensors` header (`<= 100 MiB`), zero `pickle.load` on untrusted weights, Jinja2 `autoescape=True`, zero external CDN URLs in `report.html`, zero `dangerouslySetInnerHTML` in UI. | Malformed binary rejection tests + HTML/Playwright XSS checks. | `GATE-5`, `GATE-6 (M1, M3)` |
| **NFR-07** | **OASIS SARIF 2.1.0 Compliance** | `P0` | `Must` | `G2-High` | Emitted `report.sarif` validates strictly against SARIF 2.1.0 schema with repo-relative POSIX URIs (preserving dotfiles) and traversal protection. | Schema, git-subdir URI & dotfile normalization tests. | `GATE-5` |
| **NFR-08** | **Zero-Cost Demo Sustainability** | `P2` | `Should` | `G3-Standard` | Live hosted Web Studio runs indefinitely at `₹0/month` using `DEMO_REPLAY=true` recorded L4/A100 telemetry with `< 30s` `docker compose up` cold boot. | Static build, FastAPI/WS replay test & Compose healthcheck. | `GATE-6 (M3)`, `GATE-7` |
