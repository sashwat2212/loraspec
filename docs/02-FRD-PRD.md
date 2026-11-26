# Stage 2: Functional Requirements Document (FRD / PRD), Feature Breakdown & Frozen Contracts — `loraspec`

> [!NOTE]
> **Document Status**: Locked (Stage 2)
> **Product**: **`loraspec`** — LoRA Subspace Auditor (`KnOTS`) & Per-Token Top-$k$ `MoLoRA` Triton Studio
> **Upstream References**: [00-CHARTER.md](./00-CHARTER.md) | [01-BRD.md](./01-BRD.md)
> **Downstream References**: [03-ARCHITECTURE.md](./03-ARCHITECTURE.md) | [SDD-01](./04-DESIGN-SPECTRAL-AUDITOR.md)..[SDD-05](./04-DESIGN-CLI-CI-BENCHMARKS.md) | [05-IMPLEMENTATION-PLAN.md](./05-IMPLEMENTATION-PLAN.md)

---

## 1. Release Cutlines & Delivery Milestones

| Milestone / Release | Scope & Channels | Budget Tier | Shipped User & Engineering Value |
|---|---|---|---|
| **Stage 5 Walking Skeleton (`v0.1.0-alpha`)** | `EPIC-01` (core), `EPIC-06` (JSON/SARIF/Manifest), `EPIC-07` (CLI core & `GpuBudgetGuard`): `loraspec doctor`, `inspect`, `audit --tiers 0`, `report`. Pure-Python `.safetensors` + PEFT reader, factorized thin-QR/SVD + Grassmann math (with NumPy fast-path), `RULE-CONF-001..004`, `RULE-SPEC-001..004`, `MergeRiskScore`, Grade `A–F`. | **₹0** | Immediately executable, offline-capable CLI & strict-typed SDK with full CI (`ruff`, `pyright`, `pytest`). |
| **M1 (`v0.1.0` Auditor + `KnOTS` + SARIF Action)** | Completes `EPIC-01`, `EPIC-02`, `EPIC-06` (`report.html`, `action.yml`, `benchmarks/planted_collisions/`). Scalable `KnOTS` joint-SVD merger (`knots_ties`, `knots_dare_ties`, `knots_linear`, `use_gks`), zero-CDN HTML Certification Card, `$0` CPU GitHub Marketplace Action. | **₹0** | Full pre-flight PR gate + drop-in `KnOTS` static alignment merger usable by any `peft` team. |
| **M2 (`v0.2.0` Triton `Arrow-SGMV` Kernel)** | Completes `EPIC-03`: `Arrow` prototype extractor + `@triton.jit` two-stage Top-$k$ `Arrow-SGMV` CUDA kernel (mixed ranks $r \in \{8,16,32,64\}$) + PyTorch MPS/CPU fallback + property-based parity suite + `loraspec bench` L4 roofline harness. | **₹400–₹700** (L4) | Custom GPU kernel outperforming `peft` eager Arrow `>=2.0x–3.0x` with verified FP16/BF16 numerical equivalence. |
| **M3 (`v0.3.0` `vLLM V1` + React 19 Studio)** | Completes `EPIC-04`, `EPIC-05`: `[project.entry-points."vllm.general_plugins"]` worker hook, FastAPI `/v1/chat/completions` + `/ws/telemetry`, React 19 + TypeScript + Vite + Tailwind dark-mode Studio, `DEMO_REPLAY=true`, `docker compose up`. | **₹350–₹500** (L4) | End-to-end live token-colorized multi-LoRA serving + zero-cost hosted `DEMO_REPLAY` demo. |
| **M4 (`v1.0.0` L4/A100 Study & 5-Channel Launch)** | Completes `EPIC-07` (`recipes/molora_knots_benchmark/`): 6 domain LoRAs + 4-arm accuracy/throughput study across L4 & A100, `docs/TECHNICAL-REPORT.md`, `scripts/export_social_assets.py`, PyPI OIDC + Marketplace `v1`. | **₹750–₹1,200** (L4+A100) | Complete empirical proof-of-work strictly inside `₹2,000–₹3,000` ceiling. |

---

## 2. 100% End-to-End Traceability Matrix (`BRD` → `FRD` → `Feature` → `Arch` → `SDD` → `Milestone`)

| BRD ID | FRD ID(s) | Epic | Feature ID & Name | Arch Component | Stage 4 SDD | Delivery Phase |
|---|---|---|---|---|---|---|
| `BRD-REQ-01` | `FRD-REQ-01`, `FRD-REQ-02` | `EPIC-01` | `FEAT-01`: Bounds-Safe `.safetensors` & PEFT Config Reader | `ARCH-COMP-03` | `SDD-01` | **Stage 5** |
| `BRD-REQ-02` | `FRD-REQ-03`, `FRD-REQ-04` | `EPIC-01` | `FEAT-02`: Config Parity, Scaling (`rsLoRA`) & Target Placement Audit | `ARCH-COMP-04` | `SDD-01` | **Stage 5** |
| `BRD-REQ-03` | `FRD-REQ-05`, `FRD-REQ-06` | `EPIC-01` | `FEAT-03`: Factorized QR+SVD Spectral Health & Rank-Collapse Engine | `ARCH-COMP-04` | `SDD-01` | **Stage 5** |
| `BRD-REQ-04` | `FRD-REQ-07`, `FRD-REQ-08` | `EPIC-01` | `FEAT-04`: Grassmann Principal Angles & Pairwise Collision Analyzer | `ARCH-COMP-04` | `SDD-01` | **Stage 5** |
| `BRD-REQ-05` | `FRD-REQ-09` | `EPIC-01` | `FEAT-05`: `MergeRiskScore (0–100)`, Grade `A–F` & Policy Recommender | `ARCH-COMP-10` | `SDD-01` | **Stage 5** |
| `BRD-REQ-06` | `FRD-REQ-10`, `FRD-REQ-11` | `EPIC-02` | `FEAT-06`: Factorized `KnOTS` Joint-SVD Aligner (Mixed Ranks) | `ARCH-COMP-05` | `SDD-02` | **M1** |
| `BRD-REQ-07` | `FRD-REQ-12`, `FRD-REQ-13` | `EPIC-02` | `FEAT-07`: `knots_ties` / `knots_dare_ties` / `knots_linear` + `GenKnowSub` | `ARCH-COMP-05` | `SDD-02` | **M1** |
| `BRD-REQ-08` | `FRD-REQ-14`, `FRD-REQ-15` | `EPIC-02` | `FEAT-08`: Energy Retention, Reconstruction RMSE & PEFT Serializer | `ARCH-COMP-05` | `SDD-02` | **M1** |
| `BRD-REQ-09` | `FRD-REQ-16`, `FRD-REQ-17` | `EPIC-03` | `FEAT-09`: `Arrow` Spectral Prototype Extractor & Top-$k$ Router Gate | `ARCH-COMP-06` | `SDD-03` | **M2** |
| `BRD-REQ-10` | `FRD-REQ-18`, `FRD-REQ-19` | `EPIC-03` | `FEAT-10`: `@triton.jit` Two-Stage Top-$k$ `Arrow-SGMV` CUDA Kernel | `ARCH-COMP-07` | `SDD-03` | **M2** |
| `BRD-REQ-11` | `FRD-REQ-20` | `EPIC-03` | `FEAT-11`: PyTorch MPS/CPU Reference Backend & Auto-Dispatcher | `ARCH-COMP-07` | `SDD-03` | **M2** |
| `BRD-REQ-12` | `FRD-REQ-21` | `EPIC-03` | `FEAT-12`: Kernel Equivalence Suite & Roofline `loraspec bench` | `ARCH-COMP-07` | `SDD-03`, `SDD-05` | **M2** |
| `BRD-REQ-13` | `FRD-REQ-22`, `FRD-REQ-23` | `EPIC-04` | `FEAT-13`: `vLLM V1` Zero-Fork `general_plugins` Worker Integration | `ARCH-COMP-08` | `SDD-04` | **M3** |
| `BRD-REQ-14` | `FRD-REQ-24`, `FRD-REQ-25` | `EPIC-04` | `FEAT-14`: FastAPI `/v1/chat/completions` + `/ws/telemetry` Gateway | `ARCH-COMP-09` | `SDD-04` | **M3** |
| `BRD-REQ-15` | `FRD-REQ-26` | `EPIC-04` | `FEAT-15`: `DEMO_REPLAY=true` Zero-Cost Recorded Stream Engine | `ARCH-COMP-09` | `SDD-04` | **M3** |
| `BRD-REQ-16` | `FRD-REQ-27`, `FRD-REQ-28`, `FRD-REQ-29` | `EPIC-05` | `FEAT-16`: React 19 Studio (Observatory, `KnOTS` Lab, Token Arena, Roofline) | `ARCH-COMP-12` | `SDD-04` | **M3** |
| `BRD-REQ-17` | `FRD-REQ-30` | `EPIC-05` | `FEAT-17`: Single-Command `Dockerfile` + `docker-compose.yml` Stack | `ARCH-COMP-09`, `ARCH-COMP-12` | `SDD-04` | **M3** |
| `BRD-REQ-18` | `FRD-REQ-31`, `FRD-REQ-32` | `EPIC-06` | `FEAT-18`: JSON, SARIF 2.1.0, Offline HTML (`Jinja2`) & `manifest.json` | `ARCH-COMP-11` | `SDD-05` | **Stage 5** (HTML M1) |
| `BRD-REQ-19` | `FRD-REQ-33` | `EPIC-06` | `FEAT-19`: Typer CLI Surface, `loraspec.toml` (`extra="forbid"`) & Exit `0–5` | `ARCH-COMP-01` | `SDD-05` | **Stage 5** |
| `BRD-REQ-20` | `FRD-REQ-34` | `EPIC-06` | `FEAT-20`: `$0` CPU GitHub Marketplace Action (`action.yml`) | `ARCH-COMP-11` | `SDD-05` | **M1** |
| `BRD-REQ-21` | `FRD-REQ-35` | `EPIC-07` | `FEAT-21`: Atomic `GpuBudgetGuard` & `--smoke < ₹10` Circuit Breaker | `ARCH-COMP-02` | `SDD-05` | **Stage 5** |
| `BRD-REQ-22` | `FRD-REQ-36` | `EPIC-07` | `FEAT-22`: 4-Arm L4/A100 Empirical Study & Social Asset Exporter | `ARCH-COMP-10` | `SDD-05` | **M4** |

---

## 3. Detailed Functional Requirements (`FRD-REQ-01` – `FRD-REQ-36`)

### `EPIC-01`: Safetensors Ingestion & Spectral Subspace Auditing
- **`FRD-REQ-01` — Safe Header Parsing & Bounds Validation (`FEAT-01` | Stage 5)**
  - *Input*: Path to adapter directory (containing `adapter_config.json` + `adapter_model.safetensors`) or direct `.safetensors` file.
  - *Processing*: Read 8-byte little-endian `uint64` header length `N`. Reject immediately (`AdapterParseError`, exit code `2`) if `N == 0`, `N > 100 * 1024 * 1024`, or `8 + N > file_size`. Parse UTF-8 JSON header; verify every tensor entry's `[begin, end]` byte slice satisfies `0 <= begin <= end <= payload_bytes` and `end - begin == prod(shape) * dtype_byte_width`. Validate supported dtypes (`F32`, `F16`, `BF16`).
  - *Output*: Canonical `LoRAAdapterSpec` Pydantic object + SHA-256 file fingerprint.
- **`FRD-REQ-02` — `lora_A` / `lora_B` Layer Pairing & Shape Normalization (`FEAT-01` | Stage 5)**
  - *Processing*: Match keys following PEFT conventions (`base_model.model...lora_A[.adapter].weight` and `lora_B[.adapter].weight`). Validate `lora_A` has shape `(r, d_in)` and `lora_B` has shape `(d_out, r)` (transpose-detect anomalies if swapped). Reject unmatched orphan halves cleanly.
  - *Output*: Ordered list of `LoRALayerSpec` entries with `layer_name`, `module_type` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, etc.), `in_features`, `out_features`, `rank`.
- **`FRD-REQ-03` — Multi-Adapter Config & Base-Model Drift Verification (`FEAT-02` | Stage 5)**
  - *Processing*: Across $\ge 2$ inspected adapters, assert `base_model_name_or_path` matches identically (`RULE-CONF-001` `CRITICAL` if mismatched). Flag missing/zero `lora_alpha` or corrupt config (`RULE-CONF-002` `HIGH`). Flag mixed `rank` or mixed `use_rslora` flags across fleet (`RULE-CONF-004` `MEDIUM`).
- **`FRD-REQ-04` — All-Linear Placement Check (`FEAT-02` | Stage 5)**
  - *Processing*: Inspect `target_modules` against attention projections (`q/k/v/o_proj`) and MLP projections (`gate/up/down_proj`). If MLP projections are omitted, emit `RULE-CONF-003` (`HIGH`) citing Thinking Machines *LoRA Without Regret* capacity findings.
- **`FRD-REQ-05` — Factorized Thin-QR + SVD Computation (`FEAT-03` | Stage 5)**
  - *Processing*: Compute scaling $s = \alpha / r$ (or $\alpha / \sqrt{r}$ if `use_rslora=True`). Perform QR decomposition $s B = Q_B R_B$ and $A^T = Q_A R_A$, then SVD on $r \times r$ matrix $R_B R_A^T = U_c \Sigma V_c^T$. Never allocate dense $d_{\text{out}} \times d_{\text{in}}$ $\Delta W$. Use NumPy/LAPACK fast-path when available; deterministic pure-Python Modified Gram-Schmidt + two-sided Jacobi SVD fallback otherwise.
- **`FRD-REQ-06` — Spectral Health & Rank-Collapse Rules (`FEAT-03` | Stage 5)**
  - *Processing*: For each layer compute $\|\Delta W\|_2 = \sigma_1$, $\|\Delta W\|_F = \sqrt{\sum \sigma_i^2}$, effective stable rank $r_{\text{eff}} = \|\Delta W\|_F^2 / \sigma_1^2$, utilization $r_{\text{eff}} / r$, condition ratio $\sigma_1 / \sigma_r$, and top-1 energy concentration $\sigma_1^2 / \sum \sigma_i^2$.
  - *Rules*:
    - `RULE-SPEC-001` (`CRITICAL`): Non-finite (`NaN`/`Inf`) weights OR dead zero layer ($\|\Delta W\|_F \le 10^{-12}$ when non-zero expected).
    - `RULE-SPEC-002` (`HIGH`): Rank collapse when mean $r_{\text{eff}} / r < 0.20$ or $\ge 30\%$ of layers have top-1 singular energy $> 0.85$ (with $r \ge 4$).
    - `RULE-SPEC-003` (`HIGH`): Spectral norm spike when layer $\|\Delta W\|_2 > \max(10.0, 5.0 \cdot \text{median}_\ell(\|\Delta W^{(\ell)}\|_2))$.
- **`FRD-REQ-07` — Pairwise Grassmann Principal Angles & Similarity (`FEAT-04` | Stage 5)**
  - *Processing*: For each shared layer $\ell$ and adapter pair $(a, b)$, form active column orthonormal bases $U_a = Q_{B,a} U_{c,a}[:, :k_a]$ and $U_b = Q_{B,b} U_{c,b}[:, :k_b]$ (retaining singular directions with $\sigma_i > 10^{-6} \sigma_1$). Compute SVD of $M = U_a^T U_b$, obtaining principal cosines $c_i = \text{clip}(\sigma_i(M), 0, 1)$, principal angles $\theta_i = \arccos(c_i)$, **Grassmann chordal similarity** $\frac{1}{\min(k_a,k_b)} \sum c_i^2$, and **geodesic distance** $\sqrt{\sum \theta_i^2}$. Compute equivalently for row bases $V_a, V_b$ and report the directional overlap.
- **`FRD-REQ-08` — Sign-Conflict & Subspace Collision Diagnostics (`FEAT-04` | Stage 5)**
  - *Processing*: Measure sign-conflict rate across aligned/dominant singular directions: fraction of overlapping basis/factor entries with opposite sign weighted by magnitude. Trigger `RULE-SPEC-004`:
    - `CRITICAL` if `chordal_similarity >= 0.70` AND `sign_conflict_rate >= 0.35` on $\ge 20\%$ of shared layers.
    - `HIGH` if `chordal_similarity >= 0.45` (or `cos_principal_angle_max >= 0.85` with conflict $\ge 0.25$).
- **`FRD-REQ-09` — Deterministic `MergeRiskScore`, Grade (`A–F`) & Recommendations (`FEAT-05` | Stage 5)**
  - *Processing*: Compute bounded `merge_risk_score` $\in [0.0, 100.0]$ (formula locked in `SDD-01`) and assign deterministic Grade `A–F` alongside explicit `recommendations` list (`SAFE_LINEAR_MERGE`, `USE_KNOTS_ALIGNMENT`, `USE_TOP_K_MOLORA`, `RETRAIN_COLLAPSED_LORA`).

### `EPIC-02`: Scalable `KnOTS` Joint-SVD Subspace Aligner
- **`FRD-REQ-10` — Factorized Joint-SVD Shared Basis Extraction (`FEAT-06` | M1)**
  - *Input*: $K \ge 2$ adapters matching base architecture, target rank $r_{\text{out}}$.
  - *Processing*: For each layer $\ell$, concatenate scaled left factors $B_{\text{cat}} = [s_1 B_1, \dots, s_K B_K] \in \mathbb{R}^{d_{\text{out}} \times R_{\text{sum}}}$ ($R_{\text{sum}} = \sum r_m$). Compute thin QR $B_{\text{cat}} = Q_U R_U$ and thin QR of block-stacked row factors $A_{\text{cat}}^T$, or compute Gram matrix SVD across $R_{\text{sum}} \times R_{\text{sum}}$ core to extract shared orthogonal basis $U \Sigma$ and aligned task matrices $\{V_m^T\}_{m=1}^K$ without allocating $d_{\text{out}} \times K d_{\text{in}}$.
- **`FRD-REQ-11` — Heterogeneous Rank Harmonization (`FEAT-06` | M1)**
  - *Processing*: Supports arbitrary mixes of $r_m \in \{4, 8, 16, 32, 64, 128\}$ inside $R_{\text{sum}} = \sum r_m$, projecting every adapter into the common $R_{\text{sum}}$-dimensional (or $\min(d_{\text{out}}, R_{\text{sum}})$) shared singular basis before trimming/merging and compressing back to `target_rank`.
- **`FRD-REQ-12` — Aligned `knots_ties`, `knots_dare_ties`, `knots_linear` Operators (`FEAT-07` | M1)**
  - *Processing*:
    - `knots_linear`: weighted sum $\sum_m w_m V_m^T$ in aligned basis.
    - `knots_ties`: trim bottom $(1 - \text{density})$ magnitudes per $V_m^T$, elect majority weighted sign $\gamma = \text{sign}(\sum_m w_m V_m^T)$, disjoint mean over entries matching $\gamma$.
    - `knots_dare_ties`: Bernoulli drop with rate $p_{\text{dare}}$ and $(1 - p_{\text{dare}})^{-1}$ rescale (deterministic `rng_seed`) followed by `knots_ties`.
- **`FRD-REQ-13` — `GenKnowSub` Residual Subtraction (`FEAT-07` | M1)**
  - *Processing*: Optional `--gks-adapter <path>` and `--gks-lambda` subtracts general-knowledge update $\lambda \Delta W_{\text{gen}}$ in factorized concatenated space prior to `KnOTS` alignment or `Arrow` prototype extraction (`arXiv:2505.10939`).
- **`FRD-REQ-14` — Singular Energy Retention & Reconstruction Metrics (`FEAT-08` | M1)**
  - *Processing*: Compute retained singular energy ratio $\eta = \sum_{i=1}^{r_{\text{out}}} \sigma_i^2 / \sum_{i=1}^{R_{\text{sum}}} \sigma_i^2$, per-task reconstruction RMSE $\|\Delta W_{\text{merged}} - w_m \Delta W_m\|_F / \|\Delta W_m\|_F$, and pre- vs post-`KnOTS` sign-conflict reduction %.
- **`FRD-REQ-15` — PEFT-Compatible Export & Merge Manifest (`FEAT-08` | M1)**
  - *Output*: Valid `adapter_model.safetensors` (`lora_A`, `lora_B` factorized at `r_out`, `lora_alpha = r_out` or configured scaling), `adapter_config.json`, and `knots_merge_manifest.json` loadable directly by `peft.PeftModel.from_pretrained()`.

### `EPIC-03`: Per-Token Top-$k$ `Arrow` Spectral Routing & `@triton.jit` `Arrow-SGMV` Kernel
- **`FRD-REQ-16` — `Arrow` Spectral Prototype Extraction (`FEAT-09` | M2)**
  - *Processing*: Extract unit-norm dominant right singular vector $p_m^{(\ell)} = v_1^{(m,\ell)} \in \mathbb{R}^{d_{\text{in}}}$ per adapter $m$ and target linear layer $\ell$ (with optional `GenKnowSub` subtraction before SVD). Persist prototype bank tensor `[num_adapters, d_in]` per layer.
- **`FRD-REQ-17` — Temperature-Scaled Sign-Invariant Top-$k$ Router (`FEAT-09` | M2)**
  - *Processing*: Given activation $X \in \mathbb{R}^{T \times d_{\text{in}}}$, compute $\hat{X} = X / (\|X\|_2 + \epsilon)$, scores $Z = |\hat{X} P^T| \in \mathbb{R}^{T \times K}$, select `top_k` expert indices $E \in \mathbb{Z}^{T \times k}$ and normalize routing weights $G = \text{softmax}(Z_{\text{top}_k} / \tau) \in \mathbb{R}^{T \times k}$. Compute per-token router Shannon entropy $H_t = -\sum_{j=1}^k g_{t,j} \log(g_{t,j} + \epsilon)$.
- **`FRD-REQ-18` — Deterministic Token-to-Expert Segment Sorting (`FEAT-10` | M2)**
  - *Processing*: Flatten $(T, k)$ routing assignments into $T \cdot k$ `(token_id, slot_idx, expert_id, weight)` items; argsort by `expert_id` on GPU/CPU to build contiguous `sorted_token_ids`, `sorted_weights`, and cumulative `expert_offsets[K+1]` without Python loops.
- **`FRD-REQ-19` — Two-Stage Fused `@triton.jit` `Arrow-SGMV` Kernel (`FEAT-10` | M2)**
  - *Processing*:
    - Stage 1 (`_arrow_sgmv_shrink_kernel`): Grid `(cdiv(max_seg_tokens, BLOCK_M), cdiv(max_rank, BLOCK_R), num_experts)`; gathers $X[\text{token\_id}]$, loads $A_e[:, :r_e]$ masked by `offs_r < rank_e`, accumulates in FP32, writes scratchpad $H \in \mathbb{R}^{(T \cdot k) \times r_{\max}}$.
    - Stage 2 (`_arrow_sgmv_expand_reduce_kernel`): Grid `(cdiv(T, BLOCK_M), cdiv(d_out, BLOCK_N))` or segment expand + deterministic gather-sum across the $k$ slots per token; scales by $g_{t,j} \cdot s_e$, multiplies $H_{t,j} B_e$, and writes $Y_t += \sum_{j=1}^k \Delta y_{t,j}$ in FP32 before final cast to input dtype (guarantees batch-invariant reduction over $k$ slots without non-deterministic float `atomic_add` across experts!).
- **`FRD-REQ-20` — Automatic PyTorch CPU/MPS Reference Fallback (`FEAT-11` | M2)**
  - *Processing*: `ArrowMoLoRADispatcher` inspects `device.type` and Triton availability (`loraspec.kernels.has_triton_cuda()`). Runs `@triton.jit` on CUDA (`>= SM 80` / Ada L4 `SM 89` / Ampere A100 `SM 80`), PyTorch batched index-select reference on MPS/CPU, and pure-Python reference on tiny unit diagnostics.
- **`FRD-REQ-21` — Property-Based Equivalence & Roofline `bench` (`FEAT-12` | M2)**
  - *Processing*: Asserts `max_abs_err <= 1e-3` (FP16) / `<= 5e-3` (BF16) across $T \in \{1, 7, 64, 513, 2048\}$, $K \in \{2, 4, 6\}$, mixed $r_m \in \{8, 16, 32, 64\}$, $k \in \{1..4\}$, and skewed routing (empty expert buckets). Measures kernel ms, effective TFLOP/s, HBM GB/s vs theoretical peak (L4 `300 GB/s`, A100 `2039 GB/s`), and peak VRAM vs `peft` eager Arrow.

### `EPIC-04`: `vLLM V1` Plugin, Streaming Telemetry Gateway & `DEMO_REPLAY`
- **`FRD-REQ-22` — `vLLM V1` General Plugin Entry-Point Hook (`FEAT-13` | M3)**
  - *Processing*: Declares `[project.entry-points."vllm.general_plugins"]` `loraspec = "loraspec.serving.vllm_plugin:register"`. On vLLM worker boot (when `LORASPEC_MOLORA_ENABLE=1`), registers multi-adapter `Arrow` prototype buffers and wraps target linear projections with `ArrowMoLoRADispatcher`.
- **`FRD-REQ-23` — Non-Blocking Token Routing Telemetry Capture (`FEAT-13` | M3)**
  - *Processing*: Captures per-step selected adapter IDs, Top-$k$ softmax weights, router entropy, and step ITL in a pre-allocated ring buffer (zero GPU-CPU sync inside inner layers; single async copy per decode token step).
- **`FRD-REQ-24` — OpenAI-Compatible `/v1/chat/completions` Gateway (`FEAT-14` | M3)**
  - *Processing*: FastAPI endpoint serving SSE token chunks extended with optional `loraspec_routing` payload (`top_k_adapters`, `top_k_weights`, `router_entropy`, `itl_ms`), plus `/api/v1/adapters`, `/api/v1/audit`, and `/api/v1/knots/preview`.
- **`FRD-REQ-25` — Real-Time `/ws/telemetry` WebSocket Bus (`FEAT-14` | M3)**
  - *Processing*: Multiplexes token-by-token generation events, VRAM gauge snapshots, and TTFT/ITL histograms over WebSockets to all connected Studio clients.
- **`FRD-REQ-26` — `DEMO_REPLAY=true` Zero-Cost Recorded Telemetry Mode (`FEAT-15` | M3)**
  - *Processing*: When `DEMO_REPLAY=true` (or `--replay` CLI), loads pre-recorded verified L4 & A100 telemetry JSONL sessions (`assets/demo_replays/*.jsonl`), pacing token streams at authentic measured ITL (`--speed 1.0x | 2.0x | instant`) with zero CUDA dependency.

### `EPIC-05`: Interactive React 19 Web Studio & Container Stack
- **`FRD-REQ-27` — Subspace Observatory & `KnOTS` Lab Views (`FEAT-16` | M3)**
  - *Processing*: Renders interactive Grassmann Chordal Similarity Matrix, per-layer $r_{\text{eff}}$ vs nominal $r$ bar charts, singular value scree plots, `RULE-*` SARIF evidence drawer, and `KnOTS` conflict reduction before/after comparison.
- **`FRD-REQ-28` — Live Token-Colorized `MoLoRA` Generation Arena (`FEAT-16` | M3)**
  - *Processing*: Colorizes each streamed output token dynamically by its top-1 dominant adapter (`sql-lora` emerald, `rust-lora` orange, `json-lora` cyan, `math-lora` violet, `python-lora` amber, `base` slate) with opacity proportional to routing confidence $g_{t,1}$; clicking/hovering any token reveals the full Top-$k$ bar chart, router entropy, and per-layer breakdown.
- **`FRD-REQ-29` — L4 vs A100 Roofline & 4-Arm Ablation Explorer (`FEAT-16` | M3)**
  - *Processing*: Interactive comparison chart plotting Base vs `peft` TIES vs `KnOTS` vs Triton Top-$k$ `MoLoRA` accuracy (`pass@1`), ITL (ms/tok), throughput (tok/s), and peak VRAM across L4 and A100.
- **`FRD-REQ-30` — Single-Command `Dockerfile` + `docker-compose.yml` (`FEAT-17` | M3)**
  - *Processing*: Multi-stage Node 22 + Python 3.12 container image bundling built React static assets served by FastAPI (`DEMO_REPLAY=true` default or optional GPU runtime profile).

### `EPIC-06`: CLI, Reporting, Config & `$0` CPU GitHub Marketplace Action
- **`FRD-REQ-31` — Synchronized JSON, SARIF 2.1.0 & `manifest.json` (`FEAT-18` | Stage 5)**
  - *Processing*: Writes `AuditReport` (`report.json`), `RunManifest` (`manifest.json`), and `report.sarif` conforming strictly to SARIF 2.1.0 with repo-relative normalized artifact URIs (including dotfiles).
- **`FRD-REQ-32` — Zero-CDN Offline HTML Certification Card (`FEAT-18` | M1)**
  - *Processing*: Jinja2 (`autoescape=True`) renders self-contained `report.html` (`< 250 KB`, 0 external HTTP requests) featuring Grade `A–F` badge, `MergeRiskScore` gauge, inline SVG Grassmann collision heatmap, rank utilization chart, and findings table.
- **`FRD-REQ-33` — CLI Surface, `loraspec.toml` (`extra="forbid"`) & Exit Codes `0–5` (`FEAT-19` | Stage 5)**
  - *Processing*: Exposes `doctor`, `inspect`, `audit`, `merge`, `bench`, `serve`, `replay`, `report` with deterministic exit codes `0–5`.
- **`FRD-REQ-34` — GitHub Marketplace Composite Action (`action.yml`) (`FEAT-20` | M1)**
  - *Processing*: Runs `uvx loraspec audit` on `ubuntu-latest` CPU in `< 20s`, outputs `grade`, `merge-risk-score`, `sarif-file`, `html-file`, `exit-code`, uploads SARIF to GitHub Code Scanning and HTML to workflow artifacts.

### `EPIC-07`: Cost Controls & Controlled Empirical GPU Study
- **`FRD-REQ-35` — Atomic `GpuBudgetGuard` & `--smoke < ₹10` Circuit Breaker (`FEAT-21` | Stage 5)**
  - *Processing*: `asyncio.Lock`-guarded hourly rate & cost accumulator (`L4 = ₹57.82/hr`, `A100 = ₹223.02/hr`, local CPU/MPS = `₹0.0`). Aborts before dispatching any sweep step exceeding `max_cost_inr` (`EXIT_BUDGET_EXCEEDED = 3`) and preserves partial telemetry.
- **`FRD-REQ-36` — Controlled 4-Arm Study & Social Asset Generator (`FEAT-22` | M4)**
  - *Processing*: Runs reproducible 4-arm benchmark across 6 domain LoRAs (`SQL`, `Rust`, `JSON`, `Python`, `Math`, `General`) on L4 + 1.5h A100 pass, writes `docs/TECHNICAL-REPORT.md`, and generates 2×-DPI figures + 60fps UI `.mp4` via `scripts/export_social_assets.py`.

---

## 4. Frozen CLI Contract

```text
loraspec [GLOBAL OPTIONS] <COMMAND> [COMMAND OPTIONS]

Commands:
  doctor     Verify Python, CPU/MPS/CUDA backends, optional spectral/triton/vllm stacks & cache health.
  inspect    Safely parse PEFT adapter directory or .safetensors weights into canonical LoRAAdapterSpec.
  audit      Run Tier 0-1 config, spectral rank-collapse & Grassmann collision audit -> JSON/SARIF/HTML.
  merge      Run scalable KnOTS joint-SVD subspace alignment merge across >=2 LoRAs ($0 CPU).
  bench      Benchmark Arrow-SGMV kernel parity, HBM bandwidth, latency & VRAM vs PyTorch/peft baselines.
  serve      Launch FastAPI OpenAI-compatible + /ws/telemetry Studio gateway (live GPU or DEMO_REPLAY).
  replay     Replay a recorded L4/A100 MoLoRA telemetry JSONL trace in terminal or WebSocket stream.
  report     Validate and re-render JSON, SARIF 2.1.0, or offline HTML from an existing report.json.
```

### Deterministic Exit Codes
| Code | Constant | Meaning |
|---|---|---|
| `0` | `EXIT_PASS` | Operation succeeded with zero findings meeting `--fail-on` threshold. |
| `1` | `EXIT_AUDIT_FAIL` | Completed audit/benchmark flagged findings $\ge$ `--fail-on` severity (or `Grade D/F` policy failure). |
| `2` | `EXIT_CONFIG_ERROR` | Invalid CLI args, malformed `loraspec.toml`, corrupt `.safetensors`, or missing adapter config. |
| `3` | `EXIT_BUDGET_EXCEEDED` | Hard `--max-cost-inr` / `--max-gpu-hours` / `--smoke` ceiling triggered (`GpuBudgetGuard`). |
| `4` | `EXIT_RUNTIME_ERROR` | Missing optional CUDA/Triton/vLLM runtime or unrecoverable kernel/serving failure. |
| `5` | `EXIT_PARITY_ERROR` | Triton vs. PyTorch numerical equivalence check violated tolerance (`atol`/`rtol`). |

### Declarative Configuration (`loraspec.toml`)
```toml
[project]
fail_on = "high"                     # critical | high | medium | low
default_backend = "auto"             # auto | triton_cuda | torch_mps | torch_cpu | pure_python

[audit]
rank_collapse_ratio = 0.20           # Flag HIGH if mean r_eff / r < 0.20
top1_energy_threshold = 0.85         # Flag HIGH if >=30% layers exceed 85% energy in sigma_1
spectral_spike_multiplier = 5.0      # Flag HIGH if ||dW||_2 > 5x layer median
collision_high_chordal = 0.45        # Grassmann chordal similarity threshold (HIGH)
collision_crit_chordal = 0.70        # Grassmann chordal similarity threshold (CRITICAL)
sign_conflict_threshold = 0.35       # Directional sign conflict trigger
require_all_linear = true            # Enforce LoRA Without Regret attention+MLP warning

[knots]
default_method = "knots_ties"        # knots_ties | knots_dare_ties | knots_linear
ties_density = 0.50
dare_drop_rate = 0.20
min_energy_retention = 0.90
rng_seed = 42

[molora]
top_k = 2
router_temperature = 1.0
use_gks = false

[budget]
max_cost_inr = 2500.0                # Hard INR cap (<= 3000 ceiling incl. 18% GST)
smoke_cap_inr = 10.0                 # Mandatory --smoke dry-run cap
l4_rate_inr_per_hr = 57.82           # Rs 49/hr + 18% GST
a100_rate_inr_per_hr = 223.02        # Rs 189/hr + 18% GST

[studio]
host = "127.0.0.1"
port = 8000
demo_replay = true
```

---

## 5. Frozen Python SDK & HTTP/WebSocket API Contracts

All Python models use **Pydantic v2** (`extra="forbid"` on user configs, strict enums, ISO-8601 UTC timestamps). Canonical SDK models live in [`src/loraspec/core/schemas.py`](../src/loraspec/core/schemas.py):
- `Severity`, `AdapterFormat`, `LoRALayerSpec`, `LoRAAdapterSpec`, `LayerSpectralMetrics`, `PairwiseCollisionMetrics`, `AuditFinding`, `KnOTSMergeSummary`, `MoLoRARouteTelemetry`, `RunManifest`, `AuditReport`.

### FastAPI Studio REST & WebSocket Endpoints (M3 / `SDD-04`)
| Endpoint | Method / Protocol | Request / Query | Response Payload |
|---|---|---|---|
| `/health` | `GET` | — | `{"status":"ok","demo_replay":bool,"backend":str,"version":str}` |
| `/api/v1/adapters` | `GET` | — | List of loaded `LoRAAdapterSpec` objects |
| `/api/v1/audit` | `POST` | `{"adapter_paths":[...],"fail_on":"high"}` | Complete `AuditReport` JSON |
| `/api/v1/knots/preview` | `POST` | `{"adapters":[...],"method":"knots_ties","target_rank":16}` | `KnOTSMergeSummary` + pre/post conflict comparison |
| `/v1/chat/completions` | `POST` (SSE) | Standard OpenAI body + `{"top_k":2,"temperature_router":1.0,"stream":true}` | OpenAI chunks + `loraspec_routing` (`MoLoRARouteTelemetry`) |
| `/ws/telemetry` | `WebSocket` | `?replay_id=...&speed=1.0` | Live JSON frames (`token`, `routing`, `gpu_stats`, `done`) |

---

## 6. Non-Functional Requirements (`NFR-01` – `NFR-08`)

| NFR ID | Category | Requirement | Verification |
|---|---|---|---|
| **NFR-01** | **Memory Safety** | Spectral auditing & `KnOTS` alignment never materialize dense $d_{\text{out}} \times d_{\text{in}}$ weight updates ($O((d_{\text{out}}+d_{\text{in}})\sum r_m)$ bound). | Peak RSS / `tracemalloc` test on 4096-dim synthetic layers. |
| **NFR-02** | **Audit Latency** | Inspecting + auditing 4 `Qwen3-4B` adapters on 4-vCPU CI completes in `< 10.0s`; `action.yml` end-to-end `< 20s`. | Timed benchmark in CI workflow. |
| **NFR-03** | **Kernel Accuracy** | Triton `Arrow-SGMV` matches PyTorch FP32 reference within `atol <= 1e-3` (FP16) and `<= 5e-3` (BF16). | Randomized property-based CUDA parity tests. |
| **NFR-04** | **Deterministic Reproducibility** | Same weights + `loraspec.toml` + `rng_seed` yield bitwise-identical `MergeRiskScore`, Grade, SARIF, and `KnOTS` factors. | Multi-run determinism unit tests. |
| **NFR-05** | **Cross-OS Compatibility** | Core CLI & unit tests pass on macOS `arm64` (without CUDA/Triton) and Linux `x86_64` (`ubuntu-latest`). | CI matrix + local sandbox verification. |
| **NFR-06** | **Security & XSS Hygiene** | Bounds-checked `.safetensors` header (`<= 100 MB`), zero `pickle.load` execution on untrusted weights, Jinja2 `autoescape=True`, zero external CDN URLs in `report.html`. | Malformed binary tests + HTML XSS tests. |
| **NFR-07** | **SARIF Compliance** | Generated `report.sarif` strictly validates against OASIS SARIF 2.1.0 JSON schema with git-repo-relative URIs. | Schema & git-subdir URI tests. |
| **NFR-08** | **Zero-Cost Demo Sustainability** | Live hosted Web Studio runs indefinitely at `₹0/month` using `DEMO_REPLAY=true` recorded L4/A100 telemetry. | Static build + replay smoke check. |
