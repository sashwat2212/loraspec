# Stage 4 / SDD-05: CLI, Synchronized Reporting, GitHub Action & L4/A100 Empirical Plan — `loraspec`

> [!NOTE]
> **Subsystem**: `SDD-05` (`ARCH-COMP-01` CLI/Config + `ARCH-COMP-02` `GpuBudgetGuard`/Manifest + `ARCH-COMP-11` SARIF/HTML/Action + `ARCH-COMP-10` Empirical Study)
> **Linked Features**: `FEAT-18`, `FEAT-19`, `FEAT-20`, `FEAT-21`, `FEAT-22`
> **Upstream Docs**: [02-FRD-PRD.md](./02-FRD-PRD.md) | [03-ARCHITECTURE.md](./03-ARCHITECTURE.md)
> **Package Modules**: `src/loraspec/cli/`, `src/loraspec/core/config.py`, `src/loraspec/core/budget.py`, `src/loraspec/reporting/`, `action.yml`, `recipes/molora_knots_benchmark/`

---

## 1. CLI Architecture & Strict Config (`ARCH-COMP-01`)

Built with **Typer + Rich + Pydantic v2** (`extra="forbid"` on `loraspec.toml`).
- **Global & Subcommand Surface**: `doctor`, `inspect`, `audit`, `merge`, `bench`, `serve`, `replay`, `report`.
- **Stage 5 vs. Stage 6 Behavior**:
  - Working immediately in Stage 5: `--version`, `doctor`, `inspect`, `audit --tiers 0`, `report`.
  - Clean Stage 6 stubs (`merge` [M1], `bench` [M2], `serve`/`replay` [M3]) print exact milestone guidance and exit deterministically with `EXIT_CONFIG_ERROR (2)` until implemented in M1–M3.

---

## 2. Atomic `GpuBudgetGuard` & Reproducibility Manifest (`ARCH-COMP-02`)

```python
class GpuBudgetGuard:
    """Async-lock guarded INR & GPU-hour circuit breaker (EXIT_BUDGET_EXCEEDED = 3)."""
```
- Built-in verified rate card (E2E Networks incl. 18% GST):
  - `local_cpu` / `local_mps` / `demo_replay`: **`₹0.00 / hr`**
  - `e2e_l4_24gb`: **`₹57.82 / hr`** (`₹49/hr + 18% GST`)
  - `e2e_a100_80gb`: **`₹223.02 / hr`** (`₹189/hr + 18% GST`)
- Mandatory `--smoke` flag on all cloud GPU recipes (`recipes/molora_knots_benchmark/`) enforces `smoke_cap_inr <= 10.0` (halting cleanly after 1–2 micro-batches/prompts to verify pipeline wiring before launching full runs).
- Every run writes an immutable `RunManifest` (`manifest.json`) hashing every `.safetensors` file (`adapter_sha256s`), `loraspec_version`, `git_sha`, `host_os`, `host_arch`, `accelerator_backend`, `rng_seed`, `total_gpu_cost_inr`, and `cli_args`.

---

## 3. Synchronized Reporting & `$0` CPU GitHub Marketplace Action (`ARCH-COMP-11`)

### 3.1 `report.json` & `report.sarif` (OASIS SARIF 2.1.0)
- `report.json`: Direct `AuditReport.model_dump_json(indent=2)` serialization (`schema_version = "1.0.0"`).
- `report.sarif`: Maps every `AuditFinding` (`RULE-CONF-*`, `RULE-SPEC-*`, `RULE-KNOTS-*`) to SARIF 2.1.0:
  - `tool.driver.name = "loraspec"`, `informationUri = "https://github.com/sashwat2212/loraspec"`
  - Severity mapping: `CRITICAL` / `HIGH` → `"error"`, `MEDIUM` → `"warning"`, `LOW` / `INFO` → `"note"`
  - **Git-Repo-Relative URI Normalization**: Walks upward for `.git` so auditing a subdirectory (`./adapters/sql_lora`) emits clean PR-annotatable URIs (`adapters/sql_lora/adapter_config.json`) while preserving dotfiles (`.env`, `.github/...`).

### 3.2 Zero-CDN Offline HTML Certification Card (`report.html` — M1)
- Jinja2 (`autoescape=True`) renders a single `< 250 KB` standalone HTML card with **0 external script/CSS/font CDNs**:
  - Grade `A–F` + `MergeRiskScore (0–100)` gauge (inline SVG)
  - Pairwise Grassmann Chordal Similarity heatmap (inline SVG)
  - Per-layer effective rank utilization bars ($r_{\text{eff}}/r$)
  - Expandable findings table & `RunManifest` SHA-256 verification strip

### 3.3 Root `action.yml` (GitHub Marketplace `sashwat2212/loraspec@v1` — M1)
- Composite Action running natively on free `ubuntu-latest` CPU runners (`< 20s` cold start):
  - Installs via `uvx --from loraspec` (or pinned PR workspace during CI)
  - Runs `loraspec audit <paths> --sarif loraspec.sarif --html loraspec.html --fail-on high`
  - Uploads `loraspec.sarif` to `github/codeql-action/upload-sarif@v3`
  - Uploads `loraspec.html` + `report.json` via `actions/upload-artifact@v4`
  - Exposes outputs: `grade`, `merge-risk-score`, `sarif-file`, `html-file`, `report-json`, `exit-code`

---

## 4. Controlled L4/A100 Empirical Study Design (`recipes/molora_knots_benchmark/` — M4)

### 4.1 Model & Heterogeneous Adapter Suite (`₹350–₹550` L4)
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` (or `Qwen/Qwen3-4B-Instruct`, verified fits cleanly in 24 GB L4 BF16 + multi-LoRA memory).
- **6 Heterogeneous Domain LoRAs** (or verified curated HF domain adapters fine-tuned lightly/standardized cleanly with all-linear placement per *LoRA Without Regret*):
  1. `sql-lora` ($r=16$): Text-to-SQL (`Spider` / `WikiSQL` style schema queries)
  2. `rust-lora` ($r=32$): Idiomatic Rust systems & trait generation
  3. `json-lora` ($r=8$): Strict JSON Schema structured extraction
  4. `python-lora` ($r=32$): Algorithmic Python (`MBPP`-style unit-tested code)
  5. `math-lora` ($r=16$): Step-by-step verifiable GSM math reasoning
  6. `general-lora` ($r=16$): General instruction baseline (for `GenKnowSub` subtraction testing)

### 4.2 Four-Arm Controlled Evaluation (`₹750–₹1,050` L4 + `₹335–₹446` A100)
Evaluated across **single-domain held-out sets** AND a **compositional multi-domain prompt suite** (requiring SQL + Rust + JSON or Python + Math inside single generations):

| Arm | Configuration | Key Metrics Measured |
|---|---|---|
| **Arm 0** | Unadapted Base Model (`Qwen3-4B` / `Qwen2.5-3B`) | `pass@1`, schema validity %, ITL (ms/tok), VRAM |
| **Arm 1** | `peft` Static Merge (`linear` & unaligned `ties`) | Pre-merge `loraspec` collision score, multi-task accuracy drop, ITL |
| **Arm 2** | `loraspec merge` (`KnOTS` Joint-SVD + `TIES`, $r_{\text{out}}=32$) | Pre/post sign conflict reduction, energy retention $\eta$, single/multi-task lift |
| **Arm 3** | Per-Token Top-$k$ `MoLoRA` (`Arrow` + `@triton.jit` `Arrow-SGMV` via `vLLM V1`) | Compositional `pass@1`, domain-aligned token routing purity, throughput & VRAM vs **`peft` eager Arrow** on **L4 + A100** |

### 4.3 Build-in-Public Visual Asset Pipeline (`scripts/export_social_assets.py`)
Generates publication-grade `2×`-DPI figures (`peft` eager stall vs Triton speedup, Grassmann chordal heatmap, L4 vs A100 roofline, 4-arm ablation bar chart) and 60fps Playwright `.mp4` clip of the live token-colorized Studio Arena for LinkedIn, X, and `README.md`.
