# Controlled 4-Arm L4 & A100 Multi-LoRA Empirical Study (`recipes/molora_knots_benchmark/`)

Scheduled for **Stage 6 Milestone M4** (per [docs/04-DESIGN-CLI-CI-BENCHMARKS.md](../../docs/04-DESIGN-CLI-CI-BENCHMARKS.md)) strictly inside the `₹2,000–₹3,000` E2E Networks GPU ceiling (`--smoke < ₹10` pre-flight gate enforced via `GpuBudgetGuard`):

- **Arm 0**: Base Model (`Qwen2.5-3B-Instruct` / `Qwen3-4B-Instruct`)
- **Arm 1**: `peft` Static Merge (`linear` & unaligned `ties`)
- **Arm 2**: `loraspec merge` (`KnOTS` Joint-SVD + `TIES`)
- **Arm 3**: Per-Token Top-$k$ `MoLoRA` (`Arrow` + `@triton.jit` `Arrow-SGMV` via `vLLM V1`) + speedup/VRAM comparison against `peft` eager Arrow across **L4 (`300 GB/s`)** and **A100 (`2039 GB/s`)**.
