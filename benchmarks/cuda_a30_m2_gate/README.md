# Milestone 2 (`M2`) — Live NVIDIA A30 CUDA & Triton Hardware Gate Proof (`GATE-6 [M2]`)

This directory stores the exact reproducible hardware verification artifacts captured on **E2E Cloud `NVIDIA A30` (`24 GB HBM2`)** for Milestone 2 (`v0.2.0`) under `GpuBudgetGuard`.

## 1. Environment & Stack

| Component | Exact Verified Value |
|---|---|
| **GPU Hardware** | NVIDIA A30 (`24,576 MiB` HBM2, Ampere `SM 8.0`, `933 GB/s` peak HBM bandwidth) |
| **Driver / CUDA** | NVIDIA Driver `570.133.20` (`CUDA 12.8`) |
| **Host OS** | Ubuntu `24.04.3 LTS` (`x86_64`, Kernel `6.8.0-79-generic`) |
| **Python / Package Versions** | Python `3.12.3`, `torch==2.10.0+cu128`, `triton==3.6.0`, `numpy==2.5.5`, `safetensors==0.8.0` |
| **Budget Profile** | `e2e_a30_24gb` (`₹106.20/hr` = `Rs. 90/hr + 18% GST`) |

## 2. Verification & Static Gate Results

| Check | Result |
|---|---|
| `uv run --offline --no-sync ruff check .` | `All checks passed!` (`0` errors) |
| `uv run --offline --no-sync ruff format --check .` | `59 files already formatted` |
| `uv run --offline --no-sync pyright` | `0 errors, 0 warnings, 0 informations` |
| `uv run --offline --no-sync pytest -v` | **`84 passed in 80.69s`** (`0` failed, `0` skipped — includes all `tests/gpu/*.py` `@triton.jit` CUDA tests) |

## 3. Roofline & Parity Summary (`report_bench.json` / `manifest.json`)

- **Command**: `uv run --offline --no-sync loraspec bench --synthetic --no-smoke --hardware e2e_a30_24gb --batch-sizes 1,4,16,32,64 --seq-len 64 --dtypes fp16,bf16 --out-dir /tmp/a30-roofline`
- **Problem Geometry**: `K = 8` heterogeneous experts (`ranks = [8, 16, 32, 64, 8, 16, 32, 64]`), `top_k = 2`, `d_in = 4096`, `d_out = 2048`, `seq_len = 64` (`T = 64 .. 4096` tokens).
- **Numerical Equivalence (`KPI-ENG-02`)**:
  - `fp16`: `2.60e-4 .. 4.36e-4` max abs error vs `float32` reference (`<= 1.00e-3` threshold)
  - `bf16`: `2.00e-3 .. 3.86e-3` max abs error vs `float32` reference (`<= 5.00e-3` threshold)
- **Latency Speedup & Scratchpad VRAM Savings (`KPI-RES-03`)**:
  - **Suite-Wide Mean Speedup vs `peft` Eager Arrow**: **`2.0048x`** (`1.50x–1.58x` at launch-bound `B=1..4` → `1.80x–1.82x` at `B=16` → **`2.43x–2.48x` at `B=32`** → **`2.66x–2.76x` at `B=64`**)
  - **Mean Peak Scratchpad VRAM Reduction vs `peft` Eager Arrow**: **`76.75%`** (`674.69 MB` → `162.16 MB` at `B=64, S=64`)
- **Measured Sweep Cost (`KPI-ENG-05`)**: `₹0.1723` INR (`--smoke`: `₹0.1108` INR); total `~1 hr` live `A30` provisioning/debugging session spend `~₹106.20` INR incl. 18% GST (`<= ₹3,000` budget ceiling).

