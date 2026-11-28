"""CUDA roofline and speedup benchmark tests (skipped cleanly without NVIDIA CUDA)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loraspec.kernels import has_triton_cuda
from loraspec.kernels.bench import run_kernel_parity_and_roofline_bench

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not has_triton_cuda(), reason="Requires NVIDIA CUDA + Triton"),
]


def test_bench_l4_roofline_and_peft_speedup(tmp_path: Path) -> None:
    summary = run_kernel_parity_and_roofline_bench(
        synthetic=True,
        top_k=2,
        batch_sizes=[16, 32],
        seq_len=64,
        dtypes=["fp16", "bf16"],
        hardware="auto",
        smoke=False,
        out_dir=tmp_path / "cuda_bench",
    )
    assert summary.points
    sgmv_pts = [p for p in summary.points if p.arm_name == "triton_arrow_sgmv"]
    assert sgmv_pts
    for pt in sgmv_pts:
        assert pt.achieved_tflops > 0.0
        assert pt.hbm_bandwidth_gb_s > 0.0
        tol = 1e-3 if pt.dtype == "fp16" else 5e-3
        assert pt.max_abs_err_vs_ref <= tol
    assert summary.speedup_vs_peft_eager >= 2.0
    assert summary.vram_savings_pct >= 30.0
