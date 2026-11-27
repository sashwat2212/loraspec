"""Unit tests for budget-guarded roofline `loraspec bench` CLI and error exits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from loraspec.cli.app import app
from loraspec.core.budget import BudgetReservation, GpuBudgetGuard
from loraspec.core.errors import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PARITY_ERROR,
    EXIT_PASS,
    BudgetExceededError,
    KernelParityError,
)
from loraspec.core.schemas import BenchmarkSuiteSummary, RunManifest
from loraspec.inspect.safetensors_reader import load_peft_adapter
from loraspec.kernels.bench import run_kernel_parity_and_roofline_bench

runner = CliRunner()


def test_cli_bench_smoke_cpu_budget_guarded(
    orthogonal_adapter_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    path_a, path_b = orthogonal_adapter_pair

    # 1. Synthetic --smoke succeeds on local CPU/MPS at ₹0 and writes bench JSON + manifest
    syn_out = tmp_path / "syn_smoke"
    res_syn = runner.invoke(
        app,
        [
            "bench",
            "--synthetic",
            "--smoke",
            "--top-k",
            "2",
            "--batch-sizes",
            "1,4",
            "--seq-len",
            "4",
            "--dtypes",
            "fp16,bf16",
            "--out-dir",
            str(syn_out),
        ],
    )
    assert res_syn.exit_code == EXIT_PASS, res_syn.stdout + res_syn.stderr
    bench_report = BenchmarkSuiteSummary.model_validate_json(
        (syn_out / "report_bench.json").read_text(encoding="utf-8")
    )
    manifest = RunManifest.model_validate_json(
        (syn_out / "manifest.json").read_text(encoding="utf-8")
    )
    assert bench_report.manifest == manifest
    assert manifest.total_gpu_cost_inr == 0.0
    assert len(bench_report.points) >= 3
    assert {p.arm_name for p in bench_report.points} == {
        "peft_eager_arrow",
        "vllm_static",
        "triton_arrow_sgmv",
    }
    assert bench_report.speedup_vs_peft_eager > 0.0
    assert "Speedup vs PEFT Eager" in res_syn.stdout

    # 2. Real PEFT adapters + --gks-adapter succeeds
    real_out = tmp_path / "real_smoke"
    res_real = runner.invoke(
        app,
        [
            "bench",
            str(path_a),
            str(path_b),
            "--gks-adapter",
            str(path_a),
            "--gks-lambda",
            "0.25",
            "--top-k",
            "2",
            "--batch-sizes",
            "1,2",
            "--seq-len",
            "4",
            "--dtypes",
            "fp32",
            "--smoke",
            "--out-dir",
            str(real_out),
        ],
    )
    assert res_real.exit_code == EXIT_PASS, res_real.stdout + res_real.stderr
    real_rep = BenchmarkSuiteSummary.model_validate_json(
        (real_out / "report_bench.json").read_text(encoding="utf-8")
    )
    assert set(real_rep.manifest.adapter_sha256s) == {"sql_lora", "rust_lora"}

    # 3. Single adapter with default top_k=2 exits EXIT_CONFIG_ERROR (2)
    single_res = runner.invoke(app, ["bench", str(path_a)])
    assert single_res.exit_code == EXIT_CONFIG_ERROR

    # 4. Missing paths and missing --synthetic exits 2
    empty_res = runner.invoke(app, ["bench"])
    assert empty_res.exit_code == EXIT_CONFIG_ERROR

    # 5. Passing both paths and --synthetic exits 2
    both_res = runner.invoke(app, ["bench", str(path_a), str(path_b), "--synthetic"])
    assert both_res.exit_code == EXIT_CONFIG_ERROR

    # 6. Budget exhaustion mid-sweep exits 3 and saves completed partial artifacts
    # Step 1 commits 10s L4 (₹0.1606 <= ₹0.30); step 2 projects 0.1606 + 0.2409 > ₹0.30 -> abort!
    class _FixedStepGuard(GpuBudgetGuard):
        async def reconcile(self, reservation: BudgetReservation, actual_seconds: float) -> float:
            _ = actual_seconds
            return await super().reconcile(reservation, 10.0)

    partial_out = tmp_path / "partial_budget"
    loaded = [load_peft_adapter(path_a), load_peft_adapter(path_b)]
    mid_guard = _FixedStepGuard(max_cost_inr=0.30, smoke=False)
    with pytest.raises(BudgetExceededError) as exc_b:
        run_kernel_parity_and_roofline_bench(
            loaded,
            top_k=2,
            batch_sizes=[1, 2, 4],
            seq_len=4,
            dtypes=["fp32"],
            hardware="e2e_l4_24gb",
            smoke=False,
            max_cost_inr=0.30,
            out_dir=partial_out,
            estimated_step_seconds=15.0,
            budget_guard=mid_guard,
        )
    assert exc_b.value.exit_code == EXIT_BUDGET_EXCEEDED
    assert (partial_out / "report_bench.json").is_file()
    assert (partial_out / "manifest.json").is_file()
    partial_rep = BenchmarkSuiteSummary.model_validate_json(
        (partial_out / "report_bench.json").read_text(encoding="utf-8")
    )
    assert len(partial_rep.points) == 3  # 1 completed batch point (3 arms) saved before abort
    assert partial_rep.manifest.total_gpu_cost_inr == pytest.approx(
        57.82 * (10.0 / 3600.0), abs=1e-4
    )

    # CLI exit code 3 on immediate budget exceed (smoke_cap_inr=0.10, step=15s -> 0.24 > 0.10)
    toml_tight = tmp_path / "tight_cli.toml"
    toml_tight.write_text("[budget]\nmax_cost_inr = 0.10\nsmoke_cap_inr = 0.10\n", encoding="utf-8")
    cli_budget_res = runner.invoke(
        app,
        [
            "bench",
            "--synthetic",
            "--hardware",
            "e2e_l4_24gb",
            "--smoke",
            "--config",
            str(toml_tight),
            "--out-dir",
            str(tmp_path / "cli_budget_partial"),
        ],
    )
    assert cli_budget_res.exit_code == EXIT_BUDGET_EXCEEDED

    # 7. Parity error raises KernelParityError with exit_code == 5
    with pytest.raises(KernelParityError) as exc_p:
        run_kernel_parity_and_roofline_bench(
            synthetic=True,
            batch_sizes=[1],
            seq_len=2,
            dtypes=["fp32"],
            smoke=True,
            parity_tolerances={"fp32": -1.0},
        )
    assert exc_p.value.exit_code == EXIT_PARITY_ERROR

    # 8. Unwritable --out-dir (regular file) exits 2
    bad_target = tmp_path / "not_a_dir.json"
    bad_target.write_text(json.dumps({}), encoding="utf-8")
    bad_out = runner.invoke(app, ["bench", "--synthetic", "--smoke", "--out-dir", str(bad_target)])
    assert bad_out.exit_code == EXIT_CONFIG_ERROR
