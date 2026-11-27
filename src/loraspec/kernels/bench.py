"""Budget-guarded numerical parity and roofline microbenchmark harness (`loraspec bench`)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import math
import random
import time
import tracemalloc
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from loraspec.core.budget import GpuBudgetGuard
from loraspec.core.errors import BudgetExceededError, ConfigError, KernelParityError
from loraspec.core.manifest import build_run_manifest
from loraspec.core.schemas import (
    BenchmarkSuiteSummary,
    KernelBenchmarkPoint,
    RunManifest,
)
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights
from loraspec.kernels import has_mps, has_triton_cuda, optional_torch
from loraspec.kernels.dispatch import ArrowMoLoRADispatcher
from loraspec.kernels.reference import (
    arrow_molora_reference,
    peft_eager_arrow_forward,
    pure_python_arrow_reference,
    static_lora_forward,
)
from loraspec.kernels.router import ArrowPrototypeBank, extract_arrow_prototypes

BenchDtype = Literal["fp16", "bf16", "fp32"]

_PARITY_TOLERANCE_BY_DTYPE: dict[BenchDtype, float] = {
    "fp16": 1e-3,
    "bf16": 5e-3,
    "fp32": 1e-5,
}

_VALID_HARDWARE_INPUTS: frozenset[str] = frozenset(
    {"auto", "local_cpu", "local_mps", "demo_replay", "e2e_l4_24gb", "e2e_a100_80gb"}
)


def _normalize_bench_dtype(raw: str) -> BenchDtype:
    norm = raw.strip().lower()
    alias_map: dict[str, BenchDtype] = {
        "fp16": "fp16",
        "f16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "f32": "fp32",
        "float32": "fp32",
    }
    if norm not in alias_map:
        raise ConfigError(f"Unsupported benchmark dtype '{raw}'. Supported: fp16, bf16, fp32.")
    return alias_map[norm]


def resolve_bench_hardware(hardware: str) -> str:
    """Resolve `auto` or validate explicit hardware profile against GpuBudgetGuard rate table."""
    if hardware not in _VALID_HARDWARE_INPUTS:
        raise ConfigError(
            f"Unsupported --hardware '{hardware}'. Choose from: {sorted(_VALID_HARDWARE_INPUTS)}."
        )
    if hardware != "auto":
        return hardware
    if has_triton_cuda():
        torch_mod = optional_torch()
        if torch_mod is not None:
            try:
                dev_name = str(torch_mod.cuda.get_device_name(0)).upper()
                if "A100" in dev_name:
                    return "e2e_a100_80gb"
            except Exception:
                pass
        return "e2e_l4_24gb"
    if has_mps():
        return "local_mps"
    return "local_cpu"


def _build_synthetic_prototype_bank(
    *,
    num_experts: int,
    d_in: int,
    d_out: int,
    seed: int,
) -> ArrowPrototypeBank:
    rng = random.Random(seed)
    base_ranks = [8, 16, 32, 64]
    ranks = [base_ranks[i % len(base_ranks)] for i in range(num_experts)]
    r_max = max(ranks)
    scalings = [
        1.0 if i % 2 == 0 else (16.0 / math.sqrt(float(ranks[i]))) for i in range(num_experts)
    ]

    protos: Matrix = []
    a_packed: list[Matrix] = []
    b_packed: list[Matrix] = []

    for exp_idx in range(num_experts):
        r_e = ranks[exp_idx]
        p_raw = [rng.gauss(0.0, 1.0) for _ in range(d_in)]
        p_norm = max(math.sqrt(sum(v * v for v in p_raw)), 1e-12)
        protos.append([v / p_norm for v in p_raw])

        a_mat: Matrix = []
        for r in range(r_max):
            if r < r_e:
                a_mat.append([rng.gauss(0.0, 0.05) for _ in range(d_in)])
            else:
                a_mat.append([0.0] * d_in)
        a_packed.append(a_mat)

        b_mat: Matrix = []
        for _ in range(d_out):
            row = [rng.gauss(0.0, 0.05) if r < r_e else 0.0 for r in range(r_max)]
            b_mat.append(row)
        b_packed.append(b_mat)

    layer_name = "synthetic.layers.0.mlp.down_proj"
    return ArrowPrototypeBank(
        adapter_ids=tuple(f"expert_{i}" for i in range(num_experts)),
        layer_names=(layer_name,),
        prototypes={layer_name: protos},
        a_packed={layer_name: a_packed},
        b_packed={layer_name: b_packed},
        ranks={layer_name: ranks},
        scalings={layer_name: scalings},
        in_features={layer_name: d_in},
        out_features={layer_name: d_out},
        warnings=(),
    )


def _write_bench_artifacts(
    out_dir: Path,
    summary: BenchmarkSuiteSummary,
    manifest: RunManifest,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_bench.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _compute_roofline_metrics(
    *,
    arm_name: Literal["peft_eager_arrow", "vllm_static", "triton_arrow_sgmv"],
    num_tokens: int,
    top_k: int,
    d_in: int,
    d_out: int,
    ranks: Sequence[int],
    expert_indices: Sequence[Sequence[int]],
    dtype: BenchDtype,
    latency_ms: float,
) -> tuple[float, float, float]:
    bpe = 4 if dtype == "fp32" else 2
    num_experts = len(ranks)
    r_max = max(ranks, default=1)
    latency_s = max(latency_ms / 1000.0, 1e-9)

    if arm_name == "vllm_static":
        r0 = max(ranks[0], 1)
        flops = float(2 * num_tokens * (d_in + d_out) * r0)
        moved_bytes = float(
            (num_tokens * d_in + num_tokens * d_out * 2 + r0 * (d_in + d_out)) * bpe
            + num_tokens * r0 * 4 * 2
        )
        scratchpad_mb = float(num_tokens * (r0 + d_out) * 4) / (1024.0 * 1024.0)
    elif arm_name == "peft_eager_arrow":
        router_flops = 2 * num_tokens * d_in * num_experts
        gemm_flops = sum(2 * num_tokens * (d_in + d_out) * max(r, 1) for r in ranks)
        flops = float(router_flops + gemm_flops)
        moved_bytes = float(
            (num_tokens * d_in + num_experts * d_in + num_tokens * d_out * 2) * bpe
            + sum((r * (d_in + d_out)) * bpe for r in ranks)
            + num_experts * num_tokens * d_out * 4 * 2
        )
        scratchpad_mb = float(num_experts * num_tokens * d_out * 4) / (1024.0 * 1024.0)
    else:
        # triton_arrow_sgmv
        router_flops = 2 * num_tokens * d_in * num_experts
        active_rank_sum = sum(
            max(ranks[exp_id], 0) for tok_exps in expert_indices for exp_id in tok_exps
        )
        sgmv_flops = 2 * (d_in + d_out) * active_rank_sum
        flops = float(router_flops + sgmv_flops)
        active_experts = {exp_id for tok_exps in expert_indices for exp_id in tok_exps}
        weight_bytes = sum(ranks[e] * (d_in + d_out) * bpe for e in active_experts)
        moved_bytes = float(
            (num_tokens * d_in + num_experts * d_in + num_tokens * d_out * 2) * bpe
            + weight_bytes
            + num_tokens * top_k * r_max * 4 * 2
            + num_tokens * top_k * d_out * 4 * 2
        )
        scratchpad_mb = float(num_tokens * top_k * (r_max + d_out) * 4) / (1024.0 * 1024.0)

    tflops = flops / (latency_s * 1e12)
    bw_gb_s = moved_bytes / (latency_s * 1e9)
    return round(tflops, 6), round(bw_gb_s, 4), scratchpad_mb


def _measure_python_arm(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    analytical_mb: float,
) -> tuple[Any, float, float]:
    for _ in range(warmup):
        _ = fn()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        result: Any = None
        for _ in range(iters):
            result = fn()
        elapsed_ms = ((time.perf_counter() - t0) / float(iters)) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    peak_mb = max(analytical_mb, float(peak_bytes) / (1024.0 * 1024.0))
    return result, max(elapsed_ms, 1e-4), round(peak_mb, 6)


def _measure_torch_arm(
    torch_mod: Any,
    device: Any,
    fn: Any,
    *,
    warmup: int,
    iters: int,
    analytical_mb: float,
) -> tuple[Any, float, float]:
    for _ in range(warmup):
        _ = fn()

    dev_type = str(getattr(device, "type", device)).lower()
    if dev_type == "cuda":
        torch_mod.cuda.synchronize(device)
        torch_mod.cuda.reset_peak_memory_stats(device)
        base_alloc = int(torch_mod.cuda.memory_allocated(device))
        start_ev = torch_mod.cuda.Event(enable_timing=True)
        end_ev = torch_mod.cuda.Event(enable_timing=True)
        start_ev.record()
        result: Any = None
        for _ in range(iters):
            result = fn()
        end_ev.record()
        torch_mod.cuda.synchronize(device)
        elapsed_ms = float(start_ev.elapsed_time(end_ev)) / float(iters)
        peak_alloc = int(torch_mod.cuda.max_memory_allocated(device))
        delta_mb = max(0.0, float(peak_alloc - base_alloc) / (1024.0 * 1024.0))
        return result, max(elapsed_ms, 1e-4), round(max(analytical_mb, delta_mb), 6)

    if dev_type == "mps":
        mps_mod = getattr(torch_mod, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "synchronize"):
            mps_mod.synchronize()
        t0 = time.perf_counter()
        result = None
        for _ in range(iters):
            result = fn()
        if mps_mod is not None and hasattr(mps_mod, "synchronize"):
            mps_mod.synchronize()
        elapsed_ms = ((time.perf_counter() - t0) / float(iters)) * 1000.0
        return result, max(elapsed_ms, 1e-4), round(analytical_mb, 6)

    return _measure_python_arm(fn, warmup=0, iters=iters, analytical_mb=analytical_mb)


def _compute_suite_ratios(
    points: Sequence[KernelBenchmarkPoint],
) -> tuple[float, float]:
    peft_map = {(p.batch_size, p.dtype): p for p in points if p.arm_name == "peft_eager_arrow"}
    sgmv_map = {(p.batch_size, p.dtype): p for p in points if p.arm_name == "triton_arrow_sgmv"}
    speedups: list[float] = []
    vram_savings: list[float] = []
    for key, sgmv_pt in sgmv_map.items():
        peft_pt = peft_map.get(key)
        if peft_pt is None:
            continue
        if sgmv_pt.latency_ms > 0.0:
            speedups.append(peft_pt.latency_ms / sgmv_pt.latency_ms)
        if peft_pt.peak_vram_mb > 0.0:
            vram_savings.append(
                ((peft_pt.peak_vram_mb - sgmv_pt.peak_vram_mb) / peft_pt.peak_vram_mb) * 100.0
            )
    mean_speedup = sum(speedups) / float(len(speedups)) if speedups else 1.0
    mean_savings = sum(vram_savings) / float(len(vram_savings)) if vram_savings else 0.0
    return round(mean_speedup, 4), round(mean_savings, 4)


def _check_positive_int(val: object, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < 1:
        raise ConfigError(f"{name} must be an integer >= 1, got {val!r}.")
    return val


def _check_positive_float(val: object, name: str) -> float:
    if (
        isinstance(val, bool)
        or not isinstance(val, (int, float))
        or not math.isfinite(float(val))
        or float(val) <= 0.0
    ):
        raise ConfigError(f"{name} must be finite and > 0.0, got {val!r}.")
    return float(val)


def _check_finite_float(
    val: object, name: str, *, min_val: float, max_val: float | None = None
) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ConfigError(f"{name} must be numeric, got {val!r}.")
    fval = float(val)
    if not math.isfinite(fval) or fval < min_val or (max_val is not None and fval > max_val):
        bound = f"[{min_val}, {max_val}]" if max_val is not None else f">= {min_val}"
        raise ConfigError(f"{name} must be finite and {bound}, got {val!r}.")
    return fval


async def async_run_kernel_parity_and_roofline_bench(
    loaded_adapters: Sequence[LoadedLoRAWeights] | None = None,
    *,
    synthetic: bool = False,
    top_k: int = 2,
    temperature: float = 1.0,
    batch_sizes: Sequence[int] = (1, 4, 16, 32, 64),
    seq_len: int = 32,
    dtypes: Sequence[str] = ("fp16", "bf16"),
    hardware: str = "auto",
    smoke: bool = True,
    max_cost_inr: float = 2500.0,
    smoke_cap_inr: float = 10.0,
    gks_adapter: LoadedLoRAWeights | None = None,
    gks_lambda: float = 1.0,
    out_dir: Path | None = None,
    seed: int = 42,
    cli_args: Sequence[str] | None = None,
    estimated_step_seconds: float = 15.0,
    budget_guard: GpuBudgetGuard | None = None,
    parity_tolerances: dict[BenchDtype, float] | None = None,
) -> BenchmarkSuiteSummary:
    """Run parity check + roofline microbenchmark across peft, static vLLM, and Arrow-SGMV."""
    top_k_val = _check_positive_int(top_k, "top_k")
    temp_val = _check_positive_float(temperature, "temperature")
    seq_len_val = _check_positive_int(seq_len, "seq_len")
    max_cost_val = _check_finite_float(max_cost_inr, "max_cost_inr", min_val=0.0, max_val=3000.0)
    est_sec_val = _check_finite_float(estimated_step_seconds, "estimated_step_seconds", min_val=0.0)

    if not batch_sizes:
        raise ConfigError("batch_sizes cannot be empty.")
    resolved_batches: list[int] = [_check_positive_int(bs, "batch_size") for bs in batch_sizes]

    if not dtypes:
        raise ConfigError("dtypes cannot be empty.")
    resolved_dtypes: list[BenchDtype] = [_normalize_bench_dtype(item) for item in dtypes]

    resolved_hw = resolve_bench_hardware(hardware)
    guard = budget_guard or GpuBudgetGuard(
        max_cost_inr=float(max_cost_val),
        smoke=smoke,
        smoke_cap_inr=float(smoke_cap_inr),
    )

    if synthetic and loaded_adapters:
        raise ConfigError("Specify either loaded_adapters or synthetic=True, not both.")
    if not synthetic and not loaded_adapters:
        raise ConfigError("Provide >= 1 LoRA adapter path or pass --synthetic.")

    torch_mod = optional_torch()
    use_cuda = has_triton_cuda()
    use_mps = has_mps() and not use_cuda

    if synthetic:
        num_experts = max(4, top_k_val)
        if torch_mod is not None and use_cuda:
            syn_d_in, syn_d_out = 512, 512
        elif torch_mod is not None:
            syn_d_in, syn_d_out = 128, 128
        else:
            syn_d_in, syn_d_out = 48, 48
        bank = _build_synthetic_prototype_bank(
            num_experts=num_experts, d_in=syn_d_in, d_out=syn_d_out, seed=seed
        )
        adapter_sha256s = {"synthetic_molora": f"synthetic-k{num_experts}-d{syn_d_in}-seed{seed}"}
        root_hint = Path.cwd()
    else:
        assert loaded_adapters is not None
        if top_k_val > len(loaded_adapters):
            raise ConfigError(
                f"top_k={top_k_val} exceeds number of loaded adapters ({len(loaded_adapters)})."
            )
        bank = extract_arrow_prototypes(
            loaded_adapters,
            gks_adapter=gks_adapter,
            gks_lambda=gks_lambda,
            layer_policy="union",
            backend="auto",
        )
        adapter_sha256s = {
            item.spec.adapter_id: item.spec.weights_sha256 for item in loaded_adapters
        }
        root_hint = loaded_adapters[0].spec.root_path

    if smoke:
        resolved_batches = resolved_batches[:2]
        if torch_mod is None and len(resolved_dtypes) > 1:
            resolved_dtypes = resolved_dtypes[:1]
        eff_seq_len = min(seq_len_val, 8) if torch_mod is None else seq_len_val
        warmup_iters = 1
        timed_iters = 2
    else:
        eff_seq_len = seq_len_val
        warmup_iters = 3
        timed_iters = 8

    target_layer = bank.layer_names[0]
    d_in = bank.in_features[target_layer]
    d_out = bank.out_features[target_layer]
    ranks_list = bank.ranks[target_layer]
    scalings_list = bank.scalings[target_layer]
    protos_mat = bank.prototypes[target_layer]
    a_packed_mat = bank.a_packed[target_layer]
    b_packed_mat = bank.b_packed[target_layer]

    tol_table = dict(_PARITY_TOLERANCE_BY_DTYPE)
    if parity_tolerances is not None:
        tol_table.update(parity_tolerances)

    dispatcher = ArrowMoLoRADispatcher()
    points: list[KernelBenchmarkPoint] = []
    rng = random.Random(seed + 99)

    def _finalize_summary() -> tuple[BenchmarkSuiteSummary, RunManifest]:
        manifest = build_run_manifest(
            adapter_sha256s=adapter_sha256s,
            root_hint=root_hint,
            rng_seed=seed,
            total_gpu_cost_inr=guard.spent_cost_inr,
            cli_args=list(cli_args) if cli_args is not None else [],
        )
        speedup, vram_sav = _compute_suite_ratios(points)
        summary = BenchmarkSuiteSummary(
            manifest=manifest,
            points=points,
            speedup_vs_peft_eager=speedup,
            vram_savings_pct=vram_sav,
            arm_accuracy_pass_at_1={},
        )
        return summary, manifest

    for bs in resolved_batches:
        num_tokens = bs * eff_seq_len
        for bench_dt in resolved_dtypes:
            try:
                res = await guard.reserve_or_raise(resolved_hw, float(est_sec_val))
            except BudgetExceededError:
                if out_dir is not None:
                    partial_summary, partial_manifest = _finalize_summary()
                    _write_bench_artifacts(out_dir, partial_summary, partial_manifest)
                raise

            t_step0 = time.perf_counter()
            try:
                if torch_mod is not None:
                    if use_cuda:
                        exec_device = torch_mod.device("cuda")
                        dispatch_backend: Literal["triton_cuda", "torch_mps", "torch_cpu"] = (
                            "triton_cuda"
                        )
                    elif use_mps:
                        exec_device = torch_mod.device("mps")
                        dispatch_backend = "torch_mps"
                    else:
                        exec_device = torch_mod.device("cpu")
                        dispatch_backend = "torch_cpu"

                    torch_mod.manual_seed(seed + bs)
                    dtype_map: dict[BenchDtype, Any] = {
                        "fp16": torch_mod.float16,
                        "bf16": torch_mod.bfloat16,
                        "fp32": torch_mod.float32,
                    }
                    dtype_obj = dtype_map[bench_dt]

                    x_t = (
                        torch_mod.randn((num_tokens, d_in), dtype=dtype_obj, device=exec_device)
                        * 0.5
                    )
                    base_t = (
                        torch_mod.randn((num_tokens, d_out), dtype=dtype_obj, device=exec_device)
                        * 0.25
                    )
                    proto_t, a_t, b_t, ranks_t, scalings_t = bank.layer_tensors(
                        target_layer, device=exec_device, dtype=dtype_obj
                    )

                    # FP32 reference evaluated on identical quantized input values
                    ref_out, _ = arrow_molora_reference(
                        x=x_t.to(dtype=torch_mod.float32),
                        base_out=base_t.to(dtype=torch_mod.float32),
                        prototypes=proto_t,
                        a_weights=a_t.to(dtype=torch_mod.float32),
                        b_weights=b_t.to(dtype=torch_mod.float32),
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k_val,
                        temperature=temp_val,
                        layer_name=target_layer,
                    )

                    # Parity check on ArrowMoLoRADispatcher
                    cand_out, cand_tel = dispatcher.forward(
                        x=x_t,
                        base_out=base_t,
                        prototypes=proto_t,
                        a_weights=a_t,
                        b_weights=b_t,
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k_val,
                        temperature=temp_val,
                        backend=dispatch_backend,
                        layer_name=target_layer,
                    )
                    max_abs_err = float(
                        torch_mod.max(
                            torch_mod.abs(cand_out.to(dtype=torch_mod.float32) - ref_out)
                        ).item()
                    )
                    tol = tol_table[bench_dt]
                    if not math.isfinite(max_abs_err) or max_abs_err > tol:
                        raise KernelParityError(
                            f"Kernel parity failure at batch={bs}, seq_len={eff_seq_len}, "
                            f"dtype={bench_dt}: max_abs_err={max_abs_err:.6g} > tol={tol:.6g}."
                        )

                    # Measure 3 arms
                    _, _, peft_mb_est = _compute_roofline_metrics(
                        arm_name="peft_eager_arrow",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    peft_call = functools.partial(
                        peft_eager_arrow_forward,
                        x=x_t,
                        base_out=base_t,
                        prototypes=proto_t,
                        a_weights=a_t,
                        b_weights=b_t,
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k_val,
                        temperature=temp_val,
                        layer_name=target_layer,
                    )
                    _, peft_ms, peft_vram = _measure_torch_arm(
                        torch_mod,
                        exec_device,
                        peft_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=peft_mb_est,
                    )
                    peft_tf, peft_bw, _ = _compute_roofline_metrics(
                        arm_name="peft_eager_arrow",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=peft_ms,
                    )

                    _, _, static_mb_est = _compute_roofline_metrics(
                        arm_name="vllm_static",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    static_call = functools.partial(
                        static_lora_forward,
                        x=x_t,
                        base_out=base_t,
                        a_weight=a_t[0],
                        b_weight=b_t[0],
                        scaling=scalings_list[0],
                        rank=ranks_list[0],
                    )
                    _, static_ms, static_vram = _measure_torch_arm(
                        torch_mod,
                        exec_device,
                        static_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=static_mb_est,
                    )
                    static_tf, static_bw, _ = _compute_roofline_metrics(
                        arm_name="vllm_static",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=static_ms,
                    )

                    _, _, sgmv_mb_est = _compute_roofline_metrics(
                        arm_name="triton_arrow_sgmv",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    sgmv_call = functools.partial(
                        dispatcher.forward,
                        x=x_t,
                        base_out=base_t,
                        prototypes=proto_t,
                        a_weights=a_t,
                        b_weights=b_t,
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k_val,
                        temperature=temp_val,
                        backend=dispatch_backend,
                        layer_name=target_layer,
                    )
                    _, sgmv_ms, sgmv_vram = _measure_torch_arm(
                        torch_mod,
                        exec_device,
                        sgmv_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=sgmv_mb_est,
                    )
                    sgmv_tf, sgmv_bw, _ = _compute_roofline_metrics(
                        arm_name="triton_arrow_sgmv",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=sgmv_ms,
                    )
                else:
                    x_py = [[rng.gauss(0.0, 1.0) for _ in range(d_in)] for _ in range(num_tokens)]
                    base_py = [
                        [rng.gauss(0.0, 1.0) for _ in range(d_out)] for _ in range(num_tokens)
                    ]

                    ref_py, _ = pure_python_arrow_reference(
                        x=x_py,
                        base_out=base_py,
                        prototypes=protos_mat,
                        a_weights=a_packed_mat,
                        b_weights=b_packed_mat,
                        ranks=ranks_list,
                        scalings=scalings_list,
                        top_k=top_k_val,
                        temperature=temp_val,
                        layer_name=target_layer,
                    )
                    cand_py, cand_tel = dispatcher.forward(
                        x=x_py,
                        base_out=base_py,
                        prototypes=protos_mat,
                        a_weights=a_packed_mat,
                        b_weights=b_packed_mat,
                        ranks=ranks_list,
                        scalings=scalings_list,
                        top_k=top_k_val,
                        temperature=temp_val,
                        backend="pure_python",
                        layer_name=target_layer,
                    )
                    max_abs_err = max(
                        (
                            abs(c - r)
                            for row_c, row_r in zip(cand_py, ref_py, strict=True)
                            for c, r in zip(row_c, row_r, strict=True)
                        ),
                        default=0.0,
                    )
                    tol = tol_table[bench_dt]
                    if not math.isfinite(max_abs_err) or max_abs_err > tol:
                        raise KernelParityError(
                            f"Kernel parity failure at batch={bs}, seq_len={eff_seq_len}, "
                            f"dtype={bench_dt}: max_abs_err={max_abs_err:.6g} > tol={tol:.6g}."
                        )

                    _, _, peft_mb_est = _compute_roofline_metrics(
                        arm_name="peft_eager_arrow",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    peft_py_call = functools.partial(
                        peft_eager_arrow_forward,
                        x=x_py,
                        base_out=base_py,
                        prototypes=protos_mat,
                        a_weights=a_packed_mat,
                        b_weights=b_packed_mat,
                        ranks=ranks_list,
                        scalings=scalings_list,
                        top_k=top_k_val,
                        temperature=temp_val,
                        layer_name=target_layer,
                    )
                    _, peft_ms, peft_vram = _measure_python_arm(
                        peft_py_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=peft_mb_est,
                    )
                    peft_tf, peft_bw, _ = _compute_roofline_metrics(
                        arm_name="peft_eager_arrow",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=peft_ms,
                    )

                    _, _, static_mb_est = _compute_roofline_metrics(
                        arm_name="vllm_static",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    static_py_call = functools.partial(
                        static_lora_forward,
                        x=x_py,
                        base_out=base_py,
                        a_weight=a_packed_mat[0],
                        b_weight=b_packed_mat[0],
                        scaling=scalings_list[0],
                        rank=ranks_list[0],
                    )
                    _, static_ms, static_vram = _measure_python_arm(
                        static_py_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=static_mb_est,
                    )
                    static_tf, static_bw, _ = _compute_roofline_metrics(
                        arm_name="vllm_static",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=static_ms,
                    )

                    _, _, sgmv_mb_est = _compute_roofline_metrics(
                        arm_name="triton_arrow_sgmv",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=1.0,
                    )
                    sgmv_py_call = functools.partial(
                        dispatcher.forward,
                        x=x_py,
                        base_out=base_py,
                        prototypes=protos_mat,
                        a_weights=a_packed_mat,
                        b_weights=b_packed_mat,
                        ranks=ranks_list,
                        scalings=scalings_list,
                        top_k=top_k_val,
                        temperature=temp_val,
                        backend="pure_python",
                        layer_name=target_layer,
                    )
                    _, sgmv_ms, sgmv_vram = _measure_python_arm(
                        sgmv_py_call,
                        warmup=warmup_iters,
                        iters=timed_iters,
                        analytical_mb=sgmv_mb_est,
                    )
                    sgmv_tf, sgmv_bw, _ = _compute_roofline_metrics(
                        arm_name="triton_arrow_sgmv",
                        num_tokens=num_tokens,
                        top_k=top_k_val,
                        d_in=d_in,
                        d_out=d_out,
                        ranks=ranks_list,
                        expert_indices=cand_tel.expert_indices,
                        dtype=bench_dt,
                        latency_ms=sgmv_ms,
                    )

                points.append(
                    KernelBenchmarkPoint(
                        hardware_profile=resolved_hw,
                        arm_name="peft_eager_arrow",
                        batch_size=bs,
                        seq_len=eff_seq_len,
                        dtype=bench_dt,
                        latency_ms=round(peft_ms, 4),
                        achieved_tflops=peft_tf,
                        hbm_bandwidth_gb_s=peft_bw,
                        peak_vram_mb=peft_vram,
                        max_abs_err_vs_ref=0.0,
                    )
                )
                points.append(
                    KernelBenchmarkPoint(
                        hardware_profile=resolved_hw,
                        arm_name="vllm_static",
                        batch_size=bs,
                        seq_len=eff_seq_len,
                        dtype=bench_dt,
                        latency_ms=round(static_ms, 4),
                        achieved_tflops=static_tf,
                        hbm_bandwidth_gb_s=static_bw,
                        peak_vram_mb=static_vram,
                        max_abs_err_vs_ref=0.0,
                    )
                )
                points.append(
                    KernelBenchmarkPoint(
                        hardware_profile=resolved_hw,
                        arm_name="triton_arrow_sgmv",
                        batch_size=bs,
                        seq_len=eff_seq_len,
                        dtype=bench_dt,
                        latency_ms=round(sgmv_ms, 4),
                        achieved_tflops=sgmv_tf,
                        hbm_bandwidth_gb_s=sgmv_bw,
                        peak_vram_mb=sgmv_vram,
                        max_abs_err_vs_ref=round(max_abs_err, 8),
                    )
                )
            except Exception:
                await guard.release(res)
                raise
            else:
                await guard.reconcile(res, max(time.perf_counter() - t_step0, 0.0))

    summary, manifest = _finalize_summary()
    if out_dir is not None:
        _write_bench_artifacts(out_dir, summary, manifest)
    return summary


def run_kernel_parity_and_roofline_bench(
    loaded_adapters: Sequence[LoadedLoRAWeights] | None = None,
    *,
    synthetic: bool = False,
    top_k: int = 2,
    temperature: float = 1.0,
    batch_sizes: Sequence[int] = (1, 4, 16, 32, 64),
    seq_len: int = 32,
    dtypes: Sequence[str] = ("fp16", "bf16"),
    hardware: str = "auto",
    smoke: bool = True,
    max_cost_inr: float = 2500.0,
    smoke_cap_inr: float = 10.0,
    gks_adapter: LoadedLoRAWeights | None = None,
    gks_lambda: float = 1.0,
    out_dir: Path | None = None,
    seed: int = 42,
    cli_args: Sequence[str] | None = None,
    estimated_step_seconds: float = 15.0,
    budget_guard: GpuBudgetGuard | None = None,
    parity_tolerances: dict[BenchDtype, float] | None = None,
) -> BenchmarkSuiteSummary:
    """Synchronous entry point for kernel parity and roofline benchmark sweep."""
    coro = async_run_kernel_parity_and_roofline_bench(
        loaded_adapters=loaded_adapters,
        synthetic=synthetic,
        top_k=top_k,
        temperature=temperature,
        batch_sizes=batch_sizes,
        seq_len=seq_len,
        dtypes=dtypes,
        hardware=hardware,
        smoke=smoke,
        max_cost_inr=max_cost_inr,
        smoke_cap_inr=smoke_cap_inr,
        gks_adapter=gks_adapter,
        gks_lambda=gks_lambda,
        out_dir=out_dir,
        seed=seed,
        cli_args=cli_args,
        estimated_step_seconds=estimated_step_seconds,
        budget_guard=budget_guard,
        parity_tolerances=parity_tolerances,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(asyncio.run, coro)
        return fut.result()
