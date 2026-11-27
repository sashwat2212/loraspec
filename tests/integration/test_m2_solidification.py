"""Comprehensive Milestone 2 (M2) solidification tests and Triton kernel AST invariant checks."""

from __future__ import annotations

import ast
import asyncio
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from loraspec.cli.app import app
from loraspec.core.budget import GpuBudgetGuard
from loraspec.core.errors import (
    EXIT_BUDGET_EXCEEDED,
    EXIT_CONFIG_ERROR,
    EXIT_PARITY_ERROR,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    BudgetExceededError,
    KernelParityError,
    RuntimeExecutionError,
)
from loraspec.core.schemas import BenchmarkSuiteSummary, RunManifest
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.kernels import (
    ArrowMoLoRADispatcher,
    build_segment_metadata,
    extract_arrow_prototypes,
    has_mps,
    has_triton_cuda,
    peft_eager_arrow_forward,
    pure_python_arrow_reference,
    route_topk_tokens,
)

runner = CliRunner()

M2_LAYERS: tuple[str, ...] = (
    "base_model.model.layers.0.self_attn.q_proj",
    "base_model.model.layers.0.mlp.gate_proj",
    "base_model.model.layers.0.mlp.up_proj",
    "base_model.model.layers.0.mlp.down_proj",
)


def _write_adapter(
    root: Path,
    *,
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    tensors: dict[str, tuple[Matrix, str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "bias": "none",
        "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def test_triton_kernel_ast_invariants() -> None:
    """AST static verification of all 6 Triton correctness invariants in arrow_sgmv.py."""
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src" / "loraspec" / "kernels" / "arrow_sgmv.py"
    src_text = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src_text)

    # 1. Verify @triton.autotune configs use BLOCK_* >= 16 and powers of 2
    config_lists: dict[str, list[ast.Call]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "_SHRINK_AUTOTUNE_CONFIGS",
                "_EXPAND_AUTOTUNE_CONFIGS",
            }:
                assert isinstance(node.value, ast.List)
                calls: list[ast.Call] = []
                for elt in node.value.elts:
                    assert isinstance(elt, ast.Call)
                    calls.append(elt)
                config_lists[target.id] = calls

    assert set(config_lists) == {"_SHRINK_AUTOTUNE_CONFIGS", "_EXPAND_AUTOTUNE_CONFIGS"}
    for name, calls in config_lists.items():
        assert len(calls) >= 2, f"{name} must provide >= 2 autotune configs"
        for call in calls:
            assert call.args and isinstance(call.args[0], ast.Dict)
            block_dict = call.args[0]
            keys = [
                k.value
                for k in block_dict.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            vals = [
                v.value
                for v in block_dict.values
                if isinstance(v, ast.Constant)
                and isinstance(v.value, int)
                and not isinstance(v.value, bool)
            ]
            kv = dict(zip(keys, vals, strict=True))
            required_keys = (
                {"BLOCK_M", "BLOCK_K", "BLOCK_R"}
                if "SHRINK" in name
                else {"BLOCK_M", "BLOCK_N", "BLOCK_R"}
            )
            assert required_keys.issubset(kv.keys())
            for bname, bval in kv.items():
                assert bval >= 16 and _is_power_of_two(bval), (
                    f"{name} has invalid {bname}={bval} (< 16 or not power of 2)"
                )

    # Locate both @triton.jit kernels and triton_arrow_sgmv_forward
    funcs: dict[str, ast.FunctionDef] = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_arrow_sgmv_shrink_kernel" in funcs
    assert "_arrow_sgmv_expand_kernel" in funcs
    assert "triton_arrow_sgmv_forward" in funcs

    shrink_fn = funcs["_arrow_sgmv_shrink_kernel"]
    expand_fn = funcs["_arrow_sgmv_expand_kernel"]
    fwd_fn = funcs["triton_arrow_sgmv_forward"]

    for kfn in (shrink_fn, expand_fn):
        dec_names = [ast.unparse(d) for d in kfn.decorator_list]
        assert any("triton.autotune" in d for d in dec_names)
        assert any("triton.jit" in d for d in dec_names)

        # 2. Verify tl.int64 casts in pointer arithmetic
        to_int64_calls = [
            n
            for n in ast.walk(kfn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "to"
            and any(ast.unparse(a) == "tl.int64" for a in n.args)
        ]
        assert len(to_int64_calls) >= 5, f"{kfn.name} missing tl.int64 pointer index casts"

        # 3. Verify early return guards
        early_returns = [
            ast.unparse(n)
            for n in ast.walk(kfn)
            if isinstance(n, ast.If) and any(isinstance(b, ast.Return) for b in n.body)
        ]
        assert any("seg_len <= 0" in stmt for stmt in early_returns)
        assert any("pid_m * BLOCK_M >= seg_len" in stmt for stmt in early_returns)
        assert any("rank_e <= 0" in stmt for stmt in early_returns)
        if kfn is shrink_fn:
            assert any("pid_r * BLOCK_R >= rank_e" in stmt for stmt in early_returns)

        # 4. Verify rank axis masking offs_r < rank_e and other=0.0 on tl.load
        compares = [ast.unparse(n) for n in ast.walk(kfn) if isinstance(n, ast.Compare)]
        assert any("offs_r < rank_e" in cmp_src for cmp_src in compares)
        load_calls = [
            n
            for n in ast.walk(kfn)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) == "tl.load"
            and any(kw.arg == "mask" for kw in n.keywords)
        ]
        assert load_calls
        for lcall in load_calls:
            other_kw = next((kw for kw in lcall.keywords if kw.arg == "other"), None)
            assert other_kw is not None, f"Masked tl.load in {kfn.name} must pass other=0.0/0"

        # 5. Verify tl.float32 accumulation and IS_FP32 -> input_precision="ieee"
        zeros_calls = [
            ast.unparse(n)
            for n in ast.walk(kfn)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.zeros"
        ]
        assert any("tl.float32" in z for z in zeros_calls)
        dot_calls = [
            n for n in ast.walk(kfn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.dot"
        ]
        assert any(
            any(
                kw.arg == "input_precision"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "ieee"
                for kw in dcall.keywords
            )
            for dcall in dot_calls
        )

    # Stage 2 casts h_tile.to(b_tile.dtype) when not IS_FP32
    expand_src = ast.unparse(expand_fn)
    assert "h_tile.to(b_tile.dtype)" in expand_src
    assert "Y_slots_ptr" in expand_src

    # 6. Zero tl.atomic_add anywhere in arrow_sgmv.py and deterministic slot sum in forward
    all_calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any("atomic_add" in c for c in all_calls)
    fwd_src = ast.unparse(fwd_fn)
    assert "y_slots.sum(dim=1)" in fwd_src


def test_m2_end_to_end_heterogeneous_molora_pure_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """E2E M2 verification on heterogeneous adapters without requiring torch/CUDA."""
    d_in, d_out = 1024, 1024
    # Build 3 adapters with ranks 4, 8, 16 and mixed rsLoRA in O((d_out+d_in)*r)
    specs = [("sql", 4, 4.0, False, 0), ("rust", 8, 8.0, True, 16), ("json", 16, 16.0, False, 40)]
    dirs: list[Path] = []
    for name, rk, alpha, use_rs, offset in specs:
        tensors: dict[str, tuple[Matrix, str]] = {}
        for lidx, prefix in enumerate(M2_LAYERS):
            shift = offset + lidx * 4
            a_mat = [[1.0 if j == (shift + r) else 0.0 for j in range(d_in)] for r in range(rk)]
            b_mat = [[1.5 if i == (shift + c) else 0.0 for c in range(rk)] for i in range(d_out)]
            tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
            tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")
        dirs.append(
            _write_adapter(
                tmp_path / name, rank=rk, alpha=alpha, use_rslora=use_rs, tensors=tensors
            )
        )

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        loaded = [load_peft_adapter(p) for p in dirs]
        bank = extract_arrow_prototypes(loaded, layer_policy="union")
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(bank.layer_names) == 4

    with capsys.disabled():
        print(
            f"\n[M2 Prototype Extraction] 3 adapters x 4 layers x d={d_in} (r=4,8,16): "
            f"elapsed={elapsed:.3f}s, peak_ram={peak_bytes / (1024 * 1024):.2f} MiB"
        )
    assert elapsed < 10.0
    assert peak_bytes < 150 * 1024 * 1024

    # Small end-to-end routing + segment sorting + dispatcher equivalence
    d_small = 16
    layer0 = M2_LAYERS[0]
    small_dirs: list[Path] = []
    for name, rk, alpha, use_rs, offset in [
        ("s_sql", 2, 2.0, False, 0),
        ("s_rust", 4, 8.0, True, 4),
        ("s_json", 6, 6.0, False, 9),
    ]:
        t_map = {
            f"{layer0}.lora_A.weight": (
                [[1.0 if j == (offset + r) else 0.0 for j in range(d_small)] for r in range(rk)],
                "F32",
            ),
            f"{layer0}.lora_B.weight": (
                [[0.25 if i == (offset + c) else 0.0 for c in range(rk)] for i in range(d_small)],
                "F32",
            ),
        }
        small_dirs.append(
            _write_adapter(tmp_path / name, rank=rk, alpha=alpha, use_rslora=use_rs, tensors=t_map)
        )

    small_loaded = [load_peft_adapter(p) for p in small_dirs]
    small_bank = extract_arrow_prototypes(small_loaded)
    protos = small_bank.prototypes[layer0]
    a_p = small_bank.a_packed[layer0]
    b_p = small_bank.b_packed[layer0]
    ranks = small_bank.ranks[layer0]
    scalings = small_bank.scalings[layer0]

    x_in: Matrix = [
        [2.0 if j == 0 else (0.5 if j == 4 else 0.0) for j in range(d_small)],
        [3.0 if j == 9 else (0.3 if j == 4 else 0.0) for j in range(d_small)],
    ]
    base_in: Matrix = [[0.1 * t for _ in range(d_small)] for t in range(2)]

    e_ids, w_mat, ent = route_topk_tokens(x_in, protos, top_k=2, temperature=1.0)
    assert len(ent) == 2 and all(v >= 0.0 for v in ent)
    layout = build_segment_metadata(e_ids, w_mat, num_experts=3)
    assert layout.num_tokens == 2 and layout.top_k == 2

    y_disp, tel = ArrowMoLoRADispatcher().forward(
        x_in, base_in, protos, a_p, b_p, ranks, scalings, top_k=2, backend="auto", layer_name=layer0
    )
    y_ref, _ = pure_python_arrow_reference(
        x_in, base_in, protos, a_p, b_p, ranks, scalings, top_k=2, layer_name=layer0
    )
    y_peft, _ = peft_eager_arrow_forward(
        x_in, base_in, protos, a_p, b_p, ranks, scalings, top_k=2, layer_name=layer0
    )
    for rd, rr, rp in zip(y_disp, y_ref, y_peft, strict=True):
        assert rd == pytest.approx(rr, abs=1e-12)
        assert rd == pytest.approx(rp, abs=1e-12)
    assert tel.layer_name == layer0

    # CLI bench end-to-end artifact verification and exit codes 0, 2, 3, 4, 5
    out_bench = tmp_path / "bench_e2e"
    res0 = runner.invoke(
        app,
        [
            "bench",
            str(small_dirs[0]),
            str(small_dirs[1]),
            str(small_dirs[2]),
            "--top-k",
            "2",
            "--batch-sizes",
            "1,2",
            "--seq-len",
            "4",
            "--dtypes",
            "fp16,bf16",
            "--smoke",
            "--out-dir",
            str(out_bench),
        ],
    )
    assert res0.exit_code == EXIT_PASS, res0.stdout + res0.stderr
    rep = BenchmarkSuiteSummary.model_validate_json(
        (out_bench / "report_bench.json").read_text("utf-8")
    )
    man = RunManifest.model_validate_json((out_bench / "manifest.json").read_text("utf-8"))
    assert rep.manifest == man
    assert set(man.adapter_sha256s) == {"s_sql", "s_rust", "s_json"}

    assert runner.invoke(app, ["bench", str(small_dirs[0])]).exit_code == EXIT_CONFIG_ERROR
    guard = GpuBudgetGuard(max_cost_inr=10.0, smoke=True, smoke_cap_inr=10.0)
    with pytest.raises(BudgetExceededError) as exc_b:
        asyncio.run(guard.reserve_or_raise("e2e_l4_24gb", 3600.0))
    assert exc_b.value.exit_code == EXIT_BUDGET_EXCEEDED
    assert RuntimeExecutionError("cuda absent").exit_code == EXIT_RUNTIME_ERROR
    assert KernelParityError("err").exit_code == EXIT_PARITY_ERROR
    assert isinstance(has_triton_cuda(), bool)


def test_m2_pytorch_cpu_mps_dispatcher_parity(tmp_path: Path) -> None:
    """Verify full PyTorch CPU (and Apple MPS if available) parity across fp16/bf16/fp32."""
    torch_mod: Any = pytest.importorskip("torch")
    d_in, d_out, r_max, num_experts, num_tokens = 24, 20, 8, 4, 17
    torch_mod.manual_seed(7)

    x_fp32 = torch_mod.randn((num_tokens, d_in), dtype=torch_mod.float32) * 0.5
    base_fp32 = torch_mod.randn((num_tokens, d_out), dtype=torch_mod.float32) * 0.25
    protos = torch_mod.nn.functional.normalize(
        torch_mod.randn((num_experts, d_in), dtype=torch_mod.float32), dim=-1
    )
    ranks_list = [2, 4, 8, 3]
    a_fp32 = torch_mod.zeros((num_experts, r_max, d_in), dtype=torch_mod.float32)
    b_fp32 = torch_mod.zeros((num_experts, d_out, r_max), dtype=torch_mod.float32)
    for e_idx, rk in enumerate(ranks_list):
        a_fp32[e_idx, :rk, :] = torch_mod.randn((rk, d_in), dtype=torch_mod.float32) * 0.2
        b_fp32[e_idx, :, :rk] = torch_mod.randn((d_out, rk), dtype=torch_mod.float32) * 0.2
    ranks_t = torch_mod.tensor(ranks_list, dtype=torch_mod.int32)
    scalings_t = torch_mod.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch_mod.float32)

    dispatcher = ArrowMoLoRADispatcher()
    y_py, _ = pure_python_arrow_reference(
        x_fp32.tolist(),
        base_fp32.tolist(),
        protos.tolist(),
        a_fp32.tolist(),
        b_fp32.tolist(),
        ranks_list,
        scalings_t.tolist(),
        top_k=2,
    )
    y_cpu, tel_cpu = dispatcher.forward(
        x_fp32, base_fp32, protos, a_fp32, b_fp32, ranks_t, scalings_t, top_k=2, backend="auto"
    )
    assert tel_cpu.backend == "torch_cpu"
    for row_cpu, row_py in zip(y_cpu.tolist(), y_py, strict=True):
        assert row_cpu == pytest.approx(row_py, abs=1e-4)

    for dt, tol in ((torch_mod.float16, 1e-3), (torch_mod.bfloat16, 5e-3)):
        x_dt = x_fp32.to(dt)
        base_dt = base_fp32.to(dt)
        a_dt = a_fp32.to(dt)
        b_dt = b_fp32.to(dt)
        y_ref_dt, _ = dispatcher.forward(
            x_dt.float(),
            base_dt.float(),
            protos,
            a_dt.float(),
            b_dt.float(),
            ranks_t,
            scalings_t,
            top_k=2,
            backend="torch_cpu",
        )
        y_dt, _ = dispatcher.forward(
            x_dt,
            base_dt,
            protos,
            a_dt,
            b_dt,
            ranks_t,
            scalings_t,
            top_k=2,
            backend="torch_cpu",
        )
        assert float(torch_mod.max(torch_mod.abs(y_dt.float() - y_ref_dt)).item()) <= tol

    if has_mps():
        y_mps, tel_mps = dispatcher.forward(
            x_fp32.to("mps"),
            base_fp32.to("mps"),
            protos.to("mps"),
            a_fp32.to("mps"),
            b_fp32.to("mps"),
            ranks_t.to("mps"),
            scalings_t.to("mps"),
            top_k=2,
            backend="auto",
        )
        assert tel_mps.backend == "torch_mps"
        assert float(torch_mod.max(torch_mod.abs(y_mps.cpu() - y_cpu)).item()) <= 1e-5

    out_torch = tmp_path / "torch_bench"
    res = runner.invoke(
        app,
        ["bench", "--synthetic", "--smoke", "--batch-sizes", "1,4", "--out-dir", str(out_torch)],
    )
    assert res.exit_code == EXIT_PASS, res.stdout + res.stderr
    assert (out_torch / "report_bench.json").is_file()

    # Explicit --hardware local_cpu executes on CPU even when Apple MPS is present
    out_cpu_explicit = tmp_path / "torch_bench_cpu"
    res_cpu = runner.invoke(
        app,
        [
            "bench",
            "--synthetic",
            "--smoke",
            "--hardware",
            "local_cpu",
            "--batch-sizes",
            "1",
            "--dtypes",
            "fp32",
            "--out-dir",
            str(out_cpu_explicit),
        ],
    )
    assert res_cpu.exit_code == EXIT_PASS, res_cpu.stdout + res_cpu.stderr
    rep_cpu = BenchmarkSuiteSummary.model_validate_json(
        (out_cpu_explicit / "report_bench.json").read_text("utf-8")
    )
    assert all(p.hardware_profile == "local_cpu" for p in rep_cpu.points)
    if not has_mps():
        res_bad_mps = runner.invoke(
            app, ["bench", "--synthetic", "--smoke", "--hardware", "local_mps"]
        )
        assert res_bad_mps.exit_code == EXIT_RUNTIME_ERROR
