"""Unit tests for factorized KnOTS joint-SVD alignment and heterogeneous-rank handling."""

from __future__ import annotations

import json
import math
import tracemalloc
from pathlib import Path

import pytest

from loraspec.core.errors import AdapterParseError, ConfigError
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.knots.aligner import (
    align_subspaces_knots,
    harmonize_heterogeneous_ranks,
    matmul_matrices,
    transpose_matrix,
)
from loraspec.knots.operators import knots_linear
from loraspec.knots.serializer import reconstruct_merged_lora


def _dense_svd_reference(mat: Matrix) -> tuple[Matrix, list[float], Matrix]:
    """Brute-force dense reference via thin_qr + small_svd for tiny synthetic tests."""
    from loraspec.core.spectral_math import small_svd, thin_qr

    m = len(mat)
    n = len(mat[0])
    if m >= n:
        q, r = thin_qr(mat)
        uc, sigmas, vc = small_svd(r)
        return matmul_matrices(q, uc), sigmas, vc
    # For wide m < n, compute SVD of mat^T then swap U, V
    q, r = thin_qr(transpose_matrix(mat))
    vc, sigmas, uc = small_svd(r)
    return uc, sigmas[:m], matmul_matrices(q, vc)


def _write_adapter_dir(
    root: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    layers: dict[str, tuple[Matrix, Matrix]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    modules = sorted({lname.rsplit(".", maxsplit=1)[-1] for lname in layers})
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "bias": "none",
        "target_modules": modules,
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tensors: dict[str, tuple[Matrix, str]] = {}
    for lname, (a_mat, b_mat) in layers.items():
        tensors[f"{lname}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{lname}.lora_B.weight"] = (b_mat, "F32")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def test_factorized_knots_matches_dense_svd() -> None:
    d_out, d_in = 14, 10
    r1, r2, r3 = 3, 2, 4
    b1 = [[math.sin(0.3 * (i + 1) * (j + 1)) for j in range(r1)] for i in range(d_out)]
    a1 = [[math.cos(0.2 * (i + 1) * (j + 2)) for j in range(d_in)] for i in range(r1)]
    b2 = [[0.7 * math.cos(0.4 * (i + 2) * (j + 1)) for j in range(r2)] for i in range(d_out)]
    a2 = [[0.9 * math.sin(0.25 * (i + 1) * (j + 1)) for j in range(d_in)] for i in range(r2)]
    b3 = [[0.5 * math.sin(0.15 * (i + 3) * (j + 2)) for j in range(r3)] for i in range(d_out)]
    a3 = [[1.1 * math.cos(0.35 * (i + 2) * (j + 1)) for j in range(d_in)] for i in range(r3)]

    scaled_bs = [
        [[val * 1.5 for val in row] for row in b1],
        [[val * 0.8 for val in row] for row in b2],
        [[val * 1.2 for val in row] for row in b3],
    ]
    a_mats = [a1, a2, a3]

    res = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["m1", "m2", "m3"],
        scaled_bs=scaled_bs,
        a_mats=a_mats,
    )
    assert res.total_sum_rank == r1 + r2 + r3

    # Build dense stacked [dW_1, dW_2, dW_3] for verification only
    dw_list = [matmul_matrices(sb, am) for sb, am in zip(scaled_bs, a_mats, strict=True)]
    dw_stack: Matrix = [dw_list[0][i] + dw_list[1][i] + dw_list[2][i] for i in range(d_out)]
    _, ref_sigmas, _ = _dense_svd_reference(dw_stack)

    k_cmp = min(len(res.sigmas), len(ref_sigmas))
    for got_s, ref_s in zip(res.sigmas[:k_cmp], ref_sigmas[:k_cmp], strict=True):
        assert got_s == pytest.approx(ref_s, abs=1e-5)

    # Verify U^T U == I (for active basis) and U @ diag(sigmas) @ V_m^T == dW_m within < 1e-5
    k_basis = len(res.sigmas)
    for col_i in range(k_basis):
        for col_j in range(k_basis):
            if res.sigmas[col_i] > 1e-9 and res.sigmas[col_j] > 1e-9:
                dot = sum(res.u_shared[r][col_i] * res.u_shared[r][col_j] for r in range(d_out))
                assert dot == pytest.approx(1.0 if col_i == col_j else 0.0, abs=1e-5)

    for dw_m, v_m_t in zip(dw_list, res.aligned_v_list, strict=True):
        scaled_v = [[res.sigmas[k] * val for val in v_m_t[k]] for k in range(k_basis)]
        recon = matmul_matrices(res.u_shared, scaled_v)
        for row_exp, row_got in zip(dw_m, recon, strict=True):
            for exp_v, got_v in zip(row_exp, row_got, strict=True):
                assert got_v == pytest.approx(exp_v, abs=1e-5)

    # Also test wide case d_out < R_sum (d_out=4, R_sum=9)
    res_wide = align_subspaces_knots(
        layer_name="layer0.wide",
        adapter_ids=["m1", "m2", "m3"],
        scaled_bs=[sb[:4] for sb in scaled_bs],
        a_mats=a_mats,
    )
    assert len(res_wide.sigmas) == 4
    for sb_w, a_w, v_m_t in zip(
        [sb[:4] for sb in scaled_bs], a_mats, res_wide.aligned_v_list, strict=True
    ):
        dw_w = matmul_matrices(sb_w, a_w)
        scaled_v_w = [[res_wide.sigmas[k] * val for val in v_m_t[k]] for k in range(4)]
        recon_w = matmul_matrices(res_wide.u_shared, scaled_v_w)
        for row_exp, row_got in zip(dw_w, recon_w, strict=True):
            for exp_v, got_v in zip(row_exp, row_got, strict=True):
                assert got_v == pytest.approx(exp_v, abs=1e-5)


def test_knots_no_dense_allocation_tracemalloc() -> None:
    d_out = 4096
    d_in = 4096
    r = 8
    num_adapters = 3
    dense_single_bytes = d_out * d_in * 8  # 128 MiB

    scaled_bs: list[Matrix] = []
    a_mats: list[Matrix] = []
    for k in range(num_adapters):
        b_m = [
            [1.0 / math.sqrt(d_out) if ((i + k) % r == j) else 0.0 for j in range(r)]
            for i in range(d_out)
        ]
        a_m = [
            [1.0 / math.sqrt(d_in) if ((j + k) % r == i) else 0.0 for j in range(d_in)]
            for i in range(r)
        ]
        scaled_bs.append(b_m)
        a_mats.append(a_m)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        aligned = align_subspaces_knots(
            layer_name="layers.0.mlp.up_proj",
            adapter_ids=["a0", "a1", "a2"],
            scaled_bs=scaled_bs,
            a_mats=a_mats,
        )
        v_merged_t = knots_linear(aligned.aligned_v_list)
        b_out, a_out, eta = reconstruct_merged_lora(
            aligned.u_shared, aligned.sigmas, v_merged_t, target_rank=16
        )
        _cur, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(b_out) == d_out
    assert len(a_out) == 16
    assert 0.0 <= eta <= 1.0
    assert peak_bytes < dense_single_bytes // 4, (
        f"KnOTS peak traced memory {peak_bytes} exceeded quarter of a single 4096x4096 dense matrix"
    )


def test_knots_heterogeneous_ranks_and_rslora_scaling(tmp_path: Path) -> None:
    d_out, d_in = 48, 40
    lname = "base_model.model.layers.0.mlp.gate_proj"

    def _make_factors(r: int, seed_offset: int) -> tuple[Matrix, Matrix]:
        a_m = [
            [math.cos(0.11 * (i + 1 + seed_offset) * (j + 1)) * 0.2 for j in range(d_in)]
            for i in range(r)
        ]
        b_m = [
            [math.sin(0.13 * (i + 1) * (j + 1 + seed_offset)) * 0.2 for j in range(r)]
            for i in range(d_out)
        ]
        return a_m, b_m

    p8 = _write_adapter_dir(
        tmp_path / "r8",
        rank=8,
        alpha=16.0,
        use_rslora=False,
        layers={lname: _make_factors(8, 1)},
    )
    p16 = _write_adapter_dir(
        tmp_path / "r16",
        rank=16,
        alpha=16.0,
        use_rslora=True,
        layers={lname: _make_factors(16, 5)},
    )
    p32 = _write_adapter_dir(
        tmp_path / "r32",
        rank=32,
        alpha=32.0,
        use_rslora=False,
        layers={lname: _make_factors(32, 9)},
    )

    loaded = [load_peft_adapter(p8), load_peft_adapter(p16), load_peft_adapter(p32)]
    layers = harmonize_heterogeneous_ranks(loaded, layer_policy="union")
    assert layers == [lname]

    scaled_bs = [
        [[val * item.spec.scaling for val in row] for row in item.layer_factors[lname][1]]
        for item in loaded
    ]
    a_mats = [item.layer_factors[lname][0] for item in loaded]
    aligned = align_subspaces_knots(
        layer_name=lname,
        adapter_ids=[item.spec.adapter_id for item in loaded],
        scaled_bs=scaled_bs,
        a_mats=a_mats,
    )
    assert aligned.total_sum_rank == 8 + 16 + 32

    v_merged_t = knots_linear(aligned.aligned_v_list)
    b_out, a_out, eta = reconstruct_merged_lora(
        aligned.u_shared, aligned.sigmas, v_merged_t, target_rank=24
    )
    assert len(b_out) == d_out
    assert len(b_out[0]) == 24
    assert len(a_out) == 24
    assert len(a_out[0]) == d_in
    assert 0.0 < eta <= 1.0


def test_knots_disjoint_layer_policies(tmp_path: Path) -> None:
    d = 12
    r = 4
    l_shared = "base_model.model.layers.0.self_attn.q_proj"
    l_only_a = "base_model.model.layers.0.mlp.gate_proj"
    l_only_b = "base_model.model.layers.1.mlp.down_proj"

    a_eye = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
    b_eye = [[1.0 if i == j else 0.0 for j in range(r)] for i in range(d)]

    pa = _write_adapter_dir(
        tmp_path / "pa",
        rank=r,
        alpha=float(r),
        layers={l_shared: (a_eye, b_eye), l_only_a: (a_eye, b_eye)},
    )
    pb = _write_adapter_dir(
        tmp_path / "pb",
        rank=r,
        alpha=float(r),
        layers={l_shared: (a_eye, b_eye), l_only_b: (a_eye, b_eye)},
    )

    loaded_a = load_peft_adapter(pa)
    loaded_b = load_peft_adapter(pb)

    assert harmonize_heterogeneous_ranks([loaded_a, loaded_b], layer_policy="union") == sorted(
        [l_shared, l_only_a, l_only_b]
    )
    assert harmonize_heterogeneous_ranks([loaded_a, loaded_b], layer_policy="intersection") == [
        l_shared
    ]

    with pytest.raises(ConfigError):
        harmonize_heterogeneous_ranks([loaded_a])
    with pytest.raises(ConfigError):
        harmonize_heterogeneous_ranks([loaded_a, loaded_b], layer_policy="invalid")  # type: ignore[arg-type]

    # Conflicting base model rejected
    pc = _write_adapter_dir(
        tmp_path / "pc",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        rank=r,
        alpha=float(r),
        layers={l_shared: (a_eye, b_eye)},
    )
    with pytest.raises(AdapterParseError, match="conflicting base models"):
        harmonize_heterogeneous_ranks([loaded_a, load_peft_adapter(pc)])

    # Empty intersection rejected
    pd = _write_adapter_dir(
        tmp_path / "pd",
        rank=r,
        alpha=float(r),
        layers={"base_model.model.layers.9.mlp.up_proj": (a_eye, b_eye)},
    )
    with pytest.raises(AdapterParseError, match="No layers survived"):
        harmonize_heterogeneous_ranks(
            [loaded_a, load_peft_adapter(pd)], layer_policy="intersection"
        )


def test_knots_zero_adapter_reconstruction() -> None:
    d_out, d_in, r = 10, 8, 3
    zero_b = [[0.0] * r for _ in range(d_out)]
    a1 = [[1.0 if j == i else 0.0 for j in range(d_in)] for i in range(r)]
    b2 = [[math.sin(0.3 * (i + 1) * (j + 1)) for j in range(r)] for i in range(d_out)]
    a2 = [[math.cos(0.2 * (i + 1) * (j + 2)) for j in range(d_in)] for i in range(r)]

    res_z = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["dead", "live"],
        scaled_bs=[zero_b, b2],
        a_mats=[a1, a2],
        backend="pure_python",
    )
    v_m = knots_linear(res_z.aligned_v_list, weights=[0.5, 0.5], backend="pure_python")
    b_out, a_out, eta = reconstruct_merged_lora(
        res_z.u_shared, res_z.sigmas, v_m, target_rank=r, backend="pure_python"
    )
    assert eta == pytest.approx(1.0, abs=1e-5)
    dw2_half = matmul_matrices([[0.5 * v for v in row] for row in b2], a2)
    recon = matmul_matrices(b_out, a_out)
    for r_exp, r_got in zip(dw2_half, recon, strict=True):
        for v_exp, v_got in zip(r_exp, r_got, strict=True):
            assert v_got == pytest.approx(v_exp, abs=1e-5)


def test_knots_numpy_vs_pure_python_parity() -> None:
    pytest.importorskip("numpy")
    d_out, d_in, r = 10, 8, 3
    zero_b = [[0.0] * r for _ in range(d_out)]
    a1 = [[1.0 if j == i else 0.0 for j in range(d_in)] for i in range(r)]
    b2 = [[math.sin(0.3 * (i + 1) * (j + 1)) for j in range(r)] for i in range(d_out)]
    a2 = [[math.cos(0.2 * (i + 1) * (j + 2)) for j in range(d_in)] for i in range(r)]

    res_py = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["dead", "live"],
        scaled_bs=[zero_b, b2],
        a_mats=[a1, a2],
        backend="pure_python",
    )
    res_np = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["dead", "live"],
        scaled_bs=[zero_b, b2],
        a_mats=[a1, a2],
        backend="numpy",
    )
    for s_np, s_py in zip(res_np.sigmas, res_py.sigmas, strict=True):
        assert s_np == pytest.approx(s_py, abs=1e-9)
