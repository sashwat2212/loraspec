"""Unit tests for KnOTS operators (TIES, DARE-TIES, linear, GenKnowSub) and diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from loraspec.core.errors import AdapterParseError, ConfigError
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.knots.aligner import align_subspaces_knots, matmul_matrices, transpose_matrix
from loraspec.knots.operators import (
    apply_genknowsub,
    knots_dare_ties,
    knots_linear,
    knots_ties,
    merge_aligned_factors,
)
from loraspec.knots.serializer import (
    compute_merge_diagnostics,
    factorized_frobenius_rmse,
    reconstruct_merged_lora,
)


def _rotation_matrix_4(theta: float) -> Matrix:
    c, s = math.cos(theta), math.sin(theta)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, c, s],
        [0.0, 0.0, -s, c],
    ]


def _make_adapter(
    root: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    rank: int,
    alpha: float,
    layers: dict[str, tuple[Matrix, Matrix]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    modules = sorted({lname.rsplit(".", maxsplit=1)[-1] for lname in layers})
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": modules,
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tensors: dict[str, tuple[Matrix, str]] = {}
    for lname, (a_mat, b_mat) in layers.items():
        tensors[f"{lname}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{lname}.lora_B.weight"] = (b_mat, "F32")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def test_knots_ties_reduces_sign_conflict() -> None:
    d_out, d_in, r = 16, 16, 4
    b1 = [[math.cos(0.3 * (i + 1) * (j + 1)) for j in range(r)] for i in range(d_out)]
    a1 = [[math.sin(0.25 * (i + 1) * (j + 2)) for j in range(d_in)] for i in range(r)]

    # Rotated & sign-flipped gauge-equivalent adapter: B2 = B1 @ (-Q), A2 = (-Q)^T @ A1
    # Unaligned parameter-space TIES suffers large sign conflict, while KnOTS aligns V1^T == V2^T
    rot = _rotation_matrix_4(0.7)
    neg_rot = [[-v for v in row] for row in rot]
    b2 = matmul_matrices(b1, neg_rot)
    a2 = matmul_matrices(transpose_matrix(neg_rot), a1)

    aligned = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["orig", "rotated"],
        scaled_bs=[b1, b2],
        a_mats=[a1, a2],
    )
    assert aligned.pre_merge_sign_conflict > 0.35

    v_merged_t, post_conflict = merge_aligned_factors(
        aligned,
        method="knots_ties",
        weights=[0.5, 0.5],
        ties_density=1.0,
    )
    assert post_conflict < 1e-6
    assert post_conflict < aligned.pre_merge_sign_conflict

    b_out, a_out, eta = reconstruct_merged_lora(
        aligned.u_shared, aligned.sigmas, v_merged_t, target_rank=r
    )
    assert eta == pytest.approx(1.0, abs=1e-5)
    rmse = factorized_frobenius_rmse(b_out, a_out, b1, a1)
    assert rmse < 1e-5


def test_knots_dare_ties_seed_determinism() -> None:
    v1 = [[float(i + 1) * 0.5 + 0.1 * j for j in range(12)] for i in range(4)]
    v2 = [[float(i + 2) * 0.4 - 0.05 * j for j in range(12)] for i in range(4)]

    out_a1 = knots_dare_ties([v1, v2], weights=[0.6, 0.4], density=0.5, drop_rate=0.25, rng_seed=42)
    out_a2 = knots_dare_ties([v1, v2], weights=[0.6, 0.4], density=0.5, drop_rate=0.25, rng_seed=42)
    out_b = knots_dare_ties([v1, v2], weights=[0.6, 0.4], density=0.5, drop_rate=0.25, rng_seed=99)

    assert out_a1 == out_a2
    assert out_a1 != out_b

    # drop_rate=0.0 matches knots_ties exactly
    ties_ref = knots_ties([v1, v2], weights=[0.6, 0.4], density=0.5)
    dare_zero = knots_dare_ties(
        [v1, v2], weights=[0.6, 0.4], density=0.5, drop_rate=0.0, rng_seed=7
    )
    assert dare_zero == ties_ref

    with pytest.raises(ConfigError):
        knots_dare_ties([v1, v2], drop_rate=1.0)
    with pytest.raises(ConfigError):
        knots_ties([v1, v2], density=0.0)
    with pytest.raises(ConfigError):
        knots_linear([v1, v2], weights=[0.0, 0.0])


def test_knots_linear_weighted_sum() -> None:
    v1 = [[2.0, -4.0], [1.0, 3.0]]
    v2 = [[6.0, 4.0], [-3.0, 7.0]]
    # Weights [3.0, 1.0] normalize to [0.75, 0.25]
    merged = knots_linear([v1, v2], weights=[3.0, 1.0])
    assert merged[0][0] == pytest.approx(0.75 * 2.0 + 0.25 * 6.0, abs=1e-12)
    assert merged[0][1] == pytest.approx(0.75 * -4.0 + 0.25 * 4.0, abs=1e-12)
    assert merged[1][0] == pytest.approx(0.75 * 1.0 + 0.25 * -3.0, abs=1e-12)
    assert merged[1][1] == pytest.approx(0.75 * 3.0 + 0.25 * 7.0, abs=1e-12)


def test_genknowsub_factorized_residual_recovery(tmp_path: Path) -> None:
    d_out, d_in = 16, 16
    lname = "base_model.model.layers.0.mlp.gate_proj"

    # Pure task rank-2 update + General knowledge rank-2 update concatenated as rank-4 task
    a_pure = [[1.0 if j == i else 0.0 for j in range(d_in)] for i in range(2)]
    b_pure = [[2.0 if row == col else 0.0 for col in range(2)] for row in range(d_out)]

    a_gen = [[1.0 if j == (i + 4) else 0.0 for j in range(d_in)] for i in range(2)]
    b_gen = [[1.5 if row == (col + 4) else 0.0 for col in range(2)] for row in range(d_out)]

    a_combined = a_pure + a_gen
    b_combined = [rp + rg for rp, rg in zip(b_pure, b_gen, strict=True)]

    task_dir = _make_adapter(
        tmp_path / "task", rank=4, alpha=4.0, layers={lname: (a_combined, b_combined)}
    )
    gen_dir = _make_adapter(tmp_path / "gen", rank=2, alpha=2.0, layers={lname: (a_gen, b_gen)})

    loaded_task = load_peft_adapter(task_dir)
    loaded_gen = load_peft_adapter(gen_dir)

    purified = apply_genknowsub([loaded_task], loaded_gen, gks_lambda=1.0)
    a_pur, b_pur = purified[0].layer_factors[lname]
    assert len(a_pur) == 6  # factorized augmentation without dense materialization

    dense_purified = matmul_matrices(b_pur, a_pur)
    dense_pure = matmul_matrices(b_pure, a_pure)
    for row_exp, row_got in zip(dense_pure, dense_purified, strict=True):
        for v_exp, v_got in zip(row_exp, row_got, strict=True):
            assert v_got == pytest.approx(v_exp, abs=1e-5)


def test_genknowsub_mismatched_adapter_rejected(tmp_path: Path) -> None:
    lname = "base_model.model.layers.0.mlp.gate_proj"
    a_16 = [[1.0 if j == i else 0.0 for j in range(16)] for i in range(2)]
    b_16 = [[1.0 if row == col else 0.0 for col in range(2)] for row in range(16)]
    a_8 = [[1.0 if j == i else 0.0 for j in range(8)] for i in range(2)]
    b_8 = [[1.0 if row == col else 0.0 for col in range(2)] for row in range(8)]

    task = load_peft_adapter(
        _make_adapter(tmp_path / "task", rank=2, alpha=2.0, layers={lname: (a_16, b_16)})
    )
    bad_dim_gen = load_peft_adapter(
        _make_adapter(tmp_path / "bad_dim", rank=2, alpha=2.0, layers={lname: (a_8, b_8)})
    )
    with pytest.raises(AdapterParseError, match="dimension mismatch"):
        apply_genknowsub([task], bad_dim_gen)

    bad_base_gen = load_peft_adapter(
        _make_adapter(
            tmp_path / "bad_base",
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            rank=2,
            alpha=2.0,
            layers={lname: (a_16, b_16)},
        )
    )
    with pytest.raises(AdapterParseError, match="base model"):
        apply_genknowsub([task], bad_base_gen)


def test_energy_retention_and_rule_knots_001_threshold() -> None:
    d = 12
    # Construct rank-4 orthogonal signal with equal singular values [2, 2, 2, 2]
    # Truncating to rank 2 retains only 50% energy (< 0.90) -> triggers RULE-KNOTS-001
    b_a = [[2.0 if i == j else 0.0 for j in range(2)] for i in range(d)]
    a_a = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(2)]
    b_b = [[2.0 if i == (j + 2) else 0.0 for j in range(2)] for i in range(d)]
    a_b = [[1.0 if j == (i + 2) else 0.0 for j in range(d)] for i in range(2)]

    aligned = align_subspaces_knots(
        layer_name="layer0.q_proj",
        adapter_ids=["a", "b"],
        scaled_bs=[b_a, b_b],
        a_mats=[a_a, a_b],
    )
    v_merged_t, post_conf = merge_aligned_factors(aligned, method="knots_ties", ties_density=1.0)

    _, _, eta_full = reconstruct_merged_lora(
        aligned.u_shared, aligned.sigmas, v_merged_t, target_rank=4
    )
    assert eta_full == pytest.approx(1.0, abs=1e-6)

    _, _, eta_trunc = reconstruct_merged_lora(
        aligned.u_shared, aligned.sigmas, v_merged_t, target_rank=2
    )
    assert eta_trunc == pytest.approx(0.5, abs=1e-5)

    summary_ok, findings_ok = compute_merge_diagnostics(
        method="knots_ties",
        adapter_ids=["a", "b"],
        target_rank=4,
        layer_energies={"layer0.q_proj": eta_full},
        layer_rmses={"layer0.q_proj": 0.0},
        pre_conflicts=[aligned.pre_merge_sign_conflict],
        post_conflicts=[post_conf],
        output_sha256="0" * 64,
        min_energy_retention=0.90,
    )
    assert summary_ok.energy_retained == pytest.approx(1.0, abs=1e-6)
    assert findings_ok == []

    summary_low, findings_low = compute_merge_diagnostics(
        method="knots_ties",
        adapter_ids=["a", "b"],
        target_rank=2,
        layer_energies={"layer0.q_proj": eta_trunc},
        layer_rmses={"layer0.q_proj": 0.1},
        pre_conflicts=[aligned.pre_merge_sign_conflict],
        post_conflicts=[post_conf],
        output_sha256="0" * 64,
        min_energy_retention=0.90,
    )
    assert summary_low.energy_retained == pytest.approx(0.5, abs=1e-5)
    assert len(findings_low) == 1
    assert findings_low[0].rule_id == "RULE-KNOTS-001"
    assert findings_low[0].severity.value == "medium"

    # Zero-energy safety: sets energy_retained=1.0 cleanly and warns
    b_zero = [[0.0] * 2 for _ in range(d)]
    _, _, eta_zero = reconstruct_merged_lora(
        aligned.u_shared, [0.0, 0.0, 0.0, 0.0], [[0.0] * d for _ in range(4)], target_rank=2
    )
    assert eta_zero == 1.0
    summary_z, findings_z = compute_merge_diagnostics(
        method="knots_linear",
        adapter_ids=["a", "b"],
        target_rank=2,
        layer_energies={"layer0.q_proj": eta_zero},
        layer_rmses={"layer0.q_proj": factorized_frobenius_rmse(b_zero, a_a, b_zero, a_a)},
        pre_conflicts=[0.0],
        post_conflicts=[0.0],
        output_sha256="0" * 64,
        zero_energy_layers=["layer0.q_proj"],
    )
    assert summary_z.energy_retained == 1.0
    assert findings_z and findings_z[0].rule_id == "RULE-KNOTS-001"


def test_factorized_frobenius_rmse_matches_dense() -> None:
    d_out, d_in = 14, 10
    b_out = [[math.sin(0.2 * (i + 1) * (j + 1)) for j in range(3)] for i in range(d_out)]
    a_out = [[math.cos(0.3 * (i + 1) * (j + 1)) for j in range(d_in)] for i in range(3)]
    b_src = [[math.cos(0.17 * (i + 2) * (j + 1)) for j in range(5)] for i in range(d_out)]
    a_src = [[math.sin(0.23 * (i + 1) * (j + 3)) for j in range(d_in)] for i in range(5)]

    scaling_src = 0.75
    got_rmse = factorized_frobenius_rmse(b_out, a_out, b_src, a_src, scaling_src=scaling_src)

    dense_out = matmul_matrices(b_out, a_out)
    b_src_scaled = [[v * scaling_src for v in row] for row in b_src]
    dense_src = matmul_matrices(b_src_scaled, a_src)
    sq_sum = sum(
        (dense_out[r][c] - dense_src[r][c]) ** 2 for r in range(d_out) for c in range(d_in)
    )
    expected_rmse = math.sqrt(sq_sum / float(d_out * d_in))
    assert got_rmse == pytest.approx(expected_rmse, abs=1e-10)


def test_isometric_sigma_weighted_ties_blocks_nullspace_rows() -> None:
    pytest.importorskip("numpy")

    # Rows 0..1 are genuine signal with small V entries (0.05) and high singular values [10.0, 5.0].
    # Rows 2..3 are nullspace rows (sigma=0.0 and sigma=5e-11 <= 1e-12*100? Let's use sigma_1=100.0
    # so 5e-11 <= 1e-10 is strictly below relative cutoff max(1e-12, 1e-12 * sigma_1)) with huge
    # unit-norm V entries (0.95) and conflicting signs.
    sigmas = [100.0, 20.0, 5e-11, 0.0]
    v1 = [
        [0.05, 0.05, 0.05, 0.05],
        [0.04, 0.04, 0.04, 0.04],
        [0.95, 0.95, 0.95, 0.95],
        [0.90, -0.90, 0.90, -0.90],
    ]
    v2 = [
        [0.05, 0.05, 0.05, 0.05],
        [0.04, 0.04, 0.04, 0.04],
        [-0.95, -0.95, -0.95, -0.95],
        [-0.90, 0.90, -0.90, 0.90],
    ]

    for backend in ("numpy", "pure_python"):
        merged = knots_ties(
            [v1, v2], weights=[0.5, 0.5], sigmas=sigmas, density=0.5, backend=backend
        )
        # Signal row 0 (100 * 0.05 = 5.0) must survive; nullspace rows must be 0.0
        assert merged[0] == pytest.approx([0.05, 0.05, 0.05, 0.05], abs=1e-12)
        assert merged[2] == [0.0, 0.0, 0.0, 0.0]
        assert merged[3] == [0.0, 0.0, 0.0, 0.0]

    # Same check for knots_dare_ties with sigmas: nullspace rows never vote or consume budget
    dare_np = knots_dare_ties(
        [v1, v2],
        weights=[0.5, 0.5],
        sigmas=sigmas,
        density=0.5,
        drop_rate=0.2,
        rng_seed=7,
        backend="numpy",
    )
    dare_py = knots_dare_ties(
        [v1, v2],
        weights=[0.5, 0.5],
        sigmas=sigmas,
        density=0.5,
        drop_rate=0.2,
        rng_seed=7,
        backend="pure_python",
    )
    assert dare_np[2] == [0.0, 0.0, 0.0, 0.0]
    assert dare_np[3] == [0.0, 0.0, 0.0, 0.0]
    for r_np, r_py in zip(dare_np, dare_py, strict=True):
        assert r_np == pytest.approx(r_py, abs=1e-12)


def test_knots_dare_ties_cross_backend_rng_parity() -> None:
    pytest.importorskip("numpy")
    v1 = [[math.sin(0.3 * (i + 1) * (j + 1)) for j in range(16)] for i in range(6)]
    v2 = [[math.cos(0.25 * (i + 2) * (j + 1)) for j in range(16)] for i in range(6)]
    sigmas = [12.0, 8.0, 4.0, 1.5, 0.5, 0.0]

    for seed in (1, 42, 1337):
        out_np = knots_dare_ties(
            [v1, v2],
            weights=[0.7, 0.3],
            sigmas=sigmas,
            density=0.5,
            drop_rate=0.25,
            rng_seed=seed,
            backend="numpy",
        )
        out_py = knots_dare_ties(
            [v1, v2],
            weights=[0.7, 0.3],
            sigmas=sigmas,
            density=0.5,
            drop_rate=0.25,
            rng_seed=seed,
            backend="pure_python",
        )
        for row_np, row_py in zip(out_np, out_py, strict=True):
            assert row_np == pytest.approx(row_py, abs=1e-12)
