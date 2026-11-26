"""Unit tests for bounds-safe .safetensors parsing and factorized QR+SVD/Grassmann math."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from loraspec.core.errors import AdapterParseError
from loraspec.core.schemas import LoRALayerSpec
from loraspec.core.spectral_math import (
    compute_layer_spectral_analysis,
    compute_pairwise_collision,
    small_svd,
    thin_qr,
)
from loraspec.inspect.safetensors_reader import (
    decode_tensor_2d,
    load_peft_adapter,
    parse_safetensors_header,
    write_safetensors,
)


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_safetensors_binary_roundtrip_dtypes(tmp_path: Path, dtype: str) -> None:
    mat = [[1.5, -2.25, 0.5], [0.0, 3.0, -0.75]]
    out_file = write_safetensors(tmp_path / f"w_{dtype}.safetensors", {"t": (mat, dtype)})
    parsed = parse_safetensors_header(out_file)
    decoded = decode_tensor_2d(out_file, parsed.tensors["t"], parsed.payload_offset)

    tol = 1e-6 if dtype == "F32" else 2e-2
    for row_exp, row_got in zip(mat, decoded, strict=True):
        for x, y in zip(row_exp, row_got, strict=True):
            assert y == pytest.approx(x, abs=tol)


def test_safetensors_bounds_and_pickle_security_rejections(tmp_path: Path) -> None:
    # 1. File too small (< 8 bytes)
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"\x01\x02")
    with pytest.raises(AdapterParseError):
        parse_safetensors_header(tiny)

    # 2. Pathological oversized header length (> file size / MAX_HEADER_BYTES)
    bomb = tmp_path / "bomb.safetensors"
    bomb.write_bytes(struct.pack("<Q", 2**50) + b"{}")
    with pytest.raises(AdapterParseError):
        parse_safetensors_header(bomb)

    # 3. Zero-length header
    zero_hdr = tmp_path / "zero.safetensors"
    zero_hdr.write_bytes(struct.pack("<Q", 0) + b"{}")
    with pytest.raises(AdapterParseError):
        parse_safetensors_header(zero_hdr)

    # 4. Out-of-bounds data_offsets
    hdr_bytes = json.dumps(
        {"a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 9999]}}
    ).encode("utf-8")
    oob = tmp_path / "oob.safetensors"
    oob.write_bytes(struct.pack("<Q", len(hdr_bytes)) + hdr_bytes + b"\x00" * 16)
    with pytest.raises(AdapterParseError):
        parse_safetensors_header(oob)

    # 5. Refuse unsafe .bin pickle checkpoints
    bin_dir = tmp_path / "unsafe_bin_adapter"
    bin_dir.mkdir()
    (bin_dir / "adapter_model.bin").write_bytes(b"fake pickle")
    with pytest.raises(AdapterParseError, match="pickle"):
        load_peft_adapter(bin_dir)


def test_thin_qr_and_jacobi_svd_accuracy() -> None:
    mat = [
        [3.0, 1.0, 0.0],
        [0.0, 2.0, -1.0],
        [1.0, 0.0, 2.0],
        [-2.0, 1.0, 1.5],
    ]
    q, r = thin_qr(mat)
    # Check Q^T Q == I_3
    for i in range(3):
        for j in range(3):
            dot = sum(q[row][i] * q[row][j] for row in range(4))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-10)
    # Check Q @ R == mat
    for row in range(4):
        for col in range(3):
            recon = sum(q[row][k] * r[k][col] for k in range(3))
            assert recon == pytest.approx(mat[row][col], abs=1e-10)

    # Jacobi SVD reconstruction on same rectangular matrix
    u, sigmas, v = small_svd(mat)
    assert sigmas == sorted(sigmas, reverse=True)
    for row in range(4):
        for col in range(3):
            recon = sum(u[row][k] * sigmas[k] * v[col][k] for k in range(3))
            assert recon == pytest.approx(mat[row][col], abs=1e-9)


def test_spectral_rank_rslora_and_gauge_invariance() -> None:
    d = 12
    r = 4
    layer = LoRALayerSpec(
        layer_name="layer0.q_proj",
        module_type="q_proj",
        in_features=d,
        out_features=d,
        rank=r,
        dtype="F32",
        a_key="a",
        b_key="b",
        a_shape=(r, d),
        b_shape=(d, r),
        param_count=2 * d * r,
    )

    # Orthonormal rank-4 with uniform singular values -> effective_stable_rank == 4.0
    a_mat = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
    b_mat = [[2.0 if row == col else 0.0 for col in range(r)] for row in range(d)]

    # Standard scaling vs rsLoRA scaling: B_rslora has alpha/sqrt(r)
    m_std, f_std = compute_layer_spectral_analysis(
        adapter_id="std", layer_spec=layer, a_mat=a_mat, b_mat=b_mat, scaling=4.0 / r
    )
    assert m_std.effective_stable_rank == pytest.approx(4.0, abs=1e-5)
    assert m_std.rank_utilization == pytest.approx(1.0, abs=1e-5)
    assert m_std.spectral_norm == pytest.approx(2.0, abs=1e-5)

    m_rs, _ = compute_layer_spectral_analysis(
        adapter_id="rs", layer_spec=layer, a_mat=a_mat, b_mat=b_mat, scaling=4.0 / math.sqrt(r)
    )
    assert m_rs.spectral_norm == pytest.approx(4.0, abs=1e-5)

    # Gauge transform (B * 2.5, A / 2.5) must preserve singular values & Grassmann subspace
    b_scaled_gauge = [[x * 2.5 for x in row] for row in b_mat]
    a_scaled_gauge = [[x / 2.5 for x in row] for row in a_mat]
    m_gauge, f_gauge = compute_layer_spectral_analysis(
        adapter_id="gauge",
        layer_spec=layer,
        a_mat=a_scaled_gauge,
        b_mat=b_scaled_gauge,
        scaling=1.0,
    )
    assert m_gauge.spectral_norm == pytest.approx(m_std.spectral_norm, abs=1e-5)

    # Identical subspace collision -> 0 deg angles, chordal 1.0, conflict 0.0
    col_same = compute_pairwise_collision(f_std, f_gauge)
    assert col_same.grassmann_chordal_similarity == pytest.approx(1.0, abs=1e-5)
    assert all(ang == pytest.approx(0.0, abs=1e-3) for ang in col_same.principal_angles_deg)
    assert col_same.sign_conflict_rate == pytest.approx(0.0, abs=1e-5)

    # Orthogonal subspace -> 90 deg angles, chordal 0.0
    a_orth = [[1.0 if j == (i + r) else 0.0 for j in range(d)] for i in range(r)]
    b_orth = [[2.0 if row == (col + r) else 0.0 for col in range(r)] for row in range(d)]
    _, f_orth = compute_layer_spectral_analysis(
        adapter_id="orth", layer_spec=layer, a_mat=a_orth, b_mat=b_orth, scaling=1.0
    )
    col_orth = compute_pairwise_collision(f_std, f_orth)
    assert col_orth.grassmann_chordal_similarity == pytest.approx(0.0, abs=1e-5)
    assert all(ang == pytest.approx(90.0, abs=1e-2) for ang in col_orth.principal_angles_deg)

    # Zero matrix SVD & zero adapter spectral handling
    zero_a = [[0.0] * d for _ in range(r)]
    zero_b = [[0.0] * r for _ in range(d)]
    m_zero, f_zero = compute_layer_spectral_analysis(
        adapter_id="zero", layer_spec=layer, a_mat=zero_a, b_mat=zero_b, scaling=1.0
    )
    assert m_zero.effective_stable_rank == 0.0
    assert m_zero.frobenius_norm == 0.0
    col_zero = compute_pairwise_collision(f_std, f_zero)
    assert col_zero.grassmann_chordal_similarity == 0.0


def test_safetensors_structural_and_pairing_edge_cases(tmp_path: Path) -> None:
    # 1. Empty safetensors (zero tensors) -> AdapterParseError
    empty_st = write_safetensors(tmp_path / "empty.safetensors", {})
    with pytest.raises(AdapterParseError, match="No paired"):
        load_peft_adapter(empty_st)

    # 2. Unpaired lora_A without lora_B -> AdapterParseError
    unpaired = write_safetensors(
        tmp_path / "unpaired.safetensors",
        {"layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32")},
    )
    with pytest.raises(AdapterParseError, match="Unpaired LoRA tensor"):
        load_peft_adapter(unpaired)

    # 3. Rank mismatch lora_A (r=2) vs lora_B (r=3) -> AdapterParseError
    mismatch = write_safetensors(
        tmp_path / "rank_mismatch.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "F32"),
        },
    )
    with pytest.raises(AdapterParseError, match="Rank mismatch"):
        load_peft_adapter(mismatch)

    # 4. 1D tensor rejected safely on decode
    hdr = json.dumps({"vec": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode(
        "utf-8"
    )
    vec_file = tmp_path / "vec1d.safetensors"
    vec_file.write_bytes(struct.pack("<Q", len(hdr)) + hdr + struct.pack("<2f", 1.0, 2.0))
    parsed = parse_safetensors_header(vec_file)
    with pytest.raises(AdapterParseError, match="2D"):
        decode_tensor_2d(vec_file, parsed.tensors["vec"], parsed.payload_offset)

    # 5. Overlapping data_offsets and malformed __metadata__ rejected
    ov_hdr = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            "b": {"dtype": "F32", "shape": [2, 2], "data_offsets": [4, 20]},
        }
    ).encode("utf-8")
    overlap_file = tmp_path / "overlap.safetensors"
    overlap_file.write_bytes(struct.pack("<Q", len(ov_hdr)) + ov_hdr + b"\x00" * 20)
    with pytest.raises(AdapterParseError, match="Overlapping"):
        parse_safetensors_header(overlap_file)

    bad_meta_hdr = json.dumps({"__metadata__": "invalid"}).encode("utf-8")
    bad_meta_file = tmp_path / "bad_meta.safetensors"
    bad_meta_file.write_bytes(struct.pack("<Q", len(bad_meta_hdr)) + bad_meta_hdr)
    with pytest.raises(AdapterParseError, match="__metadata__"):
        parse_safetensors_header(bad_meta_file)

    # 6. Zero-feature-dim tensor rejected safely on load
    zero_dim = write_safetensors(
        tmp_path / "zero_dim.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[], []], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
        },
    )
    with pytest.raises(AdapterParseError, match="Invalid empty LoRA feature dimension"):
        load_peft_adapter(zero_dim)

    # 7. Unprefixed lora_A.weight / lora_B.weight pairs cleanly
    unprefixed = write_safetensors(
        tmp_path / "unprefixed.safetensors",
        {
            "lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "lora_B.weight": ([[2.0, 0.0], [0.0, 3.0]], "F32"),
        },
    )
    loaded_unprefixed = load_peft_adapter(unprefixed)
    assert len(loaded_unprefixed.spec.layers) == 1
    assert loaded_unprefixed.spec.layers[0].rank == 2

    # 8. Multi-adapter checkpoint in single .safetensors selects cleanly without cross-pairing
    multi_st = write_safetensors(
        tmp_path / "multi.safetensors",
        {
            "layer0.q_proj.lora_A.alpha.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.alpha.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_A.beta.weight": ([[0.0, 1.0], [1.0, 0.0]], "F32"),
            "layer0.q_proj.lora_B.beta.weight": ([[0.0, 2.0], [2.0, 0.0]], "F32"),
        },
    )
    with pytest.raises(AdapterParseError, match="Multiple adapter namespaces"):
        load_peft_adapter(multi_st)
    loaded_beta = load_peft_adapter(multi_st, adapter_name="beta")
    assert loaded_beta.spec.layers[0].a_key.endswith(".beta.weight")
    assert loaded_beta.spec.metadata["available_adapter_namespaces"] == ["alpha", "beta"]

    # 9. Boolean r / lora_alpha in adapter_config.json rejected cleanly
    bool_dir = tmp_path / "bool_cfg"
    bool_dir.mkdir()
    (bool_dir / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": True, "lora_alpha": 4.0}),
        encoding="utf-8",
    )
    write_safetensors(
        bool_dir / "adapter_model.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0], [0.0]], "F32"),
        },
    )
    with pytest.raises(AdapterParseError, match="Invalid 'r'"):
        load_peft_adapter(bool_dir)

    # 10. Rank-deficient / d < r layer keeps active bases & sigmas strictly aligned
    small_d_layer = LoRALayerSpec(
        layer_name="layer0.q_proj",
        module_type="q_proj",
        in_features=4,
        out_features=2,
        rank=4,
        dtype="F32",
        a_key="a",
        b_key="b",
        a_shape=(4, 4),
        b_shape=(2, 4),
        param_count=24,
    )
    a_4x4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    b_2x4 = [[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    m_small, f_small = compute_layer_spectral_analysis(
        adapter_id="d_lt_r",
        layer_spec=small_d_layer,
        a_mat=a_4x4,
        b_mat=b_2x4,
        scaling=1.0,
    )
    assert m_small.effective_stable_rank == pytest.approx(1.25, abs=1e-5)
    assert len(f_small.active_sigmas) == 2
    assert len(f_small.u_active[0]) == 2
    assert len(f_small.v_active[0]) == 2
