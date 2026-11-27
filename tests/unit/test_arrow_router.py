"""Unit tests for Arrow spectral prototype extraction and sign-invariant Top-k routing."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from loraspec.core.errors import AdapterParseError, ConfigError, RuntimeExecutionError
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.kernels import has_mps, has_torch
from loraspec.kernels.router import (
    compute_router_entropy,
    extract_arrow_prototypes,
    route_topk_tokens,
)


def _write_test_adapter(
    root: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    rank: int,
    alpha: float,
    use_rslora: bool = False,
    use_dora: bool = False,
    tensors: dict[str, tuple[Matrix, str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "use_dora": use_dora,
        "bias": "none",
        "target_modules": ["q_proj", "gate_proj", "up_proj", "down_proj"],
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


def _cosine_abs(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu <= 1e-12 or nv <= 1e-12:
        return 0.0
    return abs(dot) / (nu * nv)


def test_arrow_prototype_extraction_and_unit_norm(tmp_path: Path) -> None:
    d_in, d_out = 12, 10
    layer_a = "base_model.model.layers.0.self_attn.q_proj"
    layer_b = "base_model.model.layers.0.mlp.down_proj"

    # Adapter 0 (rank 2): dominant right singular direction aligned with e_0
    a0 = [
        [1.0 if j == 0 else 0.0 for j in range(d_in)],
        [1.0 if j == 1 else 0.0 for j in range(d_in)],
    ]
    b0 = [
        [3.0 if (i == 0 and c == 0) else (0.5 if (i == 1 and c == 1) else 0.0) for c in range(2)]
        for i in range(d_out)
    ]

    # Adapter 1 (rank 4, rsLoRA): dominant right singular vector along [0, 0, 0.6, 0.8, ...]
    a1 = [
        [0.6 if j == 2 else (0.8 if j == 3 else 0.0) for j in range(d_in)],
        [1.0 if j == 4 else 0.0 for j in range(d_in)],
        [1.0 if j == 5 else 0.0 for j in range(d_in)],
        [1.0 if j == 6 else 0.0 for j in range(d_in)],
    ]
    b1 = [
        [4.0 if (i == 2 and c == 0) else (0.2 if (i == c + 2) else 0.0) for c in range(4)]
        for i in range(d_out)
    ]

    # Adapter 2 (rank 2): zero-norm layer_a to verify safe zero prototype + warning
    a2 = [[0.0] * d_in for _ in range(2)]
    b2 = [[0.0] * 2 for _ in range(d_out)]

    p0 = _write_test_adapter(
        tmp_path / "sql",
        rank=2,
        alpha=4.0,
        use_rslora=False,
        tensors={
            f"{layer_a}.lora_A.weight": (a0, "F32"),
            f"{layer_a}.lora_B.weight": (b0, "F32"),
            f"{layer_b}.lora_A.weight": (a0, "F32"),
            f"{layer_b}.lora_B.weight": (b0, "F32"),
        },
    )
    p1 = _write_test_adapter(
        tmp_path / "rust",
        rank=4,
        alpha=8.0,
        use_rslora=True,
        tensors={
            f"{layer_a}.lora_A.weight": (a1, "F32"),
            f"{layer_a}.lora_B.weight": (b1, "F32"),
        },
    )
    p2 = _write_test_adapter(
        tmp_path / "empty_layer",
        rank=2,
        alpha=2.0,
        tensors={
            f"{layer_a}.lora_A.weight": (a2, "F32"),
            f"{layer_a}.lora_B.weight": (b2, "F32"),
        },
    )

    loaded = [load_peft_adapter(p0), load_peft_adapter(p1), load_peft_adapter(p2)]
    bank = extract_arrow_prototypes(loaded, layer_policy="union")

    assert bank.adapter_ids == ("sql", "rust", "empty_layer")
    assert bank.layer_names == (layer_b, layer_a)
    assert bank.ranks[layer_a] == [2, 4, 2]
    assert bank.ranks[layer_b] == [2, 0, 0]
    assert bank.scalings[layer_a][0] == pytest.approx(2.0)
    assert bank.scalings[layer_a][1] == pytest.approx(8.0 / math.sqrt(4.0))

    # Prototypes for non-zero adapters are unit L2 norm and match dominant v1 (>= 0.9999)
    protos_a = bank.prototypes[layer_a]
    assert math.sqrt(sum(v * v for v in protos_a[0])) == pytest.approx(1.0, abs=1e-6)
    assert math.sqrt(sum(v * v for v in protos_a[1])) == pytest.approx(1.0, abs=1e-6)
    expected_v1_0 = [1.0 if j == 0 else 0.0 for j in range(d_in)]
    expected_v1_1 = [0.6 if j == 2 else (0.8 if j == 3 else 0.0) for j in range(d_in)]
    assert _cosine_abs(protos_a[0], expected_v1_0) >= 0.9999
    assert _cosine_abs(protos_a[1], expected_v1_1) >= 0.9999

    # Zero-norm & missing-in-union emit safe zero vectors and warnings (never NaN)
    assert protos_a[2] == [0.0] * d_in
    assert bank.prototypes[layer_b][1] == [0.0] * d_in
    assert len(bank.warnings) >= 3
    assert all(math.isfinite(v) for row in protos_a for v in row)

    # Packed A/B zero-padded past r_m
    assert len(bank.a_packed[layer_a][0]) == 4
    assert bank.a_packed[layer_a][0][2] == [0.0] * d_in
    assert len(bank.b_packed[layer_a][0][0]) == 4
    assert bank.b_packed[layer_a][0][0][2:] == [0.0, 0.0]

    # Intersection policy drops layer_b
    bank_inter = extract_arrow_prototypes(loaded, layer_policy="intersection")
    assert bank_inter.layer_names == (layer_a,)

    # Validation failures
    with pytest.raises(ConfigError):
        extract_arrow_prototypes([])
    with pytest.raises(ConfigError):
        extract_arrow_prototypes(loaded, layer_policy=cast(Any, "outer"))

    if not has_torch():
        with pytest.raises(RuntimeExecutionError):
            bank.layer_tensors(layer_a)
    else:
        import importlib

        torch_mod: Any = importlib.import_module("torch")
        pt, at, bt, rt, st = bank.layer_tensors(layer_a, dtype=torch_mod.float16)
        assert tuple(pt.shape) == (3, d_in)
        assert tuple(at.shape) == (3, 4, d_in)
        assert tuple(bt.shape) == (3, d_out, 4)
        assert tuple(rt.shape) == (3,)
        assert tuple(st.shape) == (3,)
        assert pt.dtype == torch_mod.float32
        assert at.dtype == torch_mod.float16
        with pytest.raises(ConfigError):
            bank.layer_tensors("unknown.layer")


def test_arrow_prototype_with_genknowsub(tmp_path: Path) -> None:
    d_in, d_out = 10, 10
    layer = "base_model.model.layers.0.mlp.gate_proj"

    # General adapter dominates along basis e_0 (sigma = 20.0)
    a_gen = [[1.0 if j == 0 else 0.0 for j in range(d_in)]]
    b_gen = [[20.0 if i == 0 else 0.0] for i in range(d_out)]

    # Task A = pure_A (e_1, sigma=2.0) + gen (e_0, sigma=20.0)
    a_task_a = [
        [1.0 if j == 0 else 0.0 for j in range(d_in)],
        [1.0 if j == 1 else 0.0 for j in range(d_in)],
    ]
    b_task_a = [
        [20.0 if (i == 0 and c == 0) else (2.0 if (i == 1 and c == 1) else 0.0) for c in range(2)]
        for i in range(d_out)
    ]

    # Task B = pure_B (e_2, sigma=2.0) + gen (e_0, sigma=20.0)
    a_task_b = [
        [1.0 if j == 0 else 0.0 for j in range(d_in)],
        [1.0 if j == 2 else 0.0 for j in range(d_in)],
    ]
    b_task_b = [
        [20.0 if (i == 0 and c == 0) else (2.0 if (i == 2 and c == 1) else 0.0) for c in range(2)]
        for i in range(d_out)
    ]

    p_gen = _write_test_adapter(
        tmp_path / "gen",
        rank=1,
        alpha=1.0,
        tensors={
            f"{layer}.lora_A.weight": (a_gen, "F32"),
            f"{layer}.lora_B.weight": (b_gen, "F32"),
        },
    )
    p_a = _write_test_adapter(
        tmp_path / "task_a",
        rank=2,
        alpha=2.0,
        tensors={
            f"{layer}.lora_A.weight": (a_task_a, "F32"),
            f"{layer}.lora_B.weight": (b_task_a, "F32"),
        },
    )
    p_b = _write_test_adapter(
        tmp_path / "task_b",
        rank=2,
        alpha=2.0,
        tensors={
            f"{layer}.lora_A.weight": (a_task_b, "F32"),
            f"{layer}.lora_B.weight": (b_task_b, "F32"),
        },
    )

    loaded_tasks = [load_peft_adapter(p_a), load_peft_adapter(p_b)]
    loaded_gks = load_peft_adapter(p_gen)

    # Without GenKnowSub, both prototypes collapse onto the general direction e_0
    bank_raw = extract_arrow_prototypes(loaded_tasks)
    raw_protos = bank_raw.prototypes[layer]
    assert _cosine_abs(raw_protos[0], raw_protos[1]) > 0.99

    # With GenKnowSub, prototypes recover orthogonal task-specific directions e_1 and e_2
    bank_gks = extract_arrow_prototypes(loaded_tasks, gks_adapter=loaded_gks, gks_lambda=1.0)
    gks_protos = bank_gks.prototypes[layer]
    assert _cosine_abs(gks_protos[0], [1.0 if j == 1 else 0.0 for j in range(d_in)]) >= 0.9999
    assert _cosine_abs(gks_protos[1], [1.0 if j == 2 else 0.0 for j in range(d_in)]) >= 0.9999
    assert _cosine_abs(gks_protos[0], gks_protos[1]) <= 1e-6


def test_extract_arrow_prototypes_rejects_dora_base_dim_and_nan(tmp_path: Path) -> None:
    d = 6
    layer = "base_model.model.layers.0.self_attn.q_proj"
    base_t: dict[str, tuple[Matrix, str]] = {
        f"{layer}.lora_A.weight": ([[1.0 if j == 0 else 0.0 for j in range(d)]], "F32"),
        f"{layer}.lora_B.weight": ([[1.0] if i == 0 else [0.0] for i in range(d)], "F32"),
    }
    p_ok = _write_test_adapter(tmp_path / "ok", rank=1, alpha=1.0, tensors=base_t)
    p_dora = _write_test_adapter(
        tmp_path / "dora", rank=1, alpha=1.0, use_dora=True, tensors=base_t
    )
    p_other_base = _write_test_adapter(
        tmp_path / "other_base",
        base_model="meta-llama/Llama-3.1-8B",
        rank=1,
        alpha=1.0,
        tensors=base_t,
    )
    p_bad_dim = _write_test_adapter(
        tmp_path / "bad_dim",
        rank=1,
        alpha=1.0,
        tensors={
            f"{layer}.lora_A.weight": ([[1.0] * 8], "F32"),
            f"{layer}.lora_B.weight": ([[1.0] for _ in range(8)], "F32"),
        },
    )
    p_nan = _write_test_adapter(
        tmp_path / "nan",
        rank=1,
        alpha=1.0,
        tensors={
            f"{layer}.lora_A.weight": ([[math.nan] + [0.0] * (d - 1)], "F32"),
            f"{layer}.lora_B.weight": ([[1.0] for _ in range(d)], "F32"),
        },
    )

    ok_w = load_peft_adapter(p_ok)
    with pytest.raises(AdapterParseError):
        extract_arrow_prototypes([ok_w, load_peft_adapter(p_dora)])
    with pytest.raises(AdapterParseError):
        extract_arrow_prototypes([ok_w, load_peft_adapter(p_other_base)])
    with pytest.raises(AdapterParseError):
        extract_arrow_prototypes([ok_w, load_peft_adapter(p_bad_dim)])
    with pytest.raises(AdapterParseError):
        extract_arrow_prototypes([ok_w, load_peft_adapter(p_nan)])


def test_topk_routing_sign_invariance_and_entropy(tmp_path: Path) -> None:
    d_in, d_out = 8, 8
    layer = "base_model.model.layers.0.self_attn.q_proj"
    a0 = [[1.0 if j == 0 else 0.0 for j in range(d_in)]]
    b0 = [[2.0 if i == 0 else 0.0] for i in range(d_out)]
    a1 = [[1.0 if j == 1 else 0.0 for j in range(d_in)]]
    b1 = [[2.0 if i == 1 else 0.0] for i in range(d_out)]
    a2 = [[1.0 if j == 2 else 0.0 for j in range(d_in)]]
    b2 = [[2.0 if i == 2 else 0.0] for i in range(d_out)]

    # Sign-flipped adapters (-B0, A0) and (B1, -A1)
    b0_neg = [[-v for v in row] for row in b0]
    a1_neg = [[-v for v in row] for row in a1]

    p_pos = [
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "p0",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a0, "F32"),
                    f"{layer}.lora_B.weight": (b0, "F32"),
                },
            )
        ),
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "p1",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a1, "F32"),
                    f"{layer}.lora_B.weight": (b1, "F32"),
                },
            )
        ),
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "p2",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a2, "F32"),
                    f"{layer}.lora_B.weight": (b2, "F32"),
                },
            )
        ),
    ]
    p_flip = [
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "f0",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a0, "F32"),
                    f"{layer}.lora_B.weight": (b0_neg, "F32"),
                },
            )
        ),
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "f1",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a1_neg, "F32"),
                    f"{layer}.lora_B.weight": (b1, "F32"),
                },
            )
        ),
        load_peft_adapter(
            _write_test_adapter(
                tmp_path / "f2",
                rank=1,
                alpha=1.0,
                tensors={
                    f"{layer}.lora_A.weight": (a2, "F32"),
                    f"{layer}.lora_B.weight": (b2, "F32"),
                },
            )
        ),
    ]

    bank_pos = extract_arrow_prototypes(p_pos)
    bank_flip = extract_arrow_prototypes(p_flip)

    x_tokens: Matrix = [
        [2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # primarily expert 0, then 1
        [0.0, 0.3, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # primarily expert 2, then 1
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zero activation -> tie break [0, 1]
    ]
    e_pos, w_pos, h_pos = route_topk_tokens(
        x_tokens, bank_pos.prototypes[layer], top_k=2, temperature=0.5
    )
    e_flip, w_flip, h_flip = route_topk_tokens(
        x_tokens, bank_flip.prototypes[layer], top_k=2, temperature=0.5
    )

    assert e_pos == e_flip == [[0, 1], [2, 1], [0, 1]]
    for r_p, r_f in zip(w_pos, w_flip, strict=True):
        assert sum(r_p) == pytest.approx(1.0, abs=1e-6)
        assert r_p == pytest.approx(r_f, abs=1e-6)
    assert h_pos == pytest.approx(h_flip, abs=1e-6)
    assert all(h >= 0.0 and math.isfinite(h) for h in h_pos)

    # top_k=1 yields clamped zero entropy
    _, w1, h1 = route_topk_tokens(x_tokens, bank_pos.prototypes[layer], top_k=1, temperature=1.0)
    assert w1 == [[1.0], [1.0], [1.0]]
    assert h1 == [0.0, 0.0, 0.0]

    # Empty T == 0 returns empty outputs cleanly
    e0, w0, h0 = route_topk_tokens([], bank_pos.prototypes[layer], top_k=2, temperature=1.0)
    assert e0 == [] and w0 == [] and h0 == []
    assert compute_router_entropy([]) == []

    # PyTorch CPU (and MPS if available) matches pure-Python deterministically
    if has_torch():
        import importlib

        torch_mod: Any = importlib.import_module("torch")
        devices = ["cpu"]
        if has_mps():
            devices.append("mps")
        for dev in devices:
            x_t = torch_mod.tensor(x_tokens, dtype=torch_mod.float32, device=dev)
            p_t = torch_mod.tensor(bank_pos.prototypes[layer], dtype=torch_mod.float32, device=dev)
            et, wt, ht = route_topk_tokens(x_t, p_t, top_k=2, temperature=0.5)
            assert et.dtype == torch_mod.int32
            assert wt.dtype == torch_mod.float32
            assert ht.dtype == torch_mod.float32
            assert et.detach().cpu().tolist() == e_pos
            for row_t, row_py in zip(wt.detach().cpu().tolist(), w_pos, strict=True):
                assert row_t == pytest.approx(row_py, abs=1e-6)
            assert ht.detach().cpu().tolist() == pytest.approx(h_pos, abs=1e-6)


def test_invalid_topk_or_temperature_raises_config_error() -> None:
    protos = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    x = [[1.0, 2.0, 0.0, 0.0]]

    for bad_k in (0, -1, 3, cast(Any, True)):
        with pytest.raises(ConfigError):
            route_topk_tokens(x, protos, top_k=bad_k, temperature=1.0)

    for bad_temp in (0.0, -0.5, float("nan"), float("inf"), cast(Any, False)):
        with pytest.raises(ConfigError):
            route_topk_tokens(x, protos, top_k=1, temperature=bad_temp)

    with pytest.raises(ConfigError):
        route_topk_tokens([[1.0, 2.0]], protos, top_k=1, temperature=1.0)

    with pytest.raises(ConfigError):
        compute_router_entropy([[0.5, 0.5]], eps=0.0)
