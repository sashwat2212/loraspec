"""Unit tests for segment sorting, PyTorch/pure-Python reference parity & auto-dispatch."""

from __future__ import annotations

import importlib
import math
from typing import Any, cast

import pytest

from loraspec.core.errors import EXIT_RUNTIME_ERROR, ConfigError, RuntimeExecutionError
from loraspec.core.spectral_math import Matrix
from loraspec.kernels import has_mps, has_torch, has_triton_cuda
from loraspec.kernels.arrow_sgmv import build_segment_metadata, triton_arrow_sgmv_forward
from loraspec.kernels.dispatch import ArrowMoLoRADispatcher
from loraspec.kernels.reference import (
    arrow_molora_reference,
    peft_eager_arrow_forward,
    pure_python_arrow_reference,
    static_lora_forward,
)


def _manual_dense_arrow(
    x: Matrix,
    base_out: Matrix,
    expert_ids: list[list[int]],
    routing_weights: list[list[float]],
    a_weights: list[Matrix],
    b_weights: list[Matrix],
    ranks: list[int],
    scalings: list[float],
) -> Matrix:
    t_cnt = len(x)
    d_out = len(base_out[0])
    d_in = len(x[0])
    out = [list(row) for row in base_out]
    for t in range(t_cnt):
        for exp_id, w in zip(expert_ids[t], routing_weights[t], strict=True):
            rk = ranks[exp_id]
            sc = scalings[exp_id]
            for o in range(d_out):
                acc = 0.0
                for r in range(rk):
                    h_tr = sum(x[t][j] * a_weights[exp_id][r][j] for j in range(d_in))
                    acc += b_weights[exp_id][o][r] * h_tr
                out[t][o] += w * sc * acc
    return out


def test_segment_sorting_roundtrip_and_empty_buckets() -> None:
    # Expert 1 and Expert 3 intentionally receive 0 tokens
    expert_ids = [[2, 0], [0, 2], [2, 0], [0, 2]]
    weights = [[0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1]]

    layout = build_segment_metadata(expert_ids, weights, num_experts=4)
    assert layout.num_tokens == 4
    assert layout.top_k == 2
    assert layout.num_experts == 4
    assert layout.expert_counts == [4, 0, 4, 0]
    assert layout.expert_offsets == [0, 4, 4, 8, 8]
    assert layout.max_segment_len == 4

    # Reconstruct original [T, k] assignment tensors from sorted buffers
    reconstructed_ids = [[-1, -1] for _ in range(4)]
    reconstructed_w = [[0.0, 0.0] for _ in range(4)]
    for exp_id in range(4):
        start = layout.expert_offsets[exp_id]
        end = layout.expert_offsets[exp_id + 1]
        for pos in range(start, end):
            t_id = layout.sorted_token_ids[pos]
            s_id = layout.sorted_slots[pos]
            reconstructed_ids[t_id][s_id] = exp_id
            reconstructed_w[t_id][s_id] = layout.sorted_weights[pos]

    assert reconstructed_ids == expert_ids
    assert reconstructed_w == weights

    # Invalid expert id or num_experts raises ConfigError
    with pytest.raises(ConfigError):
        build_segment_metadata([[0, 5]], [[0.5, 0.5]], num_experts=4)
    with pytest.raises(ConfigError):
        build_segment_metadata(expert_ids, weights, num_experts=0)

    if has_torch():
        torch_mod: Any = importlib.import_module("torch")
        devices = ["cpu"]
        if has_mps():
            devices.append("mps")
        for dev in devices:
            e_t = torch_mod.tensor(expert_ids, dtype=torch_mod.int32, device=dev)
            w_t = torch_mod.tensor(weights, dtype=torch_mod.float32, device=dev)
            t_layout = build_segment_metadata(e_t, w_t, num_experts=4)
            assert t_layout.expert_counts.detach().cpu().tolist() == [4, 0, 4, 0]
            assert t_layout.expert_offsets.detach().cpu().tolist() == [0, 4, 4, 8, 8]
            assert t_layout.sorted_token_ids.detach().cpu().tolist() == layout.sorted_token_ids
            assert t_layout.sorted_slots.detach().cpu().tolist() == layout.sorted_slots
            assert t_layout.sorted_weights.detach().cpu().tolist() == pytest.approx(
                layout.sorted_weights, abs=1e-6
            )


def test_zero_token_early_exit() -> None:
    protos = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    a_w = [
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    ]
    b_w = [
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ]
    ranks = [1, 2]
    scalings = [1.0, 1.5]

    seg_empty = build_segment_metadata([], [], num_experts=2)
    assert seg_empty.num_tokens == 0
    assert seg_empty.max_segment_len == 0
    assert seg_empty.expert_offsets == [0, 0, 0]

    y_empty, tel_empty = pure_python_arrow_reference(
        [], [], protos, a_w, b_w, ranks, scalings, top_k=2
    )
    assert y_empty == []
    assert tel_empty.num_tokens == 0
    assert tel_empty.backend == "pure_python"

    dispatcher = ArrowMoLoRADispatcher()
    y_disp, tel_disp = dispatcher.forward([], [], protos, a_w, b_w, ranks, scalings, top_k=2)
    assert y_disp == []
    assert tel_disp.num_tokens == 0

    if has_torch():
        torch_mod: Any = importlib.import_module("torch")
        x0 = torch_mod.empty((0, 4), dtype=torch_mod.float32)
        b0 = torch_mod.empty((0, 3), dtype=torch_mod.float32)
        y_t, tel_t = dispatcher.forward(x0, b0, protos, a_w, b_w, ranks, scalings, top_k=2)
        assert tuple(y_t.shape) == (0, 3)
        assert tel_t.num_tokens == 0
        assert tel_t.backend == "torch_cpu"


def test_auto_dispatcher_cpu_mps_matches_pure_python() -> None:
    d_in, d_out, r_max = 6, 5, 4
    protos = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    ]
    # Heterogeneous ranks: [1, 4, 2, 0] (expert 3 has rank 0 / union empty slot)
    ranks = [1, 4, 2, 0]
    scalings = [2.0, 1.5, 0.75, 0.0]

    a_weights: list[Matrix] = []
    b_weights: list[Matrix] = []
    for e_idx, rk in enumerate(ranks):
        a_mat = [
            [
                math.cos(0.3 * (r + 1) * (j + 1) + e_idx) * 0.25 if r < rk else 0.0
                for j in range(d_in)
            ]
            for r in range(r_max)
        ]
        b_mat = [
            [
                math.sin(0.2 * (o + 1) * (r + 1) + e_idx) * 0.35 if r < rk else 0.0
                for r in range(r_max)
            ]
            for o in range(d_out)
        ]
        a_weights.append(a_mat)
        b_weights.append(b_mat)

    x_2d: Matrix = [
        [1.5, 0.4, -0.2, 0.1, 0.3, -0.1],
        [-0.2, 2.1, 0.5, -0.3, 0.2, 0.4],
        [0.1, -0.3, 1.8, 0.2, -0.4, 0.5],
        [0.5, 0.5, 0.0, 0.0, 0.1, 0.2],
    ]
    base_2d: Matrix = [[0.1 * (t + 1) + 0.05 * o for o in range(d_out)] for t in range(4)]

    dispatcher = ArrowMoLoRADispatcher()
    y_py, tel_py = dispatcher.forward(
        x_2d,
        base_2d,
        protos,
        a_weights,
        b_weights,
        ranks,
        scalings,
        top_k=2,
        temperature=0.8,
        backend="auto",
        layer_name="layers.0.q_proj",
    )
    assert tel_py.backend == "pure_python"
    assert tel_py.layer_name == "layers.0.q_proj"
    assert tel_py.num_tokens == 4

    expected_dense = _manual_dense_arrow(
        x_2d,
        base_2d,
        tel_py.expert_indices,
        tel_py.routing_weights,
        a_weights,
        b_weights,
        ranks,
        scalings,
    )
    for row_got, row_exp in zip(y_py, expected_dense, strict=True):
        assert row_got == pytest.approx(row_exp, abs=1e-10)

    # PEFT eager baseline matches Arrow-MoLoRA math identically
    y_peft_py, tel_peft_py = peft_eager_arrow_forward(
        x_2d, base_2d, protos, a_weights, b_weights, ranks, scalings, top_k=2, temperature=0.8
    )
    assert tel_peft_py.expert_indices == tel_py.expert_indices
    for row_got, row_exp in zip(y_peft_py, y_py, strict=True):
        assert row_got == pytest.approx(row_exp, abs=1e-10)

    # Static baseline matches expert 0 with full weight
    y_static = static_lora_forward(
        x_2d, base_2d, a_weights[0], b_weights[0], scalings[0], rank=ranks[0]
    )
    expected_static = _manual_dense_arrow(
        x_2d, base_2d, [[0]] * 4, [[1.0]] * 4, a_weights, b_weights, ranks, scalings
    )
    for row_got, row_exp in zip(y_static, expected_static, strict=True):
        assert row_got == pytest.approx(row_exp, abs=1e-10)

    # 3D [B, S, d_in] support in pure Python
    x_3d = [x_2d[:2], x_2d[2:]]
    base_3d = [base_2d[:2], base_2d[2:]]
    y_3d_py, tel_3d_py = dispatcher.forward(
        x_3d, base_3d, protos, a_weights, b_weights, ranks, scalings, top_k=2, temperature=0.8
    )
    assert len(y_3d_py) == 2 and len(y_3d_py[0]) == 2
    for row_got, row_exp in zip(y_3d_py[0] + y_3d_py[1], y_py, strict=True):
        assert row_got == pytest.approx(row_exp)
    assert tel_3d_py.num_tokens == 4

    # Invalid backend rejected
    with pytest.raises(ConfigError):
        dispatcher.forward(
            x_2d,
            base_2d,
            protos,
            a_weights,
            b_weights,
            ranks,
            scalings,
            backend=cast(Any, "invalid_backend"),
        )

    if has_torch():
        torch_mod: Any = importlib.import_module("torch")
        devices = [("cpu", "torch_cpu")]
        if has_mps():
            devices.append(("mps", "torch_mps"))

        for dev, expected_backend in devices:
            x_t = torch_mod.tensor(x_2d, dtype=torch_mod.float32, device=dev)
            base_t = torch_mod.tensor(base_2d, dtype=torch_mod.float32, device=dev)
            p_t = torch_mod.tensor(protos, dtype=torch_mod.float32, device=dev)
            a_t = torch_mod.tensor(a_weights, dtype=torch_mod.float32, device=dev)
            b_t = torch_mod.tensor(b_weights, dtype=torch_mod.float32, device=dev)
            r_t = torch_mod.tensor(ranks, dtype=torch_mod.int32, device=dev)
            s_t = torch_mod.tensor(scalings, dtype=torch_mod.float32, device=dev)

            y_auto, tel_auto = dispatcher.forward(
                x_t, base_t, p_t, a_t, b_t, r_t, s_t, top_k=2, temperature=0.8, backend="auto"
            )
            assert tel_auto.backend == expected_backend
            assert tel_auto.expert_indices == tel_py.expert_indices
            for row_t, row_py in zip(y_auto.detach().cpu().tolist(), y_py, strict=True):
                assert row_t == pytest.approx(row_py, abs=1e-5)

            y_ref_alias, tel_ref_alias = dispatcher.forward(
                x_t,
                base_t,
                p_t,
                a_t,
                b_t,
                r_t,
                s_t,
                top_k=2,
                temperature=0.8,
                backend="torch_reference",
            )
            assert tel_ref_alias.backend == expected_backend
            assert float(torch_mod.max(torch_mod.abs(y_ref_alias - y_auto)).item()) <= 1e-6

            # 3D Tensor support
            y_3d_t, tel_3d_t = arrow_molora_reference(
                x_t.reshape(2, 2, d_in),
                base_t.reshape(2, 2, d_out),
                p_t,
                a_t,
                b_t,
                r_t,
                s_t,
                top_k=2,
                temperature=0.8,
            )
            assert tuple(y_3d_t.shape) == (2, 2, d_out)
            assert tel_3d_t.num_tokens == 4
            assert (
                float(torch_mod.max(torch_mod.abs(y_3d_t.reshape(4, d_out) - y_auto)).item())
                <= 1e-6
            )


def test_forced_triton_without_cuda_raises_runtime_error() -> None:
    protos = [[1.0, 0.0], [0.0, 1.0]]
    a_w = [[[1.0, 0.0]], [[0.0, 1.0]]]
    b_w = [[[1.0], [0.0]], [[0.0], [1.0]]]
    ranks = [1, 1]
    scalings = [1.0, 1.0]
    x = [[1.0, 0.5]]
    base = [[0.0, 0.0]]

    dispatcher = ArrowMoLoRADispatcher()
    if not has_triton_cuda():
        with pytest.raises(RuntimeExecutionError) as exc_info:
            dispatcher.forward(
                x, base, protos, a_w, b_w, ranks, scalings, top_k=1, backend="triton_cuda"
            )
        assert exc_info.value.exit_code == EXIT_RUNTIME_ERROR

        seg = build_segment_metadata([[0]], [[1.0]], num_experts=2)
        with pytest.raises(RuntimeExecutionError) as exc_sgmv:
            triton_arrow_sgmv_forward(x, base, a_w, b_w, ranks, scalings, seg)
        assert exc_sgmv.value.exit_code == EXIT_RUNTIME_ERROR

    if not has_mps():
        with pytest.raises(RuntimeExecutionError) as exc_mps:
            dispatcher.forward(
                x, base, protos, a_w, b_w, ranks, scalings, top_k=1, backend="torch_mps"
            )
        assert exc_mps.value.exit_code == EXIT_RUNTIME_ERROR

    if not has_torch():
        with pytest.raises(RuntimeExecutionError) as exc_cpu:
            dispatcher.forward(
                x, base, protos, a_w, b_w, ranks, scalings, top_k=1, backend="torch_cpu"
            )
        assert exc_cpu.value.exit_code == EXIT_RUNTIME_ERROR
        with pytest.raises(RuntimeExecutionError):
            arrow_molora_reference(x, base, protos, a_w, b_w, ranks, scalings)
