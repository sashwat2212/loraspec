"""Property-based CUDA Triton Arrow-SGMV vs PyTorch FP32 equivalence suite."""

from __future__ import annotations

import math
import random
from typing import Any

import pytest

from loraspec.kernels import has_triton_cuda
from loraspec.kernels.arrow_sgmv import build_segment_metadata, triton_arrow_sgmv_forward
from loraspec.kernels.dispatch import ArrowMoLoRADispatcher
from loraspec.kernels.reference import arrow_molora_reference
from loraspec.kernels.router import route_topk_tokens

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not has_triton_cuda(), reason="Requires NVIDIA CUDA + Triton"),
]


def test_triton_fp16_bf16_mixed_rank_property_parity() -> None:
    torch: Any = pytest.importorskip("torch")
    rng = random.Random(2026)

    token_counts = [1, 3, 17, 128, 1025, 2048]
    mixed_rank_pool = [8, 16, 32, 64]
    dtypes = [("fp16", torch.float16, 1e-3), ("bf16", torch.bfloat16, 5e-3)]

    case_count = 0
    for dt_name, dt_obj, tol in dtypes:
        for num_tokens in token_counts:
            for top_k in (1, 2, 3, 4):
                for trial in range(3):
                    num_experts = max(top_k, rng.choice([2, 4, 6]))
                    ranks_list = [rng.choice(mixed_rank_pool) for _ in range(num_experts)]
                    r_max = max(ranks_list)
                    d_in = rng.choice([64, 128, 256])
                    d_out = rng.choice([64, 128, 256])

                    torch.manual_seed(1000 + case_count)
                    x = torch.randn((num_tokens, d_in), dtype=dt_obj, device="cuda") * 0.5
                    base_out = torch.randn((num_tokens, d_out), dtype=dt_obj, device="cuda") * 0.2
                    prototypes = torch.nn.functional.normalize(
                        torch.randn((num_experts, d_in), dtype=torch.float32, device="cuda"),
                        dim=-1,
                    )

                    a_packed = torch.zeros((num_experts, r_max, d_in), dtype=dt_obj, device="cuda")
                    b_packed = torch.zeros((num_experts, d_out, r_max), dtype=dt_obj, device="cuda")
                    a_scale = 1.0 / math.sqrt(float(d_in))
                    for e_idx, r_e in enumerate(ranks_list):
                        b_scale = 0.1 / math.sqrt(float(max(r_e, 1)))
                        a_packed[e_idx, :r_e, :] = (
                            torch.randn((r_e, d_in), dtype=dt_obj, device="cuda") * a_scale
                        )
                        b_packed[e_idx, :, :r_e] = (
                            torch.randn((d_out, r_e), dtype=dt_obj, device="cuda") * b_scale
                        )

                    ranks_t = torch.tensor(ranks_list, dtype=torch.int32, device="cuda")
                    scalings_t = torch.tensor(
                        [1.0 if i % 2 == 0 else 2.0 for i in range(num_experts)],
                        dtype=torch.float32,
                        device="cuda",
                    )

                    ref_out, _ = arrow_molora_reference(
                        x=x.float(),
                        base_out=base_out.float(),
                        prototypes=prototypes,
                        a_weights=a_packed.float(),
                        b_weights=b_packed.float(),
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k,
                        temperature=0.75 + 0.25 * trial,
                    )
                    tri_out, tel = ArrowMoLoRADispatcher().forward(
                        x=x,
                        base_out=base_out,
                        prototypes=prototypes,
                        a_weights=a_packed,
                        b_weights=b_packed,
                        ranks=ranks_t,
                        scalings=scalings_t,
                        top_k=top_k,
                        temperature=0.75 + 0.25 * trial,
                        backend="triton_cuda",
                    )
                    assert tel.backend == "triton_cuda"
                    max_err = float(torch.max(torch.abs(tri_out.float() - ref_out)).item())
                    assert max_err <= tol, (
                        f"Case {case_count} ({dt_name}, T={num_tokens}, k={top_k}): "
                        f"max_err={max_err} > {tol}"
                    )
                    case_count += 1

    assert case_count >= 100


def test_triton_skewed_empty_expert_deterministic_reduce() -> None:
    torch: Any = pytest.importorskip("torch")
    torch.manual_seed(42)

    num_tokens, num_experts, d_in, d_out = 257, 6, 192, 160
    ranks_list = [8, 16, 32, 64, 8, 16]
    r_max = max(ranks_list)

    # All prototypes aligned closely to expert 0 or 5 so experts 1..4 receive 0 tokens
    x = torch.randn((num_tokens, d_in), dtype=torch.float16, device="cuda") * 0.5
    base_out = torch.randn((num_tokens, d_out), dtype=torch.float16, device="cuda") * 0.2
    prototypes = torch.zeros((num_experts, d_in), dtype=torch.float32, device="cuda")
    prototypes[0, 0] = 1.0
    prototypes[5, 0] = -1.0

    a_packed = torch.zeros((num_experts, r_max, d_in), dtype=torch.float16, device="cuda")
    b_packed = torch.zeros((num_experts, d_out, r_max), dtype=torch.float16, device="cuda")
    a_scale = 1.0 / math.sqrt(float(d_in))
    for e_idx, r_e in enumerate(ranks_list):
        b_scale = 0.1 / math.sqrt(float(max(r_e, 1)))
        a_packed[e_idx, :r_e, :] = (
            torch.randn((r_e, d_in), dtype=torch.float16, device="cuda") * a_scale
        )
        b_packed[e_idx, :, :r_e] = (
            torch.randn((d_out, r_e), dtype=torch.float16, device="cuda") * b_scale
        )

    ranks_t = torch.tensor(ranks_list, dtype=torch.int32, device="cuda")
    scalings_t = torch.ones((num_experts,), dtype=torch.float32, device="cuda")

    expert_ids, weights, _ = route_topk_tokens(x, prototypes, top_k=2, temperature=1.0)
    layout = build_segment_metadata(expert_ids, weights, num_experts)

    out_1 = triton_arrow_sgmv_forward(x, base_out, a_packed, b_packed, ranks_t, scalings_t, layout)
    out_2 = triton_arrow_sgmv_forward(x, base_out, a_packed, b_packed, ranks_t, scalings_t, layout)
    assert torch.equal(out_1, out_2)

    ref_out, _ = arrow_molora_reference(
        x.float(),
        base_out.float(),
        prototypes,
        a_packed.float(),
        b_packed.float(),
        ranks_t,
        scalings_t,
        top_k=2,
        temperature=1.0,
    )
    assert float(torch.max(torch.abs(out_1.float() - ref_out)).item()) <= 1e-3
