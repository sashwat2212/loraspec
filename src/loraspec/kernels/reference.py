"""Cross-platform PyTorch (CPU/MPS/CUDA) & pure-Python Arrow-MoLoRA references & baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from loraspec.core.errors import ConfigError, RuntimeExecutionError
from loraspec.core.schemas import MoLoRARouteTelemetryBatch
from loraspec.core.spectral_math import Matrix
from loraspec.kernels import optional_torch
from loraspec.kernels.arrow_sgmv import build_segment_metadata
from loraspec.kernels.router import compute_router_entropy, route_topk_tokens

TelemetryBackend = Literal["triton_cuda", "torch_mps", "torch_cpu", "pure_python"]


def _is_torch_tensor(value: Any) -> bool:
    torch_mod = optional_torch()
    return torch_mod is not None and isinstance(value, torch_mod.Tensor)


def _infer_torch_backend(device: Any) -> TelemetryBackend:
    dev_type = str(getattr(device, "type", device)).lower()
    if dev_type == "mps":
        return "torch_mps"
    return "torch_cpu"


def arrow_molora_reference(
    x: Any,
    base_out: Any,
    prototypes: Any,
    a_weights: Any,
    b_weights: Any,
    ranks: Any,
    scalings: Any,
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    layer_name: str = "molora_layer",
    backend_label: TelemetryBackend | None = None,
) -> tuple[Any, MoLoRARouteTelemetryBatch]:
    """FP32-accumulated PyTorch reference Arrow-MoLoRA forward (100% MPS/CPU/CUDA safe)."""
    torch_mod = optional_torch()
    if torch_mod is None:
        raise RuntimeExecutionError(
            "PyTorch is required for arrow_molora_reference(); install `loraspec[kernels]`."
        )
    if not isinstance(x, torch_mod.Tensor) or not isinstance(base_out, torch_mod.Tensor):
        raise ConfigError(
            "arrow_molora_reference expects PyTorch Tensor inputs for x and base_out."
        )
    if x.ndim not in (2, 3) or base_out.ndim != x.ndim:
        raise ConfigError(
            f"x and base_out must both be 2D or 3D tensors; got {tuple(x.shape)} and "
            f"{tuple(base_out.shape)}."
        )
    if tuple(x.shape[:-1]) != tuple(base_out.shape[:-1]):
        raise ConfigError(
            f"Leading shape mismatch: x {tuple(x.shape)} vs base_out {tuple(base_out.shape)}."
        )
    if x.device != base_out.device:
        raise ConfigError(f"Device mismatch: x on {x.device}, base_out on {base_out.device}.")

    device = x.device
    resolved_label: TelemetryBackend = backend_label or _infer_torch_backend(device)

    proto_t = (
        prototypes
        if isinstance(prototypes, torch_mod.Tensor)
        else torch_mod.tensor(prototypes, dtype=torch_mod.float32, device=device)
    )
    a_t = (
        a_weights
        if isinstance(a_weights, torch_mod.Tensor)
        else torch_mod.tensor(a_weights, dtype=torch_mod.float32, device=device)
    )
    b_t = (
        b_weights
        if isinstance(b_weights, torch_mod.Tensor)
        else torch_mod.tensor(b_weights, dtype=torch_mod.float32, device=device)
    )
    ranks_t = (
        ranks
        if isinstance(ranks, torch_mod.Tensor)
        else torch_mod.tensor(ranks, dtype=torch_mod.int32, device=device)
    )
    scalings_t = (
        scalings
        if isinstance(scalings, torch_mod.Tensor)
        else torch_mod.tensor(scalings, dtype=torch_mod.float32, device=device)
    )

    for name, tensor in (
        ("prototypes", proto_t),
        ("a_weights", a_t),
        ("b_weights", b_t),
        ("ranks", ranks_t),
        ("scalings", scalings_t),
    ):
        if tensor.device != device:
            raise ConfigError(f"Device mismatch: {name} on {tensor.device}, expected {device}.")

    if (
        a_t.ndim != 3
        or b_t.ndim != 3
        or proto_t.ndim != 2
        or ranks_t.ndim != 1
        or scalings_t.ndim != 1
    ):
        raise ConfigError("Invalid tensor rank in Arrow-MoLoRA weights or prototypes.")

    num_experts, r_max, d_in = int(a_t.shape[0]), int(a_t.shape[1]), int(a_t.shape[2])
    d_out = int(b_t.shape[1])
    if (
        int(proto_t.shape[0]) != num_experts
        or int(proto_t.shape[1]) != d_in
        or int(b_t.shape[0]) != num_experts
        or int(b_t.shape[2]) != r_max
        or int(ranks_t.shape[0]) != num_experts
        or int(scalings_t.shape[0]) != num_experts
    ):
        raise ConfigError("Inconsistent expert/rank/feature dimensions across weight tensors.")
    if int(x.shape[-1]) != d_in or int(base_out.shape[-1]) != d_out:
        raise ConfigError(
            f"Layer dimension mismatch: x[-1]={int(x.shape[-1])} (expected {d_in}), "
            f"base_out[-1]={int(base_out.shape[-1])} (expected {d_out})."
        )

    x_2d = x.reshape(-1, d_in)
    base_2d = base_out.reshape(-1, d_out)
    num_tokens = int(x_2d.shape[0])

    expert_ids, routing_weights, entropy = route_topk_tokens(
        x_2d, proto_t, top_k=top_k, temperature=temperature
    )

    if num_tokens == 0:
        telemetry = MoLoRARouteTelemetryBatch(
            layer_name=layer_name,
            num_tokens=0,
            top_k=top_k,
            expert_indices=[],
            routing_weights=[],
            router_entropy=[],
            backend=resolved_label,
        )
        return base_out.clone(), telemetry

    layout = build_segment_metadata(expert_ids, routing_weights, num_experts)
    offsets_list = [int(v) for v in layout.expert_offsets.detach().cpu().tolist()]
    ranks_list = [int(v) for v in ranks_t.detach().cpu().tolist()]
    scalings_list = [float(v) for v in scalings_t.detach().cpu().tolist()]

    y_slots = torch_mod.zeros((num_tokens, top_k, d_out), dtype=torch_mod.float32, device=device)
    for exp_idx in range(num_experts):
        seg_start = offsets_list[exp_idx]
        seg_end = offsets_list[exp_idx + 1]
        rank_e = ranks_list[exp_idx]
        if seg_end <= seg_start or rank_e <= 0:
            continue
        if rank_e > r_max:
            raise ConfigError(f"Expert {exp_idx} rank {rank_e} exceeds r_max={r_max}.")

        tok_ids = layout.sorted_token_ids[seg_start:seg_end].to(dtype=torch_mod.int64)
        slot_ids = layout.sorted_slots[seg_start:seg_end].to(dtype=torch_mod.int64)
        w_seg = layout.sorted_weights[seg_start:seg_end]

        x_seg = x_2d[tok_ids].to(dtype=torch_mod.float32)
        a_e = a_t[exp_idx, :rank_e, :].to(dtype=torch_mod.float32)
        b_e = b_t[exp_idx, :, :rank_e].to(dtype=torch_mod.float32)

        h_seg = x_seg @ a_e.transpose(0, 1)
        delta_seg = (h_seg @ b_e.transpose(0, 1)) * (w_seg * scalings_list[exp_idx])[:, None]
        y_slots[tok_ids, slot_ids, :] = delta_seg

    y_out_2d = base_2d + y_slots.sum(dim=1).to(dtype=base_2d.dtype)
    telemetry = MoLoRARouteTelemetryBatch(
        layer_name=layer_name,
        num_tokens=num_tokens,
        top_k=top_k,
        expert_indices=cast(list[list[int]], expert_ids.detach().cpu().tolist()),
        routing_weights=cast(list[list[float]], routing_weights.detach().cpu().tolist()),
        router_entropy=cast(list[float], entropy.detach().cpu().tolist()),
        backend=resolved_label,
    )
    return y_out_2d.reshape(base_out.shape), telemetry


def _is_3d_seq(seq: Sequence[Any]) -> bool:
    if len(seq) == 0:
        return False
    first: Any = seq[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return False
    first_seq = cast(Sequence[Any], first)
    if len(first_seq) == 0:
        return False
    inner: Any = first_seq[0]
    return isinstance(inner, Sequence) and not isinstance(inner, (str, bytes))


def _flatten_py_matrix(seq: Sequence[Any]) -> tuple[Matrix, list[int] | None]:
    if len(seq) == 0:
        return [], None
    if _is_3d_seq(seq):
        seq_3d = cast(Sequence[Sequence[Sequence[Any]]], seq)
        batch_lengths = [len(b) for b in seq_3d]
        flat = [[float(v) for v in row] for b in seq_3d for row in b]
        return flat, batch_lengths
    first: Any = seq[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        seq_rows = cast(Sequence[Sequence[Any]], seq)
        if all(len(b) == 0 for b in seq_rows):
            return [], [0 for _ in seq_rows]
    seq_2d = cast(Sequence[Sequence[Any]], seq)
    return [[float(v) for v in row] for row in seq_2d], None


def _unflatten_py_matrix(flat: Matrix, batch_lengths: list[int] | None) -> Any:
    if batch_lengths is None:
        return flat
    out: list[Matrix] = []
    cursor = 0
    for length in batch_lengths:
        out.append(flat[cursor : cursor + length])
        cursor += length
    return out


def pure_python_arrow_reference(
    x: Sequence[Any],
    base_out: Sequence[Any],
    prototypes: Sequence[Sequence[float]],
    a_weights: Sequence[Matrix],
    b_weights: Sequence[Matrix],
    ranks: Sequence[int],
    scalings: Sequence[float],
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    layer_name: str = "molora_layer",
) -> tuple[Any, MoLoRARouteTelemetryBatch]:
    """Zero-torch exact pure-Python Arrow-MoLoRA reference supporting 2D and 3D inputs."""
    num_experts = len(prototypes)
    if num_experts < 1:
        raise ConfigError("prototypes cannot be empty.")
    if (
        len(a_weights) != num_experts
        or len(b_weights) != num_experts
        or len(ranks) != num_experts
        or len(scalings) != num_experts
    ):
        raise ConfigError("Expert count mismatch across prototypes, weights, ranks, and scalings.")

    d_in = len(prototypes[0])
    r_max = len(a_weights[0]) if a_weights[0] else 0
    d_out = len(b_weights[0]) if b_weights[0] else 0
    if d_in < 1 or r_max < 1 or d_out < 1:
        raise ConfigError("Invalid non-positive layer dimensions in pure_python_arrow_reference.")

    for exp_idx in range(num_experts):
        if len(prototypes[exp_idx]) != d_in:
            raise ConfigError("Inconsistent prototype d_in across experts.")
        if len(a_weights[exp_idx]) != r_max or any(len(r) != d_in for r in a_weights[exp_idx]):
            raise ConfigError(f"Invalid a_weights shape at expert {exp_idx}.")
        if len(b_weights[exp_idx]) != d_out or any(len(r) != r_max for r in b_weights[exp_idx]):
            raise ConfigError(f"Invalid b_weights shape at expert {exp_idx}.")
        if ranks[exp_idx] < 0 or ranks[exp_idx] > r_max:
            raise ConfigError(f"Expert {exp_idx} rank {ranks[exp_idx]} out of bounds [0, {r_max}].")

    x_2d, x_batch_lens = _flatten_py_matrix(x)
    base_2d, base_batch_lens = _flatten_py_matrix(base_out)
    if x_batch_lens != base_batch_lens or len(x_2d) != len(base_2d):
        raise ConfigError("Leading shape mismatch between x and base_out.")

    if len(x_2d) == 0:
        expert_ids_empty, weights_empty, entropy_empty = route_topk_tokens(
            [], prototypes, top_k=top_k, temperature=temperature
        )
        _ = (expert_ids_empty, weights_empty, entropy_empty)
        return _unflatten_py_matrix([], x_batch_lens), MoLoRARouteTelemetryBatch(
            layer_name=layer_name,
            num_tokens=0,
            top_k=top_k,
            expert_indices=[],
            routing_weights=[],
            router_entropy=[],
            backend="pure_python",
        )

    for row in x_2d:
        if len(row) != d_in:
            raise ConfigError(f"Expected x feature dim {d_in}, got {len(row)}.")
    for row in base_2d:
        if len(row) != d_out:
            raise ConfigError(f"Expected base_out feature dim {d_out}, got {len(row)}.")

    expert_ids_py, weights_py, entropy_py = route_topk_tokens(
        x_2d, prototypes, top_k=top_k, temperature=temperature
    )
    layout = build_segment_metadata(expert_ids_py, weights_py, num_experts)

    num_tokens = len(x_2d)
    y_slots: list[list[list[float]]] = [
        [[0.0] * d_out for _ in range(top_k)] for _ in range(num_tokens)
    ]

    offsets = cast(list[int], layout.expert_offsets)
    tok_ids = cast(list[int], layout.sorted_token_ids)
    slot_ids = cast(list[int], layout.sorted_slots)
    sorted_weights = cast(list[float], layout.sorted_weights)

    for exp_idx in range(num_experts):
        seg_start = offsets[exp_idx]
        seg_end = offsets[exp_idx + 1]
        rank_e = int(ranks[exp_idx])
        scale_e = float(scalings[exp_idx])
        if seg_end <= seg_start or rank_e <= 0 or scale_e == 0.0:
            continue

        a_e = a_weights[exp_idx][:rank_e]
        b_e = b_weights[exp_idx]
        for pos in range(seg_start, seg_end):
            tok_i = tok_ids[pos]
            slot_i = slot_ids[pos]
            coeff = sorted_weights[pos] * scale_e
            x_vec = x_2d[tok_i]
            h_vec = [sum(x_vec[j] * a_e[r][j] for j in range(d_in)) for r in range(rank_e)]
            y_slots[tok_i][slot_i] = [
                coeff * sum(b_e[o][r] * h_vec[r] for r in range(rank_e)) for o in range(d_out)
            ]

    y_out_2d: Matrix = [
        [base_2d[t][o] + sum(y_slots[t][s][o] for s in range(top_k)) for o in range(d_out)]
        for t in range(num_tokens)
    ]

    telemetry = MoLoRARouteTelemetryBatch(
        layer_name=layer_name,
        num_tokens=num_tokens,
        top_k=top_k,
        expert_indices=cast(list[list[int]], expert_ids_py),
        routing_weights=cast(list[list[float]], weights_py),
        router_entropy=cast(list[float], entropy_py),
        backend="pure_python",
    )
    return _unflatten_py_matrix(y_out_2d, x_batch_lens), telemetry


def peft_eager_arrow_forward(
    x: Any,
    base_out: Any,
    prototypes: Any,
    a_weights: Any,
    b_weights: Any,
    ranks: Any,
    scalings: Any,
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    layer_name: str = "molora_layer",
) -> tuple[Any, MoLoRARouteTelemetryBatch]:
    """Eager un-fused `peft`-style Arrow baseline allocating per-expert [T, d_out] outputs."""
    if _is_torch_tensor(x):
        torch_mod = optional_torch()
        assert torch_mod is not None
        device = x.device
        d_in = int(a_weights.shape[2])
        d_out = int(b_weights.shape[1])
        num_experts = int(a_weights.shape[0])
        x_2d = x.reshape(-1, d_in)
        base_2d = base_out.reshape(-1, d_out)
        num_tokens = int(x_2d.shape[0])

        expert_ids, routing_weights, entropy = route_topk_tokens(
            x_2d, prototypes, top_k=top_k, temperature=temperature
        )
        backend_lbl = _infer_torch_backend(device)
        if num_tokens == 0:
            return base_out.clone(), MoLoRARouteTelemetryBatch(
                layer_name=layer_name,
                num_tokens=0,
                top_k=top_k,
                expert_indices=[],
                routing_weights=[],
                router_entropy=[],
                backend=backend_lbl,
            )

        g_dense = torch_mod.zeros((num_tokens, num_experts), dtype=torch_mod.float32, device=device)
        g_dense.scatter_(1, expert_ids.to(dtype=torch_mod.int64), routing_weights)

        ranks_list = [int(v) for v in ranks.detach().cpu().tolist()]
        scalings_list = [float(v) for v in scalings.detach().cpu().tolist()]
        x_fp32 = x_2d.to(dtype=torch_mod.float32)

        # Un-fused eager per-expert projection stack matching peft Arrow scratchpad footprint
        expert_deltas: list[Any] = []
        for exp_idx in range(num_experts):
            rank_e = ranks_list[exp_idx]
            if rank_e <= 0:
                expert_deltas.append(
                    torch_mod.zeros((num_tokens, d_out), dtype=torch_mod.float32, device=device)
                )
                continue
            a_e = a_weights[exp_idx, :rank_e, :].to(dtype=torch_mod.float32)
            b_e = b_weights[exp_idx, :, :rank_e].to(dtype=torch_mod.float32)
            h_e = x_fp32 @ a_e.transpose(0, 1)
            out_e = (h_e @ b_e.transpose(0, 1)) * (
                g_dense[:, exp_idx : exp_idx + 1] * scalings_list[exp_idx]
            )
            expert_deltas.append(out_e)

        stacked = torch_mod.stack(expert_deltas, dim=0)  # [K, T, d_out] scratchpad
        y_out_2d = base_2d + stacked.sum(dim=0).to(dtype=base_2d.dtype)
        telemetry = MoLoRARouteTelemetryBatch(
            layer_name=layer_name,
            num_tokens=num_tokens,
            top_k=top_k,
            expert_indices=cast(list[list[int]], expert_ids.detach().cpu().tolist()),
            routing_weights=cast(list[list[float]], routing_weights.detach().cpu().tolist()),
            router_entropy=cast(list[float], entropy.detach().cpu().tolist()),
            backend=backend_lbl,
        )
        return y_out_2d.reshape(base_out.shape), telemetry

    # Pure-Python eager baseline
    x_2d_py, batch_lens = _flatten_py_matrix(x)
    base_2d_py, _ = _flatten_py_matrix(base_out)
    num_experts = len(prototypes)
    d_in = len(prototypes[0])
    d_out = len(b_weights[0])
    expert_ids_py, weights_py, entropy_py = route_topk_tokens(
        x_2d_py, prototypes, top_k=top_k, temperature=temperature
    )
    num_tokens = len(x_2d_py)
    if num_tokens == 0:
        return _unflatten_py_matrix([], batch_lens), MoLoRARouteTelemetryBatch(
            layer_name=layer_name,
            num_tokens=0,
            top_k=top_k,
            expert_indices=[],
            routing_weights=[],
            router_entropy=[],
            backend="pure_python",
        )

    g_dense_py: Matrix = [[0.0] * num_experts for _ in range(num_tokens)]
    for t in range(num_tokens):
        for exp_id, w in zip(expert_ids_py[t], weights_py[t], strict=True):
            g_dense_py[t][exp_id] = w

    expert_deltas_py: list[Matrix] = []
    for exp_idx in range(num_experts):
        rank_e = int(ranks[exp_idx])
        scale_e = float(scalings[exp_idx])
        if rank_e <= 0:
            expert_deltas_py.append([[0.0] * d_out for _ in range(num_tokens)])
            continue
        a_e = a_weights[exp_idx][:rank_e]
        b_e = b_weights[exp_idx]
        delta_e: Matrix = []
        for t in range(num_tokens):
            gw = g_dense_py[t][exp_idx] * scale_e
            x_vec = x_2d_py[t]
            h_vec = [sum(x_vec[j] * a_e[r][j] for j in range(d_in)) for r in range(rank_e)]
            delta_e.append(
                [gw * sum(b_e[o][r] * h_vec[r] for r in range(rank_e)) for o in range(d_out)]
            )
        expert_deltas_py.append(delta_e)

    y_out_py: Matrix = [
        [
            base_2d_py[t][o] + sum(expert_deltas_py[e][t][o] for e in range(num_experts))
            for o in range(d_out)
        ]
        for t in range(num_tokens)
    ]
    return _unflatten_py_matrix(y_out_py, batch_lens), MoLoRARouteTelemetryBatch(
        layer_name=layer_name,
        num_tokens=num_tokens,
        top_k=top_k,
        expert_indices=cast(list[list[int]], expert_ids_py),
        routing_weights=cast(list[list[float]], weights_py),
        router_entropy=cast(list[float], entropy_py),
        backend="pure_python",
    )


def static_lora_forward(
    x: Any,
    base_out: Any,
    a_weight: Any,
    b_weight: Any,
    scaling: float,
    *,
    rank: int | None = None,
) -> Any:
    """Single-adapter static LoRA forward baseline (`W_0 x + s * B A x`)."""
    if _is_torch_tensor(x):
        torch_mod = optional_torch()
        assert torch_mod is not None
        r_eff = int(a_weight.shape[0]) if rank is None else int(rank)
        if r_eff <= 0:
            return base_out.clone()
        d_in = int(a_weight.shape[1])
        d_out = int(b_weight.shape[0])
        x_2d = x.reshape(-1, d_in).to(dtype=torch_mod.float32)
        base_2d = base_out.reshape(-1, d_out)
        if int(x_2d.shape[0]) == 0:
            return base_out.clone()
        a_slice = a_weight[:r_eff, :].to(dtype=torch_mod.float32)
        b_slice = b_weight[:, :r_eff].to(dtype=torch_mod.float32)
        delta = (x_2d @ a_slice.transpose(0, 1)) @ b_slice.transpose(0, 1) * float(scaling)
        return (base_2d + delta.to(dtype=base_2d.dtype)).reshape(base_out.shape)

    x_2d_py, batch_lens = _flatten_py_matrix(x)
    base_2d_py, _ = _flatten_py_matrix(base_out)
    r_eff = len(a_weight) if rank is None else int(rank)
    if not x_2d_py or r_eff <= 0:
        return _unflatten_py_matrix(base_2d_py, batch_lens)
    d_in = len(a_weight[0])
    d_out = len(b_weight)
    a_slice_py = a_weight[:r_eff]
    out_2d: Matrix = []
    for t, x_vec in enumerate(x_2d_py):
        h_vec = [sum(x_vec[j] * a_slice_py[r][j] for j in range(d_in)) for r in range(r_eff)]
        out_2d.append(
            [
                base_2d_py[t][o]
                + float(scaling) * sum(b_weight[o][r] * h_vec[r] for r in range(r_eff))
                for o in range(d_out)
            ]
        )
    return _unflatten_py_matrix(out_2d, batch_lens)


def telemetry_to_entropy_check(telemetry: MoLoRARouteTelemetryBatch) -> list[float]:
    """Helper recomputing router entropy cleanly from telemetry routing weights."""
    if telemetry.num_tokens == 0:
        return []
    return cast(list[float], compute_router_entropy(telemetry.routing_weights))
