"""Token-to-expert segment permutation and two-stage @triton.jit Arrow-SGMV CUDA kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

from loraspec.core.errors import ConfigError, RuntimeExecutionError
from loraspec.kernels import has_triton_cuda, optional_torch, optional_triton

_F = TypeVar("_F", bound=Callable[..., Any])

_triton_mod, _tl_mod = optional_triton()

if TYPE_CHECKING:

    class tl:  # noqa: N801
        constexpr: TypeAlias = int
        float32: Any
        int64: Any

        @staticmethod
        def program_id(axis: int) -> Any: ...

        @staticmethod
        def load(*args: Any, **kwargs: Any) -> Any: ...

        @staticmethod
        def store(*args: Any, **kwargs: Any) -> None: ...

        @staticmethod
        def arange(start: int, end: Any) -> Any: ...

        @staticmethod
        def zeros(shape: tuple[int, ...], dtype: Any = None) -> Any: ...

        @staticmethod
        def dot(*args: Any, **kwargs: Any) -> Any: ...

        @staticmethod
        def trans(x: Any) -> Any: ...

    triton: Any = Any
else:
    if _triton_mod is None or _tl_mod is None:

        class _TritonShim:
            @staticmethod
            def Config(
                kwargs: dict[str, int],
                num_stages: int = 2,
                num_warps: int = 4,
            ) -> dict[str, Any]:
                return {"kwargs": kwargs, "num_stages": num_stages, "num_warps": num_warps}

            @staticmethod
            def autotune(configs: list[Any], key: list[str]) -> Callable[[_F], _F]:
                _ = (configs, key)

                def _decorator(fn: _F) -> _F:
                    return fn

                return _decorator

            @staticmethod
            def jit(fn: _F) -> _F:
                return fn

            @staticmethod
            def cdiv(x: int, y: int) -> int:
                return (x + y - 1) // y

        triton: Any = _TritonShim()
        tl: Any = cast(Any, None)
    else:
        triton = _triton_mod
        tl = _tl_mod


def _is_torch_tensor(value: Any) -> bool:
    torch_mod = optional_torch()
    return torch_mod is not None and isinstance(value, torch_mod.Tensor)


@dataclass(frozen=True)
class SegmentLayout:
    """Deterministic contiguous segment descriptor mapping `(token, slot)` pairs to experts."""

    sorted_token_ids: Any
    sorted_slots: Any
    sorted_weights: Any
    expert_counts: Any
    expert_offsets: Any
    num_tokens: int
    top_k: int
    num_experts: int
    max_segment_len: int


def _require_pos_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be an integer >= 1, got {value!r}.")
    return value


def build_segment_metadata(
    expert_ids: Any,
    routing_weights: Any,
    num_experts: int,
) -> SegmentLayout:
    """Sort Top-k token-slot assignments by expert_id without Python loops over tokens."""
    num_experts_val = _require_pos_int(num_experts, "num_experts")

    if _is_torch_tensor(expert_ids):
        torch_mod = optional_torch()
        assert torch_mod is not None
        if not _is_torch_tensor(routing_weights):
            routing_weights = torch_mod.tensor(
                routing_weights, dtype=torch_mod.float32, device=expert_ids.device
            )
        if expert_ids.ndim != 2 or routing_weights.ndim != 2:
            raise ConfigError(
                "expert_ids and routing_weights must be 2D [T, k] tensors; "
                f"got {tuple(expert_ids.shape)} and {tuple(routing_weights.shape)}."
            )
        if tuple(expert_ids.shape) != tuple(routing_weights.shape):
            raise ConfigError(
                f"Shape mismatch between expert_ids {tuple(expert_ids.shape)} "
                f"and routing_weights {tuple(routing_weights.shape)}."
            )
        num_tokens, top_k = int(expert_ids.shape[0]), int(expert_ids.shape[1])
        if top_k < 1:
            raise ConfigError(f"top_k dimension must be >= 1, got {top_k}.")
        device = expert_ids.device

        if num_tokens == 0:
            return SegmentLayout(
                sorted_token_ids=torch_mod.empty((0,), dtype=torch_mod.int32, device=device),
                sorted_slots=torch_mod.empty((0,), dtype=torch_mod.int32, device=device),
                sorted_weights=torch_mod.empty((0,), dtype=torch_mod.float32, device=device),
                expert_counts=torch_mod.zeros(
                    (num_experts_val,), dtype=torch_mod.int32, device=device
                ),
                expert_offsets=torch_mod.zeros(
                    (num_experts_val + 1,), dtype=torch_mod.int32, device=device
                ),
                num_tokens=0,
                top_k=top_k,
                num_experts=num_experts_val,
                max_segment_len=0,
            )

        flat_experts = expert_ids.reshape(-1).to(dtype=torch_mod.int64)
        flat_weights = routing_weights.reshape(-1).to(dtype=torch_mod.float32)

        if bool((flat_experts < 0).any()) or bool((flat_experts >= num_experts_val).any()):
            raise ConfigError(
                f"expert_ids contain out-of-range values outside [0, {num_experts_val})."
            )

        total_assigns = int(flat_experts.numel())
        composite_key = flat_experts * total_assigns + torch_mod.arange(
            total_assigns, device=device, dtype=torch_mod.int64
        )
        perm = torch_mod.argsort(composite_key)
        sorted_token_ids = (perm // top_k).to(dtype=torch_mod.int32)
        sorted_slots = (perm % top_k).to(dtype=torch_mod.int32)
        sorted_weights = flat_weights[perm]

        if num_experts_val <= 128:
            expert_grid = torch_mod.arange(num_experts_val, device=device, dtype=torch_mod.int64)
            expert_counts = (
                (flat_experts[:, None] == expert_grid[None, :]).sum(dim=0).to(dtype=torch_mod.int32)
            )
        elif str(device).startswith("mps"):
            count_chunks: list[Any] = []
            for start in range(0, num_experts_val, 128):
                end = min(num_experts_val, start + 128)
                chunk_grid = torch_mod.arange(start, end, device=device, dtype=torch_mod.int64)
                count_chunks.append(
                    (flat_experts[:, None] == chunk_grid[None, :])
                    .sum(dim=0)
                    .to(dtype=torch_mod.int32)
                )
            expert_counts = torch_mod.cat(count_chunks, dim=0)
        else:
            expert_counts = torch_mod.bincount(flat_experts, minlength=num_experts_val).to(
                dtype=torch_mod.int32
            )

        expert_offsets = torch_mod.zeros(
            (num_experts_val + 1,), dtype=torch_mod.int32, device=device
        )
        expert_offsets[1:] = torch_mod.cumsum(expert_counts, dim=0).to(dtype=torch_mod.int32)
        max_segment_len = int(expert_counts.max().item()) if num_experts_val > 0 else 0

        return SegmentLayout(
            sorted_token_ids=sorted_token_ids,
            sorted_slots=sorted_slots,
            sorted_weights=sorted_weights,
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            num_tokens=num_tokens,
            top_k=top_k,
            num_experts=num_experts_val,
            max_segment_len=max_segment_len,
        )

    # Pure-Python path
    if (
        not isinstance(expert_ids, Sequence)
        or isinstance(expert_ids, (str, bytes))
        or not isinstance(routing_weights, Sequence)
        or isinstance(routing_weights, (str, bytes))
    ):
        raise ConfigError("expert_ids and routing_weights must be Tensors or 2D sequences.")
    exp_rows = cast(Sequence[Sequence[Any]], expert_ids)
    w_rows = cast(Sequence[Sequence[Any]], routing_weights)
    if len(exp_rows) != len(w_rows):
        raise ConfigError("Length mismatch between expert_ids and routing_weights.")
    num_tokens = len(exp_rows)
    if num_tokens == 0:
        return SegmentLayout(
            sorted_token_ids=[],
            sorted_slots=[],
            sorted_weights=[],
            expert_counts=[0] * num_experts_val,
            expert_offsets=[0] * (num_experts_val + 1),
            num_tokens=0,
            top_k=0,
            num_experts=num_experts_val,
            max_segment_len=0,
        )

    top_k = len(exp_rows[0])
    if top_k < 1:
        raise ConfigError("top_k must be >= 1.")

    records: list[tuple[int, int, int, float]] = []
    counts = [0] * num_experts_val
    for tok_idx, (e_row, w_row) in enumerate(zip(exp_rows, w_rows, strict=True)):
        if len(e_row) != top_k or len(w_row) != top_k:
            raise ConfigError("Ragged rows in expert_ids or routing_weights.")
        for slot_idx, (exp_id, w_val) in enumerate(zip(e_row, w_row, strict=True)):
            if isinstance(exp_id, bool) or not isinstance(exp_id, int):
                raise ConfigError(f"Expert id must be int, got {exp_id!r}.")
            if not (0 <= exp_id < num_experts_val):
                raise ConfigError(f"Expert id {exp_id} out of valid range [0, {num_experts_val}).")
            counts[exp_id] += 1
            records.append((exp_id, tok_idx, slot_idx, float(w_val)))

    records.sort(key=lambda item: item[0])
    sorted_token_ids_py = [item[1] for item in records]
    sorted_slots_py = [item[2] for item in records]
    sorted_weights_py = [item[3] for item in records]
    offsets = [0] * (num_experts_val + 1)
    running = 0
    for idx, cnt in enumerate(counts):
        running += cnt
        offsets[idx + 1] = running

    return SegmentLayout(
        sorted_token_ids=sorted_token_ids_py,
        sorted_slots=sorted_slots_py,
        sorted_weights=sorted_weights_py,
        expert_counts=counts,
        expert_offsets=offsets,
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts_val,
        max_segment_len=max(counts, default=0),
    )


_SHRINK_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_K": 32, "BLOCK_R": 16}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_K": 64, "BLOCK_R": 16}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_R": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 64, "BLOCK_R": 64}, num_stages=2, num_warps=8),
]

_EXPAND_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_R": 16}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_R": 16}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_R": 32}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_R": 32}, num_stages=2, num_warps=8),
]


@triton.autotune(configs=_SHRINK_AUTOTUNE_CONFIGS, key=["d_in", "r_max"])
@triton.jit
def _arrow_sgmv_shrink_kernel(
    X_ptr: Any,
    A_ptr: Any,
    H_ptr: Any,
    sorted_token_ids_ptr: Any,
    expert_offsets_ptr: Any,
    ranks_ptr: Any,
    d_in: int,
    r_max: int,
    stride_xm: Any,
    stride_xk: Any,
    stride_ae: Any,
    stride_ar: Any,
    stride_ak: Any,
    stride_hm: Any,
    stride_hr: Any,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    if r_max <= 0:
        return
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_r = tl.program_id(2)

    pid_e_i64 = pid_e.to(tl.int64)
    seg_start = tl.load(expert_offsets_ptr + pid_e_i64)
    seg_end = tl.load(expert_offsets_ptr + pid_e_i64 + 1)
    seg_len = seg_end - seg_start
    if seg_len <= 0:
        return
    if pid_m * BLOCK_M >= seg_len:
        return

    rank_e = tl.load(ranks_ptr + pid_e_i64)
    if rank_e <= 0:
        return
    if pid_r * BLOCK_R >= rank_e:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    m_mask = offs_m < seg_len
    r_mask = offs_r < rank_e

    seg_pos = seg_start.to(tl.int64) + offs_m.to(tl.int64)
    token_ids = tl.load(sorted_token_ids_ptr + seg_pos, mask=m_mask, other=0).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    a_expert_ptr = A_ptr + pid_e_i64 * stride_ae

    for k_start in range(0, d_in, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < d_in

        x_ptrs = (
            X_ptr
            + token_ids.to(tl.int64)[:, None] * stride_xm
            + (offs_k.to(tl.int64)[None, :] * stride_xk)
        )
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        a_ptrs = (
            a_expert_ptr
            + offs_r.to(tl.int64)[:, None] * stride_ar
            + offs_k.to(tl.int64)[None, :] * stride_ak
        )
        a_tile = tl.load(a_ptrs, mask=r_mask[:, None] & k_mask[None, :], other=0.0)

        if IS_FP32:
            acc += tl.dot(x_tile, tl.trans(a_tile), input_precision="ieee")
        else:
            acc += tl.dot(x_tile, tl.trans(a_tile))

    h_ptrs = (
        H_ptr + seg_pos.to(tl.int64)[:, None] * stride_hm + offs_r.to(tl.int64)[None, :] * stride_hr
    )
    tl.store(h_ptrs, acc, mask=m_mask[:, None] & r_mask[None, :])


@triton.autotune(configs=_EXPAND_AUTOTUNE_CONFIGS, key=["d_out", "r_max"])
@triton.jit
def _arrow_sgmv_expand_kernel(
    H_ptr: Any,
    B_ptr: Any,
    Y_slots_ptr: Any,
    sorted_token_ids_ptr: Any,
    sorted_slots_ptr: Any,
    sorted_weights_ptr: Any,
    expert_offsets_ptr: Any,
    ranks_ptr: Any,
    scalings_ptr: Any,
    d_out: Any,
    r_max: Any,
    stride_hm: Any,
    stride_hr: Any,
    stride_be: Any,
    stride_bn: Any,
    stride_br: Any,
    stride_yt: Any,
    stride_ys: Any,
    stride_yn: Any,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
) -> None:
    if r_max <= 0:
        return
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    pid_e_i64 = pid_e.to(tl.int64)
    seg_start = tl.load(expert_offsets_ptr + pid_e_i64)
    seg_end = tl.load(expert_offsets_ptr + pid_e_i64 + 1)
    seg_len = seg_end - seg_start
    if seg_len <= 0:
        return
    if pid_m * BLOCK_M >= seg_len:
        return

    rank_e = tl.load(ranks_ptr + pid_e_i64)
    if rank_e <= 0:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < seg_len
    n_mask = offs_n < d_out

    seg_pos = seg_start.to(tl.int64) + offs_m.to(tl.int64)
    token_ids = tl.load(sorted_token_ids_ptr + seg_pos, mask=m_mask, other=0).to(tl.int64)
    slot_ids = tl.load(sorted_slots_ptr + seg_pos, mask=m_mask, other=0).to(tl.int64)
    weights = tl.load(sorted_weights_ptr + seg_pos, mask=m_mask, other=0.0)
    scaling_e = tl.load(scalings_ptr + pid_e_i64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_expert_ptr = B_ptr + pid_e_i64 * stride_be

    for r_start in range(0, rank_e, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        r_mask = offs_r < rank_e

        h_ptrs = (
            H_ptr
            + seg_pos.to(tl.int64)[:, None] * stride_hm
            + offs_r.to(tl.int64)[None, :] * stride_hr
        )
        h_tile = tl.load(h_ptrs, mask=m_mask[:, None] & r_mask[None, :], other=0.0)

        b_ptrs = (
            b_expert_ptr
            + offs_n.to(tl.int64)[:, None] * stride_bn
            + offs_r.to(tl.int64)[None, :] * stride_br
        )
        b_tile = tl.load(b_ptrs, mask=n_mask[:, None] & r_mask[None, :], other=0.0)

        if IS_FP32:
            acc += tl.dot(h_tile, tl.trans(b_tile), input_precision="ieee")
        else:
            acc += tl.dot(h_tile.to(b_tile.dtype), tl.trans(b_tile))

    row_scale = weights * scaling_e
    scaled_acc = row_scale[:, None] * acc

    y_ptrs = (
        Y_slots_ptr
        + token_ids.to(tl.int64)[:, None] * stride_yt
        + slot_ids.to(tl.int64)[:, None] * stride_ys
        + offs_n.to(tl.int64)[None, :] * stride_yn
    )
    tl.store(y_ptrs, scaled_acc, mask=m_mask[:, None] & n_mask[None, :])


def triton_arrow_sgmv_forward(
    x: Any,
    base_out: Any,
    a_packed: Any,
    b_packed: Any,
    ranks: Any,
    scalings: Any,
    segment_layout: SegmentLayout,
) -> Any:
    """Execute fused two-stage @triton.jit Arrow-SGMV forward pass with deterministic reduction."""
    if not has_triton_cuda():
        raise RuntimeExecutionError(
            "Triton CUDA runtime is unavailable. Use `backend='auto'` for PyTorch CPU/MPS "
            "reference execution or install `loraspec[kernels]` on a CUDA host."
        )
    torch_mod = optional_torch()
    assert torch_mod is not None

    tensors = (
        x,
        base_out,
        a_packed,
        b_packed,
        ranks,
        scalings,
        segment_layout.sorted_token_ids,
        segment_layout.sorted_slots,
        segment_layout.sorted_weights,
        segment_layout.expert_offsets,
    )
    for t in tensors:
        if not isinstance(t, torch_mod.Tensor) or not bool(t.is_cuda):
            raise RuntimeExecutionError(
                "triton_arrow_sgmv_forward requires all operands on an active CUDA device."
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
    if (
        x.dtype not in (torch_mod.float16, torch_mod.bfloat16, torch_mod.float32)
        or base_out.dtype != x.dtype
        or a_packed.dtype != x.dtype
        or b_packed.dtype != x.dtype
    ):
        raise ConfigError(
            "x, base_out, a_packed, and b_packed must share dtype in {fp16, bf16, fp32}."
        )

    num_experts, r_max, d_in = (
        int(a_packed.shape[0]),
        int(a_packed.shape[1]),
        int(a_packed.shape[2]),
    )
    d_out = int(b_packed.shape[1])
    if int(x.shape[-1]) != d_in or int(base_out.shape[-1]) != d_out:
        raise ConfigError(
            f"Layer shape mismatch: x[-1]={int(x.shape[-1])} vs d_in={d_in}, "
            f"base_out[-1]={int(base_out.shape[-1])} vs d_out={d_out}."
        )
    if (
        int(b_packed.shape[0]) != num_experts
        or int(b_packed.shape[2]) != r_max
        or int(ranks.shape[0]) != num_experts
        or int(scalings.shape[0]) != num_experts
        or segment_layout.num_experts != num_experts
    ):
        raise ConfigError("Expert dimension mismatch across packed weights and segment_layout.")

    x_2d = x.reshape(-1, d_in).contiguous()
    base_2d = base_out.reshape(-1, d_out).contiguous()
    num_tokens = int(x_2d.shape[0])
    if num_tokens != segment_layout.num_tokens:
        raise ConfigError(
            f"segment_layout.num_tokens={segment_layout.num_tokens} != x tokens={num_tokens}."
        )

    if (
        num_tokens == 0
        or segment_layout.max_segment_len == 0
        or r_max <= 0
        or d_in <= 0
        or d_out <= 0
    ):
        return base_out.clone()

    a_c = a_packed.contiguous()
    b_c = b_packed.contiguous()
    ranks_c = ranks.to(dtype=torch_mod.int32).contiguous()
    scalings_c = scalings.to(dtype=torch_mod.float32).contiguous()
    tok_ids_c = segment_layout.sorted_token_ids.to(dtype=torch_mod.int32).contiguous()
    slots_c = segment_layout.sorted_slots.to(dtype=torch_mod.int32).contiguous()
    weights_c = segment_layout.sorted_weights.to(dtype=torch_mod.float32).contiguous()
    offsets_c = segment_layout.expert_offsets.to(dtype=torch_mod.int32).contiguous()

    total_assigns = num_tokens * segment_layout.top_k
    h_sorted = torch_mod.zeros((total_assigns, r_max), dtype=torch_mod.float32, device=x_2d.device)
    y_slots = torch_mod.zeros(
        (num_tokens, segment_layout.top_k, d_out), dtype=torch_mod.float32, device=x_2d.device
    )
    is_fp32 = bool(x_2d.dtype == torch_mod.float32)

    def grid_shrink(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            num_experts,
            int(triton.cdiv(segment_layout.max_segment_len, meta["BLOCK_M"])),
            int(triton.cdiv(r_max, meta["BLOCK_R"])),
        )

    _arrow_sgmv_shrink_kernel[grid_shrink](
        x_2d,
        a_c,
        h_sorted,
        tok_ids_c,
        offsets_c,
        ranks_c,
        d_in,
        r_max,
        x_2d.stride(0),
        x_2d.stride(1),
        a_c.stride(0),
        a_c.stride(1),
        a_c.stride(2),
        h_sorted.stride(0),
        h_sorted.stride(1),
        IS_FP32=is_fp32,
    )

    def grid_expand(meta: dict[str, int]) -> tuple[int, int, int]:
        return (
            num_experts,
            int(triton.cdiv(segment_layout.max_segment_len, meta["BLOCK_M"])),
            int(triton.cdiv(d_out, meta["BLOCK_N"])),
        )

    _arrow_sgmv_expand_kernel[grid_expand](
        h_sorted,
        b_c,
        y_slots,
        tok_ids_c,
        slots_c,
        weights_c,
        offsets_c,
        ranks_c,
        scalings_c,
        d_out,
        r_max,
        h_sorted.stride(0),
        h_sorted.stride(1),
        b_c.stride(0),
        b_c.stride(1),
        b_c.stride(2),
        y_slots.stride(0),
        y_slots.stride(1),
        y_slots.stride(2),
        IS_FP32=is_fp32,
    )

    y_out_2d = (base_2d.to(dtype=torch_mod.float32) + y_slots.sum(dim=1)).to(dtype=base_2d.dtype)
    return y_out_2d.reshape(base_out.shape)
