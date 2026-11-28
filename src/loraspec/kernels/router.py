"""Zero-shot Arrow spectral prototype extraction and sign-invariant Top-k token router."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from loraspec.core.errors import AdapterParseError, ConfigError, RuntimeExecutionError
from loraspec.core.spectral_math import (
    MathBackend,
    Matrix,
    compute_layer_spectral_analysis,
    validate_math_backend,
)
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights, get_layer_scaling
from loraspec.kernels import optional_torch
from loraspec.knots.aligner import LayerPolicy, has_non_finite_matrix
from loraspec.knots.operators import apply_genknowsub


def _is_torch_tensor(value: Any) -> bool:
    torch_mod = optional_torch()
    return torch_mod is not None and isinstance(value, torch_mod.Tensor)


def _deduplicate_loaded_ids(loaded: Sequence[LoadedLoRAWeights]) -> list[LoadedLoRAWeights]:
    out: list[LoadedLoRAWeights] = []
    used: set[str] = set()
    for item in loaded:
        base_id = item.spec.adapter_id
        unique_id = base_id
        if unique_id in used:
            suffix = 1
            while f"{base_id}_{suffix}" in used:
                suffix += 1
            unique_id = f"{base_id}_{suffix}"
            item = LoadedLoRAWeights(
                spec=item.spec.model_copy(update={"adapter_id": unique_id}),
                layer_factors=item.layer_factors,
            )
        used.add(unique_id)
        out.append(item)
    return out


def _resolve_torch_dtype(torch_mod: Any, dtype: Any) -> Any:
    if dtype is None:
        return torch_mod.float32
    if isinstance(dtype, str):
        norm = dtype.strip().lower()
        mapping: dict[str, Any] = {
            "fp16": torch_mod.float16,
            "f16": torch_mod.float16,
            "float16": torch_mod.float16,
            "bf16": torch_mod.bfloat16,
            "bfloat16": torch_mod.bfloat16,
            "fp32": torch_mod.float32,
            "f32": torch_mod.float32,
            "float32": torch_mod.float32,
        }
        if norm not in mapping:
            raise ConfigError(f"Unsupported torch dtype '{dtype}'. Supported: fp16, bf16, fp32.")
        return mapping[norm]
    return dtype


@dataclass(frozen=True)
class ArrowPrototypeBank:
    """Layer-indexed Arrow unit prototypes and zero-padded heterogeneous LoRA weight banks."""

    adapter_ids: tuple[str, ...]
    layer_names: tuple[str, ...]
    prototypes: dict[str, Matrix]
    a_packed: dict[str, list[Matrix]]
    b_packed: dict[str, list[Matrix]]
    ranks: dict[str, list[int]]
    scalings: dict[str, list[float]]
    in_features: dict[str, int]
    out_features: dict[str, int]
    warnings: tuple[str, ...]

    def layer_tensors(
        self,
        layer_name: str,
        *,
        device: str | Any = "cpu",
        dtype: Any = None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Materialize `(prototypes, a_packed, b_packed, ranks, scalings)` tensors for a layer."""
        if layer_name not in self.prototypes:
            raise ConfigError(
                f"Unknown layer '{layer_name}' in ArrowPrototypeBank. "
                f"Available layers: {list(self.layer_names)}."
            )
        torch_mod = optional_torch()
        if torch_mod is None:
            raise RuntimeExecutionError(
                "PyTorch is required for ArrowPrototypeBank.layer_tensors(); "
                "install `loraspec[kernels]`."
            )
        target_dtype = _resolve_torch_dtype(torch_mod, dtype)
        prototypes_t = torch_mod.tensor(
            self.prototypes[layer_name], dtype=torch_mod.float32, device=device
        )
        a_t = torch_mod.tensor(self.a_packed[layer_name], dtype=target_dtype, device=device)
        b_t = torch_mod.tensor(self.b_packed[layer_name], dtype=target_dtype, device=device)
        ranks_t = torch_mod.tensor(self.ranks[layer_name], dtype=torch_mod.int32, device=device)
        scalings_t = torch_mod.tensor(
            self.scalings[layer_name], dtype=torch_mod.float32, device=device
        )
        return prototypes_t, a_t, b_t, ranks_t, scalings_t


def extract_arrow_prototypes(
    loaded_adapters: Sequence[LoadedLoRAWeights],
    *,
    gks_adapter: LoadedLoRAWeights | None = None,
    gks_lambda: float = 1.0,
    layer_policy: LayerPolicy = "union",
    backend: MathBackend = "auto",
) -> ArrowPrototypeBank:
    """Extract normalized dominant right-singular Arrow prototypes per layer in O((d_out+d_in)r)."""
    backend = validate_math_backend(backend)
    if not loaded_adapters:
        raise ConfigError("At least one adapter is required for Arrow prototype extraction.")
    if layer_policy not in {"union", "intersection"}:
        raise ConfigError(
            f"Invalid layer_policy '{layer_policy}'; expected 'union' or 'intersection'."
        )

    deduped = _deduplicate_loaded_ids(loaded_adapters)
    for item in deduped:
        if item.spec.use_dora:
            raise AdapterParseError(
                f"Cannot extract Arrow prototypes from DoRA adapter '{item.spec.adapter_id}': "
                "DoRA magnitude vectors are unsupported. Export standard LoRA (use_dora=False)."
            )
        if not item.spec.layers:
            raise AdapterParseError(f"Adapter '{item.spec.adapter_id}' contains zero LoRA layers.")

    base_models = {
        item.spec.base_model_name_or_path
        for item in deduped
        if item.spec.base_model_name_or_path is not None
    }
    if len(base_models) > 1:
        raise AdapterParseError(
            "Cannot extract Arrow prototypes for adapters with conflicting base models: "
            f"{sorted(base_models)}."
        )

    layer_dims: dict[str, tuple[int, int]] = {}
    for item in deduped:
        for layer in item.spec.layers:
            dims = (layer.in_features, layer.out_features)
            prev = layer_dims.get(layer.layer_name)
            if prev is not None and prev != dims:
                raise AdapterParseError(
                    f"Dimension mismatch on layer '{layer.layer_name}' in "
                    f"'{item.spec.adapter_id}': got {dims}, expected {prev}."
                )
            layer_dims[layer.layer_name] = dims
            mat_a, mat_b = item.layer_factors[layer.layer_name]
            if has_non_finite_matrix(mat_a) or has_non_finite_matrix(mat_b):
                raise AdapterParseError(
                    f"Non-finite (NaN/Inf) weights detected in '{item.spec.adapter_id}' "
                    f"layer '{layer.layer_name}'."
                )

    layer_sets = [set(item.layer_factors) for item in deduped]
    if layer_policy == "union":
        union_layers: set[str] = set()
        for lset in layer_sets:
            union_layers |= lset
        resolved_layers = sorted(union_layers)
    else:
        inter_layers = set(layer_sets[0])
        for lset in layer_sets[1:]:
            inter_layers &= lset
        resolved_layers = sorted(inter_layers)

    if not resolved_layers:
        raise AdapterParseError(
            f"No layers survived layer_policy='{layer_policy}' across adapters "
            f"{[item.spec.adapter_id for item in deduped]}."
        )

    if gks_adapter is not None:
        working = apply_genknowsub(deduped, gks_adapter, gks_lambda=gks_lambda)
    else:
        working = list(deduped)

    prototypes: dict[str, Matrix] = {}
    a_packed: dict[str, list[Matrix]] = {}
    b_packed: dict[str, list[Matrix]] = {}
    ranks: dict[str, list[int]] = {}
    scalings: dict[str, list[float]] = {}
    in_features_map: dict[str, int] = {}
    out_features_map: dict[str, int] = {}
    warnings_list: list[str] = []

    for lname in resolved_layers:
        d_in, d_out = layer_dims[lname]
        in_features_map[lname] = d_in
        out_features_map[lname] = d_out

        layer_ranks_present = [
            len(item.layer_factors[lname][0]) for item in deduped if lname in item.layer_factors
        ]
        r_max = max(layer_ranks_present, default=1)

        proto_rows: Matrix = []
        a_rows_packed: list[Matrix] = []
        b_rows_packed: list[Matrix] = []
        rank_vec: list[int] = []
        scale_vec: list[float] = []

        for orig_item, work_item in zip(deduped, working, strict=True):
            adapter_id = orig_item.spec.adapter_id
            if lname not in orig_item.layer_factors:
                proto_rows.append([0.0] * d_in)
                a_rows_packed.append([[0.0] * d_in for _ in range(r_max)])
                b_rows_packed.append([[0.0] * r_max for _ in range(d_out)])
                rank_vec.append(0)
                scale_vec.append(0.0)
                warnings_list.append(
                    f"Adapter '{adapter_id}' omits layer '{lname}' under "
                    "layer_policy='union'; emitted zero prototype and zero weights."
                )
                continue

            a_orig, b_orig = orig_item.layer_factors[lname]
            r_m = len(a_orig)
            scale_m = get_layer_scaling(orig_item.spec, lname)
            rank_vec.append(r_m)
            scale_vec.append(scale_m)

            a_pad: Matrix = [list(map(float, row)) for row in a_orig]
            for _ in range(r_max - r_m):
                a_pad.append([0.0] * d_in)
            a_rows_packed.append(a_pad)

            b_pad: Matrix = [[float(v) for v in row] + ([0.0] * (r_max - r_m)) for row in b_orig]
            b_rows_packed.append(b_pad)

            work_lspec = next(ls for ls in work_item.spec.layers if ls.layer_name == lname)
            a_work, b_work = work_item.layer_factors[lname]
            _, factors = compute_layer_spectral_analysis(
                adapter_id=work_item.spec.adapter_id,
                layer_spec=work_lspec,
                a_mat=a_work,
                b_mat=b_work,
                scaling=get_layer_scaling(work_item.spec, lname),
                backend=backend,
            )

            if (
                not factors.has_non_finite
                and factors.active_sigmas
                and factors.active_sigmas[0] > 1e-12
                and factors.v_active
                and factors.v_active[0]
            ):
                v1 = [float(row[0]) for row in factors.v_active]
                norm_v1 = math.sqrt(sum(v * v for v in v1))
                if norm_v1 > 1e-12 and math.isfinite(norm_v1):
                    proto_rows.append([v / norm_v1 for v in v1])
                else:
                    proto_rows.append([0.0] * d_in)
                    warnings_list.append(
                        f"Adapter '{adapter_id}' layer '{lname}' yielded degenerate zero-norm "
                        "top singular vector; emitted zero prototype."
                    )
            else:
                proto_rows.append([0.0] * d_in)
                warnings_list.append(
                    f"Adapter '{adapter_id}' layer '{lname}' has zero spectral energy; "
                    "emitted zero prototype vector."
                )

        prototypes[lname] = proto_rows
        a_packed[lname] = a_rows_packed
        b_packed[lname] = b_rows_packed
        ranks[lname] = rank_vec
        scalings[lname] = scale_vec

    return ArrowPrototypeBank(
        adapter_ids=tuple(item.spec.adapter_id for item in deduped),
        layer_names=tuple(resolved_layers),
        prototypes=prototypes,
        a_packed=a_packed,
        b_packed=b_packed,
        ranks=ranks,
        scalings=scalings,
        in_features=in_features_map,
        out_features=out_features_map,
        warnings=tuple(warnings_list),
    )


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be an integer >= 1, got {value!r}.")
    return value


def _require_positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ConfigError(f"{name} must be finite and > 0.0, got {value!r}.")
    return float(value)


def compute_router_entropy(routing_weights: Any, *, eps: float = 1e-8) -> Any:
    """Compute clamped non-negative per-token Shannon entropy over Top-k routing weights."""
    eps_f = _require_positive_float(eps, "eps")

    if _is_torch_tensor(routing_weights):
        torch_mod = optional_torch()
        assert torch_mod is not None
        if routing_weights.ndim != 2 or int(routing_weights.shape[1]) < 1:
            raise ConfigError(
                "routing_weights must be 2D [T, k] with k >= 1, "
                f"got {tuple(routing_weights.shape)}."
            )
        if int(routing_weights.shape[0]) == 0:
            return torch_mod.empty((0,), dtype=torch_mod.float32, device=routing_weights.device)
        w_fp32 = routing_weights.to(dtype=torch_mod.float32)
        entropy = -(w_fp32 * torch_mod.log(w_fp32 + eps_f)).sum(dim=-1)
        return entropy.clamp(min=0.0)

    if not isinstance(routing_weights, Sequence) or isinstance(routing_weights, (str, bytes)):
        raise ConfigError("routing_weights must be a PyTorch Tensor or 2D sequence of floats.")
    rows = cast(Sequence[Sequence[Any]], routing_weights)
    if len(rows) == 0:
        return []
    k = len(rows[0])
    if k < 1:
        raise ConfigError("routing_weights rows must have length >= 1.")
    out: list[float] = []
    for idx, row in enumerate(rows):
        if len(row) != k:
            raise ConfigError(f"Ragged routing_weights row at index {idx}: expected {k}.")
        h_val = -sum(float(w) * math.log(float(w) + eps_f) for w in row)
        out.append(max(0.0, h_val))
    return out


def _coerce_python_matrix(seq: Sequence[Any], expected_cols: int | None = None) -> Matrix:
    if len(seq) == 0:
        return []
    first: Any = seq[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        raise ConfigError("Expected a 2D [T, d_in] or 3D [B, S, d_in] sequence.")
    first_seq = cast(Sequence[Any], first)

    if (
        len(first_seq) > 0
        and isinstance(first_seq[0], Sequence)
        and not isinstance(first_seq[0], (str, bytes))
    ):
        # 3D [B, S, d_in] -> flatten to [T, d_in]
        flat_rows: Matrix = []
        for batch_item in seq:
            if not isinstance(batch_item, Sequence) or isinstance(batch_item, (str, bytes)):
                raise ConfigError("Invalid nested 3D activation shape.")
            batch_seq = cast(Sequence[Any], batch_item)
            for row in batch_seq:
                if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                    raise ConfigError("Invalid row inside 3D activation tensor.")
                row_seq = cast(Sequence[Any], row)
                row_f = [float(v) for v in row_seq]
                if expected_cols is not None and len(row_f) != expected_cols:
                    raise ConfigError(
                        f"Activation feature dim mismatch: expected {expected_cols}, "
                        f"got {len(row_f)}."
                    )
                expected_cols = len(row_f)
                flat_rows.append(row_f)
        return flat_rows

    if len(first_seq) == 0 and any(
        isinstance(item, Sequence) and len(cast(Sequence[Any], item)) == 0 for item in seq
    ):
        if all(isinstance(item, Sequence) and len(cast(Sequence[Any], item)) == 0 for item in seq):
            return []
        raise ConfigError("Ragged or empty inner sequence in activations.")

    mat: Matrix = []
    for row in seq:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ConfigError("Expected 2D sequence rows.")
        row_seq = cast(Sequence[Any], row)
        row_f = [float(v) for v in row_seq]
        if expected_cols is not None and len(row_f) != expected_cols:
            raise ConfigError(
                f"Activation feature dim mismatch: expected {expected_cols}, got {len(row_f)}."
            )
        expected_cols = len(row_f)
        mat.append(row_f)
    return mat


def route_topk_tokens(
    x: Any,
    prototypes: Any,
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    compute_entropy: bool = True,
) -> tuple[Any, Any, Any]:
    """Compute sign-invariant cosine Top-k expert routing, softmax weights, and entropy."""
    top_k_val = _require_positive_int(top_k, "top_k")
    temp_f = _require_positive_float(temperature, "temperature")
    eps = 1e-8

    if _is_torch_tensor(x):
        torch_mod = optional_torch()
        assert torch_mod is not None
        if x.ndim not in (2, 3):
            raise ConfigError(
                "Activation tensor x must be 2D [T, d_in] or 3D [B, S, d_in], "
                f"got {tuple(x.shape)}."
            )
        d_in = int(x.shape[-1])
        x_2d = x.reshape(-1, d_in)
        num_tokens = int(x_2d.shape[0])
        device = x.device

        if _is_torch_tensor(prototypes):
            if prototypes.device != device:
                raise ConfigError(
                    f"Device mismatch between x ({device}) and prototypes ({prototypes.device})."
                )
            proto_t = prototypes
        else:
            if not isinstance(prototypes, Sequence) or isinstance(prototypes, (str, bytes)):
                raise ConfigError("prototypes must be a Tensor or 2D sequence.")
            proto_mat = _coerce_python_matrix(cast(Sequence[Any], prototypes))
            if not proto_mat:
                raise ConfigError("prototypes cannot be empty.")
            proto_t = torch_mod.tensor(proto_mat, dtype=torch_mod.float32, device=device)

        if proto_t.ndim != 2 or int(proto_t.shape[0]) < 1 or int(proto_t.shape[1]) < 1:
            raise ConfigError(
                f"prototypes must be 2D [K, d_in] with K>=1, d_in>=1, got {tuple(proto_t.shape)}."
            )
        num_experts, proto_dim = int(proto_t.shape[0]), int(proto_t.shape[1])
        if d_in != proto_dim:
            raise ConfigError(
                "Feature dimension mismatch: "
                f"x.shape[-1]={d_in} != prototypes.shape[1]={proto_dim}."
            )
        if top_k_val > num_experts:
            raise ConfigError(f"top_k={top_k_val} exceeds number of experts K={num_experts}.")

        if num_tokens == 0:
            return (
                torch_mod.empty((0, top_k_val), dtype=torch_mod.int32, device=device),
                torch_mod.empty((0, top_k_val), dtype=torch_mod.float32, device=device),
                torch_mod.empty((0,), dtype=torch_mod.float32, device=device),
            )

        x_fp32 = x_2d.to(dtype=torch_mod.float32)
        p_fp32 = (
            proto_t if proto_t.dtype == torch_mod.float32 else proto_t.to(dtype=torch_mod.float32)
        )
        raw_abs = torch_mod.abs(x_fp32 @ p_fp32.transpose(0, 1))  # [T, K] >= 0.0
        x_norm = torch_mod.linalg.vector_norm(x_fp32, dim=-1, keepdim=True).clamp(min=eps)
        scores = raw_abs / x_norm

        if bool(x_2d.is_cuda):
            # Stable descending argsort preserves smallest-index tie-breaking without a Python loop
            sorted_idx = torch_mod.argsort(scores, dim=-1, descending=True, stable=True)[
                :, :top_k_val
            ]
            top_scores = scores.gather(-1, sorted_idx)
            expert_ids = sorted_idx.to(dtype=torch_mod.int32)
        else:
            # Deterministic smallest-index tie-breaking in pure FP32 tensor ops (MPS/CPU safe)
            work_scores = scores.clone()
            k_indices = torch_mod.arange(num_experts, device=device, dtype=torch_mod.float32)[
                None, :
            ]
            sentinel = torch_mod.full(
                (num_tokens, num_experts),
                float(num_experts),
                dtype=torch_mod.float32,
                device=device,
            )
            neg_mask_val = torch_mod.full(
                (num_tokens, 1), -1.0, dtype=torch_mod.float32, device=device
            )
            selected_ids: list[Any] = []
            selected_scores: list[Any] = []
            for _ in range(top_k_val):
                max_vals = work_scores.max(dim=-1, keepdim=True).values
                is_max = work_scores == max_vals
                cand_idx = torch_mod.where(is_max, k_indices, sentinel)
                best_idx_long = cand_idx.min(dim=-1).values.to(dtype=torch_mod.int64)
                best_val = scores.gather(-1, best_idx_long[:, None])
                selected_ids.append(best_idx_long.to(dtype=torch_mod.int32))
                selected_scores.append(best_val.squeeze(-1))
                work_scores.scatter_(-1, best_idx_long[:, None], neg_mask_val)

            expert_ids = torch_mod.stack(selected_ids, dim=-1)  # [T, top_k] int32
            top_scores = torch_mod.stack(selected_scores, dim=-1)  # [T, top_k] float32

        scaled = top_scores / temp_f
        scaled_shifted = scaled - scaled.max(dim=-1, keepdim=True).values
        exp_vals = torch_mod.exp(scaled_shifted)
        routing_weights = exp_vals / exp_vals.sum(dim=-1, keepdim=True)
        entropy = (
            compute_router_entropy(routing_weights, eps=eps)
            if compute_entropy
            else torch_mod.empty((0,), dtype=torch_mod.float32, device=device)
        )
        return expert_ids, routing_weights, entropy

    # Pure-Python path
    if _is_torch_tensor(prototypes):
        proto_mat = cast(Matrix, prototypes.detach().cpu().tolist())
    elif isinstance(prototypes, Sequence) and not isinstance(prototypes, (str, bytes)):
        proto_mat = _coerce_python_matrix(cast(Sequence[Any], prototypes))
    else:
        raise ConfigError("prototypes must be a PyTorch Tensor or 2D sequence.")

    if not proto_mat or not proto_mat[0]:
        raise ConfigError("prototypes must be non-empty 2D [K, d_in].")
    num_experts = len(proto_mat)
    proto_dim = len(proto_mat[0])
    if top_k_val > num_experts:
        raise ConfigError(f"top_k={top_k_val} exceeds number of experts K={num_experts}.")

    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)):
        raise ConfigError("x must be a PyTorch Tensor or 2D/3D sequence.")
    x_mat = _coerce_python_matrix(cast(Sequence[Any], x), expected_cols=proto_dim)
    if not x_mat:
        return [], [], []

    expert_ids_py: list[list[int]] = []
    weights_py: list[list[float]] = []
    for row in x_mat:
        norm = math.sqrt(sum(v * v for v in row))
        denom = max(norm, eps)
        x_hat_row = [v / denom for v in row]
        scores_row = [
            abs(sum(xv * pv for xv, pv in zip(x_hat_row, proto_row, strict=True)))
            for proto_row in proto_mat
        ]
        ranked = sorted(range(num_experts), key=lambda e: (-scores_row[e], e))
        chosen = ranked[:top_k_val]
        chosen_scores = [scores_row[e] / temp_f for e in chosen]
        max_s = max(chosen_scores)
        exps = [math.exp(s - max_s) for s in chosen_scores]
        sum_exp = sum(exps)
        probs = [ev / sum_exp for ev in exps]
        expert_ids_py.append(chosen)
        weights_py.append(probs)

    entropy_py = cast(list[float], compute_router_entropy(weights_py, eps=eps))
    return expert_ids_py, weights_py, entropy_py
