"""Aligned KnOTS combination operators (linear, TIES, DARE-TIES) and GenKnowSub subtraction."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, Literal, cast

from loraspec.core.errors import AdapterParseError, ConfigError
from loraspec.core.schemas import LoRAAdapterSpec, LoRALayerSpec
from loraspec.core.spectral_math import MathBackend, Matrix
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights
from loraspec.knots.aligner import (
    FactorizedKnOTSResult,
    has_non_finite_matrix,
    optional_numpy,
    scale_matrix,
)

KnotsMethod = Literal["knots_ties", "knots_dare_ties", "knots_linear"]


def normalize_weights(num_adapters: int, weights: Sequence[float] | None) -> list[float]:
    if num_adapters <= 0:
        raise ConfigError("At least one adapter is required.")
    if weights is None:
        return [1.0 / float(num_adapters)] * num_adapters
    if len(weights) != num_adapters:
        raise ConfigError(f"Expected {num_adapters} merge weights, got {len(weights)}.")
    parsed: list[float] = []
    for w in weights:
        if isinstance(w, bool) or not math.isfinite(float(w)) or float(w) < 0.0:
            raise ConfigError(f"Merge weights must be finite and non-negative, got {w!r}.")
        parsed.append(float(w))
    total = sum(parsed)
    if total <= 0.0:
        raise ConfigError("Sum of merge weights must be strictly positive.")
    return [w / total for w in parsed]


def _validate_ties_density(density: float) -> float:
    if isinstance(density, bool) or not math.isfinite(float(density)) or not (0.0 < density <= 1.0):
        raise ConfigError(f"ties_density must be in (0.0, 1.0], got {density!r}.")
    return float(density)


def _validate_dare_drop_rate(drop_rate: float) -> float:
    if (
        isinstance(drop_rate, bool)
        or not math.isfinite(float(drop_rate))
        or not (0.0 <= drop_rate < 1.0)
    ):
        raise ConfigError(f"dare_drop_rate must be in [0.0, 1.0), got {drop_rate!r}.")
    return float(drop_rate)


def _validate_aligned_shapes(aligned_v_list: Sequence[Matrix]) -> tuple[int, int]:
    if not aligned_v_list:
        raise ConfigError("aligned_v_list cannot be empty.")
    rows = len(aligned_v_list[0])
    cols = len(aligned_v_list[0][0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        raise ConfigError("Aligned V^T factors must be non-empty.")
    for idx, mat in enumerate(aligned_v_list):
        if len(mat) != rows or any(len(r) != cols for r in mat):
            raise ConfigError(
                f"Inconsistent aligned factor shape at index {idx}: expected ({rows}, {cols})."
            )
    return rows, cols


def apply_genknowsub(
    loaded_adapters: Sequence[LoadedLoRAWeights],
    gks_adapter: LoadedLoRAWeights,
    *,
    gks_lambda: float = 1.0,
) -> list[LoadedLoRAWeights]:
    """Subtract general-knowledge direction lambda * dW_0 in factorized rank space."""
    if not loaded_adapters:
        raise AdapterParseError("At least one task adapter is required for GenKnowSub.")
    if isinstance(gks_lambda, bool) or not math.isfinite(float(gks_lambda)):
        raise ConfigError(f"gks_lambda must be a finite float, got {gks_lambda!r}.")

    for lname, (a_0, b_0) in gks_adapter.layer_factors.items():
        if has_non_finite_matrix(a_0) or has_non_finite_matrix(b_0):
            raise AdapterParseError(
                f"Non-finite (NaN/Inf) weights in GenKnowSub adapter layer '{lname}'."
            )

    gks_base = gks_adapter.spec.base_model_name_or_path
    s_0 = gks_adapter.spec.scaling
    lam = float(gks_lambda)

    purified: list[LoadedLoRAWeights] = []
    for item in loaded_adapters:
        spec = item.spec
        if (
            gks_base is not None
            and spec.base_model_name_or_path is not None
            and gks_base != spec.base_model_name_or_path
        ):
            raise AdapterParseError(
                f"GenKnowSub base model '{gks_base}' does not match adapter "
                f"'{spec.adapter_id}' base model '{spec.base_model_name_or_path}'."
            )

        new_layers: list[LoRALayerSpec] = []
        new_factors: dict[str, tuple[Matrix, Matrix]] = {}
        total_params = 0

        for layer_spec in spec.layers:
            lname = layer_spec.layer_name
            a_m, b_m = item.layer_factors[lname]
            if has_non_finite_matrix(a_m) or has_non_finite_matrix(b_m):
                raise AdapterParseError(
                    f"Non-finite weights in adapter '{spec.adapter_id}' layer '{lname}'."
                )

            b_scaled = scale_matrix(b_m, spec.scaling)
            if lname in gks_adapter.layer_factors:
                a_0, b_0 = gks_adapter.layer_factors[lname]
                d_out = len(b_m)
                d_in = len(a_m[0])
                if len(b_0) != d_out or len(a_0[0]) != d_in:
                    raise AdapterParseError(
                        f"GenKnowSub dimension mismatch on layer '{lname}' for "
                        f"'{spec.adapter_id}': expected ({d_out}, {d_in}), "
                        f"got ({len(b_0)}, {len(a_0[0])})."
                    )
                neg_b0 = scale_matrix(b_0, -lam * s_0)
                b_aug = [rb + r0 for rb, r0 in zip(b_scaled, neg_b0, strict=True)]
                a_aug = [list(row) for row in a_m] + [list(row) for row in a_0]
            else:
                b_aug = b_scaled
                a_aug = [list(row) for row in a_m]

            new_rank = len(a_aug)
            d_in = len(a_aug[0])
            d_out = len(b_aug)
            param_count = new_rank * d_in + d_out * new_rank
            total_params += param_count
            new_factors[lname] = (a_aug, b_aug)
            new_layers.append(
                layer_spec.model_copy(
                    update={
                        "rank": new_rank,
                        "a_shape": (new_rank, d_in),
                        "b_shape": (d_out, new_rank),
                        "param_count": param_count,
                    }
                )
            )

        first_rank = new_layers[0].rank if new_layers else spec.rank
        new_spec: LoRAAdapterSpec = spec.model_copy(
            update={
                "rank": first_rank,
                "lora_alpha": float(first_rank),
                "scaling": 1.0,
                "use_rslora": False,
                "total_lora_params": total_params,
                "layers": new_layers,
                "metadata": {
                    **spec.metadata,
                    "genknowsub_applied": True,
                    "genknowsub_lambda": lam,
                },
            }
        )
        purified.append(LoadedLoRAWeights(spec=new_spec, layer_factors=new_factors))

    return purified


def knots_linear(
    aligned_v_list: Sequence[Matrix],
    weights: Sequence[float] | None = None,
    *,
    backend: MathBackend = "auto",
) -> Matrix:
    """Weighted linear combination in aligned V^T-space."""
    rows, cols = _validate_aligned_shapes(aligned_v_list)
    norm_w = normalize_weights(len(aligned_v_list), weights)

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            acc = np_mod.zeros((rows, cols), dtype=np_mod.float64)
            for mat, w in zip(aligned_v_list, norm_w, strict=True):
                acc += w * np_mod.asarray(mat, dtype=np_mod.float64)
            return cast(Matrix, acc.tolist())

    out: Matrix = [[0.0] * cols for _ in range(rows)]
    for mat, w in zip(aligned_v_list, norm_w, strict=True):
        if w == 0.0:
            continue
        for r in range(rows):
            row_src = mat[r]
            row_dst = out[r]
            for c in range(cols):
                row_dst[c] += w * row_src[c]
    return out


def _trim_ties_numpy(np_mod: Any, arr: Any, density: float) -> Any:
    if density >= 1.0:
        return arr.copy()
    total = int(arr.size)
    keep_k = max(1, math.ceil(density * total))
    cutoff_idx = max(0, total - keep_k)
    sorted_abs = np_mod.sort(np_mod.abs(arr).ravel())
    thresh = float(sorted_abs[cutoff_idx])
    if thresh == 0.0:
        return arr.copy()
    mask = (arr != 0.0) & (np_mod.abs(arr) >= thresh)
    return np_mod.where(mask, arr, 0.0)


def _trim_ties_python(mat: Matrix, density: float) -> Matrix:
    if density >= 1.0:
        return [list(row) for row in mat]
    abs_vals = sorted(abs(v) for row in mat for v in row)
    total = len(abs_vals)
    keep_k = max(1, math.ceil(density * total))
    cutoff_idx = max(0, total - keep_k)
    thresh = abs_vals[cutoff_idx]
    if thresh == 0.0:
        return [list(row) for row in mat]
    return [[v if (v != 0.0 and abs(v) >= thresh) else 0.0 for v in row] for row in mat]


def _ties_elect_and_merge_numpy(
    np_mod: Any,
    trimmed_arrs: list[Any],
    norm_w: list[float],
) -> Matrix:
    stacked = np_mod.stack(trimmed_arrs, axis=0)  # [K, R_sum, d_in]
    weights_arr = np_mod.asarray(norm_w, dtype=np_mod.float64)[:, None, None]
    weighted_sum = np_mod.sum(stacked * weights_arr, axis=0)
    elected_sign = np_mod.sign(weighted_sum)  # -1, 0, +1

    agree = (stacked != 0.0) & (np_mod.sign(stacked) == elected_sign[None, :, :])
    agree_weights = np_mod.where(agree, weights_arr, 0.0)
    denom = np_mod.sum(agree_weights, axis=0)
    numer = np_mod.sum(stacked * agree_weights, axis=0)
    merged = np_mod.divide(numer, denom, out=np_mod.zeros_like(numer), where=denom > 0.0)
    return cast(Matrix, merged.tolist())


def _ties_elect_and_merge_python(
    trimmed_mats: list[Matrix],
    norm_w: list[float],
    rows: int,
    cols: int,
) -> Matrix:
    out: Matrix = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        out_row = out[r]
        for c in range(cols):
            wsum = 0.0
            for mat, w in zip(trimmed_mats, norm_w, strict=True):
                wsum += w * mat[r][c]
            if wsum > 0.0:
                elected = 1
            elif wsum < 0.0:
                elected = -1
            else:
                continue

            numer = 0.0
            denom = 0.0
            for mat, w in zip(trimmed_mats, norm_w, strict=True):
                val = mat[r][c]
                if val == 0.0:
                    continue
                val_sign = 1 if val > 0.0 else -1
                if val_sign == elected:
                    numer += w * val
                    denom += w
            if denom > 0.0:
                out_row[c] = numer / denom
    return out


def knots_ties(
    aligned_v_list: Sequence[Matrix],
    weights: Sequence[float] | None = None,
    *,
    density: float = 0.50,
    backend: MathBackend = "auto",
) -> Matrix:
    """TIES (Trim, Elect Sign, Disjoint Merge) in aligned KnOTS V^T-space."""
    rows, cols = _validate_aligned_shapes(aligned_v_list)
    norm_w = normalize_weights(len(aligned_v_list), weights)
    valid_density = _validate_ties_density(density)

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            trimmed = [
                _trim_ties_numpy(np_mod, np_mod.asarray(mat, dtype=np_mod.float64), valid_density)
                for mat in aligned_v_list
            ]
            return _ties_elect_and_merge_numpy(np_mod, trimmed, norm_w)

    trimmed_py = [_trim_ties_python(mat, valid_density) for mat in aligned_v_list]
    return _ties_elect_and_merge_python(trimmed_py, norm_w, rows, cols)


def knots_dare_ties(
    aligned_v_list: Sequence[Matrix],
    weights: Sequence[float] | None = None,
    *,
    density: float = 0.50,
    drop_rate: float = 0.20,
    rng_seed: int = 42,
    backend: MathBackend = "auto",
) -> Matrix:
    """DARE-TIES (deterministic seeded Bernoulli drop+rescale + TIES) in aligned V^T-space."""
    rows, cols = _validate_aligned_shapes(aligned_v_list)
    norm_w = normalize_weights(len(aligned_v_list), weights)
    valid_density = _validate_ties_density(density)
    valid_drop = _validate_dare_drop_rate(drop_rate)

    if valid_drop == 0.0:
        return knots_ties(aligned_v_list, weights=norm_w, density=valid_density, backend=backend)

    keep_prob = 1.0 - valid_drop
    inv_keep = 1.0 / keep_prob

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            rng = np_mod.random.default_rng(rng_seed)
            trimmed_np: list[Any] = []
            for mat in aligned_v_list:
                arr = np_mod.asarray(mat, dtype=np_mod.float64)
                mask = rng.random((rows, cols)) < keep_prob
                rescaled = np_mod.where(mask, arr * inv_keep, 0.0)
                trimmed_np.append(_trim_ties_numpy(np_mod, rescaled, valid_density))
            return _ties_elect_and_merge_numpy(np_mod, trimmed_np, norm_w)

    rng_py = random.Random(rng_seed)
    trimmed_py: list[Matrix] = []
    for mat in aligned_v_list:
        dropped: Matrix = [
            [val * inv_keep if rng_py.random() < keep_prob else 0.0 for val in row] for row in mat
        ]
        trimmed_py.append(_trim_ties_python(dropped, valid_density))
    return _ties_elect_and_merge_python(trimmed_py, norm_w, rows, cols)


def _compute_aligned_sign_conflict(
    v_list: Sequence[Matrix],
    sigmas: Sequence[float],
) -> float:
    """Compute sigma^2-weighted pairwise sign conflict across aligned V_m^T matrices."""
    num_adapters = len(v_list)
    if num_adapters < 2:
        return 0.0

    np_mod = optional_numpy()
    if np_mod is not None:
        sig_sq = np_mod.asarray([s * s for s in sigmas], dtype=np_mod.float64)[:, None]
        arrs = [np_mod.asarray(v, dtype=np_mod.float64) for v in v_list]
        neg = 0.0
        tot = 0.0
        for i in range(num_adapters):
            for j in range(i + 1, num_adapters):
                prod = arrs[i] * arrs[j]
                weighted = sig_sq * np_mod.abs(prod)
                tot += float(np_mod.sum(weighted))
                neg += float(np_mod.sum(weighted[prod < 0.0]))
        return neg / tot if tot > 1e-12 else 0.0

    neg_py = 0.0
    tot_py = 0.0
    rows = len(v_list[0])
    cols = len(v_list[0][0]) if rows else 0
    for i in range(num_adapters):
        for j in range(i + 1, num_adapters):
            vi = v_list[i]
            vj = v_list[j]
            for r in range(rows):
                sw = sigmas[r] * sigmas[r]
                if sw <= 0.0:
                    continue
                row_i = vi[r]
                row_j = vj[r]
                for c in range(cols):
                    prod = row_i[c] * row_j[c]
                    wabs = sw * abs(prod)
                    tot_py += wabs
                    if prod < 0.0:
                        neg_py += wabs
    return neg_py / tot_py if tot_py > 1e-12 else 0.0


def merge_aligned_factors(
    aligned_res: FactorizedKnOTSResult,
    *,
    method: KnotsMethod = "knots_ties",
    weights: Sequence[float] | None = None,
    ties_density: float = 0.50,
    dare_drop_rate: float = 0.20,
    rng_seed: int = 42,
    layer_seed_offset: int = 0,
    backend: MathBackend = "auto",
) -> tuple[Matrix, float]:
    """Merge aligned V_m^T factors and return (V_merged^T, post_alignment_sign_conflict)."""
    if method not in {"knots_ties", "knots_dare_ties", "knots_linear"}:
        raise ConfigError(
            f"Unsupported KnOTS merge method '{method}'. "
            "Choose from: knots_ties, knots_dare_ties, knots_linear."
        )

    rows, cols = _validate_aligned_shapes(aligned_res.aligned_v_list)
    norm_w = normalize_weights(len(aligned_res.aligned_v_list), weights)
    effective_seed = rng_seed + layer_seed_offset

    if method == "knots_linear":
        v_merged_t = knots_linear(aligned_res.aligned_v_list, weights=norm_w, backend=backend)
        post_conflict = _compute_aligned_sign_conflict(
            aligned_res.aligned_v_list, aligned_res.sigmas
        )
        return v_merged_t, round(post_conflict, 6)

    valid_density = _validate_ties_density(ties_density)
    if method == "knots_ties":
        if backend in ("auto", "numpy"):
            np_mod = optional_numpy()
            if np_mod is not None:
                trimmed_np = [
                    _trim_ties_numpy(
                        np_mod, np_mod.asarray(mat, dtype=np_mod.float64), valid_density
                    )
                    for mat in aligned_res.aligned_v_list
                ]
                v_merged_t = _ties_elect_and_merge_numpy(np_mod, trimmed_np, norm_w)
                trimmed_lists = [cast(Matrix, a.tolist()) for a in trimmed_np]
                post_conflict = _compute_aligned_sign_conflict(trimmed_lists, aligned_res.sigmas)
                return v_merged_t, round(post_conflict, 6)

        trimmed_py = [_trim_ties_python(mat, valid_density) for mat in aligned_res.aligned_v_list]
        v_merged_t = _ties_elect_and_merge_python(trimmed_py, norm_w, rows, cols)
        post_conflict = _compute_aligned_sign_conflict(trimmed_py, aligned_res.sigmas)
        return v_merged_t, round(post_conflict, 6)

    # knots_dare_ties
    valid_drop = _validate_dare_drop_rate(dare_drop_rate)
    keep_prob = 1.0 - valid_drop
    inv_keep = 1.0 / keep_prob if valid_drop > 0.0 else 1.0

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            rng = np_mod.random.default_rng(effective_seed)
            trimmed_dare_np: list[Any] = []
            for mat in aligned_res.aligned_v_list:
                arr = np_mod.asarray(mat, dtype=np_mod.float64)
                if valid_drop > 0.0:
                    mask = rng.random((rows, cols)) < keep_prob
                    rescaled = np_mod.where(mask, arr * inv_keep, 0.0)
                else:
                    rescaled = arr.copy()
                trimmed_dare_np.append(_trim_ties_numpy(np_mod, rescaled, valid_density))
            v_merged_t = _ties_elect_and_merge_numpy(np_mod, trimmed_dare_np, norm_w)
            trimmed_lists = [cast(Matrix, a.tolist()) for a in trimmed_dare_np]
            post_conflict = _compute_aligned_sign_conflict(trimmed_lists, aligned_res.sigmas)
            return v_merged_t, round(post_conflict, 6)

    rng_py = random.Random(effective_seed)
    trimmed_dare_py: list[Matrix] = []
    for mat in aligned_res.aligned_v_list:
        if valid_drop > 0.0:
            dropped = [
                [val * inv_keep if rng_py.random() < keep_prob else 0.0 for val in row]
                for row in mat
            ]
        else:
            dropped = [list(row) for row in mat]
        trimmed_dare_py.append(_trim_ties_python(dropped, valid_density))
    v_merged_t = _ties_elect_and_merge_python(trimmed_dare_py, norm_w, rows, cols)
    post_conflict = _compute_aligned_sign_conflict(trimmed_dare_py, aligned_res.sigmas)
    return v_merged_t, round(post_conflict, 6)
