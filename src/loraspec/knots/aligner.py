"""Factorized KnOTS joint-SVD subspace alignment engine (SDD-02, zero dense d^2 allocs)."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from loraspec.core.errors import AdapterParseError, ConfigError
from loraspec.core.spectral_math import (
    MathBackend,
    Matrix,
    small_svd,
    thin_qr,
)
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights

LayerPolicy = Literal["union", "intersection"]


def optional_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def has_non_finite_matrix(mat: Matrix) -> bool:
    return any(not all(map(math.isfinite, row)) for row in mat)


def transpose_matrix(mat: Matrix) -> Matrix:
    if not mat or not mat[0]:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def matmul_matrices(a: Matrix, b: Matrix) -> Matrix:
    rows_a = len(a)
    cols_a = len(a[0]) if rows_a else 0
    cols_b = len(b[0]) if b else 0
    out: Matrix = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        row_a = a[i]
        row_out = out[i]
        for k in range(cols_a):
            aik = row_a[k]
            if aik == 0.0:
                continue
            row_b = b[k]
            for j in range(cols_b):
                row_out[j] += aik * row_b[j]
    return out


def scale_matrix(mat: Matrix, scalar: float) -> Matrix:
    return [[val * scalar for val in row] for row in mat]


@dataclass(frozen=True)
class FactorizedKnOTSResult:
    """Factorized shared left basis, singular values, and aligned task right factors."""

    layer_name: str
    u_shared: Matrix  # [d_out, k_basis]
    sigmas: list[float]  # [k_basis]
    aligned_v_list: list[Matrix]  # K matrices of [k_basis, d_in] representing V_m^T
    adapter_ids: list[str]
    total_sum_rank: int
    pre_merge_sign_conflict: float


def _hstack_matrices(mats: Sequence[Matrix]) -> Matrix:
    rows = len(mats[0])
    out: Matrix = [[] for _ in range(rows)]
    for mat in mats:
        for r in range(rows):
            out[r].extend(mat[r])
    return out


def _slice_cols(mat: Matrix, start: int, end: int) -> Matrix:
    return [row[start:end] for row in mat]


def _compute_param_sign_conflict(
    scaled_bs: Sequence[Matrix],
    a_mats: Sequence[Matrix],
) -> float:
    """Compute magnitude-weighted parameter-space sign conflict prior to KnOTS alignment."""
    num_adapters = len(scaled_bs)
    if num_adapters < 2:
        return 0.0

    np_mod = optional_numpy()
    if np_mod is not None:
        b_arrs = [np_mod.asarray(b, dtype=np_mod.float64) for b in scaled_bs]
        a_arrs = [np_mod.asarray(a, dtype=np_mod.float64) for a in a_mats]
        b_neg = 0.0
        b_tot = 0.0
        a_neg = 0.0
        a_tot = 0.0
        for i in range(num_adapters):
            for j in range(i + 1, num_adapters):
                min_r = min(int(a_arrs[i].shape[0]), int(a_arrs[j].shape[0]))
                if min_r <= 0:
                    continue
                prod_b = b_arrs[i][:, :min_r] * b_arrs[j][:, :min_r]
                abs_b = np_mod.abs(prod_b)
                b_tot += float(np_mod.sum(abs_b))
                b_neg += float(np_mod.sum(abs_b[prod_b < 0.0]))

                prod_a = a_arrs[i][:min_r, :] * a_arrs[j][:min_r, :]
                abs_a = np_mod.abs(prod_a)
                a_tot += float(np_mod.sum(abs_a))
                a_neg += float(np_mod.sum(abs_a[prod_a < 0.0]))

        parts: list[float] = []
        if b_tot > 1e-12:
            parts.append(b_neg / b_tot)
        if a_tot > 1e-12:
            parts.append(a_neg / a_tot)
        return sum(parts) / float(len(parts)) if parts else 0.0

    b_neg_py = 0.0
    b_tot_py = 0.0
    a_neg_py = 0.0
    a_tot_py = 0.0
    for i in range(num_adapters):
        for j in range(i + 1, num_adapters):
            min_r = min(len(a_mats[i]), len(a_mats[j]))
            if min_r <= 0:
                continue
            bi = scaled_bs[i]
            bj = scaled_bs[j]
            for r in range(len(bi)):
                row_i = bi[r]
                row_j = bj[r]
                for c in range(min_r):
                    prod = row_i[c] * row_j[c]
                    ap = abs(prod)
                    b_tot_py += ap
                    if prod < 0.0:
                        b_neg_py += ap

            ai = a_mats[i]
            aj = a_mats[j]
            for r in range(min_r):
                row_i = ai[r]
                row_j = aj[r]
                for vi, vj in zip(row_i, row_j, strict=True):
                    prod = vi * vj
                    ap = abs(prod)
                    a_tot_py += ap
                    if prod < 0.0:
                        a_neg_py += ap

    parts_py: list[float] = []
    if b_tot_py > 1e-12:
        parts_py.append(b_neg_py / b_tot_py)
    if a_tot_py > 1e-12:
        parts_py.append(a_neg_py / a_tot_py)
    return sum(parts_py) / float(len(parts_py)) if parts_py else 0.0


def harmonize_heterogeneous_ranks(
    loaded_adapters: Sequence[LoadedLoRAWeights],
    *,
    layer_policy: LayerPolicy = "union",
) -> list[str]:
    """Validate base compatibility and heterogeneous-rank layer schema, returning target layers."""
    if len(loaded_adapters) < 2:
        raise ConfigError(
            f"KnOTS alignment requires at least 2 adapters, received {len(loaded_adapters)}."
        )
    if layer_policy not in {"union", "intersection"}:
        raise ConfigError(
            f"Invalid layer_policy '{layer_policy}'; expected 'union' or 'intersection'."
        )

    base_models = {
        item.spec.base_model_name_or_path
        for item in loaded_adapters
        if item.spec.base_model_name_or_path is not None
    }
    if len(base_models) > 1:
        raise AdapterParseError(
            f"Cannot KnOTS-merge adapters with conflicting base models: {sorted(base_models)}."
        )

    layer_dims: dict[str, tuple[int, int]] = {}
    for item in loaded_adapters:
        if not item.spec.layers:
            raise AdapterParseError(f"Adapter '{item.spec.adapter_id}' contains zero LoRA layers.")
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

    layer_sets: list[set[str]] = [set(item.layer_factors) for item in loaded_adapters]
    if layer_policy == "union":
        merged_union: set[str] = set()
        for lset in layer_sets:
            merged_union |= lset
        resolved = sorted(merged_union)
    else:
        merged_inter = set(layer_sets[0])
        for lset in layer_sets[1:]:
            merged_inter &= lset
        resolved = sorted(merged_inter)

    if not resolved:
        raise AdapterParseError(
            f"No layers survived layer_policy='{layer_policy}' across adapters "
            f"{[item.spec.adapter_id for item in loaded_adapters]}."
        )
    return resolved


def _conditioned_core_svd(
    scaled_bs: Sequence[Matrix],
    a_mats: Sequence[Matrix],
    *,
    backend: MathBackend = "auto",
) -> tuple[Matrix, list[float], list[Matrix]]:
    """Conditioned factorized QR + R_sum x R_sum core SVD with zero singular-value squaring."""
    if not scaled_bs or len(scaled_bs) != len(a_mats):
        raise AdapterParseError("scaled_bs and a_mats must be non-empty and equal length.")

    d_out = len(scaled_bs[0])
    d_in = len(a_mats[0][0]) if a_mats[0] else 0
    if d_out <= 0 or d_in <= 0:
        raise AdapterParseError("LoRA matrices must have positive feature dimensions.")

    ranks: list[int] = []
    for idx, (b_mat, a_mat) in enumerate(zip(scaled_bs, a_mats, strict=True)):
        if len(b_mat) != d_out or not b_mat[0]:
            raise AdapterParseError(
                f"Inconsistent d_out at adapter index {idx}: expected {d_out}, got {len(b_mat)}."
            )
        r_b = len(b_mat[0])
        r_a = len(a_mat)
        if r_a != r_b or r_a <= 0:
            raise AdapterParseError(
                f"Mismatched inner rank at adapter index {idx}: B cols={r_b}, A rows={r_a}."
            )
        if any(len(row) != d_in for row in a_mat):
            raise AdapterParseError(f"Inconsistent d_in at adapter index {idx}: expected {d_in}.")
        if any(len(row) != r_b for row in b_mat):
            raise AdapterParseError(f"Ragged row length in scaled B at adapter index {idx}.")
        ranks.append(r_a)

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            b_arrs = [np_mod.asarray(b, dtype=np_mod.float64) for b in scaled_bs]
            a_arrs = [np_mod.asarray(a, dtype=np_mod.float64) for a in a_mats]
            for idx, (b_arr, a_arr) in enumerate(zip(b_arrs, a_arrs, strict=True)):
                if not (bool(np_mod.isfinite(b_arr).all()) and bool(np_mod.isfinite(a_arr).all())):
                    raise AdapterParseError(
                        f"Non-finite (NaN/Inf) entries detected at adapter index {idx}."
                    )

            b_cat = np_mod.hstack(b_arrs)  # [d_out, R_sum]
            q_u, r_u = np_mod.linalg.qr(b_cat, mode="reduced")

            m_blocks: list[Any] = []
            q_a_list: list[Any] = []
            p_a_dims: list[int] = []
            offset = 0
            for a_arr, r_m in zip(a_arrs, ranks, strict=True):
                q_am, r_am = np_mod.linalg.qr(a_arr.T, mode="reduced")
                r_um = r_u[:, offset : offset + r_m]
                m_blocks.append(r_um @ r_am.T)
                q_a_list.append(q_am)
                p_a_dims.append(int(q_am.shape[1]))
                offset += r_m

            m_core = np_mod.hstack(m_blocks)
            w_c, sigmas_arr, vt_cat = np_mod.linalg.svd(m_core, full_matrices=False)
            u_shared_arr = q_u @ w_c

            aligned_v_list: list[Matrix] = []
            p_off = 0
            for q_am, p_am in zip(q_a_list, p_a_dims, strict=True):
                vt_m = vt_cat[:, p_off : p_off + p_am]
                v_m_t_arr = vt_m @ q_am.T
                aligned_v_list.append(cast(Matrix, v_m_t_arr.tolist()))
                p_off += p_am

            return (
                cast(Matrix, u_shared_arr.tolist()),
                [float(s) for s in sigmas_arr.tolist()],
                aligned_v_list,
            )

    for idx, (b_mat, a_mat) in enumerate(zip(scaled_bs, a_mats, strict=True)):
        if has_non_finite_matrix(b_mat) or has_non_finite_matrix(a_mat):
            raise AdapterParseError(
                f"Non-finite (NaN/Inf) entries detected at adapter index {idx}."
            )

    b_cat_py = _hstack_matrices(scaled_bs)
    q_u_py, r_u_py = thin_qr(b_cat_py)

    m_blocks_py: list[Matrix] = []
    q_a_list_py: list[Matrix] = []
    offset_py = 0
    for a_mat, r_m in zip(a_mats, ranks, strict=True):
        q_am_py, r_am_py = thin_qr(transpose_matrix(a_mat))
        r_um_py = _slice_cols(r_u_py, offset_py, offset_py + r_m)
        m_blocks_py.append(matmul_matrices(r_um_py, transpose_matrix(r_am_py)))
        q_a_list_py.append(q_am_py)
        offset_py += r_m

    m_core_py = _hstack_matrices(m_blocks_py)  # [R_sum, R_sum]
    w_c_py, sigmas_full, v_c_py = small_svd(m_core_py)
    vt_cat_py = transpose_matrix(v_c_py)

    total_sum_rank = sum(ranks)
    p_u = min(d_out, total_sum_rank)
    p_a_sum = sum(min(d_in, r_m) for r_m in ranks)
    k_basis = min(p_u, p_a_sum)

    u_full_py = matmul_matrices(q_u_py, w_c_py)
    u_shared_py = [row[:k_basis] for row in u_full_py]
    sigmas_py = sigmas_full[:k_basis]

    aligned_v_py: list[Matrix] = []
    offset_v = 0
    for q_am_py, r_m in zip(q_a_list_py, ranks, strict=True):
        vt_m_py = [vt_cat_py[i][offset_v : offset_v + r_m] for i in range(k_basis)]
        aligned_v_py.append(matmul_matrices(vt_m_py, transpose_matrix(q_am_py)))
        offset_v += r_m

    return u_shared_py, sigmas_py, aligned_v_py


def align_subspaces_knots(
    *,
    layer_name: str,
    adapter_ids: Sequence[str],
    scaled_bs: Sequence[Matrix],
    a_mats: Sequence[Matrix],
    backend: MathBackend = "auto",
) -> FactorizedKnOTSResult:
    """Project K adapters on a layer into shared orthonormal U, sigmas, and aligned V_m^T."""
    if len(adapter_ids) != len(scaled_bs) or len(adapter_ids) != len(a_mats):
        raise ConfigError("adapter_ids, scaled_bs, and a_mats must have matching lengths.")

    u_shared, sigmas, aligned_v_list = _conditioned_core_svd(
        scaled_bs,
        a_mats,
        backend=backend,
    )
    total_sum_rank = sum(len(a_mat) for a_mat in a_mats)
    pre_conflict = _compute_param_sign_conflict(scaled_bs, a_mats)

    return FactorizedKnOTSResult(
        layer_name=layer_name,
        u_shared=u_shared,
        sigmas=sigmas,
        aligned_v_list=aligned_v_list,
        adapter_ids=list(adapter_ids),
        total_sum_rank=total_sum_rank,
        pre_merge_sign_conflict=round(pre_conflict, 6),
    )
