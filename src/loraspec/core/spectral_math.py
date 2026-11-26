"""Factorized rank-r QR+SVD spectral math & Grassmann principal-angle engine.

Operates strictly in O((d_out + d_in) * r) space — never materializes full dense
d_out x d_in weight matrices. Uses pure-Python Modified Gram-Schmidt + Hestenes
one-sided Jacobi SVD offline, or NumPy BLAS/LAPACK fast-path when available.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

from loraspec.core.schemas import LayerSpectralMetrics, LoRALayerSpec, PairwiseCollisionMetrics

Matrix = list[list[float]]
MathBackend = Literal["auto", "numpy", "pure_python"]


def _optional_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


@dataclass(frozen=True)
class LayerSubspaceFactors:
    adapter_id: str
    layer_name: str
    singular_values: list[float]
    u_active: Matrix  # [d_out, k_active] orthonormal columns
    v_active: Matrix  # [d_in, k_active] orthonormal columns
    active_sigmas: list[float]
    has_non_finite: bool


def _has_non_finite_matrix(mat: Matrix) -> bool:
    return any(not all(map(math.isfinite, row)) for row in mat)


def _transpose(mat: Matrix) -> Matrix:
    if not mat or not mat[0]:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
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


def _scale_matrix(mat: Matrix, scalar: float) -> Matrix:
    return [[val * scalar for val in row] for row in mat]


def thin_qr(mat: Matrix, *, tol: float = 1e-12) -> tuple[Matrix, Matrix]:
    """Modified Gram-Schmidt (2-pass re-orthogonalized) thin QR for [m, n] (m >= n or any)."""
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    if rows == 0 or cols == 0:
        return ([[] for _ in range(rows)], [[0.0] * cols for _ in range(cols)])
    q_cols: list[list[float]] = []
    r_mat: Matrix = [[0.0] * cols for _ in range(cols)]

    for j in range(cols):
        v = [mat[i][j] for i in range(rows)]
        for pass_idx in range(2):
            for i in range(j):
                qi = q_cols[i]
                dot = sum(a * b for a, b in zip(qi, v, strict=True))
                if pass_idx == 0:
                    r_mat[i][j] = dot
                else:
                    r_mat[i][j] += dot
                if dot != 0.0:
                    v = [vx - dot * qx for vx, qx in zip(v, qi, strict=True)]
        norm = math.sqrt(sum(x * x for x in v))
        r_mat[j][j] = norm
        if norm > tol:
            inv = 1.0 / norm
            q_cols.append([x * inv for x in v])
        else:
            q_cols.append([0.0] * rows)

    q_mat: Matrix = [[q_cols[j][i] for j in range(cols)] for i in range(rows)]
    return q_mat, r_mat


def small_svd(
    mat: Matrix, *, max_sweeps: int = 40, tol: float = 1e-12
) -> tuple[Matrix, list[float], Matrix]:
    """Exact Hestenes one-sided Jacobi SVD for small [m, n] matrices.

    Returns (U [m, n], sigmas [n], V [n, n]) such that mat == U @ diag(sigmas) @ V^T.
    """
    m = len(mat)
    n = len(mat[0]) if m else 0
    if m == 0 or n == 0:
        return [], [], []

    # Work on column vectors W = copy(mat) of shape [m, n]
    cols_w: list[list[float]] = [[mat[i][j] for i in range(m)] for j in range(n)]
    cols_v: list[list[float]] = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]

    for _sweep in range(max_sweeps):
        converged = True
        for p in range(n - 1):
            for q in range(p + 1, n):
                wp = cols_w[p]
                wq = cols_w[q]
                alpha = sum(x * x for x in wp)
                beta = sum(x * x for x in wq)
                gamma = sum(x * y for x, y in zip(wp, wq, strict=True))
                denom = math.sqrt(alpha * beta)
                if denom <= 1e-30:
                    continue
                if abs(gamma) > tol * denom:
                    converged = False
                    zeta = (beta - alpha) / (2.0 * gamma)
                    t = (1.0 if zeta >= 0.0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                    c = 1.0 / math.sqrt(1.0 + t * t)
                    s = c * t
                    for i in range(m):
                        w_pi = wp[i]
                        w_qi = wq[i]
                        wp[i] = c * w_pi - s * w_qi
                        wq[i] = s * w_pi + c * w_qi
                    vp = cols_v[p]
                    vq = cols_v[q]
                    for i in range(n):
                        v_pi = vp[i]
                        v_qi = vq[i]
                        vp[i] = c * v_pi - s * v_qi
                        vq[i] = s * v_pi + c * v_qi
        if converged:
            break

    norms = [math.sqrt(sum(x * x for x in col)) for col in cols_w]
    order = sorted(range(n), key=lambda idx: norms[idx], reverse=True)

    sigmas: list[float] = []
    u_cols: list[list[float]] = []
    v_cols_sorted: list[list[float]] = []

    for idx in order:
        sig = norms[idx]
        sigmas.append(sig)
        v_cols_sorted.append(cols_v[idx])
        if sig > tol:
            inv = 1.0 / sig
            u_cols.append([x * inv for x in cols_w[idx]])
        else:
            u_cols.append([0.0] * m)

    u_mat: Matrix = [[u_cols[j][i] for j in range(n)] for i in range(m)]
    v_mat: Matrix = [[v_cols_sorted[j][i] for j in range(n)] for i in range(n)]
    return u_mat, sigmas, v_mat


def _sync_orthonormal_bases(
    u_raw: Matrix,
    v_raw: Matrix,
    sigmas_raw: list[float],
    *,
    tol: float = 1e-9,
) -> tuple[Matrix, Matrix, list[float]]:
    """Orthonormalize U and V while keeping 1-to-1 column index correspondence with sigmas."""
    if not sigmas_raw or not u_raw or not v_raw or not u_raw[0] or not v_raw[0]:
        return [], [], []
    q_u, r_u = thin_qr(u_raw, tol=tol)
    q_v, r_v = thin_qr(v_raw, tol=tol)
    valid = [j for j in range(len(sigmas_raw)) if r_u[j][j] > tol and r_v[j][j] > tol]
    if not valid:
        return [], [], []
    if len(valid) == len(sigmas_raw):
        return q_u, q_v, sigmas_raw
    # Re-run thin QR strictly on synchronized surviving columns
    u_sub = [[row[j] for j in valid] for row in u_raw]
    v_sub = [[row[j] for j in valid] for row in v_raw]
    q_u_sub, _ = thin_qr(u_sub, tol=tol)
    q_v_sub, _ = thin_qr(v_sub, tol=tol)
    return q_u_sub, q_v_sub, [sigmas_raw[j] for j in valid]


def _numpy_layer_analysis(
    *,
    np_mod: Any,
    adapter_id: str,
    layer_spec: LoRALayerSpec,
    a_mat: Matrix,
    b_mat: Matrix,
    scaling: float,
) -> tuple[LayerSpectralMetrics, LayerSubspaceFactors]:
    a_arr = np_mod.asarray(a_mat, dtype=np_mod.float64)
    b_arr = np_mod.asarray(b_mat, dtype=np_mod.float64)
    if not (bool(np_mod.isfinite(a_arr).all()) and bool(np_mod.isfinite(b_arr).all())):
        return _non_finite_result(adapter_id=adapter_id, layer_spec=layer_spec)

    b_scaled = b_arr * scaling
    q_b, r_b = np_mod.linalg.qr(b_scaled, mode="reduced")
    q_a, r_a = np_mod.linalg.qr(a_arr.T, mode="reduced")
    core = r_b @ r_a.T
    u_c, sigmas_arr, vt_c = np_mod.linalg.svd(core, full_matrices=False)
    v_c = vt_c.T

    sigmas: list[float] = [float(x) for x in sigmas_arr.tolist()]
    sigma_1 = sigmas[0] if sigmas else 0.0
    frob_sq = sum(s * s for s in sigmas)
    frob_norm = math.sqrt(frob_sq)

    if sigma_1 > 1e-12:
        eff_rank = frob_sq / (sigma_1 * sigma_1)
        top1_energy = (sigma_1 * sigma_1) / frob_sq if frob_sq > 0.0 else 0.0
        min_active = max(sigmas[-1], 1e-12)
        cond_num = sigma_1 / min_active
    else:
        eff_rank = 0.0
        top1_energy = 0.0
        cond_num = 0.0
    rank_util = eff_rank / float(layer_spec.rank) if layer_spec.rank > 0 else 0.0

    active_cutoff = max(1e-9, sigma_1 * 1e-6)
    active_indices = [idx for idx, s in enumerate(sigmas) if s > active_cutoff]
    if active_indices:
        u_active_arr = q_b @ u_c[:, active_indices]
        v_active_arr = q_a @ v_c[:, active_indices]
        u_active = cast(Matrix, u_active_arr.tolist())
        v_active = cast(Matrix, v_active_arr.tolist())
        active_sigmas = [sigmas[idx] for idx in active_indices]
    else:
        u_active = []
        v_active = []
        active_sigmas = []

    metrics = LayerSpectralMetrics(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        module_type=layer_spec.module_type,
        rank=layer_spec.rank,
        singular_values=[round(s, 12) for s in sigmas],
        spectral_norm=round(sigma_1, 12),
        frobenius_norm=round(frob_norm, 12),
        effective_stable_rank=round(eff_rank, 6),
        rank_utilization=round(rank_util, 6),
        top1_energy_ratio=round(top1_energy, 6),
        condition_number=round(cond_num, 4),
        has_non_finite=False,
    )
    factors = LayerSubspaceFactors(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        singular_values=sigmas,
        u_active=u_active,
        v_active=v_active,
        active_sigmas=active_sigmas,
        has_non_finite=False,
    )
    return metrics, factors


def _non_finite_result(
    *,
    adapter_id: str,
    layer_spec: LoRALayerSpec,
) -> tuple[LayerSpectralMetrics, LayerSubspaceFactors]:
    metrics = LayerSpectralMetrics(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        module_type=layer_spec.module_type,
        rank=layer_spec.rank,
        singular_values=[0.0] * layer_spec.rank,
        spectral_norm=0.0,
        frobenius_norm=0.0,
        effective_stable_rank=0.0,
        rank_utilization=0.0,
        top1_energy_ratio=0.0,
        condition_number=0.0,
        has_non_finite=True,
    )
    factors = LayerSubspaceFactors(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        singular_values=[0.0] * layer_spec.rank,
        u_active=[],
        v_active=[],
        active_sigmas=[],
        has_non_finite=True,
    )
    return metrics, factors


def compute_layer_spectral_analysis(
    *,
    adapter_id: str,
    layer_spec: LoRALayerSpec,
    a_mat: Matrix,
    b_mat: Matrix,
    scaling: float,
    backend: MathBackend = "auto",
) -> tuple[LayerSpectralMetrics, LayerSubspaceFactors]:
    """Compute exact rank-r singular values, stable rank, and active orthonormal bases."""
    if backend in ("auto", "numpy"):
        np_mod = _optional_numpy()
        if np_mod is not None:
            return _numpy_layer_analysis(
                np_mod=np_mod,
                adapter_id=adapter_id,
                layer_spec=layer_spec,
                a_mat=a_mat,
                b_mat=b_mat,
                scaling=scaling,
            )

    if _has_non_finite_matrix(a_mat) or _has_non_finite_matrix(b_mat):
        return _non_finite_result(adapter_id=adapter_id, layer_spec=layer_spec)

    b_scaled = _scale_matrix(b_mat, scaling)
    a_t = _transpose(a_mat)

    q_b, r_b = thin_qr(b_scaled)
    q_a, r_a = thin_qr(a_t)
    core = _matmul(r_b, _transpose(r_a))  # [r, r]
    u_c, sigmas, v_c = small_svd(core)

    u_full = _matmul(q_b, u_c)  # [d_out, r]
    v_full = _matmul(q_a, v_c)  # [d_in, r]

    sigma_1 = sigmas[0] if sigmas else 0.0
    frob_sq = sum(s * s for s in sigmas)
    frob_norm = math.sqrt(frob_sq)

    if sigma_1 > 1e-12:
        eff_rank = frob_sq / (sigma_1 * sigma_1)
        top1_energy = (sigma_1 * sigma_1) / frob_sq if frob_sq > 0.0 else 0.0
        min_active = max(sigmas[-1], 1e-12)
        cond_num = sigma_1 / min_active
    else:
        eff_rank = 0.0
        top1_energy = 0.0
        cond_num = 0.0

    rank_util = eff_rank / float(layer_spec.rank) if layer_spec.rank > 0 else 0.0

    active_cutoff = max(1e-9, sigma_1 * 1e-6)
    active_indices = [idx for idx, s in enumerate(sigmas) if s > active_cutoff]
    u_active_raw = [[row[idx] for idx in active_indices] for row in u_full]
    v_active_raw = [[row[idx] for idx in active_indices] for row in v_full]
    active_sigmas_raw = [sigmas[idx] for idx in active_indices]

    # Ensure synchronized numerical orthonormality across U, V, and sigmas
    u_active, v_active, active_sigmas = _sync_orthonormal_bases(
        u_active_raw, v_active_raw, active_sigmas_raw
    )

    metrics = LayerSpectralMetrics(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        module_type=layer_spec.module_type,
        rank=layer_spec.rank,
        singular_values=[round(s, 12) for s in sigmas],
        spectral_norm=round(sigma_1, 12),
        frobenius_norm=round(frob_norm, 12),
        effective_stable_rank=round(eff_rank, 6),
        rank_utilization=round(rank_util, 6),
        top1_energy_ratio=round(top1_energy, 6),
        condition_number=round(cond_num, 4),
        has_non_finite=False,
    )
    factors = LayerSubspaceFactors(
        adapter_id=adapter_id,
        layer_name=layer_spec.layer_name,
        singular_values=sigmas,
        u_active=u_active,
        v_active=v_active,
        active_sigmas=active_sigmas,
        has_non_finite=False,
    )
    return metrics, factors


def compute_pairwise_collision(
    fa: LayerSubspaceFactors,
    fb: LayerSubspaceFactors,
    *,
    backend: MathBackend = "auto",
) -> PairwiseCollisionMetrics:
    """Compute Grassmann principal angles, chordal/geodesic distance, and sign conflict."""
    k_a = len(fa.u_active[0]) if fa.u_active and fa.u_active[0] else 0
    k_b = len(fb.u_active[0]) if fb.u_active and fb.u_active[0] else 0
    k_va = len(fa.v_active[0]) if fa.v_active and fa.v_active[0] else 0
    k_vb = len(fb.v_active[0]) if fb.v_active and fb.v_active[0] else 0

    if fa.has_non_finite or fb.has_non_finite or k_a == 0 or k_b == 0 or k_va == 0 or k_vb == 0:
        return PairwiseCollisionMetrics(
            adapter_a=fa.adapter_id,
            adapter_b=fb.adapter_id,
            layer_name=fa.layer_name,
            active_rank_a=k_a,
            active_rank_b=k_b,
            principal_angles_deg=[],
            cos_principal_angle_max=0.0,
            grassmann_chordal_similarity=0.0,
            grassmann_geodesic_dist=0.0,
            row_chordal_similarity=0.0,
            sign_conflict_rate=0.0,
        )

    np_mod = _optional_numpy() if backend in ("auto", "numpy") else None
    if np_mod is not None:
        ua = np_mod.asarray(fa.u_active, dtype=np_mod.float64)
        ub = np_mod.asarray(fb.u_active, dtype=np_mod.float64)
        va = np_mod.asarray(fa.v_active, dtype=np_mod.float64)
        vb = np_mod.asarray(fb.v_active, dtype=np_mod.float64)
        m_u_arr = ua.T @ ub
        m_v_arr = va.T @ vb
        s_u = [float(x) for x in np_mod.linalg.svd(m_u_arr, compute_uv=False).tolist()]
        s_v = [float(x) for x in np_mod.linalg.svd(m_v_arr, compute_uv=False).tolist()]
        m_u = cast(Matrix, m_u_arr.tolist())
        m_v = cast(Matrix, m_v_arr.tolist())
    else:
        # Column-space principal cosines: singular values of U_a^T U_b
        m_u = _matmul(_transpose(fa.u_active), fb.u_active)  # [k_a, k_b]
        _, s_u, _ = small_svd(m_u)
        # Row-space principal cosines: singular values of V_a^T V_b
        m_v = _matmul(_transpose(fa.v_active), fb.v_active)  # [k_va, k_vb]
        _, s_v, _ = small_svd(m_v)

    p_col = min(k_a, k_b)
    cos_u = [min(1.0, max(0.0, s)) for s in s_u[:p_col]]
    angles_rad = [math.acos(c) for c in cos_u]
    angles_deg = [round(math.degrees(a), 4) for a in angles_rad]
    chordal_col = sum(c * c for c in cos_u) / float(p_col)
    geodesic_col = math.sqrt(sum(a * a for a in angles_rad))

    p_row = min(k_va, k_vb)
    cos_v = [min(1.0, max(0.0, s)) for s in s_v[:p_row]]
    chordal_row = sum(c * c for c in cos_v) / float(p_row)

    # Gauge-invariant Frobenius cancellation ratio over active triplets
    neg_weight = 0.0
    total_abs_weight = 0.0
    rows_m = min(len(m_u), len(m_v), len(fa.active_sigmas))
    cols_m = min(len(m_u[0]) if m_u else 0, len(m_v[0]) if m_v else 0, len(fb.active_sigmas))
    for i in range(rows_m):
        sa = fa.active_sigmas[i]
        for j in range(cols_m):
            sb = fb.active_sigmas[j]
            signed_overlap = sa * sb * m_u[i][j] * m_v[i][j]
            total_abs_weight += abs(signed_overlap)
            if signed_overlap < 0.0:
                neg_weight += -signed_overlap

    sign_conflict_rate = neg_weight / total_abs_weight if total_abs_weight > 1e-12 else 0.0

    return PairwiseCollisionMetrics(
        adapter_a=fa.adapter_id,
        adapter_b=fb.adapter_id,
        layer_name=fa.layer_name,
        active_rank_a=k_a,
        active_rank_b=k_b,
        principal_angles_deg=angles_deg,
        cos_principal_angle_max=round(cos_u[0] if cos_u else 0.0, 6),
        grassmann_chordal_similarity=round(chordal_col, 6),
        grassmann_geodesic_dist=round(geodesic_col, 6),
        row_chordal_similarity=round(chordal_row, 6),
        sign_conflict_rate=round(sign_conflict_rate, 6),
    )
