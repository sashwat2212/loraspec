"""KnOTS rank truncation, factorized RMSE diagnostics, and PEFT export serializer."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from loraspec.core.errors import AdapterParseError, ConfigError
from loraspec.core.manifest import build_run_manifest, sha256_file
from loraspec.core.schemas import AuditFinding, KnOTSMergeSummary, LoRAAdapterSpec, Severity
from loraspec.core.spectral_math import (
    MathBackend,
    Matrix,
    small_svd,
    thin_qr,
    validate_math_backend,
)
from loraspec.inspect.safetensors_reader import (
    LoadedLoRAWeights,
    get_layer_scaling,
    strip_peft_base_prefix,
    write_safetensors,
)
from loraspec.knots.aligner import (
    LayerPolicy,
    align_subspaces_knots,
    harmonize_heterogeneous_ranks,
    matmul_matrices,
    optional_numpy,
    scale_matrix,
    transpose_matrix,
)
from loraspec.knots.operators import (
    KnotsMethod,
    apply_genknowsub,
    merge_aligned_factors,
    normalize_weights,
)

ExportDtype = Literal["F32", "F16", "BF16"]


def reconstruct_merged_lora(
    u_shared: Matrix,
    sigmas: Sequence[float],
    v_merged_t: Matrix,
    *,
    target_rank: int,
    backend: MathBackend = "auto",
) -> tuple[Matrix, Matrix, float]:
    """Compress U * diag(sigmas) * V_merged^T back to rank r_out without dense d_out x d_in."""
    backend = validate_math_backend(backend)
    if isinstance(target_rank, bool) or target_rank < 1:
        raise ConfigError(f"target_rank must be an integer >= 1, got {target_rank!r}.")
    if not u_shared or not u_shared[0] or not v_merged_t or not v_merged_t[0]:
        raise AdapterParseError("Cannot reconstruct merged LoRA from empty matrices.")

    d_out = len(u_shared)
    k_basis = len(u_shared[0])
    d_in = len(v_merged_t[0])
    if len(sigmas) != k_basis or len(v_merged_t) != k_basis:
        raise AdapterParseError(
            f"Dimension mismatch in reconstruct_merged_lora: U cols={k_basis}, "
            f"sigmas={len(sigmas)}, V_merged_T rows={len(v_merged_t)}."
        )

    r_out = min(target_rank, k_basis, d_in, d_out)

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            u_arr = np_mod.asarray(u_shared, dtype=np_mod.float64)  # [d_out, k_basis]
            s_arr = np_mod.asarray(sigmas, dtype=np_mod.float64)  # [k_basis]
            vm_t_arr = np_mod.asarray(v_merged_t, dtype=np_mod.float64)  # [k_basis, d_in]

            q_v, r_v = np_mod.linalg.qr(vm_t_arr.T, mode="reduced")  # q_v: [d_in, p_v]
            core = s_arr[:, None] * r_v.T  # [k_basis, p_v]
            u_tilde, sigma_tilde, vt_tilde = np_mod.linalg.svd(core, full_matrices=False)

            sigma_list = [float(x) for x in sigma_tilde.tolist()]
            total_energy = sum(x * x for x in sigma_list)
            if total_energy <= 1e-24:
                b_zero: Matrix = [[0.0] * r_out for _ in range(d_out)]
                a_zero: Matrix = [[0.0] * d_in for _ in range(r_out)]
                return b_zero, a_zero, 1.0

            kept_energy = sum(x * x for x in sigma_list[:r_out])
            energy_retained = min(1.0, kept_energy / total_energy)

            sqrt_s = np_mod.sqrt(np_mod.maximum(0.0, sigma_tilde[:r_out]))
            b_out_arr = (u_arr @ u_tilde[:, :r_out]) * sqrt_s[None, :]
            a_out_arr = sqrt_s[:, None] * (vt_tilde[:r_out, :] @ q_v.T)
            return (
                cast(Matrix, b_out_arr.tolist()),
                cast(Matrix, a_out_arr.tolist()),
                energy_retained,
            )

    q_v_py, r_v_py = thin_qr(transpose_matrix(v_merged_t))  # q_v: [d_in, k_basis]
    r_v_t_py = transpose_matrix(r_v_py)
    core_py: Matrix = [[sigmas[i] * val for val in r_v_t_py[i]] for i in range(k_basis)]
    u_tilde_py, sigma_tilde_py, v_tilde_py = small_svd(core_py)
    vt_tilde_py = transpose_matrix(v_tilde_py)

    total_energy_py = sum(s * s for s in sigma_tilde_py)
    if total_energy_py <= 1e-24:
        b_zero_py: Matrix = [[0.0] * r_out for _ in range(d_out)]
        a_zero_py: Matrix = [[0.0] * d_in for _ in range(r_out)]
        return b_zero_py, a_zero_py, 1.0

    kept_energy_py = sum(s * s for s in sigma_tilde_py[:r_out])
    energy_retained_py = min(1.0, kept_energy_py / total_energy_py)

    sqrt_s_py = [math.sqrt(max(0.0, s)) for s in sigma_tilde_py[:r_out]]
    u_tilde_k = [row[:r_out] for row in u_tilde_py]
    u_rot = matmul_matrices(u_shared, u_tilde_k)  # [d_out, r_out]
    b_out_py: Matrix = [
        [u_rot[row_idx][col_idx] * sqrt_s_py[col_idx] for col_idx in range(r_out)]
        for row_idx in range(d_out)
    ]

    vt_tilde_k = vt_tilde_py[:r_out]  # [r_out, k_basis]
    v_rot_t = matmul_matrices(vt_tilde_k, transpose_matrix(q_v_py))  # [r_out, d_in]
    a_out_py: Matrix = [
        [sqrt_s_py[row_idx] * val for val in v_rot_t[row_idx]] for row_idx in range(r_out)
    ]
    return b_out_py, a_out_py, energy_retained_py


def _sum_elementwise_prod(mat_p: Matrix, mat_q: Matrix) -> float:
    total = 0.0
    for row_p, row_q in zip(mat_p, mat_q, strict=True):
        for vp, vq in zip(row_p, row_q, strict=True):
            total += vp * vq
    return total


def factorized_frobenius_rmse(
    b_out: Matrix,
    a_out: Matrix,
    b_src: Matrix,
    a_src: Matrix,
    *,
    scaling_src: float = 1.0,
    backend: MathBackend = "auto",
) -> float:
    """Compute exact Frobenius RMSE ||B_out A_out - s B_src A_src||_F / sqrt(d_out * d_in)
    via trace inner products in rank-r space (zero dense d_out x d_in allocations).
    """
    backend = validate_math_backend(backend)
    d_out = len(b_out)
    d_in = len(a_out[0]) if a_out else 0
    if d_out == 0 or d_in == 0:
        return 0.0

    if backend in ("auto", "numpy"):
        np_mod = optional_numpy()
        if np_mod is not None:
            bx = np_mod.asarray(b_out, dtype=np_mod.float64)
            ax = np_mod.asarray(a_out, dtype=np_mod.float64)
            by = np_mod.asarray(b_src, dtype=np_mod.float64) * scaling_src
            ay = np_mod.asarray(a_src, dtype=np_mod.float64)

            norm_x_sq = float(np_mod.sum((bx.T @ bx) * (ax @ ax.T)))
            norm_y_sq = float(np_mod.sum((by.T @ by) * (ay @ ay.T)))
            cross = float(np_mod.sum((bx.T @ by) * (ax @ ay.T)))
            diff_sq = max(0.0, norm_x_sq + norm_y_sq - 2.0 * cross)
            return math.sqrt(diff_sq / float(d_out * d_in))

    by_py = scale_matrix(b_src, scaling_src) if scaling_src != 1.0 else b_src
    bx_t = transpose_matrix(b_out)
    by_t = transpose_matrix(by_py)
    ax_t = transpose_matrix(a_out)
    ay_t = transpose_matrix(a_src)

    norm_x_sq_py = _sum_elementwise_prod(matmul_matrices(bx_t, b_out), matmul_matrices(a_out, ax_t))
    norm_y_sq_py = _sum_elementwise_prod(matmul_matrices(by_t, by_py), matmul_matrices(a_src, ay_t))
    cross_py = _sum_elementwise_prod(matmul_matrices(bx_t, by_py), matmul_matrices(a_out, ay_t))
    diff_sq_py = max(0.0, norm_x_sq_py + norm_y_sq_py - 2.0 * cross_py)
    return math.sqrt(diff_sq_py / float(d_out * d_in))


def compute_merge_diagnostics(
    *,
    method: KnotsMethod,
    adapter_ids: Sequence[str],
    target_rank: int,
    layer_energies: dict[str, float],
    layer_rmses: dict[str, float],
    pre_conflicts: Sequence[float],
    post_conflicts: Sequence[float],
    output_sha256: str,
    min_energy_retention: float = 0.90,
    manifest_file_path: str | None = None,
    zero_energy_layers: Sequence[str] = (),
) -> tuple[KnOTSMergeSummary, list[AuditFinding]]:
    """Summarize KnOTS singular energy retention, RMSE, conflict reduction, and RULE-KNOTS-001."""
    if not layer_energies:
        raise ConfigError("layer_energies cannot be empty when computing merge diagnostics.")

    mean_energy = sum(layer_energies.values()) / float(len(layer_energies))
    mean_rmse = sum(layer_rmses.values()) / float(len(layer_rmses)) if layer_rmses else 0.0

    mean_pre = sum(pre_conflicts) / float(len(pre_conflicts)) if pre_conflicts else 0.0
    mean_post = sum(post_conflicts) / float(len(post_conflicts)) if post_conflicts else 0.0
    if mean_pre > 1e-12:
        raw_reduction = ((mean_pre - mean_post) / mean_pre) * 100.0
        if -1e-9 <= raw_reduction < 0.0:
            raw_reduction = 0.0
        reduction_pct = max(-100.0, min(100.0, raw_reduction))
    else:
        reduction_pct = 0.0

    findings: list[AuditFinding] = []
    if mean_energy < min_energy_retention:
        low_layers = sorted(
            lname for lname, eta in layer_energies.items() if eta < min_energy_retention
        )
        findings.append(
            AuditFinding(
                rule_id="RULE-KNOTS-001",
                title="KnOTS merged rank loses excessive singular energy",
                severity=Severity.MEDIUM,
                tier=1,
                source="loraspec",
                file_path=manifest_file_path,
                line_number=None,
                evidence=(
                    f"Mean retained singular energy {mean_energy:.4f} < threshold "
                    f"{min_energy_retention:.4f} at target_rank={target_rank} "
                    f"({len(low_layers)}/{len(layer_energies)} layers below threshold)."
                ),
                remediation=(
                    "Increase --target-rank or use dynamic Per-Token Top-k MoLoRA routing "
                    "instead of static rank compression."
                ),
            )
        )
    elif zero_energy_layers and len(zero_energy_layers) == len(layer_energies):
        findings.append(
            AuditFinding(
                rule_id="RULE-KNOTS-001",
                title="KnOTS merge yielded zero update energy",
                severity=Severity.MEDIUM,
                tier=1,
                source="loraspec",
                file_path=manifest_file_path,
                line_number=None,
                evidence=(
                    f"All {len(zero_energy_layers)} merged layer(s) produced zero spectral energy "
                    "(energy_retained=1.0)."
                ),
                remediation=(
                    "Inspect conflicting sign cancellation, merge weights, or --gks-lambda."
                ),
            )
        )

    summary = KnOTSMergeSummary(
        method=method,
        adapters_merged=list(adapter_ids),
        target_rank=target_rank,
        energy_retained=round(mean_energy, 6),
        mean_reconstruction_rmse=round(mean_rmse, 8),
        sign_conflict_reduction_pct=round(reduction_pct, 4),
        output_sha256=output_sha256,
    )
    return summary, findings


def export_peft_merged_adapter(
    *,
    out_dir: Path,
    merged_factors: dict[str, tuple[Matrix, Matrix]],
    base_model_name_or_path: str | None,
    target_modules: Sequence[str],
    dtype: ExportDtype = "F32",
    method: KnotsMethod,
    adapter_ids: Sequence[str],
    layer_energies: dict[str, float],
    layer_rmses: dict[str, float],
    pre_conflicts: Sequence[float],
    post_conflicts: Sequence[float],
    source_sha256s: dict[str, str],
    min_energy_retention: float = 0.90,
    rng_seed: int = 42,
    cli_args: Sequence[str] | None = None,
    zero_energy_layers: Sequence[str] = (),
) -> tuple[KnOTSMergeSummary, list[AuditFinding]]:
    """Serialize merged LoRA factors as PEFT adapter + knots_merge_manifest.json."""
    if not merged_factors:
        raise ConfigError("Cannot export empty merged_factors dictionary.")

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_to_write: dict[str, tuple[Matrix, str]] = {}
    layer_ranks: dict[str, int] = {}

    for lname in sorted(merged_factors):
        a_out, b_out = merged_factors[lname]
        layer_ranks[lname] = len(a_out)
        a_key = "lora_A.weight" if lname == "__root__" else f"{lname}.lora_A.weight"
        b_key = "lora_B.weight" if lname == "__root__" else f"{lname}.lora_B.weight"
        tensors_to_write[a_key] = (a_out, dtype)
        tensors_to_write[b_key] = (b_out, dtype)

    weights_path = out_dir / "adapter_model.safetensors"
    write_safetensors(
        weights_path,
        tensors_to_write,
        metadata={"format": "pt", "merged_by": "loraspec_knots"},
    )
    output_sha256 = sha256_file(weights_path)

    primary_rank = max(layer_ranks.values())
    peft_config: dict[str, object] = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_name_or_path,
        "r": primary_rank,
        "lora_alpha": float(primary_rank),
        "lora_dropout": 0.0,
        "use_rslora": False,
        "use_dora": False,
        "bias": "none",
        "target_modules": sorted(set(target_modules)),
    }
    if len(set(layer_ranks.values())) > 1:
        stripped_counts: dict[str, int] = {}
        for lname in layer_ranks:
            stripped_key = strip_peft_base_prefix(lname)
            stripped_counts[stripped_key] = stripped_counts.get(stripped_key, 0) + 1

        rank_pattern: dict[str, int] = {}
        alpha_pattern: dict[str, float] = {}
        for lname, rk in sorted(layer_ranks.items()):
            if rk == primary_rank:
                continue
            stripped_key = strip_peft_base_prefix(lname)
            pattern_key = stripped_key if stripped_counts[stripped_key] == 1 else lname
            rank_pattern[pattern_key] = rk
            alpha_pattern[pattern_key] = float(rk)

        peft_config["rank_pattern"] = rank_pattern
        peft_config["alpha_pattern"] = alpha_pattern

    (out_dir / "adapter_config.json").write_text(
        json.dumps(peft_config, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = out_dir / "knots_merge_manifest.json"
    summary, findings = compute_merge_diagnostics(
        method=method,
        adapter_ids=adapter_ids,
        target_rank=primary_rank,
        layer_energies=layer_energies,
        layer_rmses=layer_rmses,
        pre_conflicts=pre_conflicts,
        post_conflicts=post_conflicts,
        output_sha256=output_sha256,
        min_energy_retention=min_energy_retention,
        manifest_file_path=str(manifest_path.resolve()),
        zero_energy_layers=zero_energy_layers,
    )
    run_manifest = build_run_manifest(
        adapter_sha256s=source_sha256s,
        root_hint=out_dir,
        rng_seed=rng_seed,
        total_gpu_cost_inr=0.0,
        cli_args=list(cli_args) if cli_args is not None else [],
    )

    manifest_payload = {
        "schema_version": "1.0.0",
        "summary": summary.model_dump(),
        "manifest": run_manifest.model_dump(),
        "findings": [f.model_dump() for f in findings],
        "layer_energy_retained": layer_energies,
        "layer_reconstruction_rmse": layer_rmses,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return summary, findings


def _reject_non_finite_json(constant: str) -> object:
    raise ValueError(f"Non-finite JSON constant '{constant}' is not allowed.")


def load_verified_knots_summary(spec: LoRAAdapterSpec) -> KnOTSMergeSummary | None:
    """Load `knots_merge_manifest.json` iff valid and `output_sha256 == spec.weights_sha256`."""
    manifest_path = spec.root_path / "knots_merge_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        raw: object = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_json,
        )
        if not isinstance(raw, dict):
            return None
        raw_map = cast(dict[str, object], raw)
        summary_obj = raw_map.get("summary")
        if not isinstance(summary_obj, dict):
            return None
        summary = KnOTSMergeSummary.model_validate(summary_obj)
    except (OSError, UnicodeDecodeError, ValueError, ValidationError):
        return None

    if not (
        math.isfinite(summary.energy_retained)
        and math.isfinite(summary.mean_reconstruction_rmse)
        and math.isfinite(summary.sign_conflict_reduction_pct)
    ):
        return None
    if summary.output_sha256 != spec.weights_sha256:
        return None
    return summary


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


def merge_knots_adapters(
    loaded_adapters: Sequence[LoadedLoRAWeights],
    *,
    out_dir: Path,
    method: KnotsMethod = "knots_ties",
    target_rank: int = 16,
    weights: Sequence[float] | None = None,
    ties_density: float = 0.50,
    dare_drop_rate: float = 0.20,
    layer_policy: LayerPolicy = "union",
    gks_adapter: LoadedLoRAWeights | None = None,
    gks_lambda: float = 1.0,
    dtype: ExportDtype = "F32",
    rng_seed: int = 42,
    min_energy_retention: float = 0.90,
    cli_args: Sequence[str] | None = None,
    backend: MathBackend = "auto",
) -> tuple[KnOTSMergeSummary, list[AuditFinding]]:
    """Run complete factorized KnOTS merge pipeline and export PEFT artifacts."""
    backend = validate_math_backend(backend)
    if isinstance(target_rank, bool) or target_rank < 1:
        raise ConfigError(f"target_rank must be an integer >= 1, got {target_rank!r}.")
    if method not in {"knots_ties", "knots_dare_ties", "knots_linear"}:
        raise ConfigError(f"Unsupported KnOTS merge method '{method}'.")
    if dtype not in {"F32", "F16", "BF16"}:
        raise ConfigError(f"Unsupported export dtype '{dtype}'.")

    deduped = _deduplicate_loaded_ids(loaded_adapters)
    norm_weights = normalize_weights(len(deduped), weights)

    if gks_adapter is not None:
        working = apply_genknowsub(deduped, gks_adapter, gks_lambda=gks_lambda)
    else:
        working = list(deduped)

    layers = harmonize_heterogeneous_ranks(working, layer_policy=layer_policy)
    base_models = [
        item.spec.base_model_name_or_path
        for item in deduped
        if item.spec.base_model_name_or_path is not None
    ]
    shared_base = base_models[0] if base_models else None

    target_modules: set[str] = set()
    merged_factors: dict[str, tuple[Matrix, Matrix]] = {}
    layer_energies: dict[str, float] = {}
    layer_rmses: dict[str, float] = {}
    pre_conflicts: list[float] = []
    post_conflicts: list[float] = []
    zero_energy_layers: list[str] = []

    for layer_idx, lname in enumerate(layers):
        participating = [
            (idx, item) for idx, item in enumerate(working) if lname in item.layer_factors
        ]
        if not participating:
            continue

        for _, item in participating:
            for lspec in item.spec.layers:
                if lspec.layer_name == lname:
                    target_modules.add(lspec.module_type)

        sub_ids = [item.spec.adapter_id for _, item in participating]
        sub_weights_raw = [norm_weights[idx] for idx, _ in participating]
        sub_weight_sum = sum(sub_weights_raw)

        if len(participating) == 1:
            idx_only, only_item = participating[0]
            a_only, b_only = only_item.layer_factors[lname]
            layer_scale = get_layer_scaling(only_item.spec, lname)
            if norm_weights[idx_only] == 0.0:
                eff_scale = 0.0
            elif method == "knots_linear":
                eff_scale = layer_scale * norm_weights[idx_only]
            else:
                eff_scale = layer_scale
            aligned = align_subspaces_knots(
                layer_name=lname,
                adapter_ids=sub_ids,
                scaled_bs=[scale_matrix(b_only, eff_scale)],
                a_mats=[a_only],
                backend=backend,
            )
            v_merged_t = aligned.aligned_v_list[0]
            post_conflict = 0.0
        else:
            sub_weights = (
                [w / sub_weight_sum for w in sub_weights_raw]
                if sub_weight_sum > 0.0
                else [1.0 / float(len(participating))] * len(participating)
            )
            scaled_bs = [
                scale_matrix(item.layer_factors[lname][1], get_layer_scaling(item.spec, lname))
                for _, item in participating
            ]
            a_mats = [item.layer_factors[lname][0] for _, item in participating]
            aligned = align_subspaces_knots(
                layer_name=lname,
                adapter_ids=sub_ids,
                scaled_bs=scaled_bs,
                a_mats=a_mats,
                backend=backend,
            )
            v_merged_t, post_conflict = merge_aligned_factors(
                aligned,
                method=method,
                weights=sub_weights,
                ties_density=ties_density,
                dare_drop_rate=dare_drop_rate,
                rng_seed=rng_seed,
                layer_seed_offset=layer_idx,
                backend=backend,
            )

        b_out, a_out, energy = reconstruct_merged_lora(
            aligned.u_shared,
            aligned.sigmas,
            v_merged_t,
            target_rank=target_rank,
            backend=backend,
        )
        max_out_val = max(
            (abs(val) for row in b_out for val in row),
            default=0.0,
        )
        if max_out_val <= 1e-12:
            zero_energy_layers.append(lname)

        merged_factors[lname] = (a_out, b_out)
        layer_energies[lname] = energy
        pre_conflicts.append(aligned.pre_merge_sign_conflict)
        post_conflicts.append(post_conflict)

        # Weighted mean factorized RMSE across participating source adapters
        rmse_acc = 0.0
        w_acc = 0.0
        for idx, item in participating:
            a_src, b_src = item.layer_factors[lname]
            w_m = norm_weights[idx]
            rmse_m = factorized_frobenius_rmse(
                b_out,
                a_out,
                b_src,
                a_src,
                scaling_src=get_layer_scaling(item.spec, lname),
                backend=backend,
            )
            rmse_acc += w_m * rmse_m
            w_acc += w_m
        layer_rmses[lname] = rmse_acc / w_acc if w_acc > 0.0 else 0.0

    source_shas = {item.spec.adapter_id: item.spec.weights_sha256 for item in deduped}
    return export_peft_merged_adapter(
        out_dir=out_dir,
        merged_factors=merged_factors,
        base_model_name_or_path=shared_base,
        target_modules=sorted(target_modules),
        dtype=dtype,
        method=method,
        adapter_ids=[item.spec.adapter_id for item in deduped],
        layer_energies=layer_energies,
        layer_rmses=layer_rmses,
        pre_conflicts=pre_conflicts,
        post_conflicts=post_conflicts,
        source_sha256s=source_shas,
        min_energy_retention=min_energy_retention,
        rng_seed=rng_seed,
        cli_args=cli_args,
        zero_energy_layers=zero_energy_layers,
    )
