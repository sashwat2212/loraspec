"""Static config parity & factorized spectral/Grassmann collision audit rules."""

from __future__ import annotations

from statistics import median

from loraspec.core.config import AuditConfig
from loraspec.core.schemas import (
    AuditFinding,
    LayerSpectralMetrics,
    LoRAAdapterSpec,
    PairwiseCollisionMetrics,
    Severity,
)
from loraspec.core.spectral_math import (
    LayerSubspaceFactors,
    compute_layer_spectral_analysis,
    compute_pairwise_collision,
)
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights

MLP_MODULE_MARKERS: frozenset[str] = frozenset(
    {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"}
)


def _rel_path(spec: LoRAAdapterSpec, use_config: bool = False) -> str:
    target = (
        spec.config_path if (use_config and spec.config_path is not None) else spec.weights_path
    )
    return str(target)


def run_config_probes(
    adapters: list[LoRAAdapterSpec],
    cfg: AuditConfig,
) -> list[AuditFinding]:
    """Run Tier 0 PEFT config, base-model parity, scaling, and placement checks."""
    findings: list[AuditFinding] = []

    for spec in adapters:
        if spec.metadata.get("config_missing"):
            findings.append(
                AuditFinding(
                    rule_id="RULE-CONF-002",
                    title="Missing adapter_config.json for PEFT LoRA checkpoint",
                    severity=Severity.HIGH,
                    tier=0,
                    source="config_probe",
                    file_path=_rel_path(spec),
                    line_number=None,
                    evidence=(
                        f"Adapter '{spec.adapter_id}' loaded raw .safetensors without "
                        "adapter_config.json; scaling defaulted to alpha=rank."
                    ),
                    remediation=(
                        "Include a valid PEFT adapter_config.json pinning "
                        "base_model_name_or_path, r, lora_alpha, and target_modules."
                    ),
                )
            )
        elif spec.lora_alpha <= 0.0:
            findings.append(
                AuditFinding(
                    rule_id="RULE-CONF-002",
                    title="Non-positive LoRA scaling alpha",
                    severity=Severity.HIGH,
                    tier=0,
                    source="config_probe",
                    file_path=_rel_path(spec, use_config=True),
                    line_number=1,
                    evidence=(
                        f"Adapter '{spec.adapter_id}' sets lora_alpha={spec.lora_alpha} "
                        f"(scaling={spec.scaling:.4f})."
                    ),
                    remediation="Set lora_alpha > 0 in adapter_config.json.",
                )
            )

        if cfg.require_all_linear:
            modules_present = {layer.module_type for layer in spec.layers} | set(
                spec.target_modules
            )
            if not (modules_present & MLP_MODULE_MARKERS):
                findings.append(
                    AuditFinding(
                        rule_id="RULE-CONF-003",
                        title="Attention-only LoRA placement omits MLP projections",
                        severity=Severity.HIGH,
                        tier=0,
                        source="config_probe",
                        file_path=_rel_path(spec, use_config=True),
                        line_number=1,
                        evidence=(
                            f"Adapter '{spec.adapter_id}' targets only {sorted(modules_present)}; "
                            "MLP gate/up/down projections are absent."
                        ),
                        remediation=(
                            "Apply LoRA to all linear layers (attention + MLP) per "
                            "LoRA Without Regret before multi-adapter composition."
                        ),
                    )
                )

    if len(adapters) >= 2:
        base_models = {
            a.base_model_name_or_path for a in adapters if a.base_model_name_or_path is not None
        }
        if len(base_models) > 1:
            findings.append(
                AuditFinding(
                    rule_id="RULE-CONF-001",
                    title="Base model mismatch across LoRA adapter fleet",
                    severity=Severity.CRITICAL,
                    tier=0,
                    source="config_probe",
                    file_path=_rel_path(adapters[1], use_config=True),
                    line_number=1,
                    evidence=f"Conflicting base_model_name_or_path values: {sorted(base_models)}.",
                    remediation=(
                        "Ensure every adapter being merged or routed shares the exact same "
                        "base model checkpoint."
                    ),
                )
            )

        # Check shape parity on shared layer names
        layer_dims: dict[str, tuple[int, int]] = {}
        for spec in adapters:
            for layer in spec.layers:
                dims = (layer.in_features, layer.out_features)
                prev = layer_dims.get(layer.layer_name)
                if prev is not None and prev != dims:
                    findings.append(
                        AuditFinding(
                            rule_id="RULE-CONF-001",
                            title="Tensor dimension mismatch across adapters on shared layer",
                            severity=Severity.CRITICAL,
                            tier=0,
                            source="config_probe",
                            file_path=_rel_path(spec),
                            line_number=None,
                            evidence=(
                                f"Layer '{layer.layer_name}' in '{spec.adapter_id}' has "
                                f"(in={dims[0]}, out={dims[1]}), expected {prev}."
                            ),
                            remediation="Exclude mismatched architecture adapters from the fleet.",
                        )
                    )
                    break
                layer_dims[layer.layer_name] = dims

        ranks = {a.rank for a in adapters}
        rslora_modes = {a.use_rslora for a in adapters}
        if len(ranks) > 1 or len(rslora_modes) > 1:
            findings.append(
                AuditFinding(
                    rule_id="RULE-CONF-004",
                    title="Heterogeneous ranks or scaling modes across adapter fleet",
                    severity=Severity.MEDIUM,
                    tier=0,
                    source="config_probe",
                    file_path=_rel_path(adapters[0], use_config=True),
                    line_number=1,
                    evidence=(
                        f"Fleet mixes ranks={sorted(ranks)} and use_rslora={sorted(rslora_modes)}."
                    ),
                    remediation=(
                        "Use KnOTS joint-SVD alignment (`loraspec merge`) or ragged "
                        "Top-k Arrow-SGMV routing (`MoLoRA`) instead of naive linear averaging."
                    ),
                )
            )

    return findings


def run_spectral_and_collision_audit(
    loaded: list[LoadedLoRAWeights],
    cfg: AuditConfig,
) -> tuple[
    list[LayerSpectralMetrics],
    list[PairwiseCollisionMetrics],
    list[AuditFinding],
]:
    """Compute factorized QR+SVD health, Grassmann collisions, and emit RULE-* findings."""
    specs = [item.spec for item in loaded]
    findings = run_config_probes(specs, cfg)

    all_layer_metrics: list[LayerSpectralMetrics] = []
    factors_by_adapter: dict[str, dict[str, LayerSubspaceFactors]] = {}
    spec_by_id: dict[str, LoRAAdapterSpec] = {item.spec.adapter_id: item.spec for item in loaded}

    for item in loaded:
        spec = item.spec
        adapter_metrics: list[LayerSpectralMetrics] = []
        adapter_factors: dict[str, LayerSubspaceFactors] = {}

        for layer_spec in spec.layers:
            mat_a, mat_b = item.layer_factors[layer_spec.layer_name]
            lmetric, lfactor = compute_layer_spectral_analysis(
                adapter_id=spec.adapter_id,
                layer_spec=layer_spec,
                a_mat=mat_a,
                b_mat=mat_b,
                scaling=spec.scaling,
            )
            adapter_metrics.append(lmetric)
            adapter_factors[layer_spec.layer_name] = lfactor
            all_layer_metrics.append(lmetric)

        factors_by_adapter[spec.adapter_id] = adapter_factors

        # RULE-SPEC-001: NaN/Inf or Dead Zero Layer
        corrupt_layers = [
            m.layer_name for m in adapter_metrics if m.has_non_finite or m.frobenius_norm <= 1e-12
        ]
        if corrupt_layers:
            findings.append(
                AuditFinding(
                    rule_id="RULE-SPEC-001",
                    title="Non-finite (NaN/Inf) or dead zero LoRA layer weights detected",
                    severity=Severity.CRITICAL,
                    tier=1,
                    source="loraspec",
                    file_path=str(spec.weights_path),
                    line_number=None,
                    evidence=(
                        f"Adapter '{spec.adapter_id}' has dead/non-finite weights on "
                        f"{len(corrupt_layers)} layer(s), e.g. '{corrupt_layers[0]}'."
                    ),
                    remediation=(
                        "Quarantine corrupted checkpoint or inspect training gradient spikes."
                    ),
                )
            )

        # RULE-SPEC-002: Rank Collapse
        valid_ge4 = [m for m in adapter_metrics if m.rank >= 4 and not m.has_non_finite]
        if valid_ge4:
            mean_util = sum(m.rank_utilization for m in valid_ge4) / float(len(valid_ge4))
            high_top1_frac = sum(
                1 for m in valid_ge4 if m.top1_energy_ratio > cfg.top1_energy_threshold
            ) / float(len(valid_ge4))
            if mean_util < cfg.rank_collapse_ratio or high_top1_frac >= 0.30:
                findings.append(
                    AuditFinding(
                        rule_id="RULE-SPEC-002",
                        title="Severe LoRA rank collapse detected",
                        severity=Severity.HIGH,
                        tier=1,
                        source="loraspec",
                        file_path=str(spec.weights_path),
                        line_number=None,
                        evidence=(
                            f"Adapter '{spec.adapter_id}': mean rank utilization="
                            f"{mean_util:.3f} (threshold {cfg.rank_collapse_ratio:.2f}), "
                            f"{high_top1_frac * 100:.1f}% layers exceed "
                            f"{cfg.top1_energy_threshold * 100:.0f}% top-1 energy."
                        ),
                        remediation=(
                            "Truncate redundant rank via SVD compression or retrain with "
                            "higher LR / all-linear placement."
                        ),
                    )
                )

        # RULE-SPEC-003: Spectral Norm Spike
        nonzero_norms = [m.spectral_norm for m in adapter_metrics if m.spectral_norm > 0.0]
        if nonzero_norms:
            med_norm = median(nonzero_norms)
            spike_limit = max(10.0, cfg.spectral_spike_multiplier * med_norm)
            spiked = [m for m in adapter_metrics if m.spectral_norm > spike_limit]
            if spiked:
                worst = max(spiked, key=lambda m: m.spectral_norm)
                findings.append(
                    AuditFinding(
                        rule_id="RULE-SPEC-003",
                        title="Spectral norm spike detected in LoRA layer",
                        severity=Severity.HIGH,
                        tier=1,
                        source="loraspec",
                        file_path=str(spec.weights_path),
                        line_number=None,
                        evidence=(
                            f"Adapter '{spec.adapter_id}' layer '{worst.layer_name}' has "
                            f"||dW||_2={worst.spectral_norm:.4f} > threshold {spike_limit:.4f} "
                            f"(median={med_norm:.4f})."
                        ),
                        remediation=(
                            "Clamp or re-scale spiked layers before merging to prevent logit drift."
                        ),
                    )
                )

    # Pairwise Grassmann collision analysis
    collisions: list[PairwiseCollisionMetrics] = []
    adapter_ids = [item.spec.adapter_id for item in loaded]
    for i in range(len(adapter_ids)):
        for j in range(i + 1, len(adapter_ids)):
            id_a = adapter_ids[i]
            id_b = adapter_ids[j]
            fa_map = factors_by_adapter[id_a]
            fb_map = factors_by_adapter[id_b]
            shared_layers = sorted(set(fa_map) & set(fb_map))
            if not shared_layers:
                continue

            pair_metrics: list[PairwiseCollisionMetrics] = []
            for lname in shared_layers:
                fa = fa_map[lname]
                fb = fb_map[lname]
                # Skip dimension-mismatched pairs (already caught by RULE-CONF-001)
                if (
                    fa.u_active
                    and fb.u_active
                    and (
                        len(fa.u_active) != len(fb.u_active) or len(fa.v_active) != len(fb.v_active)
                    )
                ):
                    continue
                pm = compute_pairwise_collision(fa, fb)
                pair_metrics.append(pm)
                collisions.append(pm)

            if not pair_metrics:
                continue

            crit_hits = [
                pm
                for pm in pair_metrics
                if pm.grassmann_chordal_similarity >= cfg.collision_crit_chordal
                and pm.sign_conflict_rate >= cfg.sign_conflict_threshold
            ]
            high_hits = [
                pm
                for pm in pair_metrics
                if pm.grassmann_chordal_similarity >= cfg.collision_high_chordal
                or (pm.cos_principal_angle_max >= 0.85 and pm.sign_conflict_rate >= 0.25)
            ]

            crit_frac = len(crit_hits) / float(len(pair_metrics))
            if crit_frac >= 0.20:
                worst = max(crit_hits, key=lambda p: p.grassmann_chordal_similarity)
                findings.append(
                    AuditFinding(
                        rule_id="RULE-SPEC-004",
                        title="Critical Grassmann subspace collision with sign conflict",
                        severity=Severity.CRITICAL,
                        tier=1,
                        source="loraspec",
                        file_path=str(spec_by_id[id_a].weights_path),
                        line_number=None,
                        evidence=(
                            f"Pair ('{id_a}', '{id_b}'): {len(crit_hits)}/{len(pair_metrics)} "
                            f"shared layers collide severely (worst '{worst.layer_name}': "
                            f"chordal={worst.grassmann_chordal_similarity:.3f}, "
                            f"sign_conflict={worst.sign_conflict_rate:.3f})."
                        ),
                        remediation=(
                            "Do NOT linear/TIES merge blindly. Run `loraspec merge` (KnOTS "
                            "joint-SVD alignment) or deploy Per-Token Top-k `MoLoRA`."
                        ),
                    )
                )
            elif high_hits:
                worst = max(high_hits, key=lambda p: p.grassmann_chordal_similarity)
                findings.append(
                    AuditFinding(
                        rule_id="RULE-SPEC-004",
                        title="High Grassmann subspace overlap across adapter pair",
                        severity=Severity.HIGH,
                        tier=1,
                        source="loraspec",
                        file_path=str(spec_by_id[id_a].weights_path),
                        line_number=None,
                        evidence=(
                            f"Pair ('{id_a}', '{id_b}'): {len(high_hits)}/{len(pair_metrics)} "
                            f"layers show high subspace overlap (worst '{worst.layer_name}': "
                            f"chordal={worst.grassmann_chordal_similarity:.3f}, "
                            f"cos_max={worst.cos_principal_angle_max:.3f}, "
                            f"sign_conflict={worst.sign_conflict_rate:.3f})."
                        ),
                        remediation=(
                            "Align subspaces with `KnOTS` joint-SVD (`loraspec merge`) or "
                            "route dynamically per token (`MoLoRA`)."
                        ),
                    )
                )

    return all_layer_metrics, collisions, findings
