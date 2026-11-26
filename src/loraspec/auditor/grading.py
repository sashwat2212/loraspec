"""Deterministic MergeRiskScore (0-100), Grade (A-F), and deployment policy scorer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from loraspec.core.schemas import (
    AuditFinding,
    DeploymentRecommendation,
    LayerSpectralMetrics,
    PairwiseCollisionMetrics,
    Severity,
)


def compute_merge_risk_score(
    *,
    findings: Sequence[AuditFinding],
    layer_metrics: Sequence[LayerSpectralMetrics],
    collisions: Sequence[PairwiseCollisionMetrics],
) -> float:
    """Compute bounded MergeRiskScore in [0.0, 100.0] per SDD-01 Section 3.2."""
    has_crit_conf = any(
        f.rule_id == "RULE-CONF-001" and f.severity == Severity.CRITICAL for f in findings
    )
    has_nan_or_dead = any(
        f.rule_id == "RULE-SPEC-001" and f.severity == Severity.CRITICAL for f in findings
    )
    i_crit_conf_or_nan = 1.0 if (has_crit_conf or has_nan_or_dead) else 0.0

    if collisions:
        # Weighted combination of max/mean chordal collision amplified by sign conflict
        max_col = max(
            p.grassmann_chordal_similarity * (0.5 + 0.5 * p.sign_conflict_rate) for p in collisions
        )
        mean_col = sum(
            p.grassmann_chordal_similarity * (0.5 + 0.5 * p.sign_conflict_rate) for p in collisions
        ) / float(len(collisions))
        c_collision = min(1.0, 0.7 * max_col + 0.3 * mean_col)
        if any(f.rule_id == "RULE-SPEC-004" and f.severity == Severity.CRITICAL for f in findings):
            c_collision = max(c_collision, 0.9)
    else:
        c_collision = 0.0

    has_collapse = any(f.rule_id == "RULE-SPEC-002" for f in findings)
    if has_collapse:
        valid_utils = [m.rank_utilization for m in layer_metrics if m.rank >= 4]
        min_util = min(valid_utils) if valid_utils else 0.0
        r_collapse = min(1.0, max(0.5, 1.0 - min_util))
    else:
        r_collapse = 0.0

    high_or_med_cfg = [
        f
        for f in findings
        if f.rule_id in {"RULE-CONF-002", "RULE-CONF-003", "RULE-CONF-004", "RULE-SPEC-003"}
    ]
    s_spike_or_config = min(1.0, 0.5 * len(high_or_med_cfg))

    score = (
        35.0 * i_crit_conf_or_nan
        + 30.0 * c_collision
        + 20.0 * r_collapse
        + 15.0 * s_spike_or_config
    )
    return round(min(100.0, max(0.0, score)), 2)


def assign_grade(
    *,
    findings: Sequence[AuditFinding],
    merge_risk_score: float,
) -> Literal["A", "B", "C", "D", "F"]:
    """Deterministic Grade A-F mapping per SDD-01 Section 3.2."""
    has_critical = any(f.severity == Severity.CRITICAL for f in findings)
    has_high = any(f.severity == Severity.HIGH for f in findings)

    if has_critical or merge_risk_score > 60.0:
        return "F"
    if has_high or merge_risk_score > 35.0:
        return "D"
    if merge_risk_score > 15.0:
        return "C"
    if findings or merge_risk_score > 0.0:
        return "B"
    return "A"


def synthesize_recommendations(
    *,
    findings: Sequence[AuditFinding],
    collisions: Sequence[PairwiseCollisionMetrics],
) -> list[DeploymentRecommendation]:
    """Synthesize ordered deployment policy recommendations from findings and collisions."""
    recs: list[DeploymentRecommendation] = []
    rule_ids = {f.rule_id for f in findings}

    if "RULE-CONF-001" in rule_ids or "RULE-SPEC-001" in rule_ids:
        recs.append(DeploymentRecommendation.QUARANTINE_CORRUPT_ADAPTER)

    if "RULE-SPEC-002" in rule_ids:
        recs.append(DeploymentRecommendation.RETRAIN_COLLAPSED_LORA)

    has_crit_collision = any(
        f.rule_id == "RULE-SPEC-004" and f.severity == Severity.CRITICAL for f in findings
    )
    has_high_collision = any(f.rule_id == "RULE-SPEC-004" for f in findings)

    if has_crit_collision:
        recs.append(DeploymentRecommendation.USE_TOP_K_MOLORA)
        recs.append(DeploymentRecommendation.USE_KNOTS_ALIGNMENT)
    elif has_high_collision or "RULE-CONF-004" in rule_ids:
        recs.append(DeploymentRecommendation.USE_KNOTS_ALIGNMENT)
        if any(p.grassmann_chordal_similarity >= 0.45 for p in collisions):
            recs.append(DeploymentRecommendation.USE_TOP_K_MOLORA)

    if not recs:
        recs.append(DeploymentRecommendation.SAFE_LINEAR_MERGE)

    return recs
