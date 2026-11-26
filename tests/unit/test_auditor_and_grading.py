"""Unit tests for all Tier 0-1 audit probes (RULE-CONF/SPEC) and deterministic grading."""

from __future__ import annotations

import json
from pathlib import Path

from loraspec.auditor.grading import (
    assign_grade,
    compute_merge_risk_score,
    synthesize_recommendations,
)
from loraspec.auditor.probes import run_spectral_and_collision_audit
from loraspec.core.config import AuditConfig
from loraspec.core.schemas import DeploymentRecommendation
from loraspec.inspect.safetensors_reader import load_peft_adapter


def test_orthogonal_pair_passes_with_grade_a(orthogonal_adapter_pair: tuple[Path, Path]) -> None:
    path_a, path_b = orthogonal_adapter_pair
    loaded = [load_peft_adapter(path_a), load_peft_adapter(path_b)]
    metrics, collisions, findings = run_spectral_and_collision_audit(loaded, AuditConfig())

    assert len(metrics) == 8
    assert len(collisions) == 4
    assert all(c.grassmann_chordal_similarity == 0.0 for c in collisions)
    assert findings == []

    risk = compute_merge_risk_score(findings=findings, layer_metrics=metrics, collisions=collisions)
    assert risk == 0.0
    assert assign_grade(findings=findings, merge_risk_score=risk) == "A"
    assert synthesize_recommendations(findings=findings, collisions=collisions) == [
        DeploymentRecommendation.SAFE_LINEAR_MERGE
    ]


def test_colliding_conflicting_pair_triggers_critical_rule_spec_004(
    colliding_conflicting_pair: tuple[Path, Path],
) -> None:
    path_a, path_b = colliding_conflicting_pair
    loaded = [load_peft_adapter(path_a), load_peft_adapter(path_b)]
    metrics, collisions, findings = run_spectral_and_collision_audit(loaded, AuditConfig())

    rule_ids = {f.rule_id for f in findings}
    assert "RULE-SPEC-004" in rule_ids
    assert all(c.grassmann_chordal_similarity == 1.0 for c in collisions)
    assert all(c.sign_conflict_rate == 1.0 for c in collisions)

    risk = compute_merge_risk_score(findings=findings, layer_metrics=metrics, collisions=collisions)
    assert risk >= 25.0
    assert assign_grade(findings=findings, merge_risk_score=risk) == "F"
    recs = synthesize_recommendations(findings=findings, collisions=collisions)
    assert DeploymentRecommendation.USE_TOP_K_MOLORA in recs


def test_rank_collapse_spike_attention_and_nan_rules(
    rank_collapsed_adapter: Path,
    spiked_attention_only_adapter: Path,
    dead_nan_adapter: Path,
    orthogonal_adapter_pair: tuple[Path, Path],
) -> None:
    # 1. Rank collapse -> RULE-SPEC-002
    collapsed_loaded = [load_peft_adapter(rank_collapsed_adapter)]
    m_col, _, f_col = run_spectral_and_collision_audit(collapsed_loaded, AuditConfig())
    assert {f.rule_id for f in f_col} == {"RULE-SPEC-002"}
    assert all(m.effective_stable_rank == 1.0 for m in m_col)
    recs_col = synthesize_recommendations(findings=f_col, collisions=[])
    assert DeploymentRecommendation.RETRAIN_COLLAPSED_LORA in recs_col

    # 2. Spiked + attention-only -> RULE-CONF-003 + RULE-SPEC-003
    spiked_loaded = [load_peft_adapter(spiked_attention_only_adapter)]
    _, _, f_sp = run_spectral_and_collision_audit(spiked_loaded, AuditConfig())
    assert {"RULE-CONF-003", "RULE-SPEC-003"}.issubset({f.rule_id for f in f_sp})

    # 3. Dead / NaN weights -> RULE-SPEC-001 (CRITICAL, Grade F)
    nan_loaded = [load_peft_adapter(dead_nan_adapter)]
    m_nan, _, f_nan = run_spectral_and_collision_audit(nan_loaded, AuditConfig())
    assert "RULE-SPEC-001" in {f.rule_id for f in f_nan}
    risk_nan = compute_merge_risk_score(findings=f_nan, layer_metrics=m_nan, collisions=[])
    assert assign_grade(findings=f_nan, merge_risk_score=risk_nan) == "F"

    # 4. Base model mismatch -> RULE-CONF-001
    path_a, path_b = orthogonal_adapter_pair
    cfg_b = json.loads((path_b / "adapter_config.json").read_text(encoding="utf-8"))
    cfg_b["base_model_name_or_path"] = "meta-llama/Llama-3.1-8B-Instruct"
    (path_b / "adapter_config.json").write_text(json.dumps(cfg_b), encoding="utf-8")
    _, _, f_mismatch = run_spectral_and_collision_audit(
        [load_peft_adapter(path_a), load_peft_adapter(path_b)],
        AuditConfig(),
    )
    assert "RULE-CONF-001" in {f.rule_id for f in f_mismatch}


def test_heterogeneous_rank_and_missing_config_rules(
    orthogonal_adapter_pair: tuple[Path, Path],
    rank_collapsed_adapter: Path,
) -> None:
    path_a, _ = orthogonal_adapter_pair
    # Remove adapter_config.json from collapsed adapter -> RULE-CONF-002 + RULE-CONF-004 (r=4 vs 8)
    (rank_collapsed_adapter / "adapter_config.json").unlink()
    loaded = [load_peft_adapter(path_a), load_peft_adapter(rank_collapsed_adapter)]
    _, _, findings = run_spectral_and_collision_audit(loaded, AuditConfig())
    rule_ids = {f.rule_id for f in findings}
    assert "RULE-CONF-002" in rule_ids
    assert "RULE-CONF-004" in rule_ids


def test_grade_boundaries_zero_alpha_and_disjoint_layers(
    orthogonal_adapter_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    from loraspec.core.schemas import AuditFinding, Severity

    # 1. Exact Grade A / B / C / D / F transitions
    med = AuditFinding(
        rule_id="RULE-CONF-004",
        title="Mixed ranks",
        severity=Severity.MEDIUM,
        tier=0,
        source="config_probe",
        evidence="4 vs 8",
        remediation="Use KnOTS.",
    )
    high = AuditFinding(
        rule_id="RULE-SPEC-002",
        title="Collapse",
        severity=Severity.HIGH,
        tier=1,
        source="loraspec",
        evidence="r_eff=1",
        remediation="Retrain.",
    )
    assert assign_grade(findings=[], merge_risk_score=0.0) == "A"
    assert assign_grade(findings=[med], merge_risk_score=7.5) == "B"
    assert assign_grade(findings=[med], merge_risk_score=20.0) == "C"
    assert assign_grade(findings=[high], merge_risk_score=40.0) == "D"
    assert assign_grade(findings=[high], merge_risk_score=65.0) == "F"

    # 2. Non-positive lora_alpha -> RULE-CONF-002
    path_a, path_b = orthogonal_adapter_pair
    cfg_a = json.loads((path_a / "adapter_config.json").read_text(encoding="utf-8"))
    cfg_a["lora_alpha"] = 0.0
    (path_a / "adapter_config.json").write_text(json.dumps(cfg_a), encoding="utf-8")
    m_zero_alpha, _, f_zero_alpha = run_spectral_and_collision_audit(
        [load_peft_adapter(path_a)], AuditConfig()
    )
    assert any(f.rule_id == "RULE-CONF-002" for f in f_zero_alpha)
    assert all(m.frobenius_norm == 0.0 for m in m_zero_alpha)
    assert any(f.rule_id == "RULE-SPEC-001" for f in f_zero_alpha)

    # 3. Disjoint layers across two adapters -> 0 pairwise collisions without error
    from loraspec.inspect.safetensors_reader import write_safetensors

    disjoint_dir = tmp_path / "disjoint"
    disjoint_dir.mkdir()
    (disjoint_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                "r": 4,
                "lora_alpha": 4.0,
                "target_modules": ["k_proj", "gate_proj"],
            }
        ),
        encoding="utf-8",
    )
    d = 16
    r = 4
    write_safetensors(
        disjoint_dir / "adapter_model.safetensors",
        {
            "base_model.model.layers.9.self_attn.k_proj.lora_A.weight": (
                [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)],
                "F32",
            ),
            "base_model.model.layers.9.self_attn.k_proj.lora_B.weight": (
                [[1.0 if row == col else 0.0 for col in range(r)] for row in range(d)],
                "F32",
            ),
        },
    )
    _, dis_cols, dis_findings = run_spectral_and_collision_audit(
        [load_peft_adapter(path_b), load_peft_adapter(disjoint_dir)],
        AuditConfig(),
    )
    assert dis_cols == []
    assert not any(f.rule_id == "RULE-SPEC-004" for f in dis_findings)


def test_tiny_norm_non_dead_unsupported_peft_and_rank_mismatch(tmp_path: Path) -> None:
    from loraspec.inspect.safetensors_reader import write_safetensors

    # 1. Tiny non-zero norm (4e-9 > 1e-12) must NOT round to 0.0 and trigger RULE-SPEC-001
    tiny_dir = tmp_path / "tiny_nonzero"
    tiny_dir.mkdir()
    (tiny_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                "r": 2,
                "lora_alpha": 2.0,
                "target_modules": ["q_proj", "gate_proj"],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        tiny_dir / "adapter_model.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[4e-9, 0.0], [0.0, 0.0]], "F32"),
            "layer0.gate_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.gate_proj.lora_B.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
        },
    )
    m_tiny, _, f_tiny = run_spectral_and_collision_audit(
        [load_peft_adapter(tiny_dir)], AuditConfig()
    )
    assert m_tiny[1].frobenius_norm > 1e-12
    assert not any(f.rule_id == "RULE-SPEC-001" for f in f_tiny)

    # 2. Unsupported peft_type + missing lora_alpha + config rank mismatch -> RULE-CONF-002
    bad_cfg_dir = tmp_path / "bad_cfg"
    bad_cfg_dir.mkdir()
    (bad_cfg_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "IA3",
                "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
                "r": 8,
                "target_modules": ["q_proj", "gate_proj"],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        bad_cfg_dir / "adapter_model.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
        },
    )
    _, _, f_bad_cfg = run_spectral_and_collision_audit(
        [load_peft_adapter(bad_cfg_dir)], AuditConfig(), run_tier1=False
    )
    conf_002_titles = [f.title for f in f_bad_cfg if f.rule_id == "RULE-CONF-002"]
    assert len(conf_002_titles) == 3

    # 3. RULE-CONF-001 file_path points to conflicting config even with raw adapter in middle
    raw_mid = tmp_path / "raw_mid"
    raw_mid.mkdir()
    write_safetensors(
        raw_mid / "adapter_model.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
        },
    )
    llama_dir = tmp_path / "llama_end"
    llama_dir.mkdir()
    (llama_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
                "r": 2,
                "lora_alpha": 2.0,
                "target_modules": ["q_proj", "gate_proj"],
            }
        ),
        encoding="utf-8",
    )
    write_safetensors(
        llama_dir / "adapter_model.safetensors",
        {
            "layer0.q_proj.lora_A.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
            "layer0.q_proj.lora_B.weight": ([[1.0, 0.0], [0.0, 1.0]], "F32"),
        },
    )
    _, _, f_three = run_spectral_and_collision_audit(
        [load_peft_adapter(tiny_dir), load_peft_adapter(raw_mid), load_peft_adapter(llama_dir)],
        AuditConfig(),
        run_tier1=False,
    )
    c001 = [f for f in f_three if f.rule_id == "RULE-CONF-001"]
    assert c001 and c001[0].file_path is not None and "llama_end" in c001[0].file_path
