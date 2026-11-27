"""Unit tests for the zero-CDN, XSS-safe offline HTML Certification Card renderer."""

from __future__ import annotations

import re
from pathlib import Path

from loraspec.core.manifest import build_run_manifest
from loraspec.core.schemas import (
    AdapterFormat,
    AuditFinding,
    AuditReport,
    DeploymentRecommendation,
    KnOTSMergeSummary,
    LayerSpectralMetrics,
    LoRAAdapterSpec,
    PairwiseCollisionMetrics,
    Severity,
)
from loraspec.reporting.html_renderer import render_certification_html, write_certification_html


def _sample_report(*, inject_xss: bool = False) -> AuditReport:
    xss_payload = "<script>alert(1)</script>"
    adapter_a_id = f"sql_{xss_payload}" if inject_xss else "sql_lora"
    base_model = f"Qwen/{xss_payload}" if inject_xss else "Qwen/Qwen2.5-3B-Instruct"

    spec_a = LoRAAdapterSpec(
        adapter_id=adapter_a_id,
        adapter_format=AdapterFormat.PEFT_SAFETENSORS,
        root_path=Path("/tmp/sql_lora"),
        weights_path=Path("/tmp/sql_lora/adapter_model.safetensors"),
        config_path=Path("/tmp/sql_lora/adapter_config.json"),
        base_model_name_or_path=base_model,
        rank=16,
        lora_alpha=16.0,
        scaling=1.0,
        total_lora_params=1024,
        weights_sha256="a" * 64,
    )
    spec_b = LoRAAdapterSpec(
        adapter_id="rust_lora",
        adapter_format=AdapterFormat.PEFT_SAFETENSORS,
        root_path=Path("/tmp/rust_lora"),
        weights_path=Path("/tmp/rust_lora/adapter_model.safetensors"),
        config_path=Path("/tmp/rust_lora/adapter_config.json"),
        base_model_name_or_path=base_model,
        rank=16,
        lora_alpha=16.0,
        scaling=1.0,
        total_lora_params=1024,
        weights_sha256="b" * 64,
    )
    manifest = build_run_manifest(
        adapter_sha256s={spec_a.adapter_id: spec_a.weights_sha256, "rust_lora": "b" * 64}
    )
    evidence = f"Detected conflict with {xss_payload}" if inject_xss else "chordal=0.88"
    return AuditReport(
        manifest=manifest,
        adapters=[spec_a, spec_b],
        grade="F",
        merge_risk_score=72.5,
        findings=[
            AuditFinding(
                rule_id="RULE-SPEC-004",
                title="Critical Grassmann collision",
                severity=Severity.CRITICAL,
                tier=1,
                source="loraspec",
                file_path="adapters/sql_lora/adapter_model.safetensors",
                line_number=None,
                evidence=evidence,
                remediation="Run loraspec merge or Top-k MoLoRA.",
            )
        ],
        layer_metrics=[
            LayerSpectralMetrics(
                adapter_id=adapter_a_id,
                layer_name="base_model.model.layers.0.self_attn.q_proj",
                module_type="q_proj",
                rank=16,
                singular_values=[3.0, 2.0, 1.0],
                spectral_norm=3.0,
                frobenius_norm=3.74,
                effective_stable_rank=1.55,
                rank_utilization=0.097,
                top1_energy_ratio=0.64,
                condition_number=3.0,
            ),
            LayerSpectralMetrics(
                adapter_id="rust_lora",
                layer_name="base_model.model.layers.0.mlp.gate_proj",
                module_type="gate_proj",
                rank=16,
                singular_values=[2.0, 2.0, 2.0, 2.0],
                spectral_norm=2.0,
                frobenius_norm=4.0,
                effective_stable_rank=12.0,
                rank_utilization=0.75,
                top1_energy_ratio=0.25,
                condition_number=1.0,
            ),
        ],
        pairwise_collisions=[
            PairwiseCollisionMetrics(
                adapter_a=adapter_a_id,
                adapter_b="rust_lora",
                layer_name="base_model.model.layers.0.self_attn.q_proj",
                active_rank_a=4,
                active_rank_b=4,
                principal_angles_deg=[12.0, 20.0],
                cos_principal_angle_max=0.978,
                grassmann_chordal_similarity=0.88,
                grassmann_geodesic_dist=0.41,
                row_chordal_similarity=0.81,
                sign_conflict_rate=0.65,
            )
        ],
        knots_summary=KnOTSMergeSummary(
            method="knots_ties",
            adapters_merged=[adapter_a_id, "rust_lora"],
            target_rank=16,
            energy_retained=0.962,
            mean_reconstruction_rmse=0.0042,
            sign_conflict_reduction_pct=84.5,
            output_sha256="c" * 64,
        ),
        recommendations=[
            DeploymentRecommendation.USE_TOP_K_MOLORA,
            DeploymentRecommendation.USE_KNOTS_ALIGNMENT,
        ],
    )


def test_html_zero_external_http_urls_and_size_under_250kb(tmp_path: Path) -> None:
    report = _sample_report(inject_xss=False)
    out_file = write_certification_html(report, tmp_path / "nested" / "report.html")
    assert out_file.is_file()

    html = out_file.read_text(encoding="utf-8")
    byte_len = len(html.encode("utf-8"))
    assert byte_len < 250 * 1024

    # Zero external HTTP(S) URLs, external <script src>, <link href>, or CSS @import
    assert not re.search(r"https?://", html, flags=re.IGNORECASE)
    assert not re.search(r"<script[^>]+src=", html, flags=re.IGNORECASE)
    assert not re.search(r"<link[^>]+href=", html, flags=re.IGNORECASE)
    assert "@import" not in html.lower()

    # Core Certification Card visuals & sections present
    assert "<svg" in html
    assert "MergeRiskScore" in html or "MERGE RISK" in html
    assert "Grassmann Chordal Similarity Heatmap" in html
    assert "Effective Rank Utilization" in html
    assert "KnOTS Joint-SVD Merge Diagnostics" in html
    assert "RULE-SPEC-004" in html
    assert "a" * 64 in html


def test_html_escapes_script_tags_in_metadata() -> None:
    report = _sample_report(inject_xss=True)
    html = render_certification_html(report)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
