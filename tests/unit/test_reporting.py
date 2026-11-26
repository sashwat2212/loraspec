"""Unit tests for JSON, RunManifest, and SARIF 2.1.0 report writers."""

from __future__ import annotations

import json
from pathlib import Path

from loraspec import __version__
from loraspec.core.manifest import build_run_manifest
from loraspec.core.schemas import (
    AuditFinding,
    AuditReport,
    DeploymentRecommendation,
    Severity,
)
from loraspec.inspect.safetensors_reader import load_peft_adapter
from loraspec.reporting.json_writer import write_manifest_json, write_report_json
from loraspec.reporting.sarif_writer import (
    LORASPEC_INFO_URI,
    SARIF_SCHEMA_URI,
    build_sarif_dict,
    write_sarif,
)


def test_json_and_sarif_writers(
    orthogonal_adapter_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    path_a, _ = orthogonal_adapter_pair
    spec_a = load_peft_adapter(path_a).spec
    manifest = build_run_manifest(
        adapter_sha256s={spec_a.adapter_id: spec_a.weights_sha256},
        root_hint=path_a,
    )
    findings = [
        AuditFinding(
            rule_id="RULE-SPEC-004",
            title="Critical Grassmann collision",
            severity=Severity.CRITICAL,
            tier=1,
            source="loraspec",
            file_path="adapter_model.safetensors",
            line_number=None,
            evidence="chordal=0.95",
            remediation="Use KnOTS alignment.",
        ),
        AuditFinding(
            rule_id="RULE-CONF-004",
            title="Mixed ranks",
            severity=Severity.MEDIUM,
            tier=0,
            source="config_probe",
            file_path="./adapter_config.json",
            line_number=1,
            evidence="ranks=[4, 8]",
            remediation="Use KnOTS or Arrow-SGMV.",
        ),
        AuditFinding(
            rule_id="RULE-INFO-001",
            title="Advisory note",
            severity=Severity.INFO,
            tier=0,
            source="loraspec",
            file_path=".env",
            line_number=None,
            evidence="note",
            remediation="none",
        ),
    ]
    report = AuditReport(
        manifest=manifest,
        adapters=[spec_a],
        grade="F",
        merge_risk_score=65.0,
        findings=findings,
        recommendations=[DeploymentRecommendation.USE_TOP_K_MOLORA],
    )

    rep_path = write_report_json(report, tmp_path / "out" / "report.json")
    man_path = write_manifest_json(manifest, tmp_path / "out" / "manifest.json")
    sarif_path = write_sarif(report, tmp_path / "out" / "report.sarif")

    assert AuditReport.model_validate_json(rep_path.read_text(encoding="utf-8")) == report
    assert json.loads(man_path.read_text(encoding="utf-8"))["run_id"] == manifest.run_id

    sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
    assert sarif["$schema"] == SARIF_SCHEMA_URI
    assert sarif["version"] == "2.1.0"
    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["name"] == "loraspec"
    assert driver["semanticVersion"] == __version__
    assert driver["informationUri"] == LORASPEC_INFO_URI
    assert [r["id"] for r in driver["rules"]] == [
        "RULE-SPEC-004",
        "RULE-CONF-004",
        "RULE-INFO-001",
    ]

    results = sarif["runs"][0]["results"]
    assert [r["level"] for r in results] == ["error", "warning", "note"]
    assert [r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for r in results] == [
        "adapter_model.safetensors",
        "adapter_config.json",
        ".env",
    ]


def test_sarif_repo_relative_uris_inside_git_checkout(
    orthogonal_adapter_pair: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    path_a, _ = orthogonal_adapter_pair
    nested_dir = tmp_path / "adapters" / "sql_lora"
    nested_dir.mkdir(parents=True)
    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        (nested_dir / fname).write_bytes((path_a / fname).read_bytes())

    spec = load_peft_adapter(nested_dir).spec
    manifest = build_run_manifest(adapter_sha256s={spec.adapter_id: spec.weights_sha256})
    report = AuditReport(
        manifest=manifest,
        adapters=[spec],
        grade="F",
        merge_risk_score=40.0,
        findings=[
            AuditFinding(
                rule_id="RULE-SPEC-002",
                title="Rank collapse",
                severity=Severity.HIGH,
                tier=1,
                source="loraspec",
                file_path=str(spec.weights_path),
                line_number=None,
                evidence="r_eff=1.0",
                remediation="Truncate rank.",
            ),
            AuditFinding(
                rule_id="RULE-CONF-003",
                title="Attention only",
                severity=Severity.HIGH,
                tier=0,
                source="config_probe",
                file_path=None,
                line_number=None,
                evidence="only q_proj",
                remediation="Use all-linear.",
            ),
        ],
    )
    sarif = build_sarif_dict(report)
    uris = [
        r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for r in sarif["runs"][0]["results"]
    ]
    assert uris == ["adapters/sql_lora/adapter_model.safetensors", "adapters/sql_lora"]
